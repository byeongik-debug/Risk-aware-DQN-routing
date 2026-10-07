# risk_dqn_every_scenario.py
# Multi-Scenario DQN Agent with Action Masking

import os
import random
import numpy as np
from collections import deque

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from network_env import NetworkEnv
from topology import DESTINATION, NUM_NODES


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {
    "normal": 0.25,
    "congestion": 0.25,
    "unstable": 0.25,
    "dynamic": 0.25
}

EPISODES = 8000
GAMMA = 0.99
LR = 0.0003
BATCH_SIZE = 64
MEMORY_SIZE = 40000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9994
TARGET_UPDATE = 50

MODEL_SAVE_PATH = "risk_only_model_final.pth"
RESULT_DIR = "results"
GRAPH_DIR = os.path.join(RESULT_DIR, "training_curves")


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append(
            (state, action, reward, next_state, done, next_mask)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(np.array(next_masks))
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class RiskOnlyDQNAgent:
    def __init__(self, state_size, action_size):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LR
        )

        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.current_max_q = 0.0

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
            self.current_max_q = torch.max(q_values).item()

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        masked_q = torch.full_like(q_values, -1e9)

        for action in valid_actions:
            masked_q[action] = q_values[action]

        return int(torch.argmax(masked_q).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(
            BATCH_SIZE
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        current_q = self.policy_net(states).gather(
            1,
            actions.unsqueeze(1)
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
            next_q = masked_next_q.max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.HuberLoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            1.0
        )

        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):
        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY
        )


def sample_scenario():
    return random.choices(
        list(SCENARIO_PROBS.keys()),
        weights=list(SCENARIO_PROBS.values()),
        k=1
    )[0]


def build_action_mask(env):
    mask = np.zeros(NUM_NODES, dtype=np.float32)

    for act in env.get_valid_actions():
        mask[act] = 1.0

    return mask


def evaluate_agent(agent, scenario, test_count=50):
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    success_list = []
    reward_list = []
    risk_list = []
    hop_list = []

    for _ in range(test_count):
        env = NetworkEnv(
            use_risk=True,
            scenario=scenario
        )

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()

            action = agent.select_action(
                state,
                valid_actions
            )

            if action is None:
                break

            next_state, reward, done, _ = env.step(action)

            state = next_state
            total_reward += reward

        metrics = env.calculate_path_metrics(env.visited)

        success_list.append(
            1 if env.current_node == DESTINATION else 0
        )
        reward_list.append(total_reward)
        risk_list.append(metrics["total_risk"])
        hop_list.append(metrics["hop_count"])

    agent.epsilon = old_epsilon

    return {
        "success_rate": np.mean(success_list) * 100,
        "avg_reward": np.mean(reward_list),
        "avg_risk": np.mean(risk_list),
        "avg_hop": np.mean(hop_list),
    }


def save_training_graph(values, title, ylabel, filename):
    os.makedirs(GRAPH_DIR, exist_ok=True)

    plt.figure(figsize=(8, 4))
    plt.plot(values)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    path = os.path.join(GRAPH_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()

    print("그래프 저장:", path)


def save_all_training_graphs(
    plot_rewards,
    plot_losses,
    plot_qs,
    plot_success,
    plot_risks
):
    save_training_graph(
        plot_rewards,
        "Episode Reward",
        "Reward",
        "reward_curve.png"
    )

    save_training_graph(
        plot_losses,
        "Training Loss",
        "Loss",
        "loss_curve.png"
    )

    save_training_graph(
        plot_qs,
        "Average Q Value",
        "Q Value",
        "q_curve.png"
    )

    save_training_graph(
        plot_success,
        "Episode Success",
        "Success (%)",
        "success_curve.png"
    )

    save_training_graph(
        plot_risks,
        "Average Path Risk",
        "Risk",
        "risk_curve.png"
    )


def train_risk_only_dqn():
    base_env = NetworkEnv(
        use_risk=True,
        scenario="normal"
    )

    agent = RiskOnlyDQNAgent(
        base_env.state_size,
        base_env.action_size
    )

    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    loss_history = deque(maxlen=100)
    q_history = deque(maxlen=100)
    risk_history = deque(maxlen=100)

    plot_rewards = []
    plot_losses = []
    plot_qs = []
    plot_success = []
    plot_risks = []

    scenario_success = {
        s: deque(maxlen=100)
        for s in SCENARIOS
    }

    scenario_count = {
        s: 0
        for s in SCENARIOS
    }

    print("===== Multi-Scenario Risk-Only DQN Training Start =====")
    print(f"Scenarios   : {SCENARIOS}")
    print(f"Episodes    : {EPISODES}")
    print(f"Device      : {agent.device}")
    print(f"State size  : {base_env.state_size}")
    print(f"Action size : {base_env.action_size}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()
        scenario_count[scenario] += 1

        env = NetworkEnv(
            use_risk=True,
            scenario=scenario
        )

        state = env.reset()
        done = False
        ep_reward = 0.0

        losses = []
        q_values = []

        while not done:
            valid_actions = env.get_valid_actions()

            action = agent.select_action(
                state,
                valid_actions
            )

            if action is None:
                break

            q_values.append(agent.current_max_q)

            next_state, reward, done, info = env.step(action)

            if "next_valid_mask" in info:
                next_mask = info["next_valid_mask"]
            else:
                next_mask = build_action_mask(env)

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_mask
            )

            loss = agent.train_step()

            if loss > 0:
                losses.append(loss)

            state = next_state
            ep_reward += reward

        success = 1 if env.current_node == DESTINATION else 0
        metrics = env.calculate_path_metrics(env.visited)

        success_history.append(success)
        reward_history.append(ep_reward)
        risk_history.append(metrics["total_risk"])
        scenario_success[scenario].append(success)

        if losses:
            loss_history.append(np.mean(losses))

        if q_values:
            q_history.append(np.mean(q_values))

        plot_rewards.append(ep_reward)
        plot_success.append(success * 100)
        plot_risks.append(metrics["total_risk"])

        if losses:
            plot_losses.append(np.mean(losses))
        else:
            plot_losses.append(0.0)

        if q_values:
            plot_qs.append(np.mean(q_values))
        else:
            plot_qs.append(0.0)

        agent.decay_epsilon()

        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        if ep % 100 == 0:
            scenario_rates = []

            for sc in SCENARIOS:
                if scenario_success[sc]:
                    rate = np.mean(scenario_success[sc]) * 100
                else:
                    rate = 0.0

                scenario_rates.append(
                    f"{sc}:{rate:5.1f}%"
                )

            print(
                f"Ep {ep:5d} | "
                f"Success: {np.mean(success_history) * 100:6.2f}% | "
                f"Reward: {np.mean(reward_history):8.2f} | "
                f"Risk: {np.mean(risk_history):7.4f} | "
                f"Loss: {(np.mean(loss_history) if loss_history else 0.0):8.4f} | "
                f"Avg_Q: {(np.mean(q_history) if q_history else 0.0):8.2f} | "
                f"Eps: {agent.epsilon:.3f} | "
                + " | ".join(scenario_rates)
            )

        if ep % 3000 == 0:
            print("\n===== Intermediate Evaluation =====")

            for sc in SCENARIOS:
                result = evaluate_agent(
                    agent,
                    sc,
                    test_count=30
                )

                print(
                    f"{sc:10s} | "
                    f"Success: {result['success_rate']:6.2f}% | "
                    f"Reward: {result['avg_reward']:8.2f} | "
                    f"Risk: {result['avg_risk']:7.4f} | "
                    f"Hop: {result['avg_hop']:5.2f}"
                )

            print("===================================\n")

    torch.save(
        agent.policy_net.state_dict(),
        MODEL_SAVE_PATH
    )

    save_all_training_graphs(
        plot_rewards,
        plot_losses,
        plot_qs,
        plot_success,
        plot_risks
    )

    print("\n===== Final Evaluation =====")

    for sc in SCENARIOS:
        result = evaluate_agent(
            agent,
            sc,
            test_count=100
        )

        print(
            f"{sc:10s} | "
            f"Success: {result['success_rate']:6.2f}% | "
            f"Reward: {result['avg_reward']:8.2f} | "
            f"Risk: {result['avg_risk']:7.4f} | "
            f"Hop: {result['avg_hop']:5.2f}"
        )


if __name__ == "__main__":
    train_risk_only_dqn()