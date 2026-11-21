# train_norisk_dqn.py
# NoRisk-DQN 학습 스크립트 (실험 4용)
# Risk-DQN과 동일한 구조지만 use_risk=False → 상태에 위험도 정보 없음
# 목적: 위험 인식 보상 없을 때 성능이 얼마나 저하되는지 비교
# 사용법: python train_norisk_dqn.py <run_id>  (run_id: 1~5)

import sys, os, csv, random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from network_env import NetworkEnv
from topology import DESTINATION, NUM_NODES

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}

EPISODES      = 8000
EVAL_EPISODES = 200
GAMMA         = 0.99
LR            = 0.0003
BATCH_SIZE    = 64
MEMORY_SIZE   = 40000
EPSILON_START = 1.0
EPSILON_END   = 0.05
EPSILON_DECAY = 0.9994
TARGET_UPDATE = 50
RESULT_DIR    = "results"


class ReplayBuffer:
    """경험 리플레이 버퍼 — train_window_dqn.py와 동일"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
            torch.FloatTensor(np.array(next_masks)),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """신경망 구조 — Risk-DQN과 동일하지만 입력 state_size가 다름
    (use_risk=False이므로 위험도 벡터가 빠진 상태를 입력받음)"""
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


class NoRiskDQNAgent:
    """NoRisk-DQN 에이전트 — DQNAgent와 동일한 구조"""
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.current_max_q = 0.0

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
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
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
            next_q = masked_next_q.max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.HuberLoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def sample_scenario():
    return random.choices(list(SCENARIO_PROBS.keys()), weights=list(SCENARIO_PROBS.values()), k=1)[0]


def train(run_id):
    seed = 42 + run_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/norisk_run{run_id:02d}.pth"

    base_env = NetworkEnv(use_risk=False, scenario="normal")
    agent = NoRiskDQNAgent(base_env.state_size, base_env.action_size)

    success_history = deque(maxlen=100)
    reward_history  = deque(maxlen=100)
    loss_history    = deque(maxlen=100)
    q_history       = deque(maxlen=100)
    scenario_success = {s: deque(maxlen=100) for s in SCENARIOS}

    print(f"===== NoRisk DQN Training  Run={run_id}  Seed={seed} =====")
    print(f"Episodes={EPISODES}  Device={agent.device}  StateSize={base_env.state_size}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()
        env = NetworkEnv(use_risk=False, scenario=scenario)
        state = env.reset()
        done = False; ep_reward = 0.0; losses = []; q_values = []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break
            q_values.append(agent.current_max_q)
            next_state, reward, done, info = env.step(action)
            next_mask = info["next_valid_mask"]
            agent.memory.push(state, action, reward, next_state, done, next_mask)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state; ep_reward += reward

        success = 1 if env.current_node == DESTINATION else 0
        success_history.append(success)
        reward_history.append(ep_reward)
        scenario_success[scenario].append(success)
        if losses: loss_history.append(np.mean(losses))
        if q_values: q_history.append(np.mean(q_values))
        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0: agent.update_target()

        if ep % 100 == 0:
            scenario_rates = [
                f"{sc}:{np.mean(scenario_success[sc])*100:5.1f}%" if scenario_success[sc] else f"{sc}:  0.0%"
                for sc in SCENARIOS
            ]
            print(
                f"Ep {ep:4d} | Suc: {np.mean(success_history)*100:5.1f}% | "
                f"Rew: {np.mean(reward_history):7.2f} | "
                f"Loss: {(np.mean(loss_history) if loss_history else 0):7.4f} | "
                f"Eps: {agent.epsilon:.4f} | " + " | ".join(scenario_rates)
            )

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"\n모델 저장: {save_path}")

    # Dynamic 시나리오 평가
    agent.policy_net.eval(); agent.epsilon = 0.0
    delays, losses_eval, hops = [], [], []

    for _ in range(EVAL_EPISODES):
        env = NetworkEnv(use_risk=False, scenario="dynamic")
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state(); done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid: break
            st = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.policy_net(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid: masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))

        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses_eval.append(m["total_loss"])
            hops.append(m["hop_count"])

    result = {
        "run_id":  run_id,
        "delay":   float(np.mean(delays))       if delays else 0.0,
        "loss":    float(np.mean(losses_eval))  if losses_eval else 0.0,
        "hop":     float(np.mean(hops))         if hops else 0.0,
        "success_rate": len(delays) / EVAL_EPISODES * 100,
    }
    print(f"\n[평가-Dynamic] Run{run_id} | "
          f"Delay={result['delay']:.1f} | Loss={result['loss']:.5f} | "
          f"Hop={result['hop']:.2f} | SR={result['success_rate']:.1f}%")

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f"norisk_dynamic_run{run_id:02d}_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader(); w.writerow(result)
    print(f"결과 저장: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_norisk_dqn.py <run_id>  (run_id: 1~5)")
        sys.exit(1)
    train(int(sys.argv[1]))