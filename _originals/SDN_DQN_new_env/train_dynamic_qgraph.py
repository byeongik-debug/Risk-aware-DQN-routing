# train_dynamic_qgraph.py
# Dynamic 시나리오만으로 학습하며 Q값 수렴 그래프를 생성하는 스크립트

import os
import sys
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network_env import NetworkEnv
from topology import DESTINATION

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCENARIO           = "dynamic"
RISK_WINDOW        = 5
EPISODES           = 8000
GAMMA              = 0.99
LR                 = 0.0003
BATCH_SIZE         = 64
MEMORY_SIZE        = 40000
EPSILON_START      = 1.0
EPSILON_END        = 0.05
EPSILON_DECAY      = 0.9994
TARGET_UPDATE      = 50
MEASURE_SEED    = 12345  # Q값 측정용 고정 시드
MODEL_SAVE_PATH   = "risk_multiscenario_model.pth"
GRAPH_SAVE_PATH   = "q_convergence.png"
REWARD_GRAPH_PATH = "reward_convergence.png"
DATA_SAVE_PATH    = "training_data.npz"


class ReplayBuffer:
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


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net = QNetwork(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory     = ReplayBuffer(MEMORY_SIZE)
        self.epsilon    = EPSILON_START

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
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
        states     = states.to(self.device)
        actions    = actions.to(self.device)
        rewards    = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones      = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q      = self.target_net(next_states)
            masked_next = next_q + (1.0 - next_masks) * -1e9
            target_q    = rewards + GAMMA * masked_next.max(1)[0] * (1 - dones)

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

    def get_max_q(self, state, valid_actions):
        if not valid_actions:
            return 0.0
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
        return max(q_values[a].item() for a in valid_actions)


def measure_q(agent):
    """고정된 환경 상태에서 시작 상태의 최대 Q값 측정 (훈련 랜덤 상태 보존)"""
    rng_state = np.random.get_state()
    np.random.seed(MEASURE_SEED)

    ref_env   = NetworkEnv(use_risk=True, scenario="dynamic", risk_window=RISK_WINDOW)
    ref_state = ref_env.reset()
    valid     = ref_env.get_valid_actions()
    max_q     = agent.get_max_q(ref_state, valid)

    np.random.set_state(rng_state)
    return max_q


def save_q_graph(q_episodes, q_values):
    fig, ax = plt.subplots(figsize=(8, 4))

    # 원시 Q값 (연한 색)
    ax.plot(q_episodes, q_values, color='#A8C4E0', linewidth=0.8, alpha=0.6, label='Max Q-value')

    # 이동 평균 (진한 색)
    window = 20
    if len(q_values) >= window:
        smoothed  = np.convolve(q_values, np.ones(window) / window, mode='valid')
        smooth_ep = q_episodes[window - 1:]
        ax.plot(smooth_ep, smoothed, color='#2B6CB0', linewidth=2.0,
                label=f'Moving Average (n={window})')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Max Q-value at Start State', fontsize=12)
    ax.set_title('Q-value Convergence during Training\n(Dynamic Scenario)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(GRAPH_SAVE_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"그래프 저장: {GRAPH_SAVE_PATH}")


def save_reward_graph(ep_rewards):
    episodes = np.arange(1, len(ep_rewards) + 1)
    window   = 200

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(episodes, ep_rewards, color='#F4A97B', linewidth=0.6, alpha=0.4, label='Episode Reward')

    if len(ep_rewards) >= window:
        smoothed  = np.convolve(ep_rewards, np.ones(window) / window, mode='valid')
        smooth_ep = episodes[window - 1:]
        ax.plot(smooth_ep, smoothed, color='#C05621', linewidth=2.0,
                label=f'Moving Average (n={window})')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Episode Reward during Training\n(Dynamic Scenario)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(REWARD_GRAPH_PATH, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"보상 그래프 저장: {REWARD_GRAPH_PATH}")


def train():
    env   = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=RISK_WINDOW)
    agent = DQNAgent(env.state_size, env.action_size)

    success_history  = deque(maxlen=100)
    reward_history   = deque(maxlen=100)
    loss_history     = deque(maxlen=100)
    q_episodes       = []
    q_values         = []
    all_ep_rewards   = []
    all_ep_losses    = []
    all_ep_success   = []
    all_ep_hops      = []   # 에피소드 홉 수
    all_ep_avg_risk  = []   # 에피소드 평균 링크 위험도
    all_ep_risky_ratio = [] # 에피소드 위험 링크 선택 비율
    all_ep_epsilon   = []   # 에피소드 epsilon 값

    print("===== Risk-DQN Training (Dynamic Only) =====")
    print(f"Episodes={EPISODES}  Device={agent.device}  RiskWindow={RISK_WINDOW}")

    for ep in range(1, EPISODES + 1):
        state = env.reset()
        done  = False
        ep_reward = 0.0
        losses    = []
        ep_risks  = []
        danger_set, _ = env.get_dynamic_phase_sets()

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done, info["next_valid_mask"])
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            ep_risks.append(info["risk"])
            state = next_state
            ep_reward += reward

        ep_success = 1 if env.current_node == DESTINATION else 0
        success_history.append(ep_success)
        reward_history.append(ep_reward)
        all_ep_rewards.append(ep_reward)
        all_ep_losses.append(np.mean(losses) if losses else 0.0)
        all_ep_success.append(ep_success)
        all_ep_epsilon.append(agent.epsilon)

        # 홉 수
        hop_count = max(len(env.visited) - 1, 0)
        all_ep_hops.append(hop_count)

        # 평균 링크 위험도
        all_ep_avg_risk.append(np.mean(ep_risks) if ep_risks else 0.0)

        # 위험 링크 선택 비율
        path = env.visited
        path_edges = set()
        for i in range(len(path) - 1):
            path_edges.add(env.normalize_edge(path[i], path[i + 1]))
        risky_selected = len(path_edges & danger_set)
        risky_ratio = risky_selected / max(len(path_edges), 1)
        all_ep_risky_ratio.append(risky_ratio)
        if losses:
            loss_history.append(np.mean(losses))
        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        # 매 에피소드마다 Q값 측정
        max_q = measure_q(agent)
        q_episodes.append(ep)
        q_values.append(max_q)

        if ep % 200 == 0:
            latest_q = q_values[-1] if q_values else 0.0
            print(f"Ep {ep:5d} | "
                  f"Success: {np.mean(success_history)*100:5.1f}% | "
                  f"Reward: {np.mean(reward_history):7.2f} | "
                  f"Loss: {(np.mean(loss_history) if loss_history else 0.0):7.4f} | "
                  f"Eps: {agent.epsilon:.3f} | "
                  f"MaxQ: {latest_q:.3f}")

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"\n모델 저장: {MODEL_SAVE_PATH}")

    # 전체 Q값 데이터 저장 (나중에 다시 그릴 수 있도록)
    np.savez(DATA_SAVE_PATH,
             episodes=np.array(q_episodes),
             q_values=np.array(q_values),
             ep_rewards=np.array(all_ep_rewards),
             ep_losses=np.array(all_ep_losses),
             ep_success=np.array(all_ep_success),
             ep_hops=np.array(all_ep_hops),
             ep_avg_risk=np.array(all_ep_avg_risk),
             ep_risky_ratio=np.array(all_ep_risky_ratio),
             ep_epsilon=np.array(all_ep_epsilon))
    print(f"데이터 저장: {DATA_SAVE_PATH}")
    print(f"  keys: q_values, ep_rewards, ep_losses, ep_success, ep_hops, ep_avg_risk, ep_risky_ratio, ep_epsilon")

    save_q_graph(q_episodes, q_values)
    save_reward_graph(all_ep_rewards)
    print("===== Training Complete =====")


if __name__ == "__main__":
    train()