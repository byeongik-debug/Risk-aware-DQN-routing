# train_qos_dqn.py
# QoS-DQN 학습 스크립트 — 최종 논문에서는 사용하지 않음
#
# 설계 의도: delay+bandwidth+loss를 단일 QoS 점수로 압축한 상태로 DQN 학습

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from network_env import NetworkEnv
from topology import DESTINATION, NUM_NODES, EDGES

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}

EPISODES = 8000
GAMMA = 0.99
LR = 0.0003
BATCH_SIZE = 64
MEMORY_SIZE = 40000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9994
TARGET_UPDATE = 50
MODEL_SAVE_PATH = "qos_model_final.pth"

N_EDGES = len(EDGES)
# 상태 벡터 크기: 엣지당 단일 QoS 점수 + 현재노드·목적지·방문기록 원핫
QOS_STATE_SIZE = N_EDGES + NUM_NODES * 3


def compute_qos_state(env):
    """delay + bandwidth + loss 세 지표를 엣지당 단일 QoS 점수로 압축
    세 지표 평균 → 0(좋음)~1(나쁨) 범위의 복합 점수"""
    qos_scores = []
    for u, v in EDGES:
        e = env.normalize_edge(u, v)
        q = env.qos[e]
        delay_n = float(np.clip(q["delay"] / 120.0, 0.0, 1.0))
        bw_n    = 1.0 - float(np.clip(q["bandwidth"] / 1000.0, 0.0, 1.0))  # 대역폭은 역수
        loss_n  = float(np.clip(q["loss"] / 0.35, 0.0, 1.0))
        qos_scores.append((delay_n + bw_n + loss_n) / 3.0)

    # 위치 정보: 현재노드 원핫, 목적지 원핫, 방문 노드 원핫
    curr_onehot = np.zeros(NUM_NODES)
    curr_onehot[env.current_node] = 1.0
    dest_onehot = np.zeros(NUM_NODES)
    dest_onehot[DESTINATION] = 1.0
    visited_onehot = np.zeros(NUM_NODES)
    for node in env.visited:
        visited_onehot[node] = 1.0

    return np.concatenate([qos_scores, curr_onehot, dest_onehot, visited_onehot]).astype(np.float32)


class ReplayBuffer:
    """경험 리플레이 버퍼 — 과거 전이를 무작위 배치로 샘플링해 상관관계 제거"""
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
    """4층 MLP — 다른 DQN 파일과 동일한 구조"""
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


class QoSDQNAgent:
    """QoS 단일 점수 상태를 입력받는 DQN 에이전트"""
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
        """엡실론-그리디: 무작위 탐험(epsilon) vs 최적 액션 선택(1-epsilon)"""
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
        """벨만 방정식으로 손실 계산 후 역전파
        Huber Loss: MSE보다 이상치에 강건"""
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
            # 유효하지 않은 액션 마스킹 → 잘못된 경로 선택 방지
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
        """50 에피소드마다 타깃 네트워크를 폴리시 네트워크와 동기화"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """엡실론 지수 감쇠 → 학습이 진행될수록 탐험 줄이고 활용 늘림"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def sample_scenario():
    """4개 시나리오 중 하나를 균등 확률로 선택 — 모든 시나리오에서 고르게 학습"""
    return random.choices(list(SCENARIO_PROBS.keys()), weights=list(SCENARIO_PROBS.values()), k=1)[0]


def train():
    base_env = NetworkEnv(use_risk=False, scenario="normal")
    action_size = base_env.action_size
    agent = QoSDQNAgent(QOS_STATE_SIZE, action_size)

    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    loss_history = deque(maxlen=100)
    q_history = deque(maxlen=100)
    scenario_success = {s: deque(maxlen=100) for s in SCENARIOS}

    print("===== QoS-DQN Training Start =====")
    print(f"  state: delay+bandwidth+loss 단일 QoS 점수 per edge")
    print(f"Scenarios   : {SCENARIOS}")
    print(f"Episodes    : {EPISODES}")
    print(f"Device      : {agent.device}")
    print(f"State size  : {QOS_STATE_SIZE}")
    print(f"Action size : {action_size}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        state = compute_qos_state(env)  # NetworkEnv.get_state() 대신 QoS 전용 상태 사용
        done = False
        ep_reward = 0.0
        losses, q_values = [], []

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break
            q_values.append(agent.current_max_q)
            _, _, done, info = env.step(action)

            # 보상: 목적지 도착 시 경로 전체 평균 QoS 기반 보상 (end-to-end terminal reward)
            # 중간 스텝에서는 -0.1 스텝 패널티만 → 최단 경로 선호 유도
            if env.current_node == DESTINATION:
                path_costs = []
                for i in range(len(env.visited) - 1):
                    u, v = env.visited[i], env.visited[i + 1]
                    e = env.normalize_edge(u, v)
                    q_e = env.qos.get(e, {"delay": 120.0, "bandwidth": 0.0, "loss": 0.35})
                    d_n = float(np.clip(q_e["delay"] / 120.0, 0.0, 1.0))
                    b_n = 1.0 - float(np.clip(q_e["bandwidth"] / 1000.0, 0.0, 1.0))
                    l_n = float(np.clip(q_e["loss"] / 0.35, 0.0, 1.0))
                    path_costs.append((d_n + b_n + l_n) / 3.0)
                avg_qos = float(np.mean(path_costs)) if path_costs else 1.0
                reward = 10.0 - avg_qos * 10.0  # QoS가 좋을수록(점수 낮을수록) 보상 큼
            elif done:
                reward = -5.0  # 루프/막힘으로 실패
            else:
                reward = -0.1  # 스텝 패널티만

            next_state = compute_qos_state(env)
            next_mask = info["next_valid_mask"]
            agent.memory.push(state, action, reward, next_state, done, next_mask)
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state
            ep_reward += reward

        success = 1 if env.current_node == DESTINATION else 0
        success_history.append(success)
        reward_history.append(ep_reward)
        scenario_success[scenario].append(success)
        if losses:
            loss_history.append(np.mean(losses))
        if q_values:
            q_history.append(np.mean(q_values))

        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        if ep % 100 == 0:
            scenario_rates = [
                f"{sc}:{np.mean(scenario_success[sc])*100:5.1f}%" if scenario_success[sc] else f"{sc}:  0.0%"
                for sc in SCENARIOS
            ]
            print(
                f"Ep {ep:5d} | "
                f"Success: {np.mean(success_history)*100:6.2f}% | "
                f"Reward: {np.mean(reward_history):8.2f} | "
                f"Loss: {(np.mean(loss_history) if loss_history else 0.0):8.4f} | "
                f"Avg_Q: {(np.mean(q_history) if q_history else 0.0):8.2f} | "
                f"Eps: {agent.epsilon:.3f} | " + " | ".join(scenario_rates)
            )

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"\n모델 저장: {MODEL_SAVE_PATH}")
    print("===== Training Complete =====")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        EPISODES = int(sys.argv[1])
    train()