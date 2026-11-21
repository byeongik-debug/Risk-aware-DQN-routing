# risk_dqn_every_scenario.py
# Risk-DQN 메인 학습 스크립트 — 논문 베이스라인 모델
# 4개 시나리오(normal, congestion, unstable, dynamic)를 균등 확률로 섞어서 학습
# → 시나리오별로 따로 학습하지 않고 하나의 모델이 모든 환경에 대응하도록 일반화

import random
import numpy as np
from collections import deque

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
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}

EPISODES = 8000
GAMMA = 0.99              # 미래 보상 할인율
LR = 0.0003               # Adam 학습률
BATCH_SIZE = 64
MEMORY_SIZE = 40000       # 리플레이 버퍼 크기
EPSILON_START = 1.0       # 초기 탐험율 (100% 무작위)
EPSILON_END = 0.05        # 최소 탐험율 (5% 유지)
EPSILON_DECAY = 0.9994    # 에피소드마다 곱하는 감쇠 계수
TARGET_UPDATE = 50        # 타깃 네트워크 동기화 주기 (에피소드 단위)
MODEL_SAVE_PATH = "risk_only_model_final.pth"  # 저장 파일명


class ReplayBuffer:
    """경험 리플레이 버퍼
    연속된 전이의 상관관계를 끊어 학습 안정성 확보"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask):
        # next_mask: 다음 상태에서 유효한 액션 마스크 (이미 방문한 노드나 미연결 노드 제외)
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
            torch.FloatTensor(np.array(next_masks))
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """4층 MLP Q-네트워크 (512→256→128→action_size)
    LayerNorm: 학습 초반 발산 방지, Huber Loss와 함께 안정적 학습"""
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
    """Risk-DQN 에이전트
    policy_net: 매 스텝 업데이트 / target_net: 50 에피소드마다 동기화
    두 네트워크 분리로 학습 발산 방지 (moving target 문제 완화)"""
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
        """엡실론-그리디 액션 선택
        valid_actions만 후보로 → 방문했거나 연결 안 된 노드 선택 불가"""
        if not valid_actions:
            return None

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
            self.current_max_q = torch.max(q_values).item()

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 유효하지 않은 액션에 -무한대 마스킹 → argmax 시 반드시 valid_actions 중 선택
        masked_q = torch.full_like(q_values, -1e9)
        for action in valid_actions:
            masked_q[action] = q_values[action]

        return int(torch.argmax(masked_q).item())

    def train_step(self):
        """벨만 방정식: target_q = r + γ * max(Q_target(s')) * (1 - done)
        타깃 Q에도 액션 마스킹 적용 → 유효하지 않은 다음 액션의 Q값이 선택되지 않도록"""
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
            # 타깃 네트워크로 다음 Q값 계산 (policy_net이 아님 → 안정성)
            next_q_values = self.target_net(next_states)
            # 유효하지 않은 연결 상태의 노드 점수를 -무한대로 감쇠
            masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
            next_q = masked_next_q.max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)

        loss = nn.HuberLoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 그래디언트 클리핑
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        """타깃 네트워크를 폴리시 네트워크 가중치로 하드 업데이트"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """탐험율 지수 감쇠 — 학습이 진행될수록 더 많이 활용(exploit)"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def sample_scenario():
    """4개 시나리오 중 하나를 균등 확률(0.25씩)로 선택"""
    return random.choices(list(SCENARIO_PROBS.keys()), weights=list(SCENARIO_PROBS.values()), k=1)[0]


def evaluate_agent(agent, scenario, test_count=50):
    """중간 평가 함수 — 학습 중 3000 에피소드마다 진행 상황 확인용
    epsilon=0(그리디)로 고정해 탐험 없이 순수 정책 성능만 측정"""
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    success_list, reward_list, risk_list, hop_list = [], [], [], []

    for _ in range(test_count):
        env = NetworkEnv(use_risk=True, scenario=scenario)
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None: break

            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward

        metrics = env.calculate_path_metrics(env.visited)
        success_list.append(1 if env.current_node == DESTINATION else 0)
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


def train_risk_only_dqn():
    # use_risk=True: 상태에 시계열 위험도(5요소) 포함 — Risk-DQN의 핵심 입력
    base_env = NetworkEnv(use_risk=True, scenario="normal")
    agent = RiskOnlyDQNAgent(base_env.state_size, base_env.action_size)

    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    loss_history = deque(maxlen=100)
    q_history = deque(maxlen=100)
    risk_history = deque(maxlen=100)

    scenario_success = {s: deque(maxlen=100) for s in SCENARIOS}
    scenario_count = {s: 0 for s in SCENARIOS}

    print("===== Multi-Scenario Risk-Only DQN Training Start =====")
    print(f"Scenarios   : {SCENARIOS}")
    print(f"Episodes    : {EPISODES}")
    print(f"Device      : {agent.device}")
    print(f"State size  : {base_env.state_size}")
    print(f"Action size : {base_env.action_size}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()
        scenario_count[scenario] += 1

        env = NetworkEnv(use_risk=True, scenario=scenario)
        state = env.reset()

        done = False
        ep_reward = 0.0
        losses, q_values = [], []

        # 초기 상태에 대한 유효 액션 마스크 (첫 스텝 전에 필요)
        valid_actions = env.get_valid_actions()
        init_mask = np.zeros(NUM_NODES, dtype=np.float32)
        for act in valid_actions: init_mask[act] = 1.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None: break

            q_values.append(agent.current_max_q)
            next_state, reward, done, info = env.step(action)

            # 환경이 제공하는 next_valid_mask를 버퍼에 함께 저장
            # → 타깃 Q 계산 시 유효하지 않은 액션이 선택되지 않도록
            next_mask = info["next_valid_mask"]
            agent.memory.push(state, action, reward, next_state, done, next_mask)

            loss = agent.train_step()
            if loss > 0: losses.append(loss)

            state = next_state
            ep_reward += reward

        success = 1 if env.current_node == DESTINATION else 0
        metrics = env.calculate_path_metrics(env.visited)

        success_history.append(success)
        reward_history.append(ep_reward)
        risk_history.append(metrics["total_risk"])
        scenario_success[scenario].append(success)

        if losses: loss_history.append(np.mean(losses))
        if q_values: q_history.append(np.mean(q_values))

        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0: agent.update_target()

        if ep % 100 == 0:
            scenario_rates = [f"{sc}:{np.mean(scenario_success[sc])*100:5.1f}%" if scenario_success[sc] else f"{sc}:  0.0%" for sc in SCENARIOS]
            print(
                f"Ep {ep:5d} | "
                f"Success: {np.mean(success_history) * 100:6.2f}% | "
                f"Reward: {np.mean(reward_history):8.2f} | "
                f"Risk: {np.mean(risk_history):7.4f} | "
                f"Loss: {(np.mean(loss_history) if loss_history else 0.0):8.4f} | "
                f"Avg_Q: {(np.mean(q_history) if q_history else 0.0):8.2f} | "
                f"Eps: {agent.epsilon:.3f} | " + " | ".join(scenario_rates)
            )

        if ep % 3000 == 0:
            # 3000 에피소드마다 중간 평가 — 학습 진행 상황 확인
            print("\n===== Intermediate Evaluation =====")
            for sc in SCENARIOS:
                result = evaluate_agent(agent, sc, test_count=30)
                print(f"{sc:10s} | Success: {result['success_rate']:6.2f}% | Reward: {result['avg_reward']:8.2f} | Risk: {result['avg_risk']:7.4f} | Hop: {result['avg_hop']:5.2f}")
            print("===================================\n")

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print("\n===== Final Evaluation =====")
    for sc in SCENARIOS:
        result = evaluate_agent(agent, sc, test_count=100)
        print(f"{sc:10s} | Success: {result['success_rate']:6.2f}% | Reward: {result['avg_reward']:8.2f} | Risk: {result['avg_risk']:7.4f} | Hop: {result['avg_hop']:5.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        EPISODES = int(sys.argv[1])
        print(f"EPISODES 오버라이드: {EPISODES}")
    if len(sys.argv) > 2:
        LR = float(sys.argv[2])
        print(f"LR 오버라이드: {LR}")
    if len(sys.argv) > 3:
        GAMMA = float(sys.argv[3])
        print(f"GAMMA 오버라이드: {GAMMA}")
    train_risk_only_dqn()