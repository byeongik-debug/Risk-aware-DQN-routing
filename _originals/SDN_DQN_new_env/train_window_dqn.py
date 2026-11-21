# train_window_dqn.py
# Risk-DQN 학습 스크립트 (실험 2·3용)
# 인자로 window_size(시간 창 크기)와 run_id(반복 실험 번호)를 받아 학습
# 사용법: python train_window_dqn.py <window_size> <run_id>
# 예시:   python train_window_dqn.py 3 1

import os
import sys
import csv
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network_env import NetworkEnv
from topology import NUM_NODES, DESTINATION

EVAL_EPISODES = 200     # 학습 후 평가 에피소드 수
RESULT_DIR = "results"  # 평가 결과 저장 폴더

# 4개 시나리오를 동일 비율로 학습 → 다양한 네트워크 상황에 일반화
SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}

EPISODES = 8000         # 총 학습 에피소드
GAMMA = 0.99            # 미래 보상 할인율 (0.99 = 장기 보상 중시)
LR = 0.0003             # Adam 옵티마이저 학습률
BATCH_SIZE = 64         # 리플레이 버퍼에서 한 번에 샘플링할 크기
MEMORY_SIZE = 40000     # 리플레이 버퍼 최대 용량
EPSILON_START = 1.0     # 탐험 확률 초기값 (처음엔 100% 랜덤 행동)
EPSILON_END = 0.05      # 탐험 확률 최솟값 (학습 완료 후에도 5% 탐험 유지)
EPSILON_DECAY = 0.9994  # 에피소드마다 epsilon에 곱해지는 감소율
TARGET_UPDATE = 50      # 타겟 네트워크를 50 에피소드마다 동기화


class ReplayBuffer:
    """경험 리플레이 버퍼 — DQN 학습에서 과거 경험을 저장하고 무작위로 꺼내 학습
    순차적 상관관계를 깨뜨려 학습 안정성을 높임"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)  # 용량 초과 시 가장 오래된 것부터 삭제

    def push(self, state, action, reward, next_state, done, next_mask):
        # 한 스텝의 경험을 저장
        # next_mask: 다음 상태에서 이동 가능한 노드 마스크 (유효 액션 제한용)
        self.buffer.append((state, action, reward, next_state, done, next_mask))

    def sample(self, batch_size):
        # 무작위로 batch_size개 샘플링 → 텐서로 변환해 반환
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
    """Q-value를 출력하는 신경망
    입력: 현재 상태(state), 출력: 각 노드로 이동했을 때의 예상 누적 보상(Q값)"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.LayerNorm(512),  # 배치 정규화 대신 LayerNorm — 배치 크기에 무관하게 안정적
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),  # 출력: 각 노드(액션)의 Q값
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    """DQN 에이전트 — 정책 네트워크(policy_net)와 타겟 네트워크(target_net) 두 개 유지
    타겟 네트워크는 주기적으로만 업데이트해 학습 안정성 확보"""
    def __init__(self, state_size, action_size):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size).to(self.device)   # 실제 학습 네트워크
        self.target_net = QNetwork(state_size, action_size).to(self.device)   # 타겟 Q값 계산용
        self.target_net.load_state_dict(self.policy_net.state_dict())         # 초기에는 동일하게
        self.target_net.eval()  # 타겟 네트워크는 학습 안 함 (추론만)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.current_max_q = 0.0  # 모니터링용

    def select_action(self, state, valid_actions):
        """epsilon-greedy 정책: epsilon 확률로 탐험(랜덤), 나머지는 최적 행동 선택"""
        if not valid_actions:
            return None
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]
            self.current_max_q = torch.max(q_values).item()
        if random.random() < self.epsilon:
            return random.choice(valid_actions)  # 탐험: 랜덤 행동
        # 활용: 유효한 액션 중 Q값이 가장 높은 노드 선택
        masked_q = torch.full_like(q_values, -1e9)  # 유효하지 않은 노드는 -무한대로
        for action in valid_actions:
            masked_q[action] = q_values[action]
        return int(torch.argmax(masked_q).item())

    def train_step(self):
        """버퍼에서 배치 샘플링 후 Bellman 방정식으로 Q값 업데이트"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0  # 버퍼가 충분히 차지 않으면 학습 안 함
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        # 현재 상태에서 선택한 액션의 Q값
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # 타겟 네트워크로 다음 상태의 최대 Q값 계산
            next_q_values = self.target_net(next_states)
            # 유효하지 않은 액션은 제외 (next_mask=0이면 -무한대)
            masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
            next_q = masked_next_q.max(1)[0]
            # Bellman 방정식: 목표 Q = 즉각 보상 + 감쇄된 미래 Q (에피소드 종료 시 미래 Q=0)
            target_q = rewards + GAMMA * next_q * (1 - dones)

        # Huber Loss: MSE보다 이상값(outlier)에 강건
        loss = nn.HuberLoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 그래디언트 폭발 방지
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        # 정책 네트워크 가중치를 타겟 네트워크에 복사 (hard update)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # 에피소드마다 epsilon 감소 → 점점 탐험 줄이고 학습한 정책 활용
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def sample_scenario():
    """4개 시나리오 중 하나를 가중치에 따라 무작위 선택"""
    return random.choices(
        list(SCENARIO_PROBS.keys()),
        weights=list(SCENARIO_PROBS.values()),
        k=1,
    )[0]


def train(window_size, run_id):
    # run_id마다 다른 시드 사용 → 여러 번 실행해도 서로 다른 결과 (통계적 신뢰성)
    seed = 42 + run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/window{window_size}_run{run_id:02d}.pth"

    # use_risk=True: 상태에 위험도(risk) 정보 포함 → Risk-DQN 학습
    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=window_size)
    agent = DQNAgent(base_env.state_size, base_env.action_size)

    # 최근 100개 에피소드 이동 평균으로 학습 추이 모니터링
    success_history = deque(maxlen=100)
    reward_history = deque(maxlen=100)
    loss_history = deque(maxlen=100)

    print(f"===== Window={window_size} Run={run_id} Training Start =====")
    print(f"Episodes    : {EPISODES}")
    print(f"Device      : {agent.device}")
    print(f"Risk Window : {window_size}")
    print(f"Save Path   : {save_path}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()  # 매 에피소드마다 시나리오 무작위 선택
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=window_size)
        state = env.reset()
        done = False
        ep_reward = 0.0
        losses = []

        # 초기 유효 액션 마스크 (사용되지 않지만 버퍼 초기화 목적)
        valid_actions = env.get_valid_actions()
        init_mask = np.zeros(NUM_NODES, dtype=np.float32)
        for act in valid_actions:
            init_mask[act] = 1.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
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
        if losses:
            loss_history.append(np.mean(losses))

        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()  # 50 에피소드마다 타겟 네트워크 동기화

        if ep % 100 == 0:
            print(
                f"Ep {ep:5d} | "
                f"Success: {np.mean(success_history)*100:5.1f}% | "
                f"Reward: {np.mean(reward_history):7.2f} | "
                f"Loss: {(np.mean(loss_history) if loss_history else 0.0):7.4f} | "
                f"Eps: {agent.epsilon:.3f}"
            )

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"\n모델 저장 완료: {save_path}")

    # 학습 직후 평가 — 모든 window에서 4개 시나리오 전부 저장
    # window=1,5: Temporal Ablation(Exp 2)에서 모든 시나리오 비교 필요
    # window=3,7,10: Window Sweep(Exp 3)은 dynamic만 사용하지만 저장해두면 손해 없음
    eval_scenarios = SCENARIOS
    for scenario in eval_scenarios:
        eval_result = evaluate_model(agent, window_size, scenario)
        save_result(eval_result, window_size, run_id, scenario)


def evaluate_model(agent, window_size, scenario="dynamic", episodes=EVAL_EPISODES):
    """학습된 에이전트를 탐험 없이(epsilon=0) 평가"""
    agent.policy_net.eval()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # 평가 시 완전 greedy (학습한 정책만 사용)

    delays, losses, risks, hops = [], [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=window_size)
        env.reset()
        # 10번 QoS 업데이트 후 평가 시작 → 네트워크가 충분히 변화한 상태에서 평가
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions)
            if action is None:
                break
            state, _, done, _ = env.step(action)

        if env.current_node == DESTINATION:
            success_count += 1
            metrics = env.calculate_path_metrics(env.visited)
            delays.append(metrics["total_delay"])
            losses.append(metrics["total_loss"])
            risks.append(metrics["total_risk"])
            hops.append(metrics["hop_count"])
            hop_count = max(metrics["hop_count"], 1)
            unstable_usages.append(metrics["unstable_count"] / hop_count * 100)
            congested_usages.append(metrics["congested_count"] / hop_count * 100)

    agent.epsilon = old_epsilon
    agent.policy_net.train()

    result = {
        "scenario": scenario,
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "risk": float(np.mean(risks)) if risks else 0.0,
        "hop": float(np.mean(hops)) if hops else 0.0,
        "unstable_usage": float(np.mean(unstable_usages)) if unstable_usages else 0.0,
        "congested_usage": float(np.mean(congested_usages)) if congested_usages else 0.0,
    }
    print(
        f"\n[평가] window={window_size} scenario={scenario} | "
        f"Success={result['success_rate']:.1f}% | "
        f"Delay={result['delay']:.1f} | "
        f"Loss={result['loss']:.5f} | "
        f"Risk={result['risk']:.4f} | "
        f"Hop={result['hop']:.2f}"
    )
    return result


def save_result(result, window_size, run_id, scenario):
    """평가 결과를 CSV로 저장 — show_results.py에서 읽어서 비교 출력"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"window{window_size}_{scenario}_run{run_id:02d}_result.csv")
    fieldnames = ["window", "run_id", "scenario", "success_rate", "delay", "loss", "risk", "hop",
                  "unstable_usage", "congested_usage"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"window": window_size, "run_id": run_id, **result})
    print(f"결과 저장: {path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_window_dqn.py <window_size> <run_id> [episodes]")
        sys.exit(1)
    window_size = int(sys.argv[1])
    run_id = int(sys.argv[2])
    if len(sys.argv) >= 4:
        EPISODES = int(sys.argv[3])
        EVAL_EPISODES = 20
    train(window_size, run_id)