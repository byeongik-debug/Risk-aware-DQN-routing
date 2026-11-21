# train_risk_dqn.py
# Risk-DQN 기본 학습 스크립트
# 시간적 위험도(Temporal Risk)를 상태로 받아 최적 경로를 학습하는 DQN 에이전트

import random                    # 랜덤 선택, 샘플링에 사용
from collections import deque    # 고정 크기 큐 자료구조 (리플레이 버퍼용)

import numpy as np               # 수치 연산 라이브러리
import torch                     # 딥러닝 프레임워크
import torch.nn as nn            # 신경망 레이어 모듈
import torch.optim as optim      # 옵티마이저 모듈

from network_env import NetworkEnv   # SDN 네트워크 환경
from topology import DESTINATION     # 목적지 노드 번호

SEED = 42                # 재현성을 위한 랜덤 시드 고정
random.seed(SEED)        # Python random 시드 고정
np.random.seed(SEED)     # NumPy 시드 고정
torch.manual_seed(SEED)  # PyTorch 시드 고정

SCENARIOS      = ["normal", "congestion", "unstable", "dynamic"]                           # 학습에 사용할 4가지 시나리오
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}  # 각 시나리오 선택 확률 (균등)
RISK_WINDOW    = 5       # 시간적 위험도 계산에 사용할 과거 스텝 수

EPISODES      = 8000     # 전체 학습 에피소드 수
GAMMA         = 0.99     # 미래 보상 할인율
LR            = 0.0003   # Adam 옵티마이저 학습률
BATCH_SIZE    = 64       # 한 번에 학습할 경험 샘플 수
MEMORY_SIZE   = 40000    # 리플레이 버퍼 최대 저장 용량
EPSILON_START = 1.0      # 탐험 확률 초기값
EPSILON_END   = 0.05     # 탐험 확률 최솟값
EPSILON_DECAY = 0.9994   # 에피소드마다 탐험 확률에 곱할 감소율
TARGET_UPDATE = 50       # 타겟 네트워크 동기화 주기
MODEL_SAVE_PATH = "risk_only_model_final.pth"  # 학습 완료 후 모델 저장 경로


class ReplayBuffer:
    # 과거 경험을 저장하고 무작위로 샘플링해 학습에 사용
    # 순차적 상관관계를 제거해 학습 안정성을 높임

    def __init__(self, capacity):
        # capacity: 버퍼 최대 크기 — 초과 시 가장 오래된 경험부터 삭제
        self.buffer = deque(maxlen=capacity)  # 고정 크기 덱: 초과 시 자동으로 앞에서 삭제

    def push(self, state, action, reward, next_state, done, next_mask):
        # 한 스텝의 경험 (상태, 액션, 보상, 다음상태, 종료여부, 유효액션마스크)을 버퍼에 추가
        self.buffer.append((state, action, reward, next_state, done, next_mask))  # 튜플로 묶어 저장

    def sample(self, batch_size):
        # 버퍼에서 batch_size개를 무작위 샘플링 후 학습용 텐서로 변환해 반환
        batch = random.sample(self.buffer, batch_size)                  # 중복 없이 무작위 추출
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)  # 6개 필드로 분리
        return (
            torch.FloatTensor(np.array(states)),       # 상태 배치 텐서
            torch.LongTensor(actions),                 # 액션 인덱스 텐서
            torch.FloatTensor(rewards),                # 보상 텐서
            torch.FloatTensor(np.array(next_states)),  # 다음 상태 배치 텐서
            torch.FloatTensor(dones),                  # 종료 여부 텐서 (1.0 = 종료)
            torch.FloatTensor(np.array(next_masks)),   # 다음 상태 유효 액션 마스크 텐서
        )

    def __len__(self):
        # 현재 버퍼에 저장된 경험 수 반환
        return len(self.buffer)


class QNetwork(nn.Module):
    # Q값을 출력하는 신경망
    # 입력: 현재 상태(위험도 벡터), 출력: 각 노드로 이동했을 때의 예상 누적 보상(Q값)

    def __init__(self, state_size, action_size):
        # state_size: 입력 상태 벡터 크기, action_size: 출력 노드 수(= 전체 노드 수)
        super().__init__()           # nn.Module 초기화
        self.net = nn.Sequential(    # 순차 레이어 구성
            nn.Linear(state_size, 512),  # 입력층 → 512 히든
            nn.LayerNorm(512),           # 배치 크기에 무관하게 안정적인 정규화
            nn.ReLU(),                   # 활성화 함수
            nn.Linear(512, 256),         # 512 → 256 히든
            nn.ReLU(),                   # 활성화 함수
            nn.Linear(256, 128),         # 256 → 128 히든
            nn.ReLU(),                   # 활성화 함수
            nn.Linear(128, action_size), # 출력층: 노드 수만큼 Q값 출력
        )

    def forward(self, x):
        # 순전파: 상태 벡터를 받아 각 액션(노드)의 Q값 반환
        return self.net(x)


class DQNAgent:
    # DQN 에이전트
    # policy_net: 실제 학습되는 네트워크
    # target_net: 안정적인 Q 타겟 계산을 위해 주기적으로만 업데이트

    def __init__(self, state_size, action_size):
        # policy_net과 target_net을 동일한 초기 가중치로 생성
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU 있으면 GPU 사용
        self.policy_net = QNetwork(state_size, action_size).to(self.device)              # 실제 학습되는 정책 네트워크
        self.target_net = QNetwork(state_size, action_size).to(self.device)              # Q타겟 계산용 고정 네트워크
        self.target_net.load_state_dict(self.policy_net.state_dict())                    # 초기값을 policy_net과 동일하게
        self.target_net.eval()                                                            # 타겟 네트워크는 학습 모드 아님
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)               # Adam 옵티마이저
        self.memory     = ReplayBuffer(MEMORY_SIZE)                                      # 경험 저장 버퍼 초기화
        self.epsilon    = EPSILON_START                                                   # 탐험 확률 초기화

    def select_action(self, state, valid_actions):
        # epsilon-greedy 정책: epsilon 확률로 탐험(랜덤), 나머지는 Q값이 가장 높은 유효 노드 선택
        if not valid_actions:                                                     # 이동 가능한 노드 없으면 종료
            return None
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)     # (state_size,) → (1, state_size) 배치 차원 추가
        with torch.no_grad():                                                     # 그래디언트 계산 비활성화 (추론 전용)
            q_values = self.policy_net(state_tensor)[0]                          # 배치 차원 제거해 Q값 벡터 추출
        if random.random() < self.epsilon:                                        # 탐험: 무작위 노드 선택
            return random.choice(valid_actions)
        masked_q = torch.full_like(q_values, -1e9)                               # 유효하지 않은 액션은 -무한대로 초기화
        for action in valid_actions:                                              # 유효 액션만 실제 Q값으로 채움
            masked_q[action] = q_values[action]
        return int(torch.argmax(masked_q).item())                                 # Q값 최대 노드 선택

    def train_step(self):
        # 버퍼에서 배치 샘플링 후 Bellman 방정식으로 Q값 업데이트, 손실값 반환
        if len(self.memory) < BATCH_SIZE:                                         # 버퍼에 충분한 경험이 쌓일 때까지 학습 안 함
            return 0.0
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(BATCH_SIZE)  # 배치 샘플링
        states, actions, rewards = states.to(self.device), actions.to(self.device), rewards.to(self.device)           # GPU 이동
        next_states, dones, next_masks = next_states.to(self.device), dones.to(self.device), next_masks.to(self.device)  # GPU 이동

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # 실제 선택한 액션의 Q값 추출
        with torch.no_grad():                                                             # 타겟 계산 시 그래디언트 차단
            next_q      = self.target_net(next_states)                                   # 다음 상태의 Q값 (타겟 네트워크 사용)
            masked_next = next_q + (1.0 - next_masks) * -1e9                            # 유효하지 않은 액션 제거
            target_q    = rewards + GAMMA * masked_next.max(1)[0] * (1 - dones)         # Bellman 방정식

        loss = nn.HuberLoss()(current_q, target_q)       # Huber 손실: MSE보다 이상값에 강인
        self.optimizer.zero_grad()                        # 이전 그래디언트 초기화
        loss.backward()                                   # 역전파
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # 그래디언트 폭발 방지
        self.optimizer.step()                             # 가중치 업데이트
        return loss.item()                                # Python float으로 변환해 반환

    def update_target(self):
        # 정책 네트워크의 가중치를 타겟 네트워크에 복사 (hard update)
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        # 에피소드마다 탐험 확률 감소 → 학습이 진행될수록 학습된 정책을 더 많이 활용
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)  # 최솟값 이하로 내려가지 않도록 clamp


def train():
    # 전체 학습 루프: 4개 시나리오를 섞어 에피소드를 반복하며 에이전트 학습
    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)  # state_size, action_size 확인용
    agent    = DQNAgent(base_env.state_size, base_env.action_size)                    # 에이전트 초기화

    success_history = deque(maxlen=100)  # 최근 100 에피소드 성공 여부 추적
    reward_history  = deque(maxlen=100)  # 최근 100 에피소드 보상 추적
    loss_history    = deque(maxlen=100)  # 최근 100 에피소드 손실 추적

    print("===== Risk-DQN Training Start =====")                                        # 학습 시작 로그
    print(f"Episodes={EPISODES}  Device={agent.device}  RiskWindow={RISK_WINDOW}")      # 설정값 출력

    for ep in range(1, EPISODES + 1):                                                   # 에피소드 루프
        scenario = random.choices(list(SCENARIO_PROBS.keys()), weights=list(SCENARIO_PROBS.values()))[0]  # 시나리오 무작위 선택
        env      = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)  # 해당 시나리오 환경 생성
        state    = env.reset()                                                            # 환경 초기화 & 초기 상태 수신
        done     = False; ep_reward = 0.0; losses = []                                   # 에피소드 변수 초기화

        while not done:                                                                   # 스텝 루프 (에피소드 종료까지)
            valid_actions = env.get_valid_actions()                                       # 현재 노드에서 이동 가능한 노드 목록
            action = agent.select_action(state, valid_actions)                            # 액션 선택
            if action is None:                                                            # 이동 불가 → 에피소드 강제 종료
                break
            next_state, reward, done, info = env.step(action)                            # 환경에 액션 적용
            agent.memory.push(state, action, reward, next_state, done, info["next_valid_mask"])  # 경험 저장
            loss = agent.train_step()                                                     # 학습 1스텝
            if loss > 0:                                                                  # 학습이 실제로 이뤄진 경우만 기록
                losses.append(loss)
            state = next_state; ep_reward += reward                                       # 상태 전환, 보상 누적

        success_history.append(1 if env.current_node == DESTINATION else 0)  # 목적지 도달 여부 기록
        reward_history.append(ep_reward)                                      # 에피소드 총 보상 기록
        if losses:                                                            # 손실이 기록된 경우만
            loss_history.append(np.mean(losses))                             # 평균 손실 기록
        agent.decay_epsilon()                                                 # 탐험 확률 감소
        if ep % TARGET_UPDATE == 0:                                           # 주기적으로 타겟 네트워크 동기화
            agent.update_target()

        if ep % 100 == 0:                                                     # 100 에피소드마다 현황 출력
            print(f"Ep {ep:5d} | "
                  f"Success: {np.mean(success_history)*100:5.1f}% | "
                  f"Reward: {np.mean(reward_history):7.2f} | "
                  f"Loss: {(np.mean(loss_history) if loss_history else 0.0):7.4f} | "
                  f"Eps: {agent.epsilon:.3f}")

    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)  # 학습 완료된 정책 네트워크 저장
    print(f"\n모델 저장: {MODEL_SAVE_PATH}")                      # 저장 경로 출력
    print("===== Training Complete =====")                       # 학습 완료 로그


if __name__ == "__main__":
    train()  # 직접 실행 시 학습 시작