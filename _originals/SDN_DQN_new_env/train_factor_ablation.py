# train_factor_ablation.py
# Factor Ablation 학습 스크립트 (Exp 5)
# risk_calculator의 특정 factor를 0으로 고정한 채 처음부터 학습
# 사용법: python train_factor_ablation.py <factor> <run_id>
# factor: full / util / cong / delay_inst / loss_inst / link_type
# 예시:   python train_factor_ablation.py util 1

import os, sys, csv, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import risk_calculator  # ZEROED_FACTORS를 외부에서 제어
from network_env import NetworkEnv
from topology import NUM_NODES, DESTINATION

# ── 실험 설정 ─────────────────────────────────────────────────────────────────
EPISODES       = 8000
EVAL_EPISODES  = 200
GAMMA          = 0.99
LR             = 0.0003
BATCH_SIZE     = 64
MEMORY_SIZE    = 40000
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9994   # 원래 학습 조건 그대로
TARGET_UPDATE  = 50

SCENARIOS      = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}
RISK_WINDOW    = 5
RESULT_DIR     = "results"

# factor 이름 → ZEROED_FACTORS에 넣을 키
FACTOR_MAP = {
    "full":       [],
    "util":       ["utilization"],
    "cong":       ["congestion"],
    "delay_inst": ["delay_inst"],
    "loss_inst":  ["loss_inst"],
    "link_type":  ["link_type"],
}

# 결과 표시용 레이블
LABEL_MAP = {
    "full":       "Full",
    "util":       "-Util",
    "cong":       "-Cong",
    "delay_inst": "-dDelay",
    "loss_inst":  "-dLoss",
    "link_type":  "-Type",
}


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
            nn.Linear(state_size, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
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
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(st)[0]
            self.current_max_q = torch.max(q).item()
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        masked = torch.full_like(q, -1e9)
        for a in valid_actions:
            masked[a] = q[a]
        return int(torch.argmax(masked).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        states, actions, rewards, next_states, dones, next_masks = self.memory.sample(BATCH_SIZE)
        states, actions, rewards = states.to(self.device), actions.to(self.device), rewards.to(self.device)
        next_states, dones, next_masks = next_states.to(self.device), dones.to(self.device), next_masks.to(self.device)
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            masked_next = next_q + (1.0 - next_masks) * -1e9
            target_q = rewards + GAMMA * masked_next.max(1)[0] * (1 - dones)
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


def train(factor_key, run_id):
    # ── 핵심: 학습 전에 ZEROED_FACTORS 설정 ──────────────────────────────────
    # network_env.py가 risk_calculator를 import하므로
    # 이 값만 바꾸면 env 전체에 자동 적용됨
    risk_calculator.ZEROED_FACTORS = FACTOR_MAP[factor_key]
    label = LABEL_MAP[factor_key]

    seed = 42 + run_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/factor_{factor_key}_run{run_id:02d}.pth"

    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    agent = DQNAgent(base_env.state_size, base_env.action_size)

    success_history = deque(maxlen=100)
    reward_history  = deque(maxlen=100)
    loss_history    = deque(maxlen=100)

    print(f"===== Factor Ablation: [{label}]  Run={run_id} =====")
    print(f"Zeroed factors : {risk_calculator.ZEROED_FACTORS if risk_calculator.ZEROED_FACTORS else 'None (Full)'}")
    print(f"Episodes       : {EPISODES}  (EPSILON_DECAY={EPSILON_DECAY})")
    print(f"Device         : {agent.device}")

    for ep in range(1, EPISODES + 1):
        scenario = sample_scenario()
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        state = env.reset()
        done = False
        ep_reward = 0.0
        losses = []

        while not done:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done, info["next_valid_mask"])
            loss = agent.train_step()
            if loss > 0:
                losses.append(loss)
            state = next_state
            ep_reward += reward

        success_history.append(1 if env.current_node == DESTINATION else 0)
        reward_history.append(ep_reward)
        if losses:
            loss_history.append(np.mean(losses))

        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | "
                  f"Success: {np.mean(success_history)*100:5.1f}% | "
                  f"Reward: {np.mean(reward_history):7.2f} | "
                  f"Loss: {(np.mean(loss_history) if loss_history else 0.0):7.4f} | "
                  f"Eps: {agent.epsilon:.4f}")

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"\n모델 저장: {save_path}")

    # ── Dynamic 시나리오 평가 ─────────────────────────────────────────────────
    agent.policy_net.eval()
    agent.epsilon = 0.0
    delays, losses_eval, hops = [], [], []

    for _ in range(EVAL_EPISODES):
        env = NetworkEnv(use_risk=True, scenario="dynamic", risk_window=RISK_WINDOW)
        env.reset()
        for _ in range(10):          # 메인 실험과 동일한 warm-up 조건
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            st = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.policy_net(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses_eval.append(m["total_loss"])
            hops.append(m["hop_count"])

    result = {
        "factor":  label,
        "run_id":  run_id,
        "delay":   float(np.mean(delays))      if delays      else 0.0,
        "loss":    float(np.mean(losses_eval)) if losses_eval else 0.0,
        "hop":     float(np.mean(hops))        if hops        else 0.0,
        "success_rate": len(delays) / EVAL_EPISODES * 100,
    }

    print(f"\n[평가-Dynamic] {label} Run{run_id} | "
          f"Delay={result['delay']:.1f} | Loss={result['loss']:.5f} | Hop={result['hop']:.2f}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f"factor_{factor_key}_dynamic_run{run_id:02d}_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)
    print(f"결과 저장: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_factor_ablation.py <factor> <run_id>")
        print("factor: full / util / cong / delay_inst / loss_inst / link_type")
        sys.exit(1)
    factor_key = sys.argv[1]
    if factor_key not in FACTOR_MAP:
        print(f"Unknown factor: {factor_key}")
        print(f"Available: {list(FACTOR_MAP.keys())}")
        sys.exit(1)
    train(factor_key, int(sys.argv[2]))