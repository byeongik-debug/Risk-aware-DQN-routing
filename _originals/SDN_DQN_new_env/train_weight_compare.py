# train_weight_compare.py
# 가중치 구성 비교 실험
# 사용법: python train_weight_compare.py <weight_config> <run_id>
# weight_config: current / equal
# 예시: python train_weight_compare.py equal 1

import os, sys, csv, random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import risk_calculator
from network_env import NetworkEnv
from topology import NUM_NODES, DESTINATION

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
RISK_WINDOW   = 5
RESULT_DIR    = "results"

SCENARIOS      = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {s: 0.25 for s in SCENARIOS}

WEIGHT_CONFIGS = {
    "current":  {"util": 0.35, "cong": 0.30, "delay": 0.15, "loss": 0.10, "type": 0.10},
    "equal":    {"util": 0.20, "cong": 0.20, "delay": 0.20, "loss": 0.20, "type": 0.20},
    "temporal": {"util": 0.25, "cong": 0.20, "delay": 0.25, "loss": 0.20, "type": 0.10},
}

LABEL_MAP = {
    "current":  "W-Current",
    "equal":    "W-Equal",
    "temporal": "W-Temporal",
}


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d, nm):
        self.buffer.append((s, a, r, ns, d, nm))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d, nm = zip(*batch)
        return (torch.FloatTensor(np.array(s)), torch.LongTensor(a),
                torch.FloatTensor(r), torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d), torch.FloatTensor(np.array(nm)))

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

    def select_action(self, state, valid_actions):
        if not valid_actions:
            return None
        st = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(st)[0]
        if random.random() < self.epsilon:
            return random.choice(valid_actions)
        masked = torch.full_like(q, -1e9)
        for a in valid_actions:
            masked[a] = q[a]
        return int(torch.argmax(masked).item())

    def train_step(self):
        if len(self.memory) < BATCH_SIZE:
            return 0.0
        s, a, r, ns, d, nm = self.memory.sample(BATCH_SIZE)
        s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
        ns, d, nm = ns.to(self.device), d.to(self.device), nm.to(self.device)
        cur_q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_net(ns)
            mnq = nq + (1.0 - nm) * -1e9
            tgt = r + GAMMA * mnq.max(1)[0] * (1 - d)
        loss = nn.HuberLoss()(cur_q, tgt)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)


def train(weight_key, run_id):
    risk_calculator.CUSTOM_WEIGHTS = WEIGHT_CONFIGS[weight_key]
    risk_calculator.ZEROED_FACTORS = []
    label = LABEL_MAP[weight_key]

    seed = 42 + run_id
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/weight_{weight_key}_run{run_id:02d}.pth"

    env_tmp = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    agent = DQNAgent(env_tmp.state_size, env_tmp.action_size)

    suc_hist = deque(maxlen=100)
    rew_hist  = deque(maxlen=100)
    loss_hist = deque(maxlen=100)

    print(f"===== Weight Compare: [{label}]  Run={run_id} =====")
    print(f"Weights: {risk_calculator.CUSTOM_WEIGHTS}")
    print(f"Episodes: {EPISODES}  Device: {agent.device}")

    for ep in range(1, EPISODES + 1):
        scenario = random.choices(SCENARIOS, weights=[0.25]*4)[0]
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        state = env.reset()
        done = False; ep_reward = 0.0; losses = []

        while not done:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            if action is None: break
            ns, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, ns, done, info["next_valid_mask"])
            l = agent.train_step()
            if l > 0: losses.append(l)
            state = ns; ep_reward += reward

        suc_hist.append(1 if env.current_node == DESTINATION else 0)
        rew_hist.append(ep_reward)
        if losses: loss_hist.append(np.mean(losses))
        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0: agent.update_target()

        if ep % 100 == 0:
            print(f"Ep {ep:4d} | Suc: {np.mean(suc_hist)*100:5.1f}% | "
                  f"Rew: {np.mean(rew_hist):7.2f} | "
                  f"Loss: {(np.mean(loss_hist) if loss_hist else 0):7.4f} | "
                  f"Eps: {agent.epsilon:.4f}")

    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"\n모델 저장: {save_path}")

    # Dynamic 시나리오 평가
    agent.policy_net.eval(); agent.epsilon = 0.0
    delays, losses_eval, hops = [], [], []

    for _ in range(EVAL_EPISODES):
        env = NetworkEnv(use_risk=True, scenario="dynamic", risk_window=RISK_WINDOW)
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
        "weight_config": label,
        "run_id": run_id,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses_eval)) if losses_eval else 0.0,
        "hop":   float(np.mean(hops)) if hops else 0.0,
        "success_rate": len(delays) / EVAL_EPISODES * 100,
    }

    print(f"\n[평가-Dynamic] {label} Run{run_id} | "
          f"Delay={result['delay']:.1f} | Loss={result['loss']:.5f} | "
          f"Hop={result['hop']:.2f} | SR={result['success_rate']:.1f}%")

    os.makedirs(RESULT_DIR, exist_ok=True)
    csv_path = os.path.join(RESULT_DIR, f"weight_{weight_key}_dynamic_run{run_id:02d}_result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader(); w.writerow(result)
    print(f"결과 저장: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python train_weight_compare.py <weight_config> <run_id>")
        print(f"weight_config: {list(WEIGHT_CONFIGS.keys())}")
        sys.exit(1)
    wk = sys.argv[1]
    if wk not in WEIGHT_CONFIGS:
        print(f"Unknown config: {wk}. Available: {list(WEIGHT_CONFIGS.keys())}")
        sys.exit(1)
    train(wk, int(sys.argv[2]))