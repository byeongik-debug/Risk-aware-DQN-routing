# -*- coding: utf-8 -*-
"""125차원 고정 절제(Risk-Blind) 학습 — 표 11 NoRisk 행의 교락을 푼다.

왜 필요한가
-----------
`network_env.py` 의 `use_risk` 는 **두 가지를 동시에** 바꾼다.
  (1) 상태: 위험 50칸(`:276-279`) ↔ 원시 QoS 100칸(`:280-285`) → 125 대 175차원
  (2) 보상: 시나리오별 위험 기반(`:334-`) ↔ QoS 기반(`:326-332`)
따라서 표 11 의 `NoRisk-DQN` 행은 **입력 표현 + 보상 함수**가 함께 바뀐 팔이고,
그 행의 부호 역전을 "위험 인식의 기여" 로 읽을 수 없다(FINDINGS 22.7절).

이 스크립트는 **제안 팔(`train_factor_ablation.py` 의 `full`)과 단 한 가지만
다른** 팔을 처음부터 학습한다 — 상태의 위험 50칸을 **상수 벡터로 고정**한다.
차원(125), 신경망, 하이퍼파라미터, 시드 규칙(42+run_id), 시나리오 혼합,
`risk_window=5`, 평가 규약은 전부 같다. 보상은 `step()` 이 상태가 아니라
`self.qos[e]["risk"]` 를 직접 읽으므로 **그대로 위험 기반**이다.
→ 제안 팔과의 차는 **상태에 실린 엣지 위험 정보** 하나에만 귀속된다.

왜 0 이 아니라 평균인가
----------------------
가림 측정에서 `dest_zero` 가 무작동 통제가 아니었다(상태를 분포 밖으로 밀어낸다).
같은 이유로 상수는 **평균**을 쓴다. 두 변종을 둔다.
  - `scalar` : 50칸 전부 동일한 대역 평균 → 위험 정보 **전부** 제거
  - `edgemean`: 칸마다 그 엣지의 평균 → **정적** 엣지 서열은 남기고
                **시변** 성분만 제거 (정적/동적 분리)

가림(추론 시점 고정)과 다른 것: 가림은 학습된 정책이 그 블록을 **읽는지**를
재지만, 이 팔은 **없는 채로 학습**해 보상을 하는 것을 허용한다. 즉 가림보다
약한 상한이 아니라 다른 질문이다.

원본은 읽기만 한다. 산출은 전부 `review_results/` 아래.
사용법:
    python train_risk_blind.py const                  # 상수 벡터 추정·동결
    python train_risk_blind.py scalar 1               # 학습 (variant, run_id)
    python train_risk_blind.py edgemean 1 --episodes 60   # 비용 탐침
"""
import argparse
import csv
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

RR = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(RR, "..", "SDN_DQN_new_env"))
sys.path.insert(0, ORIG)

from network_env import NetworkEnv  # noqa: E402
from topology import DESTINATION, EDGES, NUM_NODES  # noqa: E402

# ── train_factor_ablation.py 와 동일한 학습 설정 (복사, 수정 금지) ──────────
EPISODES       = 8000
EVAL_EPISODES  = 200
GAMMA          = 0.99
LR             = 0.0003
BATCH_SIZE     = 64
MEMORY_SIZE    = 40000
EPSILON_START  = 1.0
EPSILON_END    = 0.05
EPSILON_DECAY  = 0.9994
TARGET_UPDATE  = 50
SCENARIOS      = ["normal", "congestion", "unstable", "dynamic"]
SCENARIO_PROBS = {"normal": 0.25, "congestion": 0.25, "unstable": 0.25, "dynamic": 0.25}
RISK_WINDOW    = 5

OUT = os.path.join(RR, "review_results")
MODEL_DIR = os.path.join(OUT, "models_risk_blind")
CONST_CSV = os.path.join(OUT, "risk_blind_const.csv")

N_EDGES = len(EDGES)
VARIANTS = ("scalar", "edgemean")


# ── 상수 벡터 ──────────────────────────────────────────────────────────────
def estimate_const(n_episodes=400, seed=20261115):
    """학습 분포(네 시나리오 균등 혼합)에서 엣지별 평균 위험을 추정한다.

    에피소드 안에서 위험이 누산되므로(`network_env.py:228-229`) 에피소드를
    통째로 굴려 매 스텝의 위험을 모두 모은다. 학습 때와 같은 방식으로
    시나리오를 추첨한다.
    """
    random.seed(seed)
    np.random.seed(seed)
    acc = np.zeros(N_EDGES, dtype=np.float64)
    cnt = 0
    for _ in range(n_episodes):
        scenario = random.choices(list(SCENARIO_PROBS.keys()),
                                  weights=list(SCENARIO_PROBS.values()), k=1)[0]
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        env.reset()
        done = False
        steps = 0
        while not done and steps < 40:
            acc += env.get_state()[:N_EDGES]
            cnt += 1
            valid = env.get_valid_actions()
            if not valid:
                break
            _, _, done, _ = env.step(random.choice(valid))
            steps += 1
        acc += env.get_state()[:N_EDGES]
        cnt += 1
    return acc / float(cnt)


def save_const(vec):
    os.makedirs(OUT, exist_ok=True)
    with open(CONST_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["slot", "u", "v", "edge_mean_risk"])
        for i, (u, v) in enumerate(EDGES):
            w.writerow([i, u, v, "%.10f" % vec[i]])
    print("상수 벡터 저장: %s" % CONST_CSV)
    print("  대역 평균 %.6f   최소 %.6f   최대 %.6f   sd %.6f"
          % (vec.mean(), vec.min(), vec.max(), vec.std(ddof=1)))


def load_const():
    vec = np.zeros(N_EDGES, dtype=np.float32)
    with open(CONST_CSV, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vec[int(row["slot"])] = float(row["edge_mean_risk"])
    return vec


def const_for(variant, vec):
    if variant == "scalar":
        return np.full(N_EDGES, float(vec.mean()), dtype=np.float32)
    if variant == "edgemean":
        return vec.astype(np.float32)
    raise ValueError(variant)


# ── 환경 래퍼: 상태의 위험 블록만 상수로 덮는다 ────────────────────────────
class RiskBlindEnv(NetworkEnv):
    """상태 앞 50칸을 동결 상수로 덮는다. 보상 경로는 손대지 않는다."""
    CONST = None

    def get_state(self):
        s = super().get_state()
        s[:N_EDGES] = RiskBlindEnv.CONST
        return s


# ── train_factor_ablation.py 와 동일한 에이전트 ────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *tr):
        self.buffer.append(tr)

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
        s, a, r, ns, d, nm = self.memory.sample(BATCH_SIZE)
        s, a, r = s.to(self.device), a.to(self.device), r.to(self.device)
        ns, d, nm = ns.to(self.device), d.to(self.device), nm.to(self.device)
        cur = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target_net(ns)
            nq = (nq + (1.0 - nm) * -1e9).max(1)[0]
            tgt = r + GAMMA * nq * (1 - d)
        loss = nn.HuberLoss()(cur, tgt)
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
    return random.choices(list(SCENARIO_PROBS.keys()),
                          weights=list(SCENARIO_PROBS.values()), k=1)[0]


def make_env(scenario):
    return RiskBlindEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)


def train(variant, run_id, episodes):
    RiskBlindEnv.CONST = const_for(variant, load_const())
    seed = 42 + run_id                      # train_factor_ablation.py:157 과 동일
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = "%s_run%02d" % (variant, run_id)
    if episodes != EPISODES:          # 비용 탐침은 본 산출을 덮지 않는다
        tag += "_ep%d" % episodes
    save_path = os.path.join(MODEL_DIR, tag + ".pth")

    base = make_env("normal")
    agent = DQNAgent(base.state_size, base.action_size)
    assert base.state_size == N_EDGES + NUM_NODES * 3 == 125, base.state_size

    print("===== Risk-Blind DQN  variant=%s  run=%d  seed=%d =====" % (variant, run_id, seed))
    print("Episodes=%d  StateSize=%d  Device=%s  Threads=%d"
          % (episodes, base.state_size, agent.device, torch.get_num_threads()))
    print("상수 블록: 평균 %.6f  sd %.6f (sd 0 이면 정보 전무)"
          % (RiskBlindEnv.CONST.mean(), RiskBlindEnv.CONST.std()))

    succ = deque(maxlen=100)
    rew = deque(maxlen=100)
    t0 = time.time()

    for ep in range(1, episodes + 1):
        env = make_env(sample_scenario())
        state = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done,
                              info["next_valid_mask"])
            agent.train_step()
            state = next_state
            ep_rew += reward
        succ.append(1 if env.current_node == DESTINATION else 0)
        rew.append(ep_rew)
        agent.decay_epsilon()
        if ep % TARGET_UPDATE == 0:
            agent.update_target()
        if ep % 100 == 0:
            el = time.time() - t0
            print("Ep %4d | Suc %5.1f%% | Rew %7.2f | Eps %.4f | %6.1fs (%.3fs/ep)"
                  % (ep, np.mean(succ) * 100, np.mean(rew), agent.epsilon,
                     el, el / ep), flush=True)

    torch.save(agent.policy_net.state_dict(), save_path)
    print("모델 저장: %s" % save_path)

    # ── 평가: train_factor_ablation.py:216- 과 같은 규약 (dynamic, 워밍업 10) ──
    agent.policy_net.eval()
    agent.epsilon = 0.0
    delays, losses, hops, risks = [], [], [], []
    reached = 0
    for _ in range(EVAL_EPISODES):
        env = make_env("dynamic")
        env.reset()
        for _ in range(10):
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
            reached += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
            risks.append(m.get("total_risk", float("nan")))

    row = {
        "variant": variant,
        "run_id": run_id,
        "seed": seed,
        "episodes": episodes,
        "state_size": base.state_size,
        "const_mean": "%.10f" % float(RiskBlindEnv.CONST.mean()),
        "const_sd": "%.10f" % float(RiskBlindEnv.CONST.std(ddof=0)),
        "reach_rate": "%.6f" % (reached / float(EVAL_EPISODES)),
        "delay_mean": "%.6f" % (np.mean(delays) if delays else float("nan")),
        "loss_mean": "%.6f" % (np.mean(losses) if losses else float("nan")),
        "hop_mean": "%.6f" % (np.mean(hops) if hops else float("nan")),
        "risk_mean": "%.6f" % (np.nanmean(risks) if risks else float("nan")),
        "train_sec": "%.1f" % (time.time() - t0),
        "threads": torch.get_num_threads(),
        "model": os.path.basename(save_path),
    }
    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, "risk_blind_%s.csv" % tag)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print("결과 저장: %s" % csv_path)
    for k, v in row.items():
        print("  %-12s %s" % (k, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", help="const | scalar | edgemean")
    ap.add_argument("run_id", nargs="?", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=EPISODES)
    ap.add_argument("--const-episodes", type=int, default=400)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    if a.mode == "const":
        save_const(estimate_const(a.const_episodes))
        return
    if a.mode not in VARIANTS:
        ap.error("mode must be const/scalar/edgemean")
    train(a.mode, a.run_id, a.episodes)


if __name__ == "__main__":
    main()
