# eval_baseline.py
# 학습된 Risk-DQN(risk_only_model_final.pth)을 5개 시드로 반복 평가
# 결과를 baseline_reference.csv로 저장 → 이후 모든 실험의 기준선(baseline)으로 사용
# 시드별로 환경의 무작위 QoS 변화 경로가 달라짐 → mean ± std로 통계적 신뢰도 확보

import os, csv, random
import numpy as np
import torch
import torch.nn as nn

from network_env import NetworkEnv
from topology import DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKUP_MODEL = os.path.join(BASE_DIR, "risk_only_model_final.pth")
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_CSV = os.path.join(RESULT_DIR, "baseline_reference.csv")

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EVAL_EPISODES = 200
RISK_WINDOW = 5       # 베이스라인은 window=5로 고정
SEEDS = [0, 1, 2, 3, 4]   # 5개 시드로 반복 → mean ± std


class QNetwork(nn.Module):
    """train_window_dqn.py와 동일한 구조 — 저장된 가중치를 불러오기 위해 동일해야 함"""
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


def _evaluate_one_seed(model, scenario, device, seed, episodes=EVAL_EPISODES):
    """단일 시드로 Risk-DQN 평가 → 이 시드의 평균값 반환
    모델 가중치는 고정, 환경의 무작위성(QoS 변화)만 시드로 제어"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, risks, hops = [], [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        env.reset()
        # 10번 워밍업: 네트워크가 시나리오 상태(혼잡/불안정 등)에 도달하게 함
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)[0]
            # 유효하지 않은 노드는 -무한대로 마스킹
            masked_q = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                masked_q[a] = q_values[a]
            action = int(torch.argmax(masked_q).item())
            state, _, done, _ = env.step(action)

        if env.current_node == DESTINATION:
            success_count += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            risks.append(m["total_risk"])
            hops.append(m["hop_count"])
            hop_count = max(m["hop_count"], 1)
            unstable_usages.append(m["unstable_count"] / hop_count * 100)
            congested_usages.append(m["congested_count"] / hop_count * 100)

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "risk":  float(np.mean(risks))  if risks  else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
        "unstable_usage":  float(np.mean(unstable_usages))  if unstable_usages  else 0.0,
        "congested_usage": float(np.mean(congested_usages)) if congested_usages else 0.0,
    }


def evaluate(model, scenario, device):
    """Risk-DQN을 5개 시드로 반복 평가 → mean ± std (ddof=1) 반환"""
    seed_results = [_evaluate_one_seed(model, scenario, device, s) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    r_vals = [r["risk"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "scenario": scenario,
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":      float(np.mean(d_vals)),
        "delay_std":  float(np.std(d_vals, ddof=1)),
        "loss":       float(np.mean(l_vals)),
        "loss_std":   float(np.std(l_vals, ddof=1)),
        "risk":       float(np.mean(r_vals)),
        "risk_std":   float(np.std(r_vals, ddof=1)),
        "hop":        float(np.mean(h_vals)),
        "hop_std":    float(np.std(h_vals, ddof=1)),
        "unstable_usage":  float(np.mean([r["unstable_usage"]  for r in seed_results])),
        "congested_usage": float(np.mean([r["congested_usage"] for r in seed_results])),
    }


def main():
    print("=" * 70)
    print(f"  Risk-DQN 베이스라인 평가  (seeds={SEEDS})")
    print(f"  모델: {BACKUP_MODEL}")
    print("=" * 70)

    if not os.path.exists(BACKUP_MODEL):
        print(f"[오류] 모델 파일 없음: {BACKUP_MODEL}")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(BACKUP_MODEL, map_location=device, weights_only=False))
    model.eval()  # 평가 모드 (드롭아웃/배치정규화 비활성화)

    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for sc in SCENARIOS:
        print(f"\n[평가 중] {sc}  (5 seeds × {EVAL_EPISODES} episodes)...")
        r = evaluate(model, sc, device)
        results.append(r)
        print(
            f"  Delay={r['delay']:.1f} ± {r['delay_std']:.1f} | "
            f"Loss={r['loss']:.5f} ± {r['loss_std']:.5f} | "
            f"Risk={r['risk']:.4f} ± {r['risk_std']:.4f}"
        )

    # baseline_reference.csv 저장 (std 컬럼 포함)
    fields = ["scenario", "success_rate",
              "delay", "delay_std", "loss", "loss_std",
              "risk", "risk_std", "hop", "hop_std",
              "unstable_usage", "congested_usage"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n기준 저장 완료: {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()