# eval_exp4_nowarmup.py
# Exp 4 재실험: warm-up 제거 버전
# 훈련 시 warm-up 없었으므로 평가도 동일 조건으로 맞춤
# 혼잡 링크가 delay/loss에 반영되기 전 시점 → Risk 특성의 실제 가치 측정

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
from network_env import NetworkEnv
from topology import DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RISK_MODEL   = os.path.join(BASE_DIR, "risk_only_model_final.pth")
NORISK_MODEL = os.path.join(BASE_DIR, "norisk_model_final.pth")
OUT_CSV      = os.path.join(BASE_DIR, "results", "exp4_nowarmup.csv")

SCENARIOS     = ["normal", "congestion", "unstable", "dynamic"]
EVAL_EPISODES = 200
SEEDS         = [0, 1, 2, 3, 4]


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


def evaluate_one_seed(model, use_risk, scenario, seed, episodes=EVAL_EPISODES):
    np.random.seed(seed)
    random.seed(seed)
    delays, losses, hops = [], [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=use_risk, scenario=scenario, risk_window=5)
        env.reset()
        # warm-up 없음 — 링크 QoS가 반영되기 전 시점에서 평가
        state = env.get_state()
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            st = torch.FloatTensor(state).unsqueeze(0).to(next(model.parameters()).device)
            with torch.no_grad():
                q = model(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            action = int(torch.argmax(masked).item())
            state, _, done, _ = env.step(action)

        if env.current_node == DESTINATION:
            success_count += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
    }


def evaluate_model(model_path, use_risk, label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = NetworkEnv(use_risk=use_risk, scenario="normal", risk_window=5)
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    results = {}
    for sc in SCENARIOS:
        seed_results = [evaluate_one_seed(model, use_risk, sc, s) for s in SEEDS]
        d = [r["delay"] for r in seed_results]
        l = [r["loss"]  for r in seed_results]
        h = [r["hop"]   for r in seed_results]
        results[sc] = {
            "model": label,
            "scenario": sc,
            "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
            "delay":     float(np.mean(d)),
            "delay_std": float(np.std(d, ddof=1)),
            "loss":      float(np.mean(l)),
            "loss_std":  float(np.std(l, ddof=1)),
            "hop":       float(np.mean(h)),
            "hop_std":   float(np.std(h, ddof=1)),
        }
        print(f"  [{label}] {sc}: Delay={results[sc]['delay']:.1f}±{results[sc]['delay_std']:.1f} | "
              f"Loss={results[sc]['loss']:.5f}±{results[sc]['loss_std']:.5f}")
    return results


def main():
    print("=" * 70)
    print("  Exp 4 (No Warm-up): Risk-DQN vs NoRisk-DQN")
    print("  훈련 조건 동일: warm-up 없음, 링크 QoS 반영 전 시점에서 평가")
    print("=" * 70)

    print("\n[Risk-DQN 평가 중...]")
    risk_res   = evaluate_model(RISK_MODEL,   use_risk=True,  label="Risk-DQN")

    print("\n[NoRisk-DQN 평가 중...]")
    norisk_res = evaluate_model(NORISK_MODEL, use_risk=False, label="NoRisk")

    # 결과 출력
    sep = "-" * 90
    print("\n" + "=" * 90)
    print(f"  {'Scenario':12s} | {'Model':>10s} | {'Delay (mean±std)':>18s} | "
          f"{'Loss (mean±std)':>22s} | {'Hop':>5s}")
    print(sep)

    for sc in SCENARIOS:
        r  = risk_res[sc]
        nr = norisk_res[sc]
        print(f"  {sc:12s} | {'Risk-DQN':>10s} | {r['delay']:6.1f} ± {r['delay_std']:4.1f} | "
              f"{r['loss']:7.5f} ± {r['loss_std']:.5f} | {r['hop']:5.2f}")
        dd = nr['delay'] - r['delay']
        dl = nr['loss']  - r['loss']
        dd_pct = dd / r['delay'] * 100 if r['delay'] else 0
        dl_pct = dl / r['loss']  * 100 if r['loss']  else 0
        print(f"  {sc:12s} | {'NoRisk':>10s} | {nr['delay']:6.1f} ± {nr['delay_std']:4.1f} | "
              f"{nr['loss']:7.5f} ± {nr['loss_std']:.5f} | {nr['hop']:5.2f}  "
              f"(Δdelay={dd:+.1f}/{dd_pct:+.1f}%, Δloss={dl:+.5f}/{dl_pct:+.1f}%)")
        print(sep)

    # CSV 저장
    os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)
    fields = ["model", "scenario", "success_rate",
              "delay", "delay_std", "loss", "loss_std", "hop", "hop_std"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sc in SCENARIOS:
            w.writerow({k: risk_res[sc][k]   for k in fields})
            w.writerow({k: norisk_res[sc][k] for k in fields})
    print(f"\nCSV 저장: {OUT_CSV}")


if __name__ == "__main__":
    main()