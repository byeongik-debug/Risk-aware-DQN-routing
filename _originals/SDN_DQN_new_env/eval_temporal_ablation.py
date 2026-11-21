# eval_temporal_ablation.py
# Temporal Ablation: 같은 모델(window=5로 학습)을 두 가지 조건으로 5개 시드 반복 평가
#
# 조건 A: risk_window=5 (시계열 위험도 히스토리 5칸 → temporal 정보 있음)
# 조건 B: risk_window=1 (현재 순간만 → 사실상 temporal 정보 없음)
#
# 목적: 시계열 위험도 정보가 라우팅 성능에 실제로 기여하는지 확인
# 같은 모델을 쓰므로 학습 능력 차이가 아닌 '입력 정보의 질' 차이만 측정

import os, csv, random
import numpy as np
import torch
import torch.nn as nn
from network_env import NetworkEnv
from topology import DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "risk_only_model_final.pth")
RESULT_DIR = os.path.join(BASE_DIR, "results")
SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EVAL_EPISODES = 200
SEEDS = [0, 1, 2, 3, 4]   # 5개 시드 → mean ± std


class QNetwork(nn.Module):
    """risk_dqn_every_scenario.py와 동일한 구조"""
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


def _evaluate_one_seed(model, device, scenario, risk_window, seed, episodes=EVAL_EPISODES):
    """단일 시드로 평가 → 이 시드의 평균값 반환"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, hops = [], [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=risk_window)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(state_t)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid_actions:
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


def evaluate(model, device, scenario, risk_window):
    """5개 시드로 반복 평가 → mean ± std (ddof=1) 반환"""
    seed_results = [_evaluate_one_seed(model, device, scenario, risk_window, s) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":      float(np.mean(d_vals)),
        "delay_std":  float(np.std(d_vals, ddof=1)),
        "loss":       float(np.mean(l_vals)),
        "loss_std":   float(np.std(l_vals, ddof=1)),
        "hop":        float(np.mean(h_vals)),
        "hop_std":    float(np.std(h_vals, ddof=1)),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=5)
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    print("=" * 70)
    print(f"  Temporal Ablation — 같은 모델, 두 조건 비교  (seeds={SEEDS})")
    print("  조건 A: window=5 (temporal 정보 있음)")
    print("  조건 B: window=1 (temporal 정보 없음, 현재 순간만)")
    print("=" * 70)

    results = []
    for scenario in SCENARIOS:
        print(f"\n[{scenario}]")
        for window in [5, 1]:
            r = evaluate(model, device, scenario, risk_window=window)
            r.update({"scenario": scenario, "window": window})
            results.append(r)
            print(f"  window={window} | "
                  f"Delay={r['delay']:.1f} ± {r['delay_std']:.1f} | "
                  f"Loss={r['loss']:.5f} ± {r['loss_std']:.5f}")

    # 결과 요약 출력
    sep = "-" * 70
    print("\n" + "=" * 70)
    print(f"  {'Scenario':12s} | {'Window':>6s} | {'Delay (mean±std)':>18s} | {'Loss (mean±std)':>22s}")
    print(sep)
    for r in results:
        d_str = f"{r['delay']:6.1f} ± {r['delay_std']:4.1f}"
        l_str = f"{r['loss']:7.5f} ± {r['loss_std']:.5f}"
        print(f"  {r['scenario']:12s} | {'w='+str(r['window']):>6s} | {d_str:>18s} | {l_str:>22s}")
    print(sep)

    # temporal_ablation_same_model.csv 저장 (std 컬럼 포함)
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, "temporal_ablation_same_model.csv")
    fields = ["scenario", "window", "success_rate",
              "delay", "delay_std", "loss", "loss_std", "hop", "hop_std"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV 저장: {out}")


if __name__ == "__main__":
    main()