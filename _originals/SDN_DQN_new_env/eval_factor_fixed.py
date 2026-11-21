# eval_factor_fixed.py
# 저장된 factor ablation 모델을 고정 시나리오(normal/congestion/unstable)에서 재평가
# 재학습 없이 모델 로드만 함 — dynamic 노이즈 없이 각 factor 기여도 확인

import sys, os, csv
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn

import risk_calculator
from network_env import NetworkEnv
from topology import NUM_NODES, DESTINATION

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
EVAL_EPISODES = 200
RISK_WINDOW   = 5

FACTOR_MAP = {
    "full":       [],
    "util":       ["utilization"],
    "cong":       ["congestion"],
    "delay_inst": ["delay_inst"],
    "loss_inst":  ["loss_inst"],
    "link_type":  ["link_type"],
}
LABEL_MAP = {
    "full":       "Full",
    "util":       "-Util",
    "cong":       "-Cong",
    "delay_inst": "-ΔDelay",
    "loss_inst":  "-ΔLoss",
    "link_type":  "-Type",
}
MODEL_PREFIX = {
    "full":       "window5",
    "util":       "factor_util",
    "cong":       "factor_cong",
    "delay_inst": "factor_delay_inst",
    "loss_inst":  "factor_loss_inst",
    "link_type":  "factor_link_type",
}

SCENARIOS = ["normal", "congestion", "unstable"]


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


def load_model(factor_key, run_id):
    prefix = MODEL_PREFIX[factor_key]
    path = os.path.join(MODEL_DIR, f"{prefix}_run{run_id:02d}.pth")
    if not os.path.exists(path):
        return None, None
    env_tmp = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    net = QNetwork(env_tmp.state_size, env_tmp.action_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.load_state_dict(torch.load(path, map_location=device))
    net.to(device)
    net.eval()
    return net, device


def evaluate(net, device, factor_key, scenario):
    risk_calculator.ZEROED_FACTORS = FACTOR_MAP[factor_key]
    delays, losses, hops = [], [], []

    for _ in range(EVAL_EPISODES):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            st = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = net(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))

        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])

    n = len(delays)
    if n == 0:
        return None
    return {
        "delay": float(np.mean(delays)),
        "loss":  float(np.mean(losses)),
        "hop":   float(np.mean(hops)),
        "success_rate": n / EVAL_EPISODES * 100,
        "n": n,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)
    N_RUNS = 5

    # 결과 수집: results[factor_key][scenario] = list of run dicts
    all_results = {fk: {sc: [] for sc in SCENARIOS} for fk in FACTOR_MAP}

    factors = list(FACTOR_MAP.keys())
    total = len(factors) * N_RUNS * len(SCENARIOS)
    done_count = 0

    for factor_key in factors:
        label = LABEL_MAP[factor_key]
        for run_id in range(1, N_RUNS + 1):
            net, device = load_model(factor_key, run_id)
            if net is None:
                print(f"  [없음] {MODEL_PREFIX[factor_key]}_run{run_id:02d}.pth")
                done_count += len(SCENARIOS)
                continue
            for scenario in SCENARIOS:
                done_count += 1
                print(f"  [{done_count}/{total}] {label} run{run_id} / {scenario} ...", end=" ", flush=True)
                res = evaluate(net, device, factor_key, scenario)
                if res:
                    all_results[factor_key][scenario].append(res)
                    print(f"delay={res['delay']:.1f}  loss={res['loss']:.5f}")
                else:
                    print("실패")

    # 출력
    SEP  = "-" * 100
    LINE = "=" * 100
    print("\n\n" + LINE)
    print("  Factor Ablation — Fixed Scenarios  (mean ± std, 5 seeds, ddof=1)")
    print(LINE)

    for scenario in SCENARIOS:
        print(f"\n  [[ Scenario: {scenario.upper()} ]]")
        print(f"  {'Model':<12s} | {'Delay (mean±std)':>20s} | {'vs Full':>16s} | {'Loss (mean±std)':>22s} | {'vs Full':>16s}")
        print(SEP)

        rows = []
        for factor_key in factors:
            runs = all_results[factor_key][scenario]
            if not runs:
                continue
            d_vals = [r["delay"] for r in runs]
            l_vals = [r["loss"]  for r in runs]
            d_m = float(np.mean(d_vals))
            d_s = float(np.std(d_vals, ddof=1)) if len(d_vals) > 1 else 0.0
            l_m = float(np.mean(l_vals))
            l_s = float(np.std(l_vals, ddof=1)) if len(l_vals) > 1 else 0.0
            rows.append((LABEL_MAP[factor_key], d_m, d_s, l_m, l_s, len(runs)))

        if not rows:
            continue

        base_d, base_l = rows[0][1], rows[0][3]
        for label, d_m, d_s, l_m, l_s, n in rows:
            if label == "Full":
                d_vs = "(baseline)"
                l_vs = "(baseline)"
            else:
                dd = (d_m - base_d) / base_d * 100
                dl = (l_m - base_l) / base_l * 100
                d_vs = f"{d_m - base_d:+.1f} ({dd:+.1f}%)"
                l_vs = f"{l_m - base_l:+.6f} ({dl:+.1f}%)"
            d_str = f"{d_m:6.1f} ± {d_s:4.1f}"
            l_str = f"{l_m:.5f} ± {l_s:.5f}"
            print(f"  {label:<12s} | {d_str:>20s} | {d_vs:>16s} | {l_str:>22s} | {l_vs:>16s}")
        print(SEP)

    # CSV 저장
    csv_path = os.path.join(RESULT_DIR, "factor_ablation_fixed_scenarios.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario", "model", "delay", "delay_std", "loss", "loss_std", "n"])
        for scenario in SCENARIOS:
            for factor_key in factors:
                runs = all_results[factor_key][scenario]
                if not runs:
                    continue
                d_vals = [r["delay"] for r in runs]
                l_vals = [r["loss"]  for r in runs]
                w.writerow([
                    scenario, LABEL_MAP[factor_key],
                    f"{np.mean(d_vals):.4f}", f"{np.std(d_vals, ddof=1):.4f}" if len(d_vals)>1 else "0",
                    f"{np.mean(l_vals):.6f}", f"{np.std(l_vals, ddof=1):.6f}" if len(l_vals)>1 else "0",
                    len(runs)
                ])
    print(f"\n  CSV: {csv_path}")


if __name__ == "__main__":
    main()
