# eval_window_sweep.py
# 실험 3: Window Sweep — dynamic 시나리오만 대상
#
# window=5(베이스라인) 5개 시드 mean ± std
# window=3,7,10: 각 10회 학습 결과 CSV의 mean ± std (학습 변동성 반영)
# → window_sweep_summary.csv에 std 컬럼 포함 저장

import os, csv, random
import numpy as np
import torch
import torch.nn as nn

from network_env import NetworkEnv
from topology import DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_CSV = os.path.join(BASE_DIR, "results", "baseline_reference.csv")
BACKUP_MODEL = os.path.join(BASE_DIR, "risk_only_model_final.pth")

SCENARIO = "dynamic"
EVAL_EPISODES = 200
SWEEP_WINDOWS = [3, 7, 10]
SWEEP_RUNS = 5
BASELINE_TOLERANCE = 0.30
SEEDS = [0, 1, 2, 3, 4]   # window=5 베이스라인 seed 반복 수


class QNetwork(nn.Module):
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


def _evaluate_one_seed(model, risk_window, seed, episodes=EVAL_EPISODES):
    """단일 시드로 dynamic 시나리오 평가"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, risks, hops = [], [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=risk_window)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device := next(model.parameters()).device)
            with torch.no_grad():
                q_values = model(state_tensor)[0]
            masked_q = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                masked_q[a] = q_values[a]
            action = int(torch.argmax(masked_q).item())
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

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "risk":  float(np.mean(risks))  if risks  else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
    }


def evaluate_model_seeded(model_path, risk_window):
    """window=5 베이스라인: 5개 시드 반복 → mean ± std 반환"""
    if not os.path.exists(model_path):
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=risk_window)
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    except RuntimeError as e:
        print(f"  [경고] 모델 로드 실패: {model_path}\n  {e}")
        return None
    model.eval()

    seed_results = [_evaluate_one_seed(model, risk_window, s) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    r_vals = [r["risk"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":     float(np.mean(d_vals)),
        "delay_std": float(np.std(d_vals, ddof=1)),
        "loss":      float(np.mean(l_vals)),
        "loss_std":  float(np.std(l_vals, ddof=1)),
        "risk":      float(np.mean(r_vals)),
        "hop":       float(np.mean(h_vals)),
        "hop_std":   float(np.std(h_vals, ddof=1)),
    }


def load_baseline_dynamic():
    if not os.path.exists(BASELINE_CSV):
        return None
    with open(BASELINE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("scenario") == "dynamic":
                return {"delay": float(row["delay"]), "loss": float(row["loss"])}
    return None


def main():
    print("=" * 80)
    print(f"  Window Sweep: dynamic 시나리오  (w=5: {len(SEEDS)} seeds, w=3/7/10: {SWEEP_RUNS} runs)")
    print("=" * 80)

    baseline_ref = load_baseline_dynamic()

    # --- window=5: 5개 시드 평가 ---
    print(f"\n[STEP 1] window=5 베이스라인 검증 (seeds={SEEDS})...")
    w5_result = evaluate_model_seeded(BACKUP_MODEL, risk_window=5)
    if w5_result is None:
        print("  [오류] window=5 백업 모델 평가 실패. 파이프라인 중단.")
        return

    if baseline_ref:
        for metric in ["delay", "loss"]:
            ref_val = baseline_ref[metric]
            got_val = w5_result[metric]
            diff = abs(got_val - ref_val) / ref_val if ref_val > 0 else 0
            status = "OK" if diff <= BASELINE_TOLERANCE else f"[경고] {diff*100:.1f}% 차이"
            print(f"  {metric}: 기준={ref_val:.4f}, 측정={got_val:.4f} → {status}")

    # --- window=3,7,10: 10회 학습 결과 CSV 읽어 mean ± std 계산 ---
    result_dir = os.path.join(BASE_DIR, "results")
    sweep_results = {}
    for window in SWEEP_WINDOWS:
        print(f"\n[STEP 2] window={window} 결과 수집 중 ({SWEEP_RUNS} runs)...")
        run_vals = []
        for run_id in range(1, SWEEP_RUNS + 1):
            p = os.path.join(result_dir, f"window{window}_dynamic_run{run_id:02d}_result.csv")
            if not os.path.exists(p):
                print(f"  [경고] 없음: run={run_id}")
                continue
            row = list(csv.DictReader(open(p)))[0]
            run_vals.append({k: float(row[k]) for k in ["delay", "loss", "risk", "hop"]})

        if run_vals:
            sweep_results[window] = run_vals
            print(f"  {len(run_vals)}개 유효, avg_delay={np.mean([x['delay'] for x in run_vals]):.1f}")
        else:
            print(f"  [오류] 유효 결과 없음")

    # --- 결과 출력 ---
    sep = "-" * 80
    print("\n" + "=" * 80)
    print(f"  {'Window':>8s} | {'Delay (mean±std)':>18s} | {'Loss (mean±std)':>22s} | {'N':>4s}")
    print(sep)
    print(f"  {'w=5':>8s} | {w5_result['delay']:6.1f} ± {w5_result['delay_std']:4.1f} | "
          f"{w5_result['loss']:7.5f} ± {w5_result['loss_std']:.5f} | {len(SEEDS):>4d}")
    for window in SWEEP_WINDOWS:
        if window not in sweep_results:
            print(f"  {'w='+str(window):>8s} | {'N/A':>18s} | {'N/A':>22s} | {'0':>4s}")
            continue
        vals = sweep_results[window]
        d_vals = [x["delay"] for x in vals]
        l_vals = [x["loss"]  for x in vals]
        print(f"  {'w='+str(window):>8s} | {np.mean(d_vals):6.1f} ± {np.std(d_vals, ddof=1):4.1f} | "
              f"{np.mean(l_vals):7.5f} ± {np.std(l_vals, ddof=1):.5f} | {len(vals):>4d}")
    print(sep)

    # --- window_sweep_summary.csv 저장 (std 포함) ---
    os.makedirs(result_dir, exist_ok=True)
    out_path = os.path.join(result_dir, "window_sweep_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["window", "delay", "delay_std", "loss", "loss_std", "risk", "hop", "hop_std", "n_runs"])
        writer.writerow([5, w5_result["delay"], w5_result["delay_std"],
                         w5_result["loss"], w5_result["loss_std"],
                         w5_result["risk"], w5_result["hop"], w5_result["hop_std"], len(SEEDS)])
        for window in SWEEP_WINDOWS:
            if window in sweep_results:
                vals = sweep_results[window]
                d_vals = [x["delay"] for x in vals]
                l_vals = [x["loss"]  for x in vals]
                r_vals = [x["risk"]  for x in vals]
                h_vals = [x["hop"]   for x in vals]
                writer.writerow([window,
                                 np.mean(d_vals), np.std(d_vals, ddof=1),
                                 np.mean(l_vals), np.std(l_vals, ddof=1),
                                 np.mean(r_vals),
                                 np.mean(h_vals), np.std(h_vals, ddof=1),
                                 len(vals)])
    print(f"\nCSV 저장: {out_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        BACKUP_MODEL = sys.argv[1]
    main()