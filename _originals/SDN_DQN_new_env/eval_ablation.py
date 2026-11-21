# eval_ablation.py
# 실험 2: Temporal Ablation — 별도 학습된 모델 비교 (올바른 Ablation)
#
# Proposed  (window=5): Temporal Risk 포함 상태로 처음부터 학습
# NoTemporal(window=1): Temporal Risk = 0인 상태로 처음부터 학습
#                       (delay_instability, loss_instability가 항상 0)
#
# 각 5회 학습 run의 결과 CSV를 읽어 mean ± std 계산
# 결과 저장: results/ablation_summary.csv → show_results.py Exp 2에서 사용

import os
import csv

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
N_RUNS = 5   # 각 window에서 5회 학습 run


def load_runs(window_size, scenario, n_runs=N_RUNS):
    """window{w}_{scenario}_run{id}_result.csv 파일들을 읽어 run별 결과 반환"""
    rows = []
    for run_id in range(1, n_runs + 1):
        csv_path = os.path.join(RESULT_DIR, f"window{window_size}_{scenario}_run{run_id:02d}_result.csv")
        if not os.path.exists(csv_path):
            print(f"  [경고] 없음: window={window_size} {scenario} run={run_id}")
            continue
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rows.append({k: float(row[k]) for k in ["success_rate", "delay", "loss", "risk", "hop"]})
    return rows


def compute_stats(rows, keys=("delay", "loss", "risk", "hop")):
    """run 목록에서 mean ± std (ddof=1) 계산"""
    if not rows:
        return None
    stats = {"n": len(rows)}
    for k in keys:
        vals = [r[k] for r in rows]
        stats[k]          = float(np.mean(vals))
        stats[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    stats["success_rate"] = float(np.mean([r["success_rate"] for r in rows]))
    return stats


def main():
    print("=" * 80)
    print(f"  Temporal Ablation: Proposed(w=5) vs NoTemporal(w=1)  (각 {N_RUNS}회 학습 run)")
    print("  ※ 같은 모델로 평가 window만 바꾸는 것이 아니라, 각각 처음부터 별도 학습한 결과")
    print("=" * 80)

    all_stats = {"w5": {}, "w1": {}}

    sep = "-" * 80
    print(f"\n  {'Scenario':12s} | {'Window':>12s} | {'Delay (mean±std)':>18s} | {'Loss (mean±std)':>22s} | "
          f"{'Risk (mean±std)':>18s} | {'N':>4s}")
    print(sep)

    for sc in SCENARIOS:
        for w, label, key in [(5, "Proposed(w=5)", "w5"), (1, "NoTemporal(w=1)", "w1")]:
            rows = load_runs(w, sc)
            stats = compute_stats(rows)
            all_stats[key][sc] = stats

            if stats:
                d_str = f"{stats['delay']:6.1f} ± {stats['delay_std']:4.1f}"
                l_str = f"{stats['loss']:7.5f} ± {stats['loss_std']:.5f}"
                r_str = f"{stats['risk']:6.4f} ± {stats['risk_std']:.4f}"
                print(f"  {sc:12s} | {label:>12s} | {d_str:>18s} | {l_str:>22s} | {r_str:>18s} | {stats['n']:>4d}")
            else:
                print(f"  {sc:12s} | {label:>12s} | {'N/A':>18s} | {'N/A':>22s} | {'N/A':>18s} | {'0':>4s}")
        print(sep)

    # ablation_summary.csv 저장 (std 컬럼 포함)
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, "ablation_summary.csv")
    fields = ["scenario", "window",
              "delay", "delay_std", "loss", "loss_std",
              "risk", "risk_std", "hop", "hop_std",
              "success_rate", "n_runs"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sc in SCENARIOS:
            for w, key in [(5, "w5"), (1, "w1")]:
                stats = all_stats[key][sc]
                if stats is None:
                    continue
                writer.writerow({
                    "scenario": sc, "window": w,
                    "delay": stats["delay"], "delay_std": stats["delay_std"],
                    "loss": stats["loss"], "loss_std": stats["loss_std"],
                    "risk": stats["risk"], "risk_std": stats["risk_std"],
                    "hop": stats["hop"], "hop_std": stats["hop_std"],
                    "success_rate": stats["success_rate"], "n_runs": stats["n"],
                })
    print(f"\nCSV 저장: {out_path}")


if __name__ == "__main__":
    main()