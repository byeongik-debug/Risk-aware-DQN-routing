# eval_factor_summary.py
# Factor Ablation 결과 집계 (Exp 5)
# Full: 기존 window5_dynamic_run01~05 재사용 (8000ep, 5 seeds)
# 나머지 5 variant: factor_{name}_dynamic_run01~05 (8000ep, 5 seeds)

import os, sys, csv
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
N_RUNS     = 5

VARIANTS = [
    ("Full",    "window5"),
    ("-Util",   "factor_util"),
    ("-Cong",   "factor_cong"),
    ("-ΔDelay", "factor_delay_inst"),
    ("-ΔLoss",  "factor_loss_inst"),
    ("-Type",   "factor_link_type"),
]


def load_runs(prefix, n_runs=N_RUNS):
    rows = []
    for run_id in range(1, n_runs + 1):
        path = os.path.join(RESULT_DIR, f"{prefix}_dynamic_run{run_id:02d}_result.csv")
        if not os.path.exists(path):
            print(f"  [없음] {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            r = next(csv.DictReader(f))
            rows.append({"delay": float(r["delay"]), "loss": float(r["loss"]),
                         "hop": float(r["hop"])})
    return rows


def stats(rows, key):
    vals = [r[key] for r in rows]
    mean = float(np.mean(vals))
    std  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std, len(vals)


def main():
    SEP  = "-" * 96
    LINE = "=" * 96

    print(LINE)
    print("  Exp 5: Factor Ablation  (Dynamic scenario only, mean ± std, 5 seeds)")
    print("  Full: window5 8000ep (기존)  |  나머지: 8000ep 재학습  |  ddof=1")
    print(LINE)
    print(f"  {'Model':<12s} | {'Delay (mean±std)':>20s} | {'vs Full':>14s} | "
          f"{'Loss (mean±std)':>22s} | {'vs Full':>14s} | n")
    print(SEP)

    results = []
    for label, prefix in VARIANTS:
        rows = load_runs(prefix)
        if not rows:
            print(f"  {label:<12s} | 데이터 없음")
            continue
        d_m, d_s, n = stats(rows, "delay")
        l_m, l_s, _ = stats(rows, "loss")
        results.append((label, d_m, d_s, l_m, l_s, n))

    if not results:
        print("  결과 파일이 없습니다.")
        return

    base_d, base_l = results[0][1], results[0][3]

    for label, d_m, d_s, l_m, l_s, n in results:
        if label == "Full":
            d_vs = "(baseline)"
            l_vs = "(baseline)"
        else:
            dd   = (d_m - base_d) / base_d * 100
            dl   = (l_m - base_l) / base_l * 100
            d_vs = f"{d_m - base_d:+.1f} ({dd:+.1f}%)"
            l_vs = f"{l_m - base_l:+.6f} ({dl:+.1f}%)"

        d_str = f"{d_m:6.1f} ± {d_s:4.1f}"
        l_str = f"{l_m:.5f} ± {l_s:.5f}"
        print(f"  {label:<12s} | {d_str:>20s} | {d_vs:>14s} | "
              f"{l_str:>22s} | {l_vs:>14s} | n={n}")

    print(SEP)

    out = os.path.join(RESULT_DIR, "factor_ablation_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "delay", "delay_std", "loss", "loss_std", "n"])
        for label, d_m, d_s, l_m, l_s, n in results:
            w.writerow([label, f"{d_m:.4f}", f"{d_s:.4f}",
                        f"{l_m:.6f}", f"{l_s:.6f}", n])
    print(f"\n  CSV 저장: {out}")


if __name__ == "__main__":
    main()