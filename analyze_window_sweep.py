"""Re-aggregate the temporal-window sweep from the per-run CSVs and test significance.

results/window_sweep_summary.csv in SDN_DQN_new_env is stale: it reports n_runs=1 and
omits window=1 entirely, while 59 per-run result files exist.  This script rebuilds the
summary from those files and asks whether any window differs from the paper's window=5.

Window=1 matters most: a deque of maxlen 1 has zero standard deviation, so both
instability factors (delay jitter, loss jitter) are identically zero.  Window=1 is
therefore a complete ablation of the temporal mechanism, not merely a shorter memory.

Welch's t-test and the exact Student-t tail come from review_stats, which implements
them directly: scipy is unusable here (scipy 1.11.4 against numpy 2.x), and a normal-tail
approximation is optimistic at n=5.
"""

import csv
import glob
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from review_stats import mean_std, welch  # noqa: E402

RESULTS_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "SDN_DQN_new_env", "results", "window*_*_run*_result.csv",
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_results")
METRICS = ("delay", "loss", "risk", "hop", "success_rate")
BASELINE_WINDOW = 5


def load():
    """samples[(window, scenario)][metric] -> list of per-run values."""
    samples = defaultdict(lambda: defaultdict(list))
    paths = sorted(glob.glob(RESULTS_GLOB))
    if not paths:
        raise SystemExit(f"no per-run result files matched {RESULTS_GLOB}")
    for path in paths:
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (int(row["window"]), row["scenario"])
                for metric in METRICS:
                    samples[key][metric].append(float(row[metric]))
    return samples, len(paths)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    samples, n_files = load()
    print(f"read {n_files} per-run result files\n")

    summary_path = os.path.join(OUT_DIR, "window_sweep_corrected.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["scenario", "window", "n_runs"]
            + [c for m in METRICS for c in (m, f"{m}_std")]
        )
        for scenario in ("normal", "congestion", "unstable", "dynamic"):
            for window in sorted(w for (w, s) in samples if s == scenario):
                vals = samples[(window, scenario)]
                row = [scenario, window, len(vals["delay"])]
                for metric in METRICS:
                    m, sd = mean_std(vals[metric])
                    row += [f"{m:.6f}", f"{sd:.6f}"]
                writer.writerow(row)
    print(f"wrote {summary_path}")

    test_path = os.path.join(OUT_DIR, "window_sweep_significance.csv")
    with open(test_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["scenario", "metric", "window", "n", "mean", "std",
             "baseline_mean", "baseline_n", "diff_vs_w5", "t", "df", "p", "significant_p05"]
        )
        for scenario in ("normal", "congestion", "unstable", "dynamic"):
            base_key = (BASELINE_WINDOW, scenario)
            if base_key not in samples:
                continue
            print(f"\n=== {scenario}: each window vs window={BASELINE_WINDOW} ===")
            for metric in ("delay", "loss", "risk"):
                base = samples[base_key][metric]
                bm, _ = mean_std(base)
                for window in sorted(w for (w, s) in samples if s == scenario):
                    if window == BASELINE_WINDOW:
                        continue
                    vals = samples[(window, scenario)][metric]
                    m, sd = mean_std(vals)
                    diff, t, df, p = welch(vals, base)
                    sig = "yes" if (p == p and p < 0.05) else "no"
                    writer.writerow(
                        [scenario, metric, window, len(vals), f"{m:.6f}", f"{sd:.6f}",
                         f"{bm:.6f}", len(base), f"{diff:+.6f}",
                         f"{t:.4f}" if t == t else "nan",
                         f"{df:.2f}" if df == df else "nan",
                         f"{p:.4f}" if p == p else "nan", sig]
                    )
                    note = "" if len(vals) >= 2 else "  (n=1, not testable)"
                    pstr = f"p={p:.4f}" if p == p else "p=n/a"
                    print(f"  {metric:>5}  w={window:<3} n={len(vals):<2} "
                          f"{m:>10.5f} +/- {sd:<9.5f} diff={diff:+10.5f}  {pstr}  {sig}{note}")
    print(f"\nwrote {test_path}")


if __name__ == "__main__":
    main()
