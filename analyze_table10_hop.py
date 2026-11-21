# -*- coding: utf-8 -*-
"""Is Table 10's Delay/Loss column measuring quality, or path length?

calculate_path_metrics() in network_env.py accumulates per-link values along
the path and then normalizes ONE of the two columns and not the other:

    "total_delay": total_delay,              <- path SUM
    "total_loss":  total_loss / hop_count,   <- per-hop MEAN
    "total_risk":  total_risk / hop_count,   <- per-hop MEAN

So hop count pushes the two columns in OPPOSITE directions by construction: one
extra hop adds that link's delay to the sum and dilutes the loss mean. A policy
that walks longer paths therefore looks worse on Delay and better on Loss no
matter which links it picked. This script measures how much of Table 10's spread
that accounts for, using the 25 per-run result files that exist.

Because Loss is already per-hop, dividing it by hop again is meaningless. The
companion column for Loss is loss * hop -- the path total that the mean hides.

Note: the `full` key has no per-run file and no saved model, so the reference row
cannot enter this regression. That is itself a finding, reported at the end.

Usage:  python -X utf8 analyze_table10_hop.py
"""
import csv
import glob
import os

import review_stats as RS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "SDN_DQN_new_env", "results")
OUT = os.path.join(HERE, "review_results", "table10_hop_normalization.csv")


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # slope t-test
    se = (ss_res / (n - 2) / sxx) ** 0.5 if n > 2 and sxx > 0 else float("nan")
    t = b / se if se and se == se else float("nan")
    p = RS.t_two_sided_p(abs(t), n - 2) if t == t else float("nan")
    return a, b, r2, p


def main():
    runs = []
    for path in sorted(glob.glob(os.path.join(SRC, "factor_*_result.csv"))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                runs.append({
                    "factor": row["factor"],
                    "run": int(row["run_id"]),
                    "delay": float(row["delay"]),
                    "loss": float(row["loss"]),
                    "hop": float(row["hop"]),
                    "success": float(row["success_rate"]),
                })
    print("per-run files found: %d" % len(runs))
    keys = sorted(set(r["factor"] for r in runs))
    print("factors present: %s" % keys)
    print("factors MISSING: Full (no per-run csv, no saved model)\n")

    print("per-factor means (n=5 each)")
    hdr = "%-9s | %8s | %8s | %6s | %7s | %8s | %9s" % (
        "factor", "delay", "loss", "hop", "succ%", "ms/hop", "loss*hop")
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for k in keys:
        g = [r for r in runs if r["factor"] == k]
        d = sum(r["delay"] for r in g) / len(g)
        lo = sum(r["loss"] for r in g) / len(g)
        h = sum(r["hop"] for r in g) / len(g)
        sc = sum(r["success"] for r in g) / len(g)
        agg[k] = (d, lo, h, sc)
        print("%-9s | %8.2f | %8.5f | %6.3f | %7.1f | %8.2f | %9.6f"
              % (k, d, lo, h, sc, d / h, lo * h))

    print("\nregression across all %d runs" % len(runs))
    xs = [r["hop"] for r in runs]
    series = [("delay", [r["delay"] for r in runs]),
              ("loss", [r["loss"] for r in runs]),
              ("lo*hop", [r["loss"] * r["hop"] for r in runs])]
    for nm, ys in series:
        a, b, r2, p = ols(xs, ys)
        print("  %-6s = %+.5f + %+.5f * hop     R^2 = %.4f   p(slope) = %.3g"
              % (nm, a, b, r2, p))
    print("  (delay is a path sum, loss a per-hop mean -- the opposite signs"
          " are built in; lo*hop undoes the mean)")

    print("\nsame regression on the 5 factor means (n=5)")
    xs2 = [agg[k][2] for k in keys]
    for i, nm in ((0, "delay"), (1, "loss")):
        ys2 = [agg[k][i] for k in keys]
        a, b, r2, p = ols(xs2, ys2)
        print("  %-5s  R^2 = %.4f   slope %+.5f per hop   p = %.3g"
              % (nm, r2, b, p))

    print("\nthe same numbers with each column on a common basis")
    hdr2 = "%-9s | %11s | %11s | %13s | %13s | %6s" % (
        "factor", "delay(sum)", "delay/hop", "loss(perhop)", "loss*hop", "hop")
    print(hdr2)
    print("-" * len(hdr2))
    ref = min(keys, key=lambda k: agg[k][0])
    for k in keys:
        d, lo, h, _ = agg[k]
        rd, rlo, rh, _ = agg[ref]
        print("%-9s | %+10.1f%% | %+10.1f%% | %+12.1f%% | %+12.1f%% | %+5.1f%%"
              % (k,
                 100.0 * (d - rd) / rd,
                 100.0 * (d / h - rd / rh) / (rd / rh),
                 100.0 * (lo - rlo) / rlo,
                 100.0 * (lo * h - rlo * rh) / (rlo * rh),
                 100.0 * (h - rh) / rh))
    print("  (percent vs the best-delay factor, %s)" % ref)
    print("  loss(perhop) is the column Table 10 prints; loss*hop is the same")
    print("  data with the hop-count dilution removed.")

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "n", "delay_sum", "loss_per_hop", "hop",
                    "success_rate", "delay_per_hop", "loss_path_total"])
        for k in keys:
            d, lo, h, sc = agg[k]
            w.writerow([k, 5, "%.4f" % d, "%.6f" % lo, "%.4f" % h,
                        "%.2f" % sc, "%.4f" % (d / h), "%.6f" % (lo * h)])
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
