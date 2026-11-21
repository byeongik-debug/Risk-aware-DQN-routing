# -*- coding: utf-8 -*-
"""Does Table 10 support a factor ranking at all? Every pair, Holm-corrected.

Table 10 names a "most important" factor by eyeballing column means over n=5
runs. This script runs every pairwise comparison and corrects for the 15 tests.

Pairing note (measured, not assumed): train_factor_ablation.py:157-158 sets
seed = 42 + run_id ONCE, then trains 8000 episodes before the 200-episode
evaluation. The RNG state at evaluation time therefore depends on every action
taken during training, so run_id does NOT pair the evaluation conditions across
factors -- it only pairs the initialization. A paired t-test on these 5 pairs
would claim a variance reduction that is not there, so the unpaired Welch test
is the correct one. Both are printed so the difference is visible.

Metrics: the published delay (path SUM) and loss (per-hop MEAN), plus hop, plus
loss*hop (the path total the mean hides -- see section 10 of
FINDINGS_table10_f1.md). If the ranking flips between loss and loss*hop, the
published column is reporting path length.

Usage:  python -X utf8 analyze_table10_pairwise.py
Output: review_results/table10_pairwise.csv
"""
import csv
import glob
import itertools
import os

import review_stats as RS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(HERE, "..", "SDN_DQN_new_env", "results"),
       os.path.join(HERE, "review_results")]
OUT = os.path.join(HERE, "review_results", "table10_pairwise.csv")
METRICS = ["delay", "loss", "hop", "loss_x_hop"]


def paired_t(xs, ys):
    d = [a - b for a, b in zip(xs, ys)]
    n = len(d)
    m, sd = RS.mean_std(d)
    if sd == 0.0:
        return m, float("inf"), n - 1, 0.0
    t = m / (sd / n ** 0.5)
    return m, t, n - 1, RS.t_two_sided_p(abs(t), n - 1)


def load():
    data = {}
    for d in SRC:
        for path in sorted(glob.glob(os.path.join(d, "factor_*_result.csv"))):
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    k = row["factor"]
                    rec = {"run": int(row["run_id"]),
                           "delay": float(row["delay"]),
                           "loss": float(row["loss"]),
                           "hop": float(row["hop"]),
                           "success": float(row["success_rate"])}
                    rec["loss_x_hop"] = rec["loss"] * rec["hop"]
                    data.setdefault(k, []).append(rec)
    for k in data:
        data[k].sort(key=lambda r: r["run"])
    return data


def main():
    data = load()
    keys = sorted(data, key=lambda k: sum(r["delay"] for r in data[k]))
    print("factors: %d" % len(keys))
    for k in keys:
        runs = sorted(r["run"] for r in data[k])
        print("  %-12s n=%d runs=%s" % (k, len(data[k]), runs))
    print()

    hdr = "%-12s | %9s | %9s | %7s | %11s | %6s" % (
        "factor", "delay", "loss", "hop", "loss*hop", "succ%")
    print(hdr)
    print("-" * len(hdr))
    for k in keys:
        g = data[k]
        print("%-12s | %9.2f | %9.5f | %7.3f | %11.6f | %6.1f"
              % (k,
                 sum(r["delay"] for r in g) / len(g),
                 sum(r["loss"] for r in g) / len(g),
                 sum(r["hop"] for r in g) / len(g),
                 sum(r["loss_x_hop"] for r in g) / len(g),
                 sum(r["success"] for r in g) / len(g)))

    rows = []
    for met in METRICS:
        pairs = list(itertools.combinations(keys, 2))
        res = []
        for a, b in pairs:
            xs = [r[met] for r in data[a]]
            ys = [r[met] for r in data[b]]
            if len(xs) != len(ys):
                continue
            diff, tw, dfw, pw = RS.welch(xs, ys)
            _, tp, dfp, pp = paired_t(xs, ys)
            res.append({"a": a, "b": b, "diff": diff, "pw": pw, "tw": tw,
                        "dfw": dfw, "pp": pp, "tp": tp})
        adj = RS.holm_bonferroni([r["pw"] for r in res])
        for r, q in zip(res, adj):
            r["pw_holm"] = q
        print("\n=== %s === (%d pairs, Holm over all %d)"
              % (met, len(res), len(res)))
        h2 = "%-12s %-12s | %11s | %9s | %9s | %9s | %4s" % (
            "A", "B", "A-B", "Welch p", "Holm p", "paired p", "sig")
        print(h2)
        print("-" * len(h2))
        for r in sorted(res, key=lambda x: x["pw"]):
            star = "***" if r["pw_holm"] < 0.05 else (
                " * " if r["pw"] < 0.05 else "   ")
            print("%-12s %-12s | %+11.5f | %9.4f | %9.4f | %9.4f | %4s"
                  % (r["a"], r["b"], r["diff"], r["pw"], r["pw_holm"],
                     r["pp"], star))
            rows.append([met, r["a"], r["b"], "%.6f" % r["diff"],
                         "%.5f" % r["tw"], "%.2f" % r["dfw"],
                         "%.5f" % r["pw"], "%.5f" % r["pw_holm"],
                         "%.5f" % r["pp"],
                         "yes" if r["pw_holm"] < 0.05 else "no"])
        nraw = sum(1 for r in res if r["pw"] < 0.05)
        nadj = sum(1 for r in res if r["pw_holm"] < 0.05)
        npair = sum(1 for r in res if r["pp"] < 0.05)
        print("  significant: raw %d/%d, Holm %d/%d  (paired raw %d/%d)"
              % (nraw, len(res), nadj, len(res), npair, len(res)))

    print("\n=== how many runs would it take? ===")
    print("Minimum detectable effect at alpha=.05 with the observed spread,")
    print("and the n per group needed for 80% power on the OBSERVED gap.")
    h3 = "%-10s %-12s %-12s | %11s | %11s | %7s" % (
        "metric", "A", "B", "observed", "MDE", "n@80%")
    print(h3)
    print("-" * len(h3))
    for met in METRICS:
        pairs = [(keys[0], keys[-1]), (keys[0], keys[1]),
                 ("-dLoss", "-dDelay")]
        for a, b in pairs:
            xs = [r[met] for r in data[a]]
            ys = [r[met] for r in data[b]]
            obs = sum(xs) / len(xs) - sum(ys) / len(ys)
            mde = RS.min_detectable_effect(xs, ys)
            need = RS.required_n(xs, ys)
            print("%-10s %-12s %-12s | %+11.5f | %+11.5f | %7s"
                  % (met, a, b, obs, mde,
                     need if need == need else "n/a"))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "factor_a", "factor_b", "diff_a_minus_b",
                    "welch_t", "welch_df", "welch_p", "welch_p_holm",
                    "paired_p_NOT_VALID", "significant_holm"])
        w.writerows(rows)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
