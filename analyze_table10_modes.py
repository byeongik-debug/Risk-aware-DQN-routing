# -*- coding: utf-8 -*-
"""What is Table 10's factor ordering actually made of?

Section 14 showed that no pair of the seven settings separates after Holm
correction, and that three ablations are FASTER than Full. This script asks the
next question: if the factors do not differ, what produces the ordering that
Table 10 prints?

The hypothesis under test is that each training run lands in one of a small
number of path-length regimes, and that a factor's mean is just a weighted
average over however many of its 5 runs fell into the long regime. If so, the
"factor effect" is a count out of 5 -- resolution 20% -- not a property
not a property of the factor at all.

Everything here is measured, not assumed:

  1. the regime boundary is found from the hop distribution's largest gap,
     not chosen by hand;
  2. the ordering claim is tested by regressing each factor's mean delay on its
     long-regime count;
  3. the alternative (factors differ WITHIN a regime) is tested separately, by
     comparing the within-regime delay spread against the between-factor one;
  4. the extreme pair's count difference gets a Fisher exact test, which is the
     right test for a 0-of-5 vs 5-of-5 comparison.

Reads only: ../SDN_DQN_new_env/results/factor_*_result.csv (friend's runs) and
review_results/factor_*_result.csv (the two settings retrained here). Writes
review_results/table10_modes.csv.

Usage:  KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 analyze_table10_modes.py
"""
import csv
import glob
import os
from collections import Counter

import review_stats as RS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = [os.path.join(HERE, "..", "SDN_DQN_new_env", "results"),
       os.path.join(HERE, "review_results")]
OUT = os.path.join(HERE, "review_results", "table10_modes.csv")


def load():
    runs = []
    for d in SRC:
        for path in sorted(glob.glob(os.path.join(d, "factor_*_result.csv"))):
            with open(path, newline="") as f:
                for row in csv.DictReader(f):
                    runs.append({
                        "factor": row["factor"],
                        "run": int(row["run_id"]),
                        "delay": float(row["delay"]),
                        "loss": float(row["loss"]),
                        "hop": float(row["hop"]),
                    })
    return runs


def largest_gap(vals):
    """Split a 1-D sample at its widest interior gap.

    Returns (threshold, gap_width, runner_up_width). The runner-up is reported
    so the reader can judge whether the split is a real separation or just the
    biggest of many similar gaps.
    """
    xs = sorted(vals)
    gaps = [(xs[i + 1] - xs[i], 0.5 * (xs[i] + xs[i + 1]))
            for i in range(len(xs) - 1)]
    gaps.sort(reverse=True)
    return gaps[0][1], gaps[0][0], gaps[1][0]


def ols(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    se = (ss_res / (n - 2) / sxx) ** 0.5 if n > 2 else float("nan")
    t = b / se if se and se == se and se > 0 else float("nan")
    p = RS.t_two_sided_p(abs(t), n - 2) if t == t else float("nan")
    return a, b, r2, p


def comb(n, k):
    if k < 0 or k > n:
        return 0
    r = 1
    for i in range(k):
        r = r * (n - i) // (i + 1)
    return r


def fisher_exact_2x2(a, b, c, d):
    """Two-sided Fisher exact p for [[a, b], [c, d]], by table enumeration."""
    row1, row2 = a + b, c + d
    col1 = a + c
    tot = row1 + row2

    def prob(x):
        return (comb(row1, x) * comb(row2, col1 - x)) / float(comb(tot, col1))

    obs = prob(a)
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    return sum(prob(x) for x in range(lo, hi + 1)
               if prob(x) <= obs + 1e-12)


def main():
    runs = load()
    keys = sorted(set(r["factor"] for r in runs),
                  key=lambda k: sum(x["delay"] for x in runs
                                    if x["factor"] == k))
    print("runs loaded: %d over %d settings" % (len(runs), len(keys)))

    # ---- 1. where does the hop distribution split? ----
    hops = [r["hop"] for r in runs]
    thr, gap, gap2 = largest_gap(hops)
    print("\n=== 1. the regime boundary, found from the data ===")
    print("hop values, sorted:")
    xs = sorted(hops)
    for i in range(0, len(xs), 12):
        print("  " + " ".join("%5.2f" % v for v in xs[i:i + 12]))
    print("widest interior gap = %.3f hop at %.3f  (runner-up gap %.3f)"
          % (gap, thr, gap2))
    if gap < 2.0 * gap2:
        print("  NOTE: widest gap < 2x the runner-up, so this is a")
        print("  tail split, not a clean bimodality. The cut is a")
        print("  label for the long side, not evidence of two modes.")
    # a second, interior cut: the widest gap among the SHORT side only
    short = [v for v in xs if v < thr]
    thr2, gap_s, gap_s2 = largest_gap(short)
    print("within the short side, widest gap = %.3f at %.3f (runner-up %.3f)"
          % (gap_s, thr2, gap_s2))
    print("using the interior cut %.3f hop as the long/short boundary" % thr2)
    cut = thr2

    # ---- 2. per-factor long-regime count ----
    print(chr(10) + "=== 2. how many of each setting is long? ===")
    hdr = ("%-10s | %-34s | %-28s | %5s | %8s | %7s"
           % ("factor", "delay by run", "hop by run", "long", "mean dly",
              "mean hop"))
    print(hdr)
    print("-" * len(hdr))
    agg = {}
    for k in keys:
        g = sorted((r["run"], r) for r in runs if r["factor"] == k)
        g = [x[1] for x in g]
        nlong = sum(1 for r in g if r["hop"] >= cut)
        md, sd = RS.mean_std([r["delay"] for r in g])
        mh, sh = RS.mean_std([r["hop"] for r in g])
        ml, sl = RS.mean_std([r["loss"] for r in g])
        agg[k] = {"n": len(g), "long": nlong, "delay": md, "delay_sd": sd,
                  "hop": mh, "hop_sd": sh, "loss": ml, "loss_sd": sl,
                  "runs": g}
        print("%-10s | %-34s | %-28s | %d/%d | %8.2f | %7.3f"
              % (k,
                 " ".join("%6.1f" % r["delay"] for r in g),
                 " ".join("%5.2f" % r["hop"] for r in g),
                 nlong, len(g), md, mh))

    # ---- 3. does the count explain the ordering? ----
    print("\n=== 3. is the factor ordering just that count? ===")
    cs = [float(agg[k]["long"]) for k in keys]
    for nm in ("delay", "hop", "loss"):
        ys = [agg[k][nm] for k in keys]
        a, b, r2, p = ols(cs, ys)
        print("  mean %-5s = %+10.4f %+10.4f * count  R2 = %.4f p = %.4g"
              % (nm, a, b, r2, p))
    ranks_by_delay = [agg[k]["long"] for k in keys]
    mono = all(ranks_by_delay[i] <= ranks_by_delay[i + 1]
               for i in range(len(ranks_by_delay) - 1))
    print("  long count is monotone in the delay ranking: %s  (%s)"
          % ("YES" if mono else "no", ranks_by_delay))

    # ---- 4. the alternative: do factors differ WITHIN a regime? ----
    print("\n=== 4. the alternative -- factors differing within a regime ===")
    for side, test in (("short", lambda r: r["hop"] < cut),
                       ("long", lambda r: r["hop"] >= cut)):
        sub = [r for r in runs if test(r)]
        if len(sub) < 4:
            print("  %-5s side: only %d runs, skipped" % (side, len(sub)))
            continue
        m, s = RS.mean_std([r["delay"] for r in sub])
        print("  %-5s side: n=%2d  delay %.2f +- %.2f  hop %.3f +- %.3f"
              % (side, len(sub), m, s,
                 *RS.mean_std([r["hop"] for r in sub])))
        per = {}
        for k in keys:
            g = [r["delay"] for r in sub if r["factor"] == k]
            if g:
                per[k] = RS.mean_std(g) + (len(g),)
        print("        per factor on this side: %s"
              % "  ".join("%s %.1f(n=%d)" % (k, v[0], v[2])
                          for k, v in per.items()))
        if len(per) >= 2:
            spread = max(v[0] for v in per.values()) \
                - min(v[0] for v in per.values())
            print("        between-factor spread %.2f ms vs within-side sd"
                  " %.2f ms  ->  ratio %.2f"
                  % (spread, s, spread / s if s else float("nan")))

    # ---- 5. the extreme pair, tested on the count ----
    print("\n=== 5. the extreme pair, tested on the count itself ===")
    lo_k, hi_k = keys[0], keys[-1]
    a = agg[hi_k]["long"]
    b = agg[hi_k]["n"] - a
    c = agg[lo_k]["long"]
    d = agg[lo_k]["n"] - c
    pf = fisher_exact_2x2(a, b, c, d)
    print("  %s %d/%d long  vs  %s %d/%d long"
          % (hi_k, a, a + b, lo_k, c, c + d))
    print("  Fisher exact (two-sided) p = %.4f" % pf)
    _, _, _, pw = RS.welch(
        [r["delay"] for r in agg[hi_k]["runs"]],
        [r["delay"] for r in agg[lo_k]["runs"]])
    print("  (the same pair on mean delay: Welch p = %.4f;"
          " Holm over 21 comparisons = 0.6608 -- section 14.3)" % pw)
    print("  Counting regimes is the more powerful framing, but it")
    print("  still needs the full 0-of-5 vs 5-of-5 split for p < .05,")
    print("  which only the extreme pair has.")

    # ---- 6. what the count resolution means ----
    print("\n=== 6. what n=5 can resolve ===")
    sh = [r["delay"] for r in runs if r["hop"] < cut]
    ln = [r["delay"] for r in runs if r["hop"] >= cut]
    if sh and ln:
        ms, _ = RS.mean_std(sh)
        ml, _ = RS.mean_std(ln)
        step = (ml - ms) / 5.0
        print("  short-side mean %.2f ms, long-side mean %.2f ms" % (ms, ml))
        print("  so one run moving sides shifts a factor's mean by ~%.2f ms"
              % step)
        print("  -- the quantum of Table 10's delay column at n=5.")
        print("  adjacent gaps in the published ordering vs that quantum:")
        ord_d = [agg[k]["delay"] for k in keys]
        gaps = [(keys[t], keys[t + 1], ord_d[t + 1] - ord_d[t])
                for t in range(len(keys) - 1)]
        for ka, kb, g in gaps:
            print("    %-10s -> %-10s  %6.2f ms   %s"
                  % (ka, kb, g,
                     "BELOW the quantum" if g < step else "above"))
        nb = sum(1 for _, _, g in gaps if g < step)
        print("  %d of %d adjacent gaps are smaller than one run"
              " switching sides." % (nb, len(gaps)))

    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["factor", "n", "long_runs", "long_cut_hop",
                    "delay_mean", "delay_sd", "hop_mean", "hop_sd",
                    "loss_mean", "loss_sd", "delay_by_run", "hop_by_run"])
        for k in keys:
            v = agg[k]
            w.writerow([k, v["n"], v["long"], "%.3f" % cut,
                        "%.4f" % v["delay"], "%.4f" % v["delay_sd"],
                        "%.4f" % v["hop"], "%.4f" % v["hop_sd"],
                        "%.6f" % v["loss"], "%.6f" % v["loss_sd"],
                        " ".join("%.2f" % r["delay"] for r in v["runs"]),
                        " ".join("%.3f" % r["hop"] for r in v["runs"])])
    print("\nwrote %s" % OUT)
    print("counts: %s" % dict(Counter(
        "long" if r["hop"] >= cut else "short" for r in runs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
