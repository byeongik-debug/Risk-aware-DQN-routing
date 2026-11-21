# -*- coding: utf-8 -*-
u"""Composite quantities 4.7 quotes, re-aggregated from the committed hop rows.

This does NOT re-time anything.  It reads
``review_results/decision_breakdown_runs.csv`` -- the hop-level output of
``measure_decision_breakdown.py`` -- and writes the few numbers the draft states
as sums or splits of the per-part means, so that every number in 4.7 can be
found in a committed CSV rather than in my arithmetic.

Why it is a separate script: rerunning the measurement would produce a fresh set
of timings and silently invalidate the figures already written into the draft.
Aggregation is cheap and re-runnable; timing is not.

Usage:
    python -X utf8 aggregate_decision_breakdown_derived.py [--runs <csv>]
"""
import argparse
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(RES, "decision_breakdown_runs.csv"))
    ap.add_argument("--out", default=os.path.join(RES, "decision_breakdown_derived.csv"))
    args = ap.parse_args()

    with open(args.runs, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("no rows in " + args.runs)

    def col(key):
        return [v for v in (fnum(r.get(key)) for r in rows) if v is not None]

    dec = col("decide_ms")
    dm = statistics.fmean(dec)
    out = []

    def put(key, value, n=None, note=""):
        out.append({"key": key, "value": "%.6f" % value,
                    "n": "" if n is None else n, "note": note})

    put("n_hops", float(len(rows)), len(rows), "timed hops behind every 4.7 figure")
    put("decide_ms_mean", dm, len(dec))

    # The draft used to say the non-forward half "is" tensor building plus the
    # masking loop.  Those two are measurable and they do not add up to it.
    tensor = statistics.fmean(col("tensor_ms"))
    mask = statistics.fmean(col("mask_ms"))
    fwd = statistics.fmean(col("forward_ms"))
    amax = statistics.fmean(col("argmax_ms"))
    resid = statistics.fmean(col("residual_ms"))
    put("tensor_plus_mask_ms", tensor + mask, len(rows))
    put("tensor_plus_mask_pct_of_decide", (tensor + mask) / dm * 100.0, len(rows),
        "what the draft attributed the whole non-forward half to")
    put("non_forward_pct_of_decide", (dm - fwd) / dm * 100.0, len(rows))
    put("named_share_of_non_forward_pct",
        (tensor + mask) / (dm - fwd) * 100.0, len(rows),
        "the two named costs as a share of the non-forward half")
    put("unattributed_pct_of_decide", resid / dm * 100.0, len(rows),
        "residual: decide() minus the four timed pieces")
    put("parts_accounted_pct_of_decide",
        (tensor + fwd + mask + amax) / dm * 100.0, len(rows))

    # The mechanism that could make a per-path fixed cost, measured directly.
    first = [fnum(r["decide_ms"]) for r in rows if int(float(r["hop_index"])) == 0]
    later = [fnum(r["decide_ms"]) for r in rows if int(float(r["hop_index"])) > 0]
    put("decide_ms_first_hop", statistics.fmean(first), len(first))
    put("decide_ms_later_hops", statistics.fmean(later), len(later))
    put("first_hop_premium_ms",
        statistics.fmean(first) - statistics.fmean(later), len(rows),
        "a positive value is the only thing an intercept could be; it is negative")

    # What table 12 leaves out, as charged per path rather than per hop.
    st = statistics.fmean(col("get_state_ms"))
    va = statistics.fmean(col("get_valid_ms"))
    hops = statistics.fmean([fnum(r["hops_total"]) for r in rows])
    put("get_state_plus_valid_ms_per_hop", st + va, len(rows))
    put("uncharged_ms_per_path", (st + va) * hops, len(rows),
        "mean hops %.4f" % hops)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "value", "n", "note"])
        w.writeheader()
        for r in out:
            w.writerow(r)

    for r in out:
        print("  %-36s %12s  %s" % (r["key"], r["value"], r["note"]))
    print("\nwrote %s (%d rows)" % (args.out, len(out)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
