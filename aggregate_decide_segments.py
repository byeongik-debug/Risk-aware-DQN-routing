# -*- coding: utf-8 -*-
"""Pool the three independent passes of measure_decide_segments.py.

Re-running the measurement would produce fresh timings, and the absolute
per-decision millisecond is only reproducible to ~24.5% between passes (section 5
of FINDINGS_path_setup_time).  So the passes are pooled here from what they already
wrote, without re-timing anything -- the same rule aggregate_decision_breakdown_
derived.py follows.

The point of pooling is to separate the two kinds of quantity in that measurement:

  * absolute ms per segment, which moves with the pass, and
  * each segment's *share* of the decision, which is what 4.7 (3) quotes.

So every metric is reported as mean plus the min/max across passes, and the draft
may only quote a share whose spread across passes is small enough to print.

Inputs   review_results/decide_segments_summary{,_rep2,_rep3}.csv
Output   review_results/decide_segments_derived.csv
"""

import csv
import os
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE_DIR, "review_results")
PASSES = ["decide_segments_summary.csv",
          "decide_segments_summary_rep2.csv",
          "decide_segments_summary_rep3.csv"]
OUT = os.path.join(RES, "decide_segments_derived.csv")


def read(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["metric"]] = row
    return out


def main():
    passes = [read(os.path.join(RES, p)) for p in PASSES]
    metrics = []
    for p in passes:
        for m in p:
            if m not in metrics:
                metrics.append(m)

    rows = []
    for m in metrics:
        vals = []
        for p in passes:
            if m in p:
                try:
                    vals.append(float(p[m]["value"]))
                except ValueError:
                    pass
        if not vals:
            continue
        src = next(p[m] for p in passes if m in p)
        rows.append({
            "metric": m,
            "mean": statistics.fmean(vals),
            "min": min(vals),
            "max": max(vals),
            "spread": max(vals) - min(vals),
            "n_passes": len(vals),
            "unit": src["unit"],
            "note": src["note"],
        })

    by = dict((r["metric"], r) for r in rows)

    def add(metric, value, unit, note, vals=None):
        rows.append({"metric": metric, "mean": value,
                     "min": min(vals) if vals else value,
                     "max": max(vals) if vals else value,
                     "spread": (max(vals) - min(vals)) if vals else 0.0,
                     "n_passes": len(vals) if vals else len(passes),
                     "unit": unit, "note": note})

    # The four named parts, timed in situ, as a share of the instrumented call.
    named = []
    for p in passes:
        named.append(sum(float(p["group_" + g + "_pct"]["value"])
                         for g in ("tensor", "forward", "mask", "argmax")))
    add("named_four_in_call_pct", statistics.fmean(named), "%",
        "the four parts 4.7 (3) names, timed inside the call, as a share of it",
        named)

    # The same four as a share of the *plain* call when timed in isolation -- the
    # convention the committed decomposition used.  Printing both is the point:
    # "forward is 52.8% of a decision" and "forward is 61.6% of a decision" are
    # the same work under two measurement conventions, not a disagreement.
    for g in ("tensor", "forward", "mask", "argmax"):
        vals = []
        for p in passes:
            vals.append(float(p["iso_" + g + "_ms"]["value"])
                        / float(p["plain_decide_ms"]["value"]) * 100.0)
        add("iso_" + g + "_pct_of_plain", statistics.fmean(vals), "%",
            "isolated batch timing over plain decide: the old convention", vals)

    # The manuscript's own sentence is "forward is half, the rest is tensor
    # building and the masking loop".  The committed decomposition judged that
    # sentence against isolated part timings and found the two named costs to be
    # only 36.2% of the non-forward half.  In situ the same two are a much larger
    # share, so the verdict on that sentence is a function of the convention and
    # both numbers have to be on the record.
    combos = {"tensor_plus_mask_in_call_pct": ("tensor", "mask"),
              "non_forward_in_call_pct": ("tensor", "mask", "argmax",
                                          "unnamed")}
    per_pass = {}
    for name, members in combos.items():
        vals = [sum(float(p["group_" + g + "_pct"]["value"]) for g in members)
                for p in passes]
        per_pass[name] = vals
        add(name, statistics.fmean(vals), "%",
            "in-call share: " + "+".join(members), vals)
    for name, members in (("tensor_plus_mask", ("tensor", "mask")),
                          ("unnamed", ("unnamed",)),
                          ("argmax", ("argmax",))):
        vals = []
        for i, p in enumerate(passes):
            num = sum(float(p["group_" + g + "_pct"]["value"]) for g in members)
            vals.append(num / per_pass["non_forward_in_call_pct"][i] * 100.0)
        add(name + "_share_of_non_forward_in_call_pct", statistics.fmean(vals),
            "%", "share of the non-forward part, timed in situ", vals)

    # How much cheaper each part gets when it is timed as its own batch, as a
    # share of its in-call cost.  This is the size of the measurement artefact per
    # part, and it is what the old residual was mostly made of, so it belongs in a
    # CSV rather than in a sentence.
    for g in ("tensor", "forward", "mask", "argmax"):
        vals = []
        for p in passes:
            vals.append(float(p["isolation_gain_" + g + "_ms"]["value"])
                        / float(p["group_" + g + "_ms"]["value"]) * 100.0)
        add("isolation_gain_" + g + "_pct_of_in_call", statistics.fmean(vals),
            "%", "how much cheaper this part is when timed in isolation", vals)

    # Per-decision cost of the twelve stamps as a share of a decision, so the
    # instrument's own weight is on the record next to the observer effect.
    vals = []
    for p in passes:
        vals.append(float(p["stamp_ms_per_decision"]["value"])
                    / float(p["plain_decide_ms"]["value"]) * 100.0)
    add("stamp_pct_of_decision", statistics.fmean(vals), "%",
        "twelve perf_counter stamps as a share of one decision", vals)

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "mean", "min", "max",
                                          "spread", "n_passes", "unit", "note"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    def show(metric, fmt="%.4f"):
        r = by.get(metric) or next(x for x in rows if x["metric"] == metric)
        print(("  %-34s " + fmt + "   [" + fmt + ", " + fmt + "] %s")
              % (metric, r["mean"], r["min"], r["max"], r["unit"]))

    print("3개 패스 집계 (833·833·833 홉, 재측정 없음)\n")
    print("A. 호출 안 구간 몫 — 합이 100% 가 되는 분해")
    for s in ("floattensor", "unsqueeze", "to_device", "nograd_enter", "forward",
              "index0", "nograd_exit", "full_like", "mask_loop", "argmax",
              "item_int"):
        show("seg_" + s + "_pct", "%7.3f")
    show("named_four_in_call_pct", "%7.3f")
    show("group_unnamed_pct", "%7.3f")
    print("\nB. 옛 잔차 (27.9%) 와 그 분할")
    show("old_residual_pct", "%7.3f")
    show("resid_gain_pct_min", "%7.3f")
    show("resid_gain_pct_max", "%7.3f")
    show("resid_unnamed_pct_all_to_named", "%7.3f")
    show("resid_unnamed_pct_all_to_unnamed", "%7.3f")
    print("\nC. 측정 규약이 forward 몫을 바꾼다")
    show("iso_forward_pct_of_plain", "%7.3f")
    show("seg_forward_pct", "%7.3f")
    show("non_forward_in_call_pct", "%7.3f")
    show("tensor_plus_mask_in_call_pct", "%7.3f")
    show("tensor_plus_mask_share_of_non_forward_in_call_pct", "%7.3f")
    show("unnamed_share_of_non_forward_in_call_pct", "%7.3f")
    show("argmax_share_of_non_forward_in_call_pct", "%7.3f")
    print("\nD. 계측 자체의 무게")
    show("observer_effect_pct", "%+7.3f")
    show("stamp_pct_of_decision", "%7.3f")
    show("plain_decide_ms", "%7.4f")
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
