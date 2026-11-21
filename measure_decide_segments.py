# -*- coding: utf-8 -*-
"""What is the 27.9% of a decision that decision_breakdown could not attribute?

`DRAFT_4_7_table12.md` (3) prints a decomposition of the 0.399 ms decision whose
named parts (forward 52.8%, mask 12.9%, tensor 4.2%, argmax 2.2%) sum to 72.1%,
and says of the remaining 27.9%: "재지 않았으므로 단정하지 않는다".  This script
measures it.  It starts from two defects in how that decomposition was built,
both visible in measure_decision_breakdown.walk_path:

  (1) It does not cover decide().  decide() is
          state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
          with torch.no_grad():
              q_values = model(state_tensor)[0]
          masked = torch.full_like(q_values, -1e9)
          for action in valid_actions: masked[action] = q_values[action]
          return int(torch.argmax(masked).item())
      The split timed `model(tensor)` but not the `[0]` that indexes its output,
      and nothing at all for entering/leaving the no_grad context or for the call
      frame itself.  Those pieces are *inside* the composite and *outside* every
      part, so they land in the residual by construction.

  (2) Each part was timed as its own batch of 15 repeats.  Fifteen consecutive
      FloatTensor builds see an allocator and a branch predictor in a state that
      a single build inside decide() never sees.  An isolated part can therefore
      be cheaper than the same work in situ, and the whole difference shows up as
      residual without anything being missing.

So the measurement here is structural rather than another set of isolated timings:
the same statements run once each, in order, inside one call, with perf_counter
stamps between them.  The segments then sum to the call's own wall time by
construction and the residual collapses to timer overhead, which is measured too.
Three things are reported separately and must not be conflated:

  A. in-call segment shares  -- where the 0.399 ms goes, summing to ~100%.
  B. observer effect         -- the instrumented call's total vs the plain
                                decide() median at the same hop.  If stamping
                                moves the total, A describes a different call.
  C. isolation gain          -- in-call cost of a part minus the same part timed
                                as its own 15-repeat batch.  This is the part of
                                the old residual that was never missing work at
                                all, only a cheaper measurement of the same work.

Harness: imported from measure_path_setup_time, same as decision_breakdown
(1 torch thread, REPEATS medians, WARMUP dynamic updates, risk window 5, 50
warm-up forwards).  Nothing is retrained; no original file is touched.

Outputs
  review_results/decide_segments_runs.csv     one row per timed hop
  review_results/decide_segments_summary.csv  the aggregates for 4.7 (3)
"""

import argparse
import csv
import os
import statistics
import sys
import time

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from measure_path_setup_time import (  # noqa: E402
    REPEATS, RISK_WINDOW, SCENARIOS, TORCH_WARMUP, WARMUP, decide, median_time)
from network_env import NetworkEnv  # noqa: E402
from reeval_loss_conventions import load_risk_model  # noqa: E402
from review_stats import mean_std  # noqa: E402
from topology import DESTINATION  # noqa: E402

MS = 1e3
OUT_DIR = os.path.join(BASE_DIR, "review_results")
RUNS_CSV = os.path.join(OUT_DIR, "decide_segments_runs.csv")
SUM_CSV = os.path.join(OUT_DIR, "decide_segments_summary.csv")

# The eleven segments, in execution order.  Names are the statements, not my
# interpretation of them.
SEGMENTS = [
    "floattensor",      # torch.FloatTensor(state)
    "unsqueeze",        # .unsqueeze(0)
    "to_device",        # .to(device)
    "nograd_enter",     # with torch.no_grad():  -- __enter__
    "forward",          # model(state_tensor)
    "index0",           # [0]        <- was never timed
    "nograd_exit",      # leaving the context     <- was never timed
    "full_like",        # torch.full_like(q_values, -1e9)
    "mask_loop",        # for action in valid_actions: masked[a] = q_values[a]
    "argmax",           # torch.argmax(masked)
    "item_int",         # int(.item())
]

# The four groups the old decomposition named, so C can be computed per group.
GROUPS = {
    "tensor": ["floattensor", "unsqueeze", "to_device"],
    "forward": ["forward"],
    "mask": ["full_like", "mask_loop"],
    "argmax": ["argmax", "item_int"],
}
UNNAMED = ["nograd_enter", "index0", "nograd_exit"]


def decide_segmented(model, state, valid_actions, device):
    """decide() with a stamp between every statement.

    -> (total_seconds, {segment: seconds}, action)

    The body is a statement-for-statement copy of measure_path_setup_time.decide
    with one deliberate difference: the `with torch.no_grad():` block is written as
    an explicit object plus __enter__/__exit__ so those two can be charged.  The
    work is the same work a with-statement does; `nograd_enter`+`nograd_exit` is
    therefore an upper bound on what the with-statement costs, not a new cost.
    """
    pc = time.perf_counter
    t = [pc()]
    tensor = torch.FloatTensor(state)
    t.append(pc())
    tensor = tensor.unsqueeze(0)
    t.append(pc())
    tensor = tensor.to(device)
    t.append(pc())
    guard = torch.no_grad()
    guard.__enter__()
    t.append(pc())
    out = model(tensor)
    t.append(pc())
    q_values = out[0]
    t.append(pc())
    guard.__exit__(None, None, None)
    t.append(pc())
    masked = torch.full_like(q_values, -1e9)
    t.append(pc())
    for action in valid_actions:
        masked[action] = q_values[action]
    t.append(pc())
    best = torch.argmax(masked)
    t.append(pc())
    chosen = int(best.item())
    t.append(pc())

    segs = dict((name, t[i + 1] - t[i]) for i, name in enumerate(SEGMENTS))
    return t[-1] - t[0], segs, chosen


def median_segmented(model, state, valid_actions, device, repeats=REPEATS):
    """Run the instrumented call `repeats` times and keep the *median run whole*.

    Taking a per-segment median would break additivity (the medians of the parts
    do not sum to the median of the totals), and additivity is the only reason
    this measurement answers the question.  So the run whose total is the median
    is the one reported, and the mean total is carried alongside as a check.
    """
    runs = []
    for _ in range(repeats):
        total, segs, chosen = decide_segmented(model, state, valid_actions, device)
        runs.append((total, segs, chosen))
    runs.sort(key=lambda r: r[0])
    mid = runs[len(runs) // 2]
    mean_total = statistics.fmean([r[0] for r in runs])
    return mid[0], mid[1], mid[2], mean_total


def timer_cost(n=200000):
    """-> seconds per perf_counter() call.

    Measured, not assumed: twelve stamps per call against a 0.4 ms call is only
    negligible if a stamp is sub-microsecond, and that is a property of this
    Windows build, not of the language.
    """
    pc = time.perf_counter
    best = None
    for _ in range(5):
        start = pc()
        for _ in range(n):
            pc()
        elapsed = pc() - start
        if best is None or elapsed < best:
            best = elapsed
    return best / n


def walk(env, model, device, scenario, trial, rows):
    state = env.get_state()
    done = False
    hop = 0
    hop_rows = []

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        # B: the uninstrumented composite, exactly as table 12 charges it.
        t_plain, action = median_time(
            lambda: decide(model, state, valid_actions, device))
        # A: the same call, stamped.
        t_seg, segs, _chosen, t_seg_mean = median_segmented(
            model, state, valid_actions, device)
        # C: the four named parts as their own batches -- the old method.
        t_iso_tensor, tensor = median_time(
            lambda: torch.FloatTensor(state).unsqueeze(0).to(device))

        def forward():
            with torch.no_grad():
                return model(tensor)

        t_iso_forward, out = median_time(forward)
        q_values = out[0]

        def mask():
            m = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                m[a] = q_values[a]
            return m

        t_iso_mask, masked = median_time(mask)
        t_iso_argmax, _ = median_time(lambda: int(torch.argmax(masked).item()))

        row = {
            "scenario": scenario, "trial": trial, "hop_index": hop,
            "n_valid": len(valid_actions),
            "plain_decide_ms": t_plain * MS,
            "segmented_total_ms": t_seg * MS,
            "segmented_mean_ms": t_seg_mean * MS,
            "iso_tensor_ms": t_iso_tensor * MS,
            "iso_forward_ms": t_iso_forward * MS,
            "iso_mask_ms": t_iso_mask * MS,
            "iso_argmax_ms": t_iso_argmax * MS,
        }
        for name in SEGMENTS:
            row["seg_" + name + "_ms"] = segs[name] * MS
        row["seg_sum_ms"] = sum(segs.values()) * MS
        hop_rows.append(row)

        hop += 1
        state, _, done, _ = env.step(action)

    for r in hop_rows:
        r["hops_total"] = len(hop_rows)
    rows.extend(hop_rows)
    return len(hop_rows), env.current_node == DESTINATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    runs_csv = RUNS_CSV.replace(".csv", args.tag + ".csv")
    sum_csv = SUM_CSV.replace(".csv", args.tag + ".csv")

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model = load_risk_model()
    model.eval()

    dummy = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    dummy.reset()
    warm = torch.FloatTensor(dummy.get_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(TORCH_WARMUP):
            model(warm)

    stamp_s = timer_cost()
    print("torch {}, {} thread(s), {} repeats/median, {} trials/scenario".format(
        torch.__version__, torch.get_num_threads(), REPEATS, args.trials))
    print("perf_counter(): {:.1f} ns/call  ->  12 stamps = {:.5f} ms/decision"
          .format(stamp_s * 1e9, stamp_s * 12 * MS))

    rows = []
    reached_all = 0
    paths_all = 0
    for scenario in args.scenarios:
        reached = 0
        for trial in range(args.trials):
            env = NetworkEnv(use_risk=True, scenario=scenario,
                             risk_window=RISK_WINDOW)
            env.reset()
            for _ in range(WARMUP):
                env.update_dynamic_qos()
            _hops, ok = walk(env, model, device, scenario, trial, rows)
            reached += 1 if ok else 0
        paths_all += args.trials
        reached_all += reached
        print("  {:<11s} {:d}/{:d} paths reached destination"
              .format(scenario, reached, args.trials))

    fields = (["scenario", "trial", "hop_index", "hops_total", "n_valid",
               "plain_decide_ms", "segmented_total_ms", "segmented_mean_ms",
               "seg_sum_ms"]
              + ["seg_" + s + "_ms" for s in SEGMENTS]
              + ["iso_tensor_ms", "iso_forward_ms", "iso_mask_ms",
                 "iso_argmax_ms"])
    with open(runs_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, r[k]) for k in fields))

    n = len(rows)
    seg_mean = dict((s, statistics.fmean([r["seg_" + s + "_ms"] for r in rows]))
                    for s in SEGMENTS)
    total = statistics.fmean([r["segmented_total_ms"] for r in rows])
    seg_sum = statistics.fmean([r["seg_sum_ms"] for r in rows])
    plain = statistics.fmean([r["plain_decide_ms"] for r in rows])

    out = []

    def put(metric, value, unit, note):
        out.append({"metric": metric, "value": value, "unit": unit,
                    "n_hops": n, "note": note})

    print("\nA. 결정 하나의 구간별 몫 (호출 안에서, 합 = 호출 전체)")
    print("   %-14s %10s %8s" % ("구간", "ms", "%"))
    for s in SEGMENTS:
        print("   %-14s %10.6f %7.3f%%" % (s, seg_mean[s],
                                           seg_mean[s] / total * 100.0))
        put("seg_" + s + "_ms", seg_mean[s], "ms", "in-call segment mean")
        put("seg_" + s + "_pct", seg_mean[s] / total * 100.0, "%",
            "share of instrumented call total")
    print("   %-14s %10.6f %7.3f%%  (구간 합)" % ("sum", seg_sum,
                                                 seg_sum / total * 100.0))
    print("   %-14s %10.6f" % ("call total", total))
    print("   미귀속 잔차 = %.6f ms (%.3f%%)  <- 스탬프 사이에만 남는다"
          % (total - seg_sum, (total - seg_sum) / total * 100.0))
    put("seg_sum_ms", seg_sum, "ms", "sum of the eleven segments")
    put("segmented_total_ms", total, "ms", "instrumented call wall time")
    put("unattributed_ms", total - seg_sum, "ms",
        "total minus the sum of segments")
    put("unattributed_pct", (total - seg_sum) / total * 100.0, "%",
        "residual of the segmented decomposition")
    put("stamp_ns", stamp_s * 1e9, "ns", "perf_counter cost per call")
    put("stamp_ms_per_decision", stamp_s * 12 * MS, "ms", "12 stamps")

    grp = {}
    for g, members in GROUPS.items():
        grp[g] = sum(seg_mean[m] for m in members)
    unnamed_ms = sum(seg_mean[m] for m in UNNAMED)
    print("\n   옛 분해가 이름 붙인 넷 + 이름 없던 셋")
    for g in ("tensor", "forward", "mask", "argmax"):
        print("   %-14s %10.6f %7.3f%%" % (g, grp[g], grp[g] / total * 100.0))
        put("group_" + g + "_ms", grp[g], "ms", "in-call group total")
        put("group_" + g + "_pct", grp[g] / total * 100.0, "%", "in-call share")
    print("   %-14s %10.6f %7.3f%%  (%s)"
          % ("unnamed", unnamed_ms, unnamed_ms / total * 100.0,
             "+".join(UNNAMED)))
    put("group_unnamed_ms", unnamed_ms, "ms",
        "nograd enter/exit + [0] indexing: never timed before")
    put("group_unnamed_pct", unnamed_ms / total * 100.0, "%", "in-call share")

    print("\nB. 관측자 효과 — 스탬프가 호출 전체를 움직이는가")
    print("   계측 안 한 decide() 중앙값 평균 %.6f ms" % plain)
    print("   스탬프 박은 호출       평균 %.6f ms  (%+.3f%%)"
          % (total, (total - plain) / plain * 100.0))
    put("plain_decide_ms", plain, "ms", "uninstrumented composite, same hops")
    put("observer_effect_pct", (total - plain) / plain * 100.0, "%",
        "instrumented total vs plain decide")

    print("\nC. 고립 이득 — 같은 일을 자기 배치로 재면 얼마나 싸지는가")
    print("   %-10s %10s %10s %10s %8s" % ("부분", "호출 안", "고립", "차", "차/호출안"))
    iso_sum = 0.0
    for g, col in (("tensor", "iso_tensor_ms"), ("forward", "iso_forward_ms"),
                   ("mask", "iso_mask_ms"), ("argmax", "iso_argmax_ms")):
        iso = statistics.fmean([r[col] for r in rows])
        iso_sum += iso
        d = grp[g] - iso
        print("   %-10s %10.6f %10.6f %+10.6f %7.2f%%"
              % (g, grp[g], iso, d, d / grp[g] * 100.0 if grp[g] else float("nan")))
        put("iso_" + g + "_ms", iso, "ms", "same work timed as its own batch")
        put("isolation_gain_" + g + "_ms", d, "ms", "in-call minus isolated")
    print("   %-10s %10.6f %10.6f %+10.6f"
          % ("sum(4)", sum(grp[g] for g in GROUPS), iso_sum,
             sum(grp[g] for g in GROUPS) - iso_sum))
    put("iso_sum_ms", iso_sum, "ms", "the four isolated parts, summed")
    put("isolation_gain_total_ms", sum(grp[g] for g in GROUPS) - iso_sum, "ms",
        "in-call total of the four minus their isolated total")

    # --- The old residual, split.  Two warnings about how this is allowed to be
    # read, both of which I have to state before the numbers:
    #
    #   (i) Once the observer effect is removed by proportional scaling, the split
    #       adds to the residual *exactly*, for every input.  That is an identity
    #       (sum of segments = plain, so unnamed + (named - isolated) = plain -
    #       isolated = residual), not a confirmation of anything.  What the
    #       measurement supplies is the ratio between the two terms, not the fact
    #       that they add up.
    #  (ii) Proportional scaling is an assumption: it distributes the
    #       instrumentation's cost over the eleven segments in proportion to their
    #       size.  So instead of relying on it, the same split is computed under
    #       the two extreme allocations as well -- all of the observer effect
    #       charged to the three unnamed statements, and all of it charged to the
    #       four named groups.  Those bracket every intermediate allocation, so
    #       whatever survives all three is a result and not an artefact of (ii).
    old_resid = plain - iso_sum
    named_ms = sum(grp[g] for g in GROUPS)
    obs_ms = total - plain
    print("\n   옛 잔차 = 계측 안 한 decide %.6f - 고립 4부분 합 %.6f = %+.6f ms (%.3f%%)"
          % (plain, iso_sum, old_resid, old_resid / plain * 100.0))
    print("   이 잔차를 둘로 나눈다 — 아무도 안 쟀던 세 문장 / 네 부분의 고립 이득.")
    print("   관측자 효과 %+.6f ms 를 어디에 싣느냐로 배분이 달라지므로 세 경우를 다 쓴다."
          % obs_ms)
    print("   %-22s %12s %12s %9s %9s"
          % ("배분", "무명 3문장", "고립이득", "무명 몫", "이득 몫"))
    alloc = []
    scale = plain / total if total else float("nan")
    alloc.append(("비례 (가정)", unnamed_ms * scale, named_ms * scale - iso_sum))
    alloc.append(("전부 무명에", unnamed_ms - obs_ms, named_ms - iso_sum))
    alloc.append(("전부 유명 4부분에", unnamed_ms, named_ms - obs_ms - iso_sum))
    for label, un, gain in alloc:
        print("   %-22s %12.6f %12.6f %8.1f%% %8.1f%%"
              % (label, un, gain,
                 un / old_resid * 100.0 if old_resid else float("nan"),
                 gain / old_resid * 100.0 if old_resid else float("nan")))
        key = {"비례 (가정)": "proportional", "전부 무명에": "all_to_unnamed",
               "전부 유명 4부분에": "all_to_named"}[label]
        put("resid_unnamed_ms_" + key, un, "ms", "unnamed three under " + key)
        put("resid_gain_ms_" + key, gain, "ms", "isolation gain under " + key)
        put("resid_unnamed_pct_" + key,
            un / old_resid * 100.0 if old_resid else float("nan"), "%",
            "share of the old residual")
        put("resid_gain_pct_" + key,
            gain / old_resid * 100.0 if old_resid else float("nan"), "%",
            "share of the old residual")
    gains = [g for _l, _u, g in alloc]
    uns = [u for _l, u, _g in alloc]
    print("   -> 고립 이득의 몫 %.1f~%.1f%%,  무명 3문장 %.1f~%.1f%%"
          % (min(gains) / old_resid * 100.0, max(gains) / old_resid * 100.0,
             min(uns) / old_resid * 100.0, max(uns) / old_resid * 100.0))
    print("      (스탬프 12개의 자체 비용은 %.6f ms 로 관측자 효과 %+.6f ms 의 %.1f%% 뿐"
          " — 나머지는 설명하지 않는다)"
          % (stamp_s * 12 * MS, obs_ms,
             stamp_s * 12 * MS / obs_ms * 100.0 if obs_ms else float("nan")))
    put("old_residual_ms", old_resid, "ms",
        "plain decide minus the four isolated parts: the 27.9% of 4.7 (3)")
    put("old_residual_pct", old_resid / plain * 100.0, "%", "of plain decide")
    put("resid_gain_pct_min", min(gains) / old_resid * 100.0, "%",
        "isolation gain's share of the old residual, worst allocation")
    put("resid_gain_pct_max", max(gains) / old_resid * 100.0, "%",
        "isolation gain's share of the old residual, best allocation")
    put("observer_effect_ms", obs_ms, "ms", "instrumented total minus plain")
    put("stamp_share_of_observer_pct",
        stamp_s * 12 * MS / obs_ms * 100.0 if obs_ms else float("nan"), "%",
        "the twelve stamps' own cost as a share of the observer effect")

    sd = {}
    for s in SEGMENTS:
        sd[s] = mean_std([r["seg_" + s + "_ms"] for r in rows])[1]
        put("seg_" + s + "_sd", sd[s], "ms", "sd across hops")
    put("paths_reached", reached_all, "paths",
        "of %d walked" % paths_all)

    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "unit", "n_hops",
                                          "note"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print("\nwrote %s  (%d hops)" % (runs_csv, n))
    print("wrote %s" % sum_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
