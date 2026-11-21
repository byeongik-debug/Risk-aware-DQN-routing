"""Where does the DQN controller's per-path millisecond actually go?  (table 12)

`DRAFT_4_7_table12.md` fits `per_path = 0.513 + 0.281 * decisions` to the four
scenario means and then refuses to call the 0.513 ms intercept a state-construction
cost, because that decomposition was never measured.  This script measures it, and
it starts from a design objection rather than from the number:

  measure_path_setup_time.time_dqn() builds the per-path cost as the SUM of the
  per-hop decide() calls.  Nothing else is charged.  A per-path fixed cost therefore
  has no place to live in that measurement, and an intercept cannot be one.  If the
  per-hop cost were a constant c, OLS on (decisions, per_path) would return slope c
  and intercept exactly 0.  So a non-zero intercept means only that the per-decision
  cost correlates with path length -- and the committed CSV says it does, downward:
  0.364914 ms at 6.0 decisions, 0.350314 ms at 7.6.

  One mechanism predicts that exactly.  If cost_h = c + p for the first hop of a path
  and c afterwards, then per_path = p + c*decisions: the intercept IS the first-hop
  premium and the slope IS the steady-state per-hop cost.  That is testable, so this
  script measures p directly and compares it with the refitted intercept.

Second, an asymmetry in what table 12 counts.  OSPF's per-path number comes from
env.ospf_path(), which rebuilds the networkx graph every call -- 0.17 ms of the
0.30 ms is that rebuild, and it IS charged to OSPF.  The DQN's state construction
(env.get_state(): a 50-edge Python loop plus four np.zeros(25) and a concatenate) and
its action masking input (env.get_valid_actions()) sit OUTSIDE the timed region,
because the state arrives as a return value of env.step(), which table 12 excludes as
"the network's own behaviour".  Building the observation is not the network's
behaviour; it is the controller's.  So the same work is charged on one side and not
the other, and this section is the only one in 4.7 that favours the proposal.  Both
costs are measured here and the multiple is recomputed with them charged.

Harness: imported from measure_path_setup_time so the numbers stay comparable to
table 12 (1 torch thread, REPEATS=15 medians, WARMUP=10 dynamic updates, risk window
5, 50 forward passes of warm-up before anything is timed).  Nothing is retrained and
no original file is touched; the frozen risk model is read through load_risk_model().

Outputs
  review_results/decision_breakdown_runs.csv      one row per timed hop
  review_results/decision_breakdown_summary.csv    the aggregates quoted in 4.7
"""

import argparse
import csv
import os
import random
import statistics
import sys

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from measure_path_setup_time import (  # noqa: E402
    REPEATS, RISK_WINDOW, SCENARIOS, TORCH_WARMUP, WARMUP, decide, median_time,
    time_ospf, time_qos)
from network_env import NetworkEnv  # noqa: E402
from reeval_loss_conventions import load_risk_model  # noqa: E402
from review_stats import (  # noqa: E402
    holm_bonferroni, mean_std, t_two_sided_p, welch)
from topology import DESTINATION  # noqa: E402

BOOT = 2000
BOOT_SEED = 20261120

OUT_DIR = os.path.join(BASE_DIR, "review_results")
RUNS_CSV = os.path.join(OUT_DIR, "decision_breakdown_runs.csv")
SUM_CSV = os.path.join(OUT_DIR, "decision_breakdown_summary.csv")
FITS_CSV = os.path.join(OUT_DIR, "decision_breakdown_fits.csv")

MS = 1e3

RUN_FIELDS = ["replicate", "scenario", "trial", "hop_index", "hops_total",
              "n_valid",
              "decide_ms", "tensor_ms", "forward_ms", "mask_ms", "argmax_ms",
              "components_ms", "residual_ms", "get_state_ms",
              "get_valid_ms"]


def ols(xs, ys):
    """-> (intercept, slope, r2).  Two points minimum; no scipy in this tree."""
    n = len(xs)
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx else float("nan")
    intercept = my - slope * mx
    syy = sum((y - my) ** 2 for y in ys)
    if syy:
        resid = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - resid / syy
    else:
        r2 = float("nan")
    return intercept, slope, r2, n


def ols_intercept_test(xs, ys):
    """-> (intercept, slope, r2, se_intercept, t, p, df).

    The intercept is an extrapolation to decisions = 0, which is 6-7 decisions away
    from every observed path, so its standard error carries the x-distance factor
    sqrt(1/n + xbar^2/Sxx).  Reporting the point estimate without it is what let
    0.513 ms look like a cost.
    """
    intercept, slope, r2, n = ols(xs, ys)
    df = n - 2
    if df <= 0:
        return intercept, slope, r2, float("nan"), float("nan"), float("nan"), df
    mx = statistics.fmean(xs)
    sxx = sum((x - mx) ** 2 for x in xs)
    rss = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    s2 = rss / df
    se = (s2 * (1.0 / n + mx * mx / sxx)) ** 0.5 if sxx else float("nan")
    t = intercept / se if se else float("nan")
    return intercept, slope, r2, se, t, t_two_sided_p(t, df), df


def bootstrap_intercept(xs, ys, b=BOOT, seed=BOOT_SEED):
    """-> (lo, hi, share<=0, n_usable).  Resamples paths, refits, keeps the
    intercepts.  A draw whose decision counts are all equal gives no slope and is
    dropped rather than silently counted."""
    rng = random.Random(seed)
    n = len(xs)
    out = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [xs[i] for i in idx]
        by = [ys[i] for i in idx]
        if len(set(bx)) < 2:
            continue
        a, _s, _r, _n = ols(bx, by)
        out.append(a)
    if not out:
        return float("nan"), float("nan"), float("nan"), 0
    out.sort()
    lo = out[int(0.025 * (len(out) - 1))]
    hi = out[int(0.975 * (len(out) - 1))]
    share = sum(1 for a in out if a <= 0) / float(len(out)) * 100.0
    return lo, hi, share, len(out)


def pearson(xs, ys):
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if not sxx or not syy:
        return float("nan")
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def walk_path(env, model, device, scenario, trial, rows, parts=True):
    """One path.  Charges the same decide() as table 12, then splits it, and also
    times the two controller steps table 12 leaves out.

    parts=False is the control for my own instrumentation.  With parts on, each hop
    pays seven median_time() batches instead of three, which warms caches and could
    by itself move decide_ms away from the frozen harness's 0.350-0.365 ms.  The
    control keeps get_valid/get_state/decide and drops the four split timings, so
    the comparison isolates the effect of the splitting itself."""
    state = env.get_state()
    done = False
    hop = 0
    hop_rows = []

    while not done:
        t_valid, valid_actions = median_time(lambda: env.get_valid_actions())
        if not valid_actions:
            break
        t_state, _ = median_time(lambda: env.get_state())

        t_decide, action = median_time(
            lambda: decide(model, state, valid_actions, device))

        if not parts:
            nan = float("nan")
            hop_rows.append({
                "scenario": scenario, "trial": trial, "hop_index": hop,
                "n_valid": len(valid_actions),
                "decide_ms": t_decide * MS, "tensor_ms": nan,
                "forward_ms": nan, "mask_ms": nan, "argmax_ms": nan,
                "components_ms": nan, "residual_ms": nan,
                "get_state_ms": t_state * MS, "get_valid_ms": t_valid * MS,
            })
            hop += 1
            state, _, done, _ = env.step(action)
            continue

        # The same work, split.  Timed after decide() on the unchanged state, so the
        # pieces see exactly the inputs the composite call saw.
        def build():
            return torch.FloatTensor(state).unsqueeze(0).to(device)

        t_tensor, tensor = median_time(build)

        def forward():
            with torch.no_grad():
                return model(tensor)

        t_forward, out = median_time(forward)
        q_values = out[0]

        def mask():
            masked = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                masked[a] = q_values[a]
            return masked

        t_mask, masked = median_time(mask)
        t_argmax, _ = median_time(lambda: int(torch.argmax(masked).item()))

        comp = t_tensor + t_forward + t_mask + t_argmax
        hop_rows.append({
            "scenario": scenario, "trial": trial, "hop_index": hop,
            "n_valid": len(valid_actions),
            "decide_ms": t_decide * MS, "tensor_ms": t_tensor * MS,
            "forward_ms": t_forward * MS, "mask_ms": t_mask * MS,
            "argmax_ms": t_argmax * MS, "components_ms": comp * MS,
            "residual_ms": (t_decide - comp) * MS,
            "get_state_ms": t_state * MS, "get_valid_ms": t_valid * MS,
        })
        hop += 1
        state, _, done, _ = env.step(action)

    for row in hop_rows:
        row["hops_total"] = len(hop_rows)
    rows.extend(hop_rows)
    return len(hop_rows), env.current_node == DESTINATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--no-parts", dest="parts", action="store_false",
                        help="control run: time only get_valid/get_state/decide, so "
                             "the seven-timings-per-hop instrumentation cannot be "
                             "what moved decide_ms.  Writes *_noparts_*.csv so it "
                             "never overwrites the main run.")
    parser.add_argument("--replicates", type=int, default=3,
                        help="independent passes over all scenarios.  The point of "
                             "more than one is the intercept: it is the quantity "
                             "4.7 names, so it has to be shown surviving a rerun.")
    args = parser.parse_args()
    tag = "" if args.parts else "_noparts"
    runs_csv = RUNS_CSV.replace("_runs.csv", tag + "_runs.csv")
    sum_csv = SUM_CSV.replace("_summary.csv", tag + "_summary.csv")
    fits_csv = FITS_CSV.replace("_fits.csv", tag + "_fits.csv")

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

    print("torch {}, {} thread(s), {} repeats/median, {} trials/scenario".format(
        torch.__version__, torch.get_num_threads(), REPEATS, args.trials))

    rows = []
    per_path = {}      # scenario -> list of decide-sum per path (ms)
    per_state = {}     # scenario -> list of state-build sum per path (ms)
    per_valid = {}     # scenario -> list of get_valid sum per path (ms)
    decisions = {}     # scenario -> list of decision counts
    ospf_path = {}     # scenario -> list of OSPF per-path ms
    qos_path = {}      # scenario -> list of QoS-Dijkstra per-path ms

    for scenario in args.scenarios:
        per_path[scenario] = []
        per_state[scenario] = []
        per_valid[scenario] = []
        decisions[scenario] = []
        ospf_path[scenario] = []
        qos_path[scenario] = []
    fits = []

    for rep in range(args.replicates):
      rep_path = {}
      rep_dec = {}
      for scenario in args.scenarios:
        rep_path[scenario] = []
        rep_dec[scenario] = []
        reached = 0
        for trial in range(args.trials):
            env = NetworkEnv(use_risk=True, scenario=scenario,
                             risk_window=RISK_WINDOW)
            env.reset()
            for _ in range(WARMUP):
                env.update_dynamic_qos()
            before = len(rows)
            hops, ok = walk_path(env, model, device, scenario,
                                 rep * args.trials + trial, rows,
                                 parts=args.parts)
            mine = rows[before:]
            for r in mine:
                r["replicate"] = rep
            rep_path[scenario].append(sum(r["decide_ms"] for r in mine))
            rep_dec[scenario].append(hops)
            per_path[scenario].append(sum(r["decide_ms"] for r in mine))
            per_state[scenario].append(sum(r["get_state_ms"] for r in mine))
            per_valid[scenario].append(sum(r["get_valid_ms"] for r in mine))
            decisions[scenario].append(hops)
            reached += 1 if ok else 0

            # OSPF in the same process, same warmed environment state, so the
            # multiple is a within-run ratio and not a cross-run comparison.
            env2 = NetworkEnv(use_risk=True, scenario=scenario,
                              risk_window=RISK_WINDOW)
            env2.reset()
            for _ in range(WARMUP):
                env2.update_dynamic_qos()
            ospf_path[scenario].append(time_ospf(env2)["per_path"] * MS)

            # The composite-QoS Dijkstra is the comparison the draft leans on
            # ("the proposal is no slower than it"), so it has to be re-timed in
            # the same process before the state cost is added to one side only.
            env3 = NetworkEnv(use_risk=True, scenario=scenario,
                              risk_window=RISK_WINDOW)
            env3.reset()
            for _ in range(WARMUP):
                env3.update_dynamic_qos()
            qos_path[scenario].append(time_qos(env3)["per_path"] * MS)

        print("  rep {} {:<11} paths {:>3}  hops {:.3f}  decide-sum {:.4f} ms  "
              "reached {}/{}".format(
                  rep, scenario, args.trials,
                  statistics.fmean(float(d) for d in rep_dec[scenario]),
                  statistics.fmean(rep_path[scenario]), reached, args.trials))

      # One pass = one chance to reproduce the quantity 4.7 names.  Fit it here,
      # per pass, instead of only on the pooled set.
      mx = [statistics.fmean(float(d) for d in rep_dec[s]) for s in args.scenarios]
      my = [statistics.fmean(rep_path[s]) for s in args.scenarios]
      ra, rb, rr2, _rn = ols(mx, my)
      fx = []
      fy = []
      for s in args.scenarios:
          fx.extend(float(d) for d in rep_dec[s])
          fy.extend(rep_path[s])
      pa_, pb_, pr2_, _pn = ols(fx, fy)
      fits.append({"replicate": rep, "n_paths": len(fx),
                   "intercept_ms_4means": ra, "slope_ms_4means": rb,
                   "r2_4means": rr2, "intercept_ms_paths": pa_,
                   "slope_ms_paths": pb_, "r2_paths": pr2_,
                   "per_hop_ms": statistics.fmean(
                       [y / x for x, y in zip(fx, fy) if x])})
      print("  rep {} fit: per_path = {:+.4f} {:+.4f} * decisions "
            "(R2 {:.4f})".format(rep, ra, rb, rr2))

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(fits_csv, "w", newline="", encoding="utf-8") as fh:
        ff = ["replicate", "n_paths", "intercept_ms_4means", "slope_ms_4means",
              "r2_4means", "intercept_ms_paths", "slope_ms_paths", "r2_paths",
              "per_hop_ms"]
        writer = csv.DictWriter(fh, fieldnames=ff)
        writer.writeheader()
        for row in fits:
            out = dict(row)
            for key in ff[2:]:
                out[key] = "{:.6f}".format(row[key])
            writer.writerow(out)
    print("wrote " + fits_csv + " ({} replicates)".format(len(fits)))
    with open(runs_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in RUN_FIELDS:
                if key.endswith("_ms"):
                    out[key] = "{:.6f}".format(row[key])
            writer.writerow(out)
    print("\nwrote " + runs_csv + " ({} hops)".format(len(rows)))

    summary = []

    def put(scope, key, value, n=None, note=""):
        summary.append({"scope": scope, "key": key,
                        "value": "{:.6f}".format(value) if value == value else "nan",
                        "n": "" if n is None else n, "note": note})

    # ---- 1. hop position -------------------------------------------------------
    first = [r["decide_ms"] for r in rows if r["hop_index"] == 0]
    later = [r["decide_ms"] for r in rows if r["hop_index"] > 0]
    f_mean, f_sd = mean_std(first)
    l_mean, l_sd = mean_std(later)
    premium = f_mean - l_mean
    put("position", "first_hop_decide_ms", f_mean, len(first))
    put("position", "first_hop_decide_sd", f_sd, len(first))
    put("position", "later_hop_decide_ms", l_mean, len(later))
    put("position", "later_hop_decide_sd", l_sd, len(later))
    put("position", "first_hop_premium_ms", premium, None,
        "first minus later; the model cost = c + p*[hop 0] predicts "
        "intercept == this")
    put("position", "first_hop_premium_pct", premium / l_mean * 100.0 if l_mean else float("nan"))

    by_index = {}
    for r in rows:
        by_index.setdefault(r["hop_index"], []).append(r["decide_ms"])
    for idx in sorted(by_index):
        m, s = mean_std(by_index[idx])
        put("position_index", "decide_ms_hop{}".format(idx), m, len(by_index[idx]),
            "sd {:.6f}".format(s))

    # ---- 2. the intercept ------------------------------------------------------
    scen_means_x = []
    scen_means_y = []
    for scenario in args.scenarios:
        dx = statistics.fmean(float(d) for d in decisions[scenario])
        dy = statistics.fmean(per_path[scenario])
        scen_means_x.append(dx)
        scen_means_y.append(dy)
        put("scenario", "decisions_" + scenario, dx, args.trials)
        put("scenario", "per_path_decide_ms_" + scenario, dy, args.trials)
        put("scenario", "per_decision_ms_" + scenario, dy / dx if dx else float("nan"),
            args.trials)

    a4, b4, r24, se4, t4, p4, df4 = ols_intercept_test(scen_means_x, scen_means_y)
    n4 = len(scen_means_x)
    put("regression", "intercept_ms_4means", a4, n4,
        "the shape 4.7 quotes; published 0.513")
    put("regression", "slope_ms_4means", b4, n4, "published 0.281")
    put("regression", "r2_4means", r24, n4)
    put("regression", "intercept_se_4means", se4, n4, "df {}".format(df4))
    put("regression", "intercept_t_4means", t4, n4)
    put("regression", "intercept_p_4means", p4, n4, "H0: intercept = 0")

    all_x = []
    all_y = []
    for scenario in args.scenarios:
        all_x.extend(float(d) for d in decisions[scenario])
        all_y.extend(per_path[scenario])
    ap, bp, r2p, sep, tp, pp, dfp = ols_intercept_test(all_x, all_y)
    npth = len(all_x)
    put("regression", "intercept_ms_paths", ap, npth, "path-level, not 4 means")
    put("regression", "slope_ms_paths", bp, npth)
    put("regression", "r2_paths", r2p, npth)
    put("regression", "intercept_se_paths", sep, npth, "df {}".format(dfp))
    put("regression", "intercept_t_paths", tp, npth)
    put("regression", "intercept_p_paths", pp, npth, "H0: intercept = 0")

    # Does the intercept survive a rerun?  Nothing about the code or the machine
    # changed between passes, so any movement here is the quantity's own noise.
    i4 = [f["intercept_ms_4means"] for f in fits]
    ip = [f["intercept_ms_paths"] for f in fits]
    for f in fits:
        put("stability", "intercept_ms_4means_rep{}".format(f["replicate"]),
            f["intercept_ms_4means"], f["n_paths"],
            "slope {:.6f}, R2 {:.6f}".format(f["slope_ms_4means"], f["r2_4means"]))
        put("stability", "per_hop_ms_rep{}".format(f["replicate"]), f["per_hop_ms"],
            f["n_paths"])
    if len(i4) > 1:
        m4, s4 = mean_std(i4)
        put("stability", "intercept_ms_4means_mean", m4, len(i4))
        put("stability", "intercept_ms_4means_sd", s4, len(i4))
        put("stability", "intercept_ms_4means_min", min(i4), len(i4))
        put("stability", "intercept_ms_4means_max", max(i4), len(i4))
        put("stability", "intercept_ms_4means_range", max(i4) - min(i4), len(i4),
            "identical code and machine; this is the quantity's own noise")
        put("stability", "intercept_sign_flips_4means",
            float(min(i4) < 0 < max(i4)), len(i4),
            "1 = the sign is not even stable across passes")
        put("stability", "intercept_ms_paths_min", min(ip), len(ip))
        put("stability", "intercept_ms_paths_max", max(ip), len(ip))
        ph = [f["per_hop_ms"] for f in fits]
        hm, hsd = mean_std(ph)
        put("stability", "per_hop_ms_mean", hm, len(ph))
        put("stability", "per_hop_ms_sd", hsd, len(ph))
        put("stability", "per_hop_ms_spread_pct",
            (max(ph) - min(ph)) / hm * 100.0 if hm else float("nan"), len(ph),
            "per-hop cost moves this much between passes")

    blo, bhi, bshare, bn = bootstrap_intercept(all_x, all_y)
    put("regression", "intercept_boot_lo_ms", blo, bn,
        "{} resamples of the {} paths, seed {}".format(bn, npth, BOOT_SEED))
    put("regression", "intercept_boot_hi_ms", bhi, bn)
    put("regression", "intercept_boot_share_le0_pct", bshare, bn,
        "share of resamples whose intercept is <= 0")

    # Leave-one-scenario-out: four points, so dropping one is a real stress test.
    jack = []
    for drop in range(n4):
        jx = [x for i, x in enumerate(scen_means_x) if i != drop]
        jy = [y for i, y in enumerate(scen_means_y) if i != drop]
        ja, _jb, _jr, _jn = ols(jx, jy)
        jack.append(ja)
        put("jackknife", "intercept_ms_without_" + args.scenarios[drop], ja, 3)
    put("jackknife", "intercept_ms_min", min(jack), n4)
    put("jackknife", "intercept_ms_max", max(jack), n4)
    put("jackknife", "intercept_ms_range", max(jack) - min(jack), n4,
        "how far the quoted intercept moves when one scenario leaves")
    put("regression", "intercept_minus_premium_ms", a4 - premium, None,
        "0 would mean the intercept is entirely the first-hop premium")
    put("regression", "premium_share_of_intercept_pct",
        premium / a4 * 100.0 if a4 else float("nan"))

    perdec = [y / x for x, y in zip(all_x, all_y) if x]
    dcount = [x for x in all_x if x]
    put("regression", "r_perdecision_vs_decisions", pearson(dcount, perdec), len(perdec),
        "negative => long paths are cheaper per decision => positive intercept")
    const = statistics.fmean([r["decide_ms"] for r in rows])
    put("regression", "pooled_per_hop_ms", const, len(rows),
        "a constant-cost model gives intercept exactly 0 by construction")

    # ---- 3. inside one decision ------------------------------------------------
    dm = statistics.fmean([r["decide_ms"] for r in rows])
    if args.parts:
        for key in ("decide_ms", "tensor_ms", "forward_ms", "mask_ms", "argmax_ms",
                    "components_ms", "residual_ms"):
            m, s = mean_std([r[key] for r in rows])
            put("decision_parts", key, m, len(rows), "sd {:.6f}".format(s))
        for key in ("tensor_ms", "forward_ms", "mask_ms", "argmax_ms", "residual_ms"):
            m = statistics.fmean([r[key] for r in rows])
            put("decision_parts_pct", key.replace("_ms", "_pct_of_decide"),
                m / dm * 100.0 if dm else float("nan"), len(rows))
    else:
        m, s = mean_std([r["decide_ms"] for r in rows])
        put("decision_parts", "decide_ms", m, len(rows), "sd {:.6f}".format(s))

    # ---- 4. what table 12 does not charge --------------------------------------
    sm, ssd = mean_std([r["get_state_ms"] for r in rows])
    vm, vsd = mean_std([r["get_valid_ms"] for r in rows])
    put("uncharged", "get_state_ms_per_hop", sm, len(rows), "sd {:.6f}".format(ssd))
    put("uncharged", "get_valid_ms_per_hop", vm, len(rows), "sd {:.6f}".format(vsd))
    put("uncharged", "get_state_pct_of_decide", sm / dm * 100.0 if dm else float("nan"))
    put("uncharged", "uncharged_pct_of_decide",
        (sm + vm) / dm * 100.0 if dm else float("nan"))

    welch_ps = []
    for scenario in args.scenarios:
        dqn = statistics.fmean(per_path[scenario])
        st = statistics.fmean(per_state[scenario])
        va = statistics.fmean(per_valid[scenario])
        ospf = statistics.fmean(ospf_path[scenario])
        qos = statistics.fmean(qos_path[scenario])
        put("multiple", "ospf_per_path_ms_" + scenario, ospf, args.trials,
            "same process as the DQN rows")
        put("multiple", "qos_per_path_ms_" + scenario, qos, args.trials,
            "composite-QoS Dijkstra, same process")
        put("multiple", "dqn_as_table12_ms_" + scenario, dqn, args.trials)
        put("multiple", "dqn_with_state_ms_" + scenario, dqn + st + va, args.trials,
            "decide + get_state + get_valid_actions at every hop")
        put("multiple", "state_charge_pct_" + scenario,
            (st + va) / dqn * 100.0 if dqn else float("nan"), args.trials)
        put("multiple", "multiple_table12_" + scenario,
            dqn / ospf if ospf else float("nan"), args.trials)
        put("multiple", "multiple_with_state_" + scenario,
            (dqn + st + va) / ospf if ospf else float("nan"), args.trials)
        put("vs_qos", "dqn_over_qos_table12_" + scenario,
            dqn / qos if qos else float("nan"), args.trials,
            "<1 means the draft's favourable sentence holds")
        put("vs_qos", "dqn_over_qos_with_state_" + scenario,
            (dqn + st + va) / qos if qos else float("nan"), args.trials,
            "the same comparison once the observation build is charged")
        diff, tt, wdf, pv = welch(per_path[scenario], qos_path[scenario])
        put("vs_qos", "welch_diff_ms_table12_" + scenario, diff, args.trials,
            "DQN minus QoS-Dijkstra")
        put("vs_qos", "welch_p_table12_" + scenario, pv, args.trials,
            "t {:.4f}, df {:.1f}, {} paths per arm".format(tt, wdf, args.trials))
        with_state = [a + b + c for a, b, c in zip(
            per_path[scenario], per_state[scenario], per_valid[scenario])]
        diff2, tt2, wdf2, pv2 = welch(with_state, qos_path[scenario])
        put("vs_qos", "welch_diff_ms_with_state_" + scenario, diff2, args.trials)
        put("vs_qos", "welch_p_with_state_" + scenario, pv2, args.trials,
            "t {:.4f}, df {:.1f}".format(tt2, wdf2))
        welch_ps.append(("with_state_" + scenario, pv2))
        welch_ps.append(("table12_" + scenario, pv))

    # One Holm family over the eight QoS comparisons (both charging rules x four
    # scenarios); counting only the four that suit the conclusion would be the
    # family-size mistake this tree has already been caught by twice.
    order = [k for k, _p in welch_ps]
    adj = holm_bonferroni([p for _k, p in welch_ps])
    for key, pa in zip(order, adj):
        put("vs_qos", "welch_pholm_" + key, pa, 8, "Holm family 8")

    mt = [statistics.fmean(per_path[s]) / statistics.fmean(ospf_path[s])
          for s in args.scenarios]
    mw = [(statistics.fmean(per_path[s]) + statistics.fmean(per_state[s])
           + statistics.fmean(per_valid[s])) / statistics.fmean(ospf_path[s])
          for s in args.scenarios]
    put("multiple", "multiple_table12_min", min(mt), len(mt))
    put("multiple", "multiple_table12_max", max(mt), len(mt))
    put("multiple", "multiple_with_state_min", min(mw), len(mw))
    put("multiple", "multiple_with_state_max", max(mw), len(mw))

    with open(sum_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scope", "key", "value", "n", "note"])
        writer.writeheader()
        for row in summary:
            writer.writerow(row)
    print("wrote " + sum_csv + " ({} rows)".format(len(summary)))

    print("\n--- the intercept ---")
    print("  4 scenario means : per_path = {:.4f} + {:.4f} * decisions "
          "(R2 {:.4f})".format(a4, b4, r24))
    print("  path level (n={}): per_path = {:.4f} + {:.4f} * decisions "
          "(R2 {:.4f})".format(npth, ap, bp, r2p))
    print("  first hop {:.4f} ms vs later {:.4f} ms -> premium {:.4f} ms"
          .format(f_mean, l_mean, premium))
    print("  premium / intercept = {:.1f}%".format(
        premium / a4 * 100.0 if a4 else float("nan")))
    print("  r(per-decision cost, decisions) = {:+.4f}".format(
        pearson(dcount, perdec)))
    print("\n--- inside one decision (mean over {} hops) ---".format(len(rows)))
    print("  decide {:.4f} = tensor {:.4f} + forward {:.4f} + mask {:.4f} "
          "+ argmax {:.4f} (+ residual {:.4f})".format(
              dm,
              statistics.fmean([r["tensor_ms"] for r in rows]),
              statistics.fmean([r["forward_ms"] for r in rows]),
              statistics.fmean([r["mask_ms"] for r in rows]),
              statistics.fmean([r["argmax_ms"] for r in rows]),
              statistics.fmean([r["residual_ms"] for r in rows])))
    print("\n--- what table 12 leaves out ---")
    print("  get_state {:.4f} ms/hop, get_valid_actions {:.4f} ms/hop "
          "= {:.1f}% of a decision".format(
              sm, vm, (sm + vm) / dm * 100.0 if dm else float("nan")))
    print("  multiple vs OSPF: table 12 as built {:.1f}-{:.1f}x, "
          "with those charged {:.1f}-{:.1f}x".format(
              min(mt), max(mt), min(mw), max(mw)))
    print("\n--- vs the composite-QoS Dijkstra (4.7's favourable sentence) ---")
    print("  {:<12}{:>12}{:>12}{:>14}{:>12}{:>12}".format(
        "scenario", "DQN(t12)", "DQN(+state)", "QoS-Dijkstra", "ratio t12",
        "ratio +st"))
    for s in args.scenarios:
        dqn = statistics.fmean(per_path[s])
        st = statistics.fmean(per_state[s]) + statistics.fmean(per_valid[s])
        qos = statistics.fmean(qos_path[s])
        print("  {:<12}{:>12.4f}{:>12.4f}{:>14.4f}{:>12.3f}{:>12.3f}".format(
            s, dqn, dqn + st, qos, dqn / qos, (dqn + st) / qos))
    print("  Holm(8) adjusted p: " + ", ".join(
        "{} {:.4f}".format(k, pa) for k, pa in zip(order, adj)))


if __name__ == "__main__":
    main()
