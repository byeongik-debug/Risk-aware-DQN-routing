# -*- coding: utf-8 -*-
"""Table 10's sigma_ref x0.5 / x2 rows -- does the normalizer matter at all?

risk_calculator.py hardcodes two reference scales (lines 45, 56):
    delay_instability = min(np.std(delay_history) / 20.0,  1.0)
    loss_instability  = min(np.std(loss_history)  / 0.03,  1.0)

Table 10 asks for sigma_ref x0.5 and x2 rows. Retraining the policy for each
setting costs ~1.5 h x 5 runs x 4+ settings, and section 14 of
FINDINGS_table10_f1.md shows n=5 cannot separate adjacent rows anyway (39-219
runs needed). So measure the cheap question first: does the perturbation change
the RANKING of links, and does it change the PATH that gets chosen? If it does
not, retraining cannot change the routing and the rows are answerable without
it.

Nothing in SDN_DQN_new_env/ is modified. A sigma-parameterized replica of
calculate_link_risk is built here and checked against the original bit-for-bit
at the default scales; the replica reads ZEROED_FACTORS / CUSTOM_WEIGHTS live
from risk_calculator so the ablation hooks keep working.

The replica is INSTALLED as network_env.calculate_link_risk (network_env.py
does `from risk_calculator import calculate_link_risk`, binding the name in its
own namespace) and the environment then assembles the risk field itself. That
matters: the field is not just calculate_link_risk() per edge. The environment
passes a scenario-dependent `risk_type` rather than qos["type"], and afterwards
multiplies backup links by 0.60 (congestion/unstable) and dynamic support links
by 0.55 (network_env.py:236-250). An earlier version of this script rebuilt the
field by hand, read the type from the wrong key ("link_type" instead of "type")
and skipped those multipliers, which silently flattened the link_type factor
for every edge. Letting the environment build the field removes that whole
class of error, and the default setting is verified to reproduce the unpatched
environment's field exactly.

Because calculate_link_risk consumes no randomness, re-running a snapshot from
the same seed under a different sigma leaves delay/loss/utilization/bandwidth
bit-identical and changes only the risk field -- which is exactly the
comparison we want.

Usage:  python -X utf8 eval_sigma_ref_sensitivity.py [snapshots]
Output: review_results/table10_sigma_ref.csv
"""
import csv
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
OUT = os.path.join(HERE, "review_results", "table10_sigma_ref.csv")

# review_response/risk_calculator.py is the friend's pre-ablation copy -- keep
# it from shadowing the original the environment actually uses.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import risk_calculator as RC    # noqa: E402
import network_env as NE        # noqa: E402

for _m in (RC, NE):
    assert os.path.abspath(_m.__file__).startswith(ORIG), _m.__file__

ORIGINAL = NE.calculate_link_risk
SEED = 20261114
D_REF, L_REF = 20.0, 0.03
SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
SETTINGS = [("delay x0.5", D_REF * 0.5, L_REF),
            ("delay x2", D_REF * 2.0, L_REF),
            ("loss x0.5", D_REF, L_REF * 0.5),
            ("loss x2", D_REF, L_REF * 2.0),
            ("both x0.5", D_REF * 0.5, L_REF * 0.5),
            ("both x2", D_REF * 2.0, L_REF * 2.0)]


def make_risk(d_ref, l_ref):
    """The original function with the two hardcoded scales parameterized."""
    def f(utilization, is_congested, delay_history, loss_history,
          link_type="normal"):
        utilization_risk = min(max(utilization, 0.0), 1.0)
        congestion_risk = 1.0 if is_congested else 0.0
        di = (np.std(delay_history) / d_ref if len(delay_history) > 1 else 0.0)
        delay_instability = min(di, 1.0)
        li = (np.std(loss_history) / l_ref if len(loss_history) > 1 else 0.0)
        loss_instability = min(li, 1.0)
        if link_type == "unstable_congested":
            link_type_risk = 1.0
        elif link_type == "unstable":
            link_type_risk = 0.9
        elif link_type == "congested":
            link_type_risk = 0.8
        elif link_type == "backup":
            link_type_risk = 0.3
        elif link_type == "backbone":
            link_type_risk = 0.05
        else:
            link_type_risk = 0.15
        z = RC.ZEROED_FACTORS
        if "utilization" in z:
            utilization_risk = 0.0
        if "congestion" in z:
            congestion_risk = 0.0
        if "delay_inst" in z:
            delay_instability = 0.0
        if "loss_inst" in z:
            loss_instability = 0.0
        if "link_type" in z:
            link_type_risk = 0.0
        w = RC.CUSTOM_WEIGHTS if RC.CUSTOM_WEIGHTS is not None \
            else RC.DEFAULT_WEIGHTS
        risk = (w["util"] * utilization_risk + w["cong"] * congestion_risk
                + w["delay"] * delay_instability
                + w["loss"] * loss_instability
                + w["type"] * link_type_risk)
        return float(np.clip(risk, 0.0, 1.0))
    return f


def verify_replica(trials=200000):
    """At the default scales the replica must equal the original exactly."""
    rep = make_risk(D_REF, L_REF)
    rng = random.Random(1234)
    types = ["normal", "backbone", "backup", "congested", "unstable",
             "unstable_congested"]
    worst = 0.0
    for _ in range(trials):
        n = rng.randint(0, 5)
        kw = dict(utilization=rng.uniform(-0.2, 1.3),
                  is_congested=rng.random() < 0.5,
                  delay_history=[rng.uniform(0, 120) for _ in range(n)],
                  loss_history=[rng.uniform(0, 0.3) for _ in range(n)],
                  link_type=rng.choice(types))
        worst = max(worst, abs(ORIGINAL(**kw) - rep(**kw)))
    print("replica check over %d random inputs: max |diff| = %.3e"
          % (trials, worst))
    assert worst == 0.0, worst


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def build(scenario, ep, risk_fn):
    """A settled environment whose risk field the ENVIRONMENT assembled."""
    NE.calculate_link_risk = risk_fn
    try:
        random.seed(SEED + ep)
        np.random.seed(SEED + ep)
        env = NE.NetworkEnv(use_risk=True, scenario=scenario,
                            risk_window=NE.RISK_WINDOW)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
    finally:
        NE.calculate_link_risk = ORIGINAL
    return env


def field(env):
    return [float(env.qos[e]["risk"]) for e in NE.EDGES]


def verify_install(scenario, ep):
    """Installing the default replica must reproduce the untouched field."""
    a = build(scenario, ep, ORIGINAL)
    b = build(scenario, ep, make_risk(D_REF, L_REF))
    for k in ("delay", "loss", "utilization", "bandwidth", "risk"):
        xs = [float(a.qos[e][k]) for e in NE.EDGES]
        ys = [float(b.qos[e][k]) for e in NE.EDGES]
        worst = max(abs(x - y) for x, y in zip(xs, ys))
        assert worst == 0.0, (scenario, k, worst)
    assert tuple(a.ospf_path(cost_type="risk")) == \
        tuple(b.ospf_path(cost_type="risk"))
    return True


def path_quality(env, path):
    """Table 10's own columns for one path, scored by the environment.

    A changed path only matters if its measured quality differs, so report the
    same quantities Table 10 reports rather than just 'the path changed'.
    """
    m = env.calculate_path_metrics(list(path))
    hop = max(int(m["hop_count"]), 1)
    hits = max(int(m["unstable_count"]), int(m["congested_count"]))
    return (float(m["total_delay"]), float(m["total_loss"]), float(hop),
            hits / hop * 100.0, float(hits))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    verify_replica()
    for scen in SCENARIOS:
        verify_install(scen, 0)
    print("install check: default replica reproduces the untouched field"
          " exactly in all %d scenarios" % len(SCENARIOS))
    print()
    rows_out = []
    for scen in SCENARIOS:
        states = [build(scen, i, ORIGINAL) for i in range(n)]
        bvals = [field(env) for env in states]
        bpath = [tuple(env.ospf_path(cost_type="risk")) for env in states]
        bq = np.array([path_quality(env, p)
                       for env, p in zip(states, bpath)])
        bmean = float(np.mean([v for g in bvals for v in g]))
        print("=== %s (%d snapshots x %d links) ==="
              % (scen, n, len(NE.EDGES)))
        hdr = ("%-11s | %9s | %8s | %8s | %8s | %8s | %8s | %7s | %6s | %6s"
               % ("setting", "spearman", "rank chg", "path chg", "delay",
                  "loss", "hop", "Risky%", "hits", "d mean"))
        print(hdr)
        print("-" * len(hdr))
        print("%-11s | %9s | %8s | %8s | %8.2f | %8.5f | %8.3f | %6.2f%% |"
              " %6.3f | %6s"
              % ("default", "-", "-", "-", bq[:, 0].mean(), bq[:, 1].mean(),
                 bq[:, 2].mean(), bq[:, 3].mean(), bq[:, 4].mean(), "-"))
        for name, dr, lr in SETTINGS:
            f = make_risk(dr, lr)
            diffs, sps, allv, nq = [], [], [], []
            rankchg = pathchg = 0
            for i, (benv, bv, bp) in enumerate(zip(states, bvals, bpath)):
                env = build(scen, i, f)
                nv = field(env)
                allv.extend(nv)
                diffs.extend(a - b for a, b in zip(nv, bv))
                sps.append(spearman(bv, nv))
                rb, rn = ranks(bv), ranks(nv)
                rankchg += sum(1 for a, b in zip(rb, rn) if a != b)
                npath = tuple(env.ospf_path(cost_type="risk"))
                if npath != bp:
                    pathchg += 1
                # score the perturbed path on the DEFAULT environment, so the
                # yardstick is the same for every setting
                nq.append(path_quality(benv, npath))
            nq = np.array(nq)
            tot = len(NE.EDGES) * len(states)
            sp = float(np.nanmean(sps))
            print("%-11s | %9.5f | %7.1f%% | %7.1f%% | %8.2f | %8.5f |"
                  " %8.3f | %6.2f%% | %6.3f | %+6.4f"
                  % (name, sp, 100.0 * rankchg / tot,
                     100.0 * pathchg / len(states), nq[:, 0].mean(),
                     nq[:, 1].mean(), nq[:, 2].mean(), nq[:, 3].mean(),
                     nq[:, 4].mean(), float(np.mean(diffs))))
            rows_out.append([scen, name, "%.6f" % dr, "%.6f" % lr, n,
                             "%.6f" % bmean, "%.6f" % float(np.mean(allv)),
                             "%.6f" % float(np.mean(diffs)),
                             "%.6f" % float(np.max(np.abs(diffs))),
                             "%.6f" % sp,
                             "%.2f" % (100.0 * rankchg / tot),
                             "%.2f" % (100.0 * pathchg / len(states)),
                             "%.4f" % nq[:, 0].mean(),
                             "%.6f" % nq[:, 1].mean(),
                             "%.4f" % nq[:, 2].mean(),
                             "%.4f" % nq[:, 3].mean(),
                             "%.4f" % nq[:, 4].mean(),
                             "%.4f" % bq[:, 0].mean(),
                             "%.6f" % bq[:, 1].mean(),
                             "%.4f" % bq[:, 2].mean(),
                             "%.4f" % bq[:, 3].mean(),
                             "%.4f" % bq[:, 4].mean()])
        print()
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "setting", "delay_sigma_ref", "loss_sigma_ref",
                    "snapshots", "risk_mean_default", "risk_mean",
                    "risk_delta", "risk_max_abs_delta",
                    "spearman_vs_default", "rank_changed_pct",
                    "risk_path_changed_pct",
                    "delay", "loss_per_hop", "hop", "risky_pct", "risky_hits",
                    "delay_default", "loss_per_hop_default", "hop_default",
                    "risky_pct_default", "risky_hits_default"])
        w.writerows(rows_out)
    print("wrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
