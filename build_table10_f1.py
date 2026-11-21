# -*- coding: utf-8 -*-
"""Table 10's F1 column (11 settings) from a single pass of RiskValidationEnv.

Why one pass is enough
----------------------
In network_env_risk_validation.py, `risk` is a pure *output*: step() only writes
q["risk"], and every danger mechanism is generated from `random` independently of
it. So the raw time series does not depend on the weights, on sigma_ref, or on K.
Collecting it once and re-scoring offline is therefore not an approximation -- it
is strictly better than re-running per setting, because re-running would draw a
different random stream for every setting and mix that noise into the comparison.

Verification before any output
------------------------------
calculate_link_risk is wrapped so every call argument is recorded (including
link_type, which snapshot() does not expose and which differs at t=0: reset passes
the base type, step passes the danger-derived type). The recomputation is then
checked against the recorded return value; if the worst absolute error is not
below 1e-12 the script aborts and writes nothing.

Protocol -- identical to run_risk_identification_validation_v2.py
-----------------------------------------------------------------
seeds 0..9 x 30 episodes x (1 reset snapshot + 30 steps). Per snapshot the top
20% by score (k = ceil(50*0.20) = 10) is the prediction, danger_label == 1 is the
truth (n_danger = int(50*0.20) = 10). Since |pred| == |actual|, precision =
recall = F1. Per-snapshot F1 -> per-seed mean -> mean +/- sd across seeds.

Note: scipy is unusable in this environment (numpy 2.4.6 vs scipy requiring
<1.28), so statistics come from the project module review_stats.

Usage
-----
    KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 build_table10_f1.py
    KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 build_table10_f1.py --quick
"""
import argparse
import math
import os
import pickle
import random
import sys

import numpy as np

import network_env_risk_validation as RV
import review_stats as RS
from topology import EDGES

SEEDS = list(range(10))
EPISODES_PER_SEED = 30
STEPS_PER_EPISODE = 30
TOP_RATIO = 0.20
DANGER_RATIO = 0.20
N_EDGES = len(EDGES)

# risk_calculator.py hardcodes these; the local copy has no DEFAULT_WEIGHTS dict.
# The 1e-12 numeric check below is what actually guarantees they are right.
BASE_W = {"util": 0.35, "cong": 0.30, "delay": 0.15, "loss": 0.10, "type": 0.10}
SIGMA_D, SIGMA_L = 20.0, 0.03
BASE_K = RV.RISK_WINDOW  # 5

TYPE_RISK = {
    "unstable_congested": 1.0,
    "unstable": 0.9,
    "congested": 0.8,
    "backup": 0.3,
    "backbone": 0.05,
}

# weight key -> ZEROED_FACTORS name used by train_factor_ablation.py
WKEY = {
    "util": "utilization",
    "cong": "congestion",
    "delay": "delay_inst",
    "loss": "loss_inst",
    "type": "link_type",
}

RESULT_DIR = "review_results"
CACHE = os.path.join(RESULT_DIR, "_table10_rawlog.pkl")


def type_risk(name):
    return TYPE_RISK.get(name, 0.15)


# --- 1. collect the raw series ---------------------------------------------

def collect(seeds):
    """One pass over the environment, recording every calculate_link_risk call."""
    real = RV.calculate_link_risk
    rec = []

    def spy(utilization, is_congested, delay_history, loss_history,
            link_type="normal"):
        out = real(utilization=utilization, is_congested=is_congested,
                   delay_history=delay_history, loss_history=loss_history,
                   link_type=link_type)
        rec.append((float(utilization), 1.0 if is_congested else 0.0,
                    type_risk(link_type), link_type, float(out)))
        return out

    RV.calculate_link_risk = spy
    try:
        eps = []
        n_calls = 0
        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            for ep in range(EPISODES_PER_SEED):
                del rec[:]
                env = RV.RiskValidationEnv(seed=seed * 10000 + ep,
                                           danger_ratio=DANGER_RATIO)
                snaps = [env.snapshot()]
                for _ in range(STEPS_PER_EPISODE):
                    snaps.append(env.step())

                nt = len(snaps)
                if len(rec) != nt * N_EDGES:
                    raise AssertionError("call count %d != %d"
                                         % (len(rec), nt * N_EDGES))
                n_calls += len(rec)

                cols = ("delay", "loss", "util", "cong", "trisk", "risk0")
                arr = dict((k, np.zeros((nt, N_EDGES))) for k in cols)
                label = np.zeros(N_EDGES, dtype=int)
                seen_types = {}

                for t, snap in enumerate(snaps):
                    if len(snap) != N_EDGES:
                        raise AssertionError("snapshot width %d" % len(snap))
                    for j, row in enumerate(snap):
                        u, c, tr, ltname, out = rec[t * N_EDGES + j]
                        # positional join check: is this the same link?
                        if abs(out - row["risk"]) > 1e-12:
                            raise AssertionError("risk mismatch t=%d j=%d" % (t, j))
                        if abs(u - row["utilization"]) > 1e-12:
                            raise AssertionError("util mismatch t=%d j=%d" % (t, j))
                        if abs(c - float(row["is_congested"])) > 1e-12:
                            raise AssertionError("cong mismatch t=%d j=%d" % (t, j))
                        arr["delay"][t, j] = row["delay"]
                        arr["loss"][t, j] = row["loss"]
                        arr["util"][t, j] = row["utilization"]
                        arr["cong"][t, j] = c
                        arr["trisk"][t, j] = tr
                        arr["risk0"][t, j] = out
                        seen_types[ltname] = seen_types.get(ltname, 0) + 1
                        if t == 0:
                            label[j] = row["danger_label"]
                        elif label[j] != row["danger_label"]:
                            raise AssertionError("label moved t=%d j=%d" % (t, j))
                if int(label.sum()) != 10:
                    raise AssertionError("n_danger %d != 10" % int(label.sum()))
                eps.append(dict(seed=seed, ep=ep, label=label,
                                types=seen_types, **arr))
        print("  risk calls recorded: %d" % n_calls)
        return eps
    finally:
        RV.calculate_link_risk = real


# --- 2. offline re-scoring -------------------------------------------------

def rolling_std(a, K):
    """np.std(ddof=0) over a deque(maxlen=K); 0.0 while only one value is in it.

    At reset the deque is cleared and one value appended, so at time t it holds
    the last min(t+1, K) snapshot values -- exactly this window.
    """
    out = np.zeros_like(a)
    for t in range(1, a.shape[0]):
        out[t] = a[max(0, t - K + 1):t + 1].std(axis=0)
    return out


def stds(e, K):
    key = "_sd%d" % K
    if key not in e:
        e[key] = (rolling_std(e["delay"], K), rolling_std(e["loss"], K))
    return e[key]


def recompute(e, K=BASE_K, sD=SIGMA_D, sL=SIGMA_L, weights=None, zeroed=(),
              renorm=False):
    w = dict(BASE_W if weights is None else weights)
    if renorm:
        s = sum(v for k, v in w.items() if WKEY[k] not in zeroed)
        w = dict((k, (v / s if WKEY[k] not in zeroed else 0.0))
                 for k, v in w.items())

    sd, sl = stds(e, K)
    ur = np.clip(e["util"], 0.0, 1.0)   # utilization reaches 1.5 in step()
    cr = e["cong"]
    di = np.minimum(sd / sD, 1.0)
    li = np.minimum(sl / sL, 1.0)
    tr = e["trisk"]
    if "utilization" in zeroed:
        ur = 0.0
    if "congestion" in zeroed:
        cr = 0.0
    if "delay_inst" in zeroed:
        di = 0.0
    if "loss_inst" in zeroed:
        li = 0.0
    if "link_type" in zeroed:
        tr = 0.0

    risk = (w["util"] * ur + w["cong"] * cr + w["delay"] * di +
            w["loss"] * li + w["type"] * tr)
    return np.clip(risk, 0.0, 1.0)


def minmax(a):
    lo, hi = a.min(axis=1, keepdims=True), a.max(axis=1, keepdims=True)
    rng = hi - lo
    return np.where(rng < 1e-12, 0.0, (a - lo) / np.where(rng < 1e-12, 1.0, rng))


def simple_qos(e):
    return (minmax(e["delay"]) + minmax(e["loss"]) + minmax(e["util"])) / 3.0


# --- 3. top-k F1 -----------------------------------------------------------

K_TOP = max(1, int(math.ceil(N_EDGES * TOP_RATIO)))


def topk_f1(scores, label):
    pos = label.astype(bool)
    n_pos = max(int(label.sum()), 1)
    out = np.zeros(scores.shape[0])
    for t in range(scores.shape[0]):
        idx = np.argsort(-scores[t], kind="stable")[:K_TOP]
        tp = int(pos[idx].sum())
        p, r = tp / float(K_TOP), tp / float(n_pos)
        out[t] = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return out


SETTINGS = [
    ("Proposed (Full)",      {}),
    ("w/o VD",               {"zeroed": ("delay_inst",)}),
    ("w/o VL",               {"zeroed": ("loss_inst",)}),
    ("w/o VD, VL",           {"zeroed": ("delay_inst", "loss_inst")}),
    ("w/o U",                {"zeroed": ("utilization",)}),
    ("w/o IC",               {"zeroed": ("congestion",)}),
    ("w/o T",                {"zeroed": ("link_type",)}),
    ("Equal weights (0.2)",  {"weights": dict((k, 0.2) for k in BASE_W)}),
    ("sigma_ref x0.5",       {"sD": SIGMA_D * 0.5, "sL": SIGMA_L * 0.5}),
    ("sigma_ref x2",         {"sD": SIGMA_D * 2.0, "sL": SIGMA_L * 2.0}),
    ("K = 3 (K/2)",          {"K": 3}),
    ("K = 10 (2K)",          {"K": 10}),
]

# Sanity: renormalising is a monotone rescale, so it must not move F1 at all.
RENORM_CHECK = [
    ("w/o VD, VL (renorm)",  {"zeroed": ("delay_inst", "loss_inst"),
                              "renorm": True}),
    ("w/o U (renorm)",       {"zeroed": ("utilization",), "renorm": True}),
]


def per_seed_f1(eps, scorer):
    acc = {}
    for e in eps:
        acc.setdefault(e["seed"], []).append(scorer(e).mean())
    return [float(np.mean(v)) for _, v in sorted(acc.items())]


def paired_p(vals, base):
    d = [a - b for a, b in zip(vals, base)]
    dm, ds = RS.mean_std(d)
    if ds < 1e-15:
        return dm, (1.0 if abs(dm) < 1e-15 else 0.0)
    t = dm / (ds / math.sqrt(len(d)))
    return dm, RS.t_two_sided_p(abs(t), len(d) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3 seeds only")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    seeds = SEEDS[:3] if args.quick else SEEDS
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    cache = CACHE + (".quick" if args.quick else "")

    if os.path.exists(cache) and not args.no_cache:
        print("using cached raw log: %s" % cache)
        with open(cache, "rb") as f:
            eps = pickle.load(f)
    else:
        print("collecting raw series (%d seeds x %d episodes)..."
              % (len(seeds), EPISODES_PER_SEED))
        eps = collect(seeds)
        with open(cache, "wb") as f:
            pickle.dump(eps, f, protocol=4)
        print("cached: %s" % cache)

    print("episodes=%d  links/snapshot=%d  top-k=%d  window=%d"
          % (len(eps), N_EDGES, K_TOP, BASE_K))
    tc = {}
    for e in eps:
        for k, v in e["types"].items():
            tc[k] = tc.get(k, 0) + v
    print("link_type seen: %s" % sorted(tc.items(), key=lambda x: -x[1]))

    worst = 0.0
    for e in eps:
        worst = max(worst, float(np.abs(recompute(e) - e["risk0"]).max()))
    print("VERIFY recompute vs original: max |diff| = %.3e" % worst)
    if not worst < 1e-12:
        print("ABORT: recomputation does not reproduce the original risk.")
        return 1

    rows = []
    for name, kw in SETTINGS:
        vals = per_seed_f1(eps, lambda e, kw=kw: topk_f1(recompute(e, **kw),
                                                         e["label"]))
        m, s = RS.mean_std(vals)
        rows.append((name, m, s, vals))
        print("  %-22s F1 = %.4f +/- %.4f" % (name, m, s))

    base = rows[0][3]
    print("\n-- pipeline cross-checks (expect Full ~0.966, SimpleQoS ~0.936) --")
    for nm, sc in (("Delay Only", lambda e: e["delay"]),
                   ("Loss Only", lambda e: e["loss"]),
                   ("Utilization Only", lambda e: e["util"]),
                   ("Simple QoS Score", simple_qos)):
        v = per_seed_f1(eps, lambda e, sc=sc: topk_f1(sc(e), e["label"]))
        m, s = RS.mean_std(v)
        print("  %-22s F1 = %.4f +/- %.4f" % (nm, m, s))

    print("\n-- renormalisation must not move F1 (monotone rescale) --")
    for name, kw in RENORM_CHECK:
        v = per_seed_f1(eps, lambda e, kw=kw: topk_f1(recompute(e, **kw),
                                                      e["label"]))
        m, _ = RS.mean_std(v)
        ref = dict((r[0], r[1]) for r in rows)[name.replace(" (renorm)", "")]
        print("  %-22s F1 = %.4f   (vs %.4f, diff %.2e)"
              % (name, m, ref, abs(m - ref)))

    out = os.path.join(RESULT_DIR, "table10_f1.csv")
    with open(out, "w") as f:
        f.write("setting,f1_mean,f1_std,n_seeds,delta_vs_full,paired_p\n")
        for name, m, s, vals in rows:
            if name == rows[0][0]:
                f.write("\"%s\",%.6f,%.6f,%d,,\n" % (name, m, s, len(vals)))
                continue
            dm, p = paired_p(vals, base)
            f.write("\"%s\",%.6f,%.6f,%d,%+.6f,%.6g\n"
                    % (name, m, s, len(vals), dm, p))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
