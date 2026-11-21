"""Measure how much the two temporal risk factors actually contribute, and why.

The risk score assigns 25% of its weight to temporal instability (delay jitter 0.15,
loss jitter 0.10), yet removing either factor changes nothing measurable and the
temporal window (1, 3, 5, 7, 10) does not separate the policies either.  This script
locates the cause in the QoS process rather than in the learner.

In the congestion and dynamic scenarios a stressed link draws strictly positive noise
every step -- delay_noise ~ U(5, 10), loss_noise ~ U(0.012, 0.050) -- so its delay and
loss ramp monotonically into the hard clip at 120 ms and 0.35.  A signal pinned to its
ceiling has zero standard deviation, so both temporal factors collapse to exactly zero
on precisely the links the router must avoid.  The unstable scenario is the only one
with two-sided delay noise, U(-5, 10), and is therefore the only one where the temporal
factors stay informative.

Two tables are written.  The per-link-class table splits edges into the scenario's
stressed set and the rest, because the population average hides the fact that the dead
factors are concentrated on the dangerous links.  The decomposition table reports each
weighted term's share of the risk score, which is the evidence for the weight question.
"""

import csv
import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from network_env import (  # noqa: E402
    CONGESTION_LINKS,
    UNSTABLE_LINKS,
    NetworkEnv,
)
from risk_calculator import DEFAULT_WEIGHTS  # noqa: E402
from topology import EDGES  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "review_results")

# Clip ceilings from NetworkEnv.update_dynamic_qos.
DELAY_CAP, LOSS_CAP = 120.0, 0.35
# Normalisers from risk_calculator: std is divided by these before the min(.,1) cap.
DELAY_NORM, LOSS_NORM = 20.0, 0.03

SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
WINDOWS = (1, 3, 5, 7, 10)
# reset() performs one init plus one update, eval_window_sweep.py adds ten warm-ups,
# and a learned policy then takes roughly eight hops, each of which advances the QoS.
WARMUP = 10
HOPS = 8
SEEDS = (0, 1, 2, 3, 4)


def stressed_set(env):
    """The edges the scenario deliberately degrades, as update_dynamic_qos selects them."""
    if env.scenario == "congestion":
        return {env.normalize_edge(*x) for x in CONGESTION_LINKS}
    if env.scenario == "unstable":
        return {env.normalize_edge(*x) for x in UNSTABLE_LINKS}
    if env.scenario == "dynamic":
        danger, _ = env.get_dynamic_phase_sets()
        return danger
    return set()


def instability_terms(env, edge):
    """The two temporal factors exactly as risk_calculator computes them."""
    delays = list(env.delay_history[edge])
    losses = list(env.loss_history[edge])
    delay_inst = min(np.std(delays) / DELAY_NORM, 1.0) if len(delays) > 1 else 0.0
    loss_inst = min(np.std(losses) / LOSS_NORM, 1.0) if len(losses) > 1 else 0.0
    return float(delay_inst), float(loss_inst)


def link_type_risk(env, edge, stressed):
    """Reproduce risk_calculator's link-type term for this edge."""
    link_type = env.qos[edge]["type"]
    if env.scenario == "dynamic" and edge in stressed:
        return 1.0
    if env.scenario == "unstable" and edge in stressed:
        return 0.9
    if env.scenario == "congestion" and edge in stressed:
        return 0.8
    return {"backup": 0.3, "backbone": 0.05}.get(link_type, 0.15)


def rollout(scenario, window, seed):
    """Advance one environment to the point where path metrics are computed."""
    random.seed(seed)
    np.random.seed(seed)
    env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=window)
    env.reset()
    for _ in range(WARMUP + HOPS):
        env.update_dynamic_qos()
    return env


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    weights = DEFAULT_WEIGHTS

    class_rows, decomp_rows = [], []

    for scenario in SCENARIOS:
        print("=== {} (t = {} after reset) ===".format(scenario, 1 + WARMUP + HOPS))
        for window in WINDOWS:
            # Accumulate per link class over seeds.
            acc = {
                "stressed": {"n": 0, "zd": 0, "zl": 0, "dcap": 0, "lcap": 0,
                             "di": [], "li": []},
                "other": {"n": 0, "zd": 0, "zl": 0, "dcap": 0, "lcap": 0,
                          "di": [], "li": []},
            }
            terms = {k: [] for k in ("util", "cong", "delay", "loss", "type")}

            for seed in SEEDS:
                env = rollout(scenario, window, seed)
                stressed = stressed_set(env)
                for u, v in EDGES:
                    e = env.normalize_edge(u, v)
                    q = env.qos[e]
                    delay_inst, loss_inst = instability_terms(env, e)
                    bucket = acc["stressed" if e in stressed else "other"]
                    bucket["n"] += 1
                    bucket["zd"] += int(delay_inst == 0.0)
                    bucket["zl"] += int(loss_inst == 0.0)
                    bucket["dcap"] += int(q["delay"] >= DELAY_CAP - 1e-9)
                    bucket["lcap"] += int(q["loss"] >= LOSS_CAP - 1e-9)
                    bucket["di"].append(delay_inst)
                    bucket["li"].append(loss_inst)

                    terms["util"].append(min(max(q["utilization"], 0.0), 1.0))
                    terms["cong"].append(1.0 if q["is_congested"] else 0.0)
                    terms["delay"].append(delay_inst)
                    terms["loss"].append(loss_inst)
                    terms["type"].append(link_type_risk(env, e, stressed))

            for name, bucket in acc.items():
                if bucket["n"] == 0:
                    continue
                row = {
                    "scenario": scenario, "window": window, "link_class": name,
                    "n_link_samples": bucket["n"],
                    "zero_delay_jitter_pct": 100.0 * bucket["zd"] / bucket["n"],
                    "zero_loss_jitter_pct": 100.0 * bucket["zl"] / bucket["n"],
                    "delay_at_cap_pct": 100.0 * bucket["dcap"] / bucket["n"],
                    "loss_at_cap_pct": 100.0 * bucket["lcap"] / bucket["n"],
                    "mean_delay_instability": float(np.mean(bucket["di"])),
                    "mean_loss_instability": float(np.mean(bucket["li"])),
                }
                class_rows.append(row)
                print("  w={:<3} {:8} zero_jitter delay={:5.1f}% loss={:5.1f}%  "
                      "at_cap delay={:5.1f}% loss={:5.1f}%".format(
                          window, name, row["zero_delay_jitter_pct"],
                          row["zero_loss_jitter_pct"], row["delay_at_cap_pct"],
                          row["loss_at_cap_pct"]))

            weighted = {k: weights[k] * float(np.mean(v)) for k, v in terms.items()}
            total = sum(weighted.values())
            temporal = weighted["delay"] + weighted["loss"]
            decomp_rows.append({
                "scenario": scenario, "window": window,
                "mean_risk": total,
                "w_util": weighted["util"], "w_cong": weighted["cong"],
                "w_delay_inst": weighted["delay"], "w_loss_inst": weighted["loss"],
                "w_type": weighted["type"],
                "temporal_abs": temporal,
                "temporal_share_pct": (100.0 * temporal / total) if total else 0.0,
                "nominal_temporal_weight_pct": 100.0 * (weights["delay"] + weights["loss"]),
            })
            print("  w={:<3} {:8} risk={:.5f}  temporal={:.5f} ({:.1f}% of risk, "
                  "nominal weight {:.0f}%)".format(
                      window, "DECOMP", total, temporal,
                      100.0 * temporal / total if total else 0.0,
                      100.0 * (weights["delay"] + weights["loss"])))
        print()

    class_path = os.path.join(OUT_DIR, "temporal_factor_health.csv")
    with open(class_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        for row in class_rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in row.items()})
    print("wrote " + class_path)

    decomp_path = os.path.join(OUT_DIR, "risk_decomposition.csv")
    with open(decomp_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(decomp_rows[0].keys()))
        writer.writeheader()
        for row in decomp_rows:
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in row.items()})
    print("wrote " + decomp_path)


if __name__ == "__main__":
    main()
