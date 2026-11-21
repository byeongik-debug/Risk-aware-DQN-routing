# -*- coding: utf-8 -*-
"""Table 11's Risky% column: how many dangerous links CAN a path avoid?

The dynamic scenario's danger set is a union of whole named paths (PATH_A..D in
network_env.py), and every one of them starts at SOURCE and ends at DESTINATION.
Both of those nodes have degree 2, so some incident edge is dangerous in every
phase and NO path can avoid it. That gives a per-phase FLOOR on the risky-link
count, solvable as a 0/1-weight shortest path -- it is a property of the
topology, not of any policy.

This script measures the floor, then measures how close the four Dijkstra cost
functions get to it. ospf_path() is deterministic given the current qos state,
so no model is needed here; the DQN side comes from eval_ablation_risky.py.

Reported Risky% is the same quantity run_policy.py:98-99 reports:
    unstable_count / hop_count * 100
i.e. hits divided by hops -- so when the numerator sits on the floor, the whole
column is just 1/hop. That is what this script is for.

Usage:  python -X utf8 eval_risky_floor.py [episodes]
Output: review_results/table11_risky_floor.csv
"""
import csv
import os
import random
import sys
from collections import Counter

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
OUT = os.path.join(HERE, "review_results", "table11_risky_floor.csv")

# review_response/risk_calculator.py is the friend's pre-ablation copy; keep it
# from shadowing the original the environment actually uses.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import network_env as NE   # noqa: E402

assert os.path.abspath(NE.__file__).startswith(ORIG), NE.__file__

SEED = 20261114
METHODS = ["bandwidth", "delay", "hop", "risk"]
MINREF = "(floor & shortest)"
LABEL = {"bandwidth": "bandwidth (the paper's OSPF)", "delay": "delay",
         "hop": "hop", "risk": "risk", MINREF: MINREF}


def norm(u, v):
    return tuple(sorted((u, v)))


def floor_for(phase):
    """0/1 edge weights: the fewest dangerous links any s-t path must use."""
    danger = set(norm(*e) for e in NE.DYNAMIC_PHASES[phase]["danger"])
    h = nx.Graph()
    for u, v in NE.EDGES:
        h.add_edge(u, v, w=1.0 if norm(u, v) in danger else 0.0)
    best = nx.shortest_path(h, NE.SOURCE, NE.DESTINATION, weight="w")
    return sum(1 for a, b in zip(best[:-1], best[1:]) if norm(a, b) in danger)


def blank():
    return {"cnt": [], "pct": [], "hop": [], "exc": [], "floor": [],
            "delay": [], "loss": []}


def record(acc, mt, floor):
    hop = max(int(mt["hop_count"]), 1)
    # In dynamic, CONGESTION_LINKS == UNSTABLE_LINKS, so the two counts refer to
    # the same active set; max() is defensive, not a combination.
    cnt = max(int(mt["unstable_count"]), int(mt["congested_count"]))
    acc["cnt"].append(float(cnt))
    acc["pct"].append(cnt / hop * 100.0)
    acc["hop"].append(float(hop))
    acc["floor"].append(float(floor))
    acc["exc"].append(float(cnt - floor))
    acc["delay"].append(float(mt["total_delay"]))
    acc["loss"].append(float(mt["total_loss"]))
    return cnt


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    flo = dict((p, floor_for(p)) for p in NE.DYNAMIC_PHASES)
    acc = dict((m, blank()) for m in METHODS + [MINREF])
    phases = Counter()
    degs = {}

    for i in range(n):
        # per-episode seeding: rollout length decides how many draws are
        # consumed, so one seed before the loop desynchronizes the methods
        random.seed(SEED + i)
        np.random.seed(SEED + i)
        env = NE.NetworkEnv(use_risk=True, scenario="dynamic",
                            risk_window=NE.RISK_WINDOW)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        ph = env.dynamic_phase
        phases[ph] += 1
        danger = set(norm(*e) for e in NE.DYNAMIC_PHASES[ph]["danger"])

        for m in METHODS:
            record(acc[m], env.calculate_path_metrics(env.ospf_path(cost_type=m)),
                   flo[ph])

        # among floor-attaining paths, the shortest one (1000:1 lexicographic)
        h2 = nx.Graph()
        for u, v in NE.EDGES:
            h2.add_edge(u, v, w=(1000.0 if norm(u, v) in danger else 0.0) + 1.0)
        pm = nx.shortest_path(h2, NE.SOURCE, NE.DESTINATION, weight="w")
        got = record(acc[MINREF], env.calculate_path_metrics(pm), flo[ph])
        # the reference MUST attain the floor, or floor_for() is wrong
        assert got == flo[ph], (i, ph, got, flo[ph])

    g = nx.Graph()
    g.add_edges_from(NE.EDGES)
    degs = {NE.SOURCE: g.degree(NE.SOURCE),
            NE.DESTINATION: g.degree(NE.DESTINATION)}

    wmean = sum(flo[p] * c for p, c in phases.items()) / float(n)
    print("SOURCE %d degree %d, DESTINATION %d degree %d"
          % (NE.SOURCE, degs[NE.SOURCE], NE.DESTINATION, degs[NE.DESTINATION]))
    print("per-phase floor: %s" % flo)
    print("phase mix over %d episodes: %s  (weighted floor mean %.4f)"
          % (n, dict(phases), wmean))
    print()
    hdr = ("%-28s | %6s | %7s | %6s | %7s | %8s | %9s | %9s | %7s"
           % ("cost rule", "hop", "hits", "floor", "excess", "Risky%",
              "delay(sum)", "loss/hop", "at floor"))
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for m in METHODS + [MINREF]:
        a = acc[m]
        ok = 100.0 * sum(1 for e in a["exc"] if e == 0) / n
        print("%-28s | %6.3f | %7.4f | %6.4f | %+7.4f | %7.2f%% | %9.2f |"
              " %9.5f | %6.1f%%"
              % (LABEL[m], np.mean(a["hop"]), np.mean(a["cnt"]),
                 np.mean(a["floor"]), np.mean(a["exc"]), np.mean(a["pct"]),
                 np.mean(a["delay"]), np.mean(a["loss"]), ok))
        rows.append([m, n, "%.4f" % np.mean(a["hop"]),
                     "%.4f" % np.mean(a["cnt"]), "%.4f" % np.mean(a["floor"]),
                     "%.4f" % np.mean(a["exc"]), "%.4f" % np.mean(a["pct"]),
                     "%.4f" % np.std(a["pct"], ddof=1),
                     "%.4f" % np.mean(a["delay"]), "%.6f" % np.mean(a["loss"]),
                     "%.2f" % ok,
                     " ".join("%d:%d" % (int(k), v) for k, v
                              in sorted(Counter(a["exc"]).items()))])
    print()
    print("excess-hit distribution (0 = on the floor):")
    for m in METHODS + [MINREF]:
        print("  %-28s %s" % (LABEL[m], dict(Counter(acc[m]["exc"]))))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cost_rule", "episodes", "hop", "risky_hits", "floor",
                    "excess", "risky_pct", "risky_pct_sd", "delay_sum",
                    "loss_per_hop", "at_floor_pct", "excess_hist"])
        w.writerows(rows)
    print("\nwrote %s" % OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
