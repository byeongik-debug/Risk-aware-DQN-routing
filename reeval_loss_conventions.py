"""Recompute the headline loss comparison under three conventions, exactly.

analyze_loss_metric.py shows that the published loss metric is a per-hop mean, so it
falls as paths lengthen, and reconstructs what the main table would say under end-to-end
conventions.  That reconstruction works backwards from published aggregates, so it is
only an estimate: the mean of a nonlinear function of the per-link rates is not that
function of their mean.  This script replays the evaluations and computes all three
conventions from the per-link rates themselves, per episode.

The three conventions, for a path whose links drop packets at rates p_1 .. p_h:

  published   mean(p_i)            a per-link average -- what the paper reports as loss
  path_sum    sum(p_i)             the quantity delay is already reported as
  compounded  1 - prod(1 - p_i)    the chance a packet is dropped somewhere on the path

The compounded form is the one that stays a probability.  Per-link rates here reach the
0.35 clip, so a six-hop sum can exceed 1.

Replays, matching eval_qos.py and eval_baseline.py step for step (same seeds, same 200
episodes, same warm-up placement) so that the published column reproduces the committed
reference tables.  That reproduction is the script's own check that the replay is
faithful; it is asserted, not merely printed.

  OSPF       warm up 10 steps, then plan on bandwidth cost        (eval_qos.py:74)
  QoS        plan at t=0 on composite QoS cost, then warm up and
             execute the stale plan without replanning            (eval_qos.py:130)
  Risk-DQN   warm up 10 steps, then act greedily per hop          (eval_baseline.py:44)

OSPF and QoS need no checkpoint -- despite the "QoS-DQN" label in exp1_reference.csv,
eval_qos.py's QoS row is a one-shot Dijkstra planner with no network in it -- so the
default run costs a couple of minutes and no torch.  Add --with-dqn for the Risk-DQN.
"""

import argparse
import csv
import heapq
import os
import random
import sys
import time
from collections import defaultdict

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from network_env import NetworkEnv  # noqa: E402
from review_stats import holm_bonferroni, mean_std, welch  # noqa: E402
from topology import DESTINATION, EDGES  # noqa: E402

RESULTS_DIR = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "results")
OUT_DIR = os.path.join(BASE_DIR, "review_results")
RISK_MODEL = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "risk_only_model_final.pth")

SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
SEEDS = (0, 1, 2, 3, 4)
EVAL_EPISODES = 200
WARMUP = 10
RISK_WINDOW = 5
CONVENTIONS = ("published", "path_sum", "compounded")

# Adjacency for the QoS planner, as eval_qos.py builds it.
ADJ = defaultdict(list)
for _u, _v in EDGES:
    ADJ[_u].append(_v)
    ADJ[_v].append(_u)


def loss_conventions(env, path):
    """The three conventions plus delay and hop count, from the per-link rates."""
    rates = []
    delay = 0.0
    for u, v in zip(path[:-1], path[1:]):
        e = env.normalize_edge(u, v)
        if e not in env.qos:
            continue
        rates.append(env.qos[e]["loss"])
        delay += env.qos[e]["delay"]

    hop = max(len(path) - 1, 1)
    survival = 1.0
    for rate in rates:
        survival *= (1.0 - rate)

    return {
        "hop": len(path) - 1,
        "delay": delay,
        "published": sum(rates) / hop,
        "path_sum": sum(rates),
        "compounded": 1.0 - survival,
    }


def qos_plan_full_path(env):
    """eval_qos.py:31 verbatim in behaviour: Dijkstra on a composite QoS cost."""
    source = env.current_node
    dist = {source: 0.0}
    prev = {source: None}
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for nb in ADJ[node]:
            if nb == source:
                continue
            e = env.normalize_edge(node, nb)
            q = env.qos.get(e, {"delay": 120.0, "bandwidth": 0.0, "loss": 0.35})
            d_n = float(np.clip(q["delay"] / 120.0, 0.0, 1.0))
            b_n = 1.0 - float(np.clip(q["bandwidth"] / 1000.0, 0.0, 1.0))
            l_n = float(np.clip(q["loss"] / 0.35, 0.0, 1.0))
            new_cost = cost + (d_n + b_n + l_n) / 3.0
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    if DESTINATION not in prev:
        return []
    path, curr = [], DESTINATION
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return path


def seed_mean(per_episode):
    """Mean over episodes of one seed, matching how the reference tables aggregate."""
    keys = ("hop", "delay") + CONVENTIONS
    if not per_episode:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([e[k] for e in per_episode])) for k in keys}


def run_ospf_seed(scenario, seed, episodes):
    np.random.seed(seed)
    random.seed(seed)
    episodes_out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        path = env.ospf_path(cost_type="bandwidth")
        if not path or path[-1] != DESTINATION:
            continue
        episodes_out.append(loss_conventions(env, path))
    return seed_mean(episodes_out), len(episodes_out) / episodes * 100


def run_qos_seed(scenario, seed, episodes):
    np.random.seed(seed)
    random.seed(seed)
    episodes_out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        planned = qos_plan_full_path(env)
        if not planned or planned[-1] != DESTINATION:
            continue
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        for next_node in planned[1:]:
            _, _, done, _ = env.step(next_node)
            if done:
                break
        if env.current_node == DESTINATION:
            episodes_out.append(loss_conventions(env, env.visited))
    return seed_mean(episodes_out), len(episodes_out) / episodes * 100


def run_dqn_seed(model, scenario, seed, episodes):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    episodes_out = []

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
        state = env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked = torch.full_like(q_values, -1e9)
            for a in valid:
                masked[a] = q_values[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node == DESTINATION:
            episodes_out.append(loss_conventions(env, env.visited))

    return seed_mean(episodes_out), len(episodes_out) / episodes * 100


def load_risk_model():
    import torch
    import torch.nn as nn

    class QNetwork(nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_size, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, action_size),
            )

        def forward(self, x):
            return self.net(x)

    if not os.path.exists(RISK_MODEL):
        raise SystemExit("Risk-DQN checkpoint not found: " + RISK_MODEL)
    device = torch.device("cpu")
    probe = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    model = QNetwork(probe.state_size, probe.action_size).to(device)
    model.load_state_dict(torch.load(RISK_MODEL, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


def published_reference():
    """{(method, scenario): published loss} from the committed tables, for the check."""
    reference = {}
    path = os.path.join(RESULTS_DIR, "exp1_reference.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                name = "QoS" if row["model"] == "QoS-DQN" else row["model"]
                reference[(name, row["scenario"])] = float(row["loss"])
    path = os.path.join(RESULTS_DIR, "baseline_reference.csv")
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                reference[("Risk-DQN", row["scenario"])] = float(row["loss"])
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        choices=list(SCENARIOS))
    parser.add_argument("--with-dqn", action="store_true",
                        help="also replay risk_only_model_final.pth (needs torch)")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="relative gap allowed against the committed tables")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    methods = [("OSPF", run_ospf_seed), ("QoS", run_qos_seed)]
    if args.with_dqn:
        model = load_risk_model()
        methods.append(
            ("Risk-DQN", lambda sc, sd, ep: run_dqn_seed(model, sc, sd, ep)))

    reference = published_reference()
    started = time.time()
    # per_seed[(method, scenario)][key] -> one value per seed
    per_seed = {}
    success = {}

    for scenario in args.scenarios:
        for name, runner in methods:
            collected = defaultdict(list)
            rates = []
            for seed in SEEDS:
                means, rate = runner(scenario, seed, args.episodes)
                for key, value in means.items():
                    collected[key].append(value)
                rates.append(rate)
            per_seed[(name, scenario)] = collected
            success[(name, scenario)] = float(np.mean(rates))
            print("  {:<9} {:<11} hop={:5.3f} delay={:7.2f} published={:.5f} "
                  "path_sum={:.5f} compounded={:.5f}  ({:.1f} min)".format(
                      name, scenario,
                      float(np.mean(collected["hop"])),
                      float(np.mean(collected["delay"])),
                      float(np.mean(collected["published"])),
                      float(np.mean(collected["path_sum"])),
                      float(np.mean(collected["compounded"])),
                      (time.time() - started) / 60), flush=True)

    # The replay must reproduce the committed tables under the published convention.
    print("\n=== replay check against the committed reference tables ===")
    failures = []
    for (name, scenario), collected in sorted(per_seed.items()):
        expected = reference.get((name, scenario))
        if expected is None:
            continue
        got = float(np.mean(collected["published"]))
        gap = abs(got - expected) / expected if expected else float("inf")
        status = "ok" if gap <= args.tolerance else "MISMATCH"
        if gap > args.tolerance:
            failures.append((name, scenario, expected, got, gap))
        print("  {:<9} {:<11} committed={:.6f}  replay={:.6f}  gap={:+.2%}  {}".format(
            name, scenario, expected, got, (got - expected) / expected, status))

    rows = []
    for scenario in args.scenarios:
        for convention in CONVENTIONS:
            for name, _ in methods:
                values = per_seed[(name, scenario)][convention]
                mean, std = mean_std(values)
                rows.append({
                    "scenario": scenario, "convention": convention, "method": name,
                    "n_seeds": len(values), "mean": mean, "std": std,
                    "hop": float(np.mean(per_seed[(name, scenario)]["hop"])),
                    "delay": float(np.mean(per_seed[(name, scenario)]["delay"])),
                    "success_rate": success[(name, scenario)],
                })

    # OSPF is the reference the paper improves on, so every other method is tested
    # against it under each convention.
    tests = []
    for scenario in args.scenarios:
        for convention in CONVENTIONS:
            ospf = per_seed[("OSPF", scenario)][convention]
            for name, _ in methods:
                if name == "OSPF":
                    continue
                values = per_seed[(name, scenario)][convention]
                diff, t, df, p = welch(values, ospf)
                ospf_mean = mean_std(ospf)[0]
                tests.append({
                    "scenario": scenario, "convention": convention,
                    "method": name, "vs": "OSPF",
                    "diff": diff, "t": t, "df": df, "p": p,
                    "improvement_pct": -diff / ospf_mean * 100 if ospf_mean else
                    float("nan"),
                })
    for test, adjusted in zip(tests, holm_bonferroni([x["p"] for x in tests])):
        test["p_holm"] = adjusted

    out_csv = os.path.join(OUT_DIR, "loss_conventions_exact.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "scenario", "convention", "method", "n_seeds", "mean", "std",
            "hop", "delay", "success_rate"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("{:.6f}".format(v) if isinstance(v, float) else v)
                             for k, v in row.items()})

    tests_csv = os.path.join(OUT_DIR, "loss_conventions_tests.csv")
    with open(tests_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "scenario", "convention", "method", "vs", "improvement_pct",
            "diff", "t", "df", "p", "p_holm"])
        writer.writeheader()
        for test in tests:
            writer.writerow({k: ("{:.6f}".format(v) if isinstance(v, float) else v)
                             for k, v in test.items()})

    print("\n=== loss improvement over OSPF, by convention (exact, n=5 seeds) ===")
    print("{:<11} {:<9} {:>13} {:>11} {:>9} {:>9}".format(
        "scenario", "method", "convention", "improvement", "p", "p_holm"))
    for test in tests:
        print("{:<11} {:<9} {:>13} {:>10.1f}% {:>9.4f} {:>9.4f}{}".format(
            test["scenario"], test["method"], test["convention"],
            test["improvement_pct"], test["p"], test["p_holm"],
            "  *" if test["p_holm"] == test["p_holm"] and test["p_holm"] < 0.05 else ""))

    print("\nwrote {}\nwrote {}".format(out_csv, tests_csv))
    print("elapsed {:.1f} min".format((time.time() - started) / 60))

    if failures:
        print("\nthe replay did not reproduce the committed tables:")
        for name, scenario, expected, got, gap in failures:
            print("  {} {}: committed {:.6f}, replay {:.6f} ({:+.2%})".format(
                name, scenario, expected, got, (got - expected) / expected))
        raise SystemExit(
            "published convention must match the committed tables before the other two"
            " conventions mean anything")


if __name__ == "__main__":
    main()
