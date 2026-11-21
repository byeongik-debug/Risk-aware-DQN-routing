"""How long does each method take to decide a path?  (Reviewer R2-3)

The three methods do not spend their time the same way, so a single "path setup time"
number would hide the thing that matters:

  OSPF            one Dijkstra over the whole topology, once per path
  QoS-Dijkstra    the same, over the composite (delay, bandwidth, loss) cost
  Risk-DQN        one forward pass per hop, so its cost scales with path length

Since the methods also produce different path lengths (OSPF about 6.0 hops, the DQN
7.7), per-path and per-decision numbers answer different questions and both are
reported.

What is counted is controller work only.  env.step() is excluded: it advances the
simulated network, which in a real deployment is the network's own behaviour, not the
controller's compute.  Measuring it would charge the DQN for the simulator.

Two implementation artifacts are separated from algorithmic cost, because otherwise
they dominate:

  * network_env.ospf_path rebuilds a networkx Graph from EDGES on every call.  A real
    controller keeps the graph and updates weights, so the rebuild is timed apart from
    the shortest-path search.
  * the DQN's per-hop cost includes building a fresh FloatTensor and the Python loop
    that masks invalid actions.  The bare forward pass is timed separately so the
    report distinguishes "the network is slow" from "the glue around it is slow".

Thread count is a first-order measurement choice, not a detail.  This model is a
125->512->256->128->25 MLP served at batch 1, where the thread pool costs more than the
arithmetic: the bare forward pass measures 0.771 ms on 12 threads and 0.174 ms on 1, a
factor of 4.4.  A controller answering one path request at a time would pin a thread,
so --threads defaults to 1 and the 12-thread figure is kept only to show the artifact.
Reporting the many-thread number as "DQN inference cost" would overstate it fourfold.

Caveat to carry into the response: this is single-process CPU Python, not an SDN
controller.  The absolute milliseconds are not deployment figures and should not be
quoted as such.  Only the ratios, and the scaling-with-hops argument, survive the
change of platform -- and even the ratios are inflated on the DQN side by Python
interpreter overhead that a compiled serving path would not pay.
"""

import argparse
import csv
import os
import statistics
import sys
import time

import networkx as nx
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from network_env import NetworkEnv  # noqa: E402
from reeval_loss_conventions import load_risk_model, qos_plan_full_path  # noqa: E402
from review_stats import mean_std  # noqa: E402
from topology import DESTINATION, EDGES  # noqa: E402

OUT_DIR = os.path.join(BASE_DIR, "review_results")
OUT_CSV_TEMPLATE = os.path.join(OUT_DIR, "path_setup_time_threads{}.csv")

SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
WARMUP = 10
RISK_WINDOW = 5

# Repetitions of one timed operation on unchanged inputs; the median of these is the
# measurement, which keeps a scheduler hiccup from becoming a result.
REPEATS = 15
TORCH_WARMUP = 50


def median_time(function, repeats=REPEATS):
    """-> (median seconds, the function's last return value)."""
    samples = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), value


def build_ospf_graph(env, cost_type="bandwidth"):
    """network_env.ospf_path's graph construction, lifted out so it can be timed."""
    graph = nx.Graph()
    for u, v in EDGES:
        edge = env.normalize_edge(u, v)
        qos = env.qos[edge]
        if cost_type == "delay":
            cost = qos["delay"]
        elif cost_type == "hop":
            cost = 1
        elif cost_type == "bandwidth":
            cost = 1000.0 / max(qos["bandwidth"], 1e-6)
        else:
            cost = qos.get(cost_type, 1.0)
        graph.add_edge(u, v, weight=cost)
    return graph


def time_ospf(env):
    whole, path = median_time(lambda: env.ospf_path(cost_type="bandwidth"))
    rebuild, graph = median_time(lambda: build_ospf_graph(env))
    search, _ = median_time(
        lambda: nx.shortest_path(graph, source=env.current_node,
                                 target=DESTINATION, weight="weight"))
    hops = max(len(path) - 1, 0) if path else 0
    return {"per_path": whole, "graph_rebuild": rebuild, "search_only": search,
            "hops": hops, "decisions": 1}


def time_qos(env):
    whole, path = median_time(lambda: qos_plan_full_path(env))
    hops = max(len(path) - 1, 0) if path else 0
    return {"per_path": whole, "graph_rebuild": float("nan"),
            "search_only": float("nan"), "hops": hops, "decisions": 1}


def decide(model, state, valid_actions, device):
    """One hop of the DQN controller: exactly what eval_baseline.py does per step."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q_values = model(state_tensor)[0]
    masked = torch.full_like(q_values, -1e9)
    for action in valid_actions:
        masked[action] = q_values[action]
    return int(torch.argmax(masked).item())


def time_dqn(env, model, device):
    """Walk one path, charging only the decision work at each hop."""
    state = env.get_state()
    total = 0.0
    forward_total = 0.0
    per_hop = []
    done = False

    while not done:
        valid_actions = env.get_valid_actions()
        if not valid_actions:
            break

        elapsed, action = median_time(
            lambda: decide(model, state, valid_actions, device))
        tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        def forward():
            with torch.no_grad():
                return model(tensor)

        forward_elapsed, _ = median_time(forward)

        total += elapsed
        forward_total += forward_elapsed
        per_hop.append(elapsed)
        state, _, done, _ = env.step(action)

    hops = len(per_hop)
    return {"per_path": total, "forward_only": forward_total,
            "graph_rebuild": float("nan"), "search_only": float("nan"),
            "hops": hops, "decisions": hops,
            "reached": env.current_node == DESTINATION}


def run_scenario(scenario, model, device, trials):
    records = {"OSPF": [], "QoS-Dijkstra": [], "Risk-DQN": []}

    for trial in range(trials):
        for method in records:
            env = NetworkEnv(use_risk=True, scenario=scenario,
                             risk_window=RISK_WINDOW)
            env.reset()
            for _ in range(WARMUP):
                env.update_dynamic_qos()

            if method == "OSPF":
                records[method].append(time_ospf(env))
            elif method == "QoS-Dijkstra":
                records[method].append(time_qos(env))
            else:
                records[method].append(time_dqn(env, model, device))

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS))
    parser.add_argument("--threads", type=int, default=1,
                        help="torch intra-op threads; 1 is the fair setting for "
                             "batch-1 serving, see the module docstring")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    model = load_risk_model()
    model.eval()

    # First inferences pay for lazy allocation and thread-pool spin-up; charging those
    # to the first scenario would make it look several times slower than the rest.
    dummy = NetworkEnv(use_risk=True, scenario="normal", risk_window=RISK_WINDOW)
    dummy.reset()
    warm_state = torch.FloatTensor(dummy.get_state()).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(TORCH_WARMUP):
            model(warm_state)

    print("torch {} on {}, {} thread(s); {} repeats per measurement, {} trials".format(
        torch.__version__, device, torch.get_num_threads(), REPEATS, args.trials))
    print("state dim {}, action dim {}".format(
        warm_state.shape[1], model(warm_state).shape[1]))

    rows = []
    for scenario in args.scenarios:
        records = run_scenario(scenario, model, device, args.trials)
        print("\n=== {} (n={} paths per method) ===".format(scenario, args.trials))
        print("  {:<14}{:>9}{:>16}{:>18}{:>14}".format(
            "method", "hops", "per path (ms)", "per decision (ms)", "decisions"))

        for method, entries in records.items():
            per_path = [e["per_path"] * 1e3 for e in entries]
            hops = [e["hops"] for e in entries]
            decisions = [e["decisions"] for e in entries]
            path_mean, path_std = mean_std(per_path)
            hop_mean, _ = mean_std([float(h) for h in hops])
            per_decision = [p / d for p, d in zip(per_path, decisions) if d]
            dec_mean, dec_std = mean_std(per_decision)

            print("  {:<14}{:>9.3f}{:>10.4f}+-{:<6.4f}{:>12.4f}+-{:<6.4f}{:>12.2f}"
                  .format(method, hop_mean, path_mean, path_std,
                          dec_mean, dec_std,
                          statistics.fmean(float(d) for d in decisions)))

            rows.append({"torch_threads": args.threads, "scenario": scenario, "method": method,
                         "n_paths": len(entries), "hops_mean": hop_mean,
                         "per_path_ms_mean": path_mean, "per_path_ms_std": path_std,
                         "per_decision_ms_mean": dec_mean,
                         "per_decision_ms_std": dec_std,
                         "decisions_mean": statistics.fmean(
                             float(d) for d in decisions)})

            if method == "OSPF":
                rebuild, _ = mean_std([e["graph_rebuild"] * 1e3 for e in entries])
                search, _ = mean_std([e["search_only"] * 1e3 for e in entries])
                print("      of which graph rebuild {:.4f} ms, shortest-path search "
                      "{:.4f} ms ({:.0f}% rebuild)".format(
                          rebuild, search, rebuild / (rebuild + search) * 100))
                rows.append({"torch_threads": args.threads, "scenario": scenario, "method": "OSPF graph rebuild",
                             "n_paths": len(entries), "hops_mean": hop_mean,
                             "per_path_ms_mean": rebuild, "per_path_ms_std": 0.0,
                             "per_decision_ms_mean": rebuild,
                             "per_decision_ms_std": 0.0, "decisions_mean": 1.0})
                rows.append({"torch_threads": args.threads, "scenario": scenario, "method": "OSPF search only",
                             "n_paths": len(entries), "hops_mean": hop_mean,
                             "per_path_ms_mean": search, "per_path_ms_std": 0.0,
                             "per_decision_ms_mean": search,
                             "per_decision_ms_std": 0.0, "decisions_mean": 1.0})

            if method == "Risk-DQN":
                forward, _ = mean_std([e["forward_only"] * 1e3 for e in entries])
                reached = sum(1 for e in entries if e.get("reached"))
                print("      of which bare forward passes {:.4f} ms ({:.0f}% of the "
                      "decision cost); reached destination {}/{}".format(
                          forward, forward / path_mean * 100, reached, len(entries)))
                rows.append({"torch_threads": args.threads, "scenario": scenario, "method": "Risk-DQN forward only",
                             "n_paths": len(entries), "hops_mean": hop_mean,
                             "per_path_ms_mean": forward, "per_path_ms_std": 0.0,
                             "per_decision_ms_mean": forward / max(hop_mean, 1.0),
                             "per_decision_ms_std": 0.0, "decisions_mean": hop_mean})

        ospf_path_ms = mean_std([e["per_path"] * 1e3
                                 for e in records["OSPF"]])[0]
        dqn_path_ms = mean_std([e["per_path"] * 1e3
                                for e in records["Risk-DQN"]])[0]
        print("  Risk-DQN / OSPF per-path cost: {:.1f}x".format(
            dqn_path_ms / ospf_path_ms))

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = OUT_CSV_TEMPLATE.format(args.threads)
    fields = ["torch_threads", "scenario", "method", "n_paths", "hops_mean",
              "per_path_ms_mean",
              "per_path_ms_std", "per_decision_ms_mean", "per_decision_ms_std",
              "decisions_mean"]
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in ("hops_mean", "per_path_ms_mean", "per_path_ms_std",
                        "per_decision_ms_mean", "per_decision_ms_std",
                        "decisions_mean"):
                out[key] = "{:.6f}".format(row[key])
            writer.writerow(out)

    print("\nwrote " + out_csv)
    print("\nRead these as ratios on one machine, not as deployment latencies.  The")
    print("DQN pays per hop and the planners pay once, so the gap widens with path")
    print("length -- that scaling, not the millisecond count, is the reviewer's point.")


if __name__ == "__main__":
    main()
