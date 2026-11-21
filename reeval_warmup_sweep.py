"""Sweep the evaluation warm-up length to show when the temporal risk signal dies.

diagnose_temporal_factors.py establishes that a stressed link in the congestion and
dynamic scenarios draws strictly positive delay and loss noise every step, so it ramps
monotonically into the hard clip at 120 ms / 0.35.  Once pinned there its standard
deviation is identically zero and both temporal risk factors vanish.

The ramp needs roughly ten steps to reach the ceiling, and that is exactly the length of
the warm-up that train_window_dqn.py's evaluate_model and eval_window_sweep.py perform
before scoring.  Training itself uses no warm-up: train() steps straight out of reset().
So the policies learn on a regime where delay jitter separates stressed from safe links
by a factor of about sixteen, and are then scored on a regime where that ratio inverts.

This script holds the trained models fixed -- no retraining -- and varies only the
warm-up, which is the one knob that decides whether the temporal factors carry
information at scoring time.  If the window parameter matters at all, it should separate
the policies at short warm-ups and stop separating them as the links saturate.

Default cost is about half an hour on CPU; --runs-per-window and --episodes trade
precision for time.
"""

import argparse
import csv
import glob
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from network_env import CONGESTION_LINKS, UNSTABLE_LINKS, NetworkEnv  # noqa: E402
from topology import DESTINATION, EDGES  # noqa: E402

MODEL_DIR = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "models")
OUT_DIR = os.path.join(BASE_DIR, "review_results")

WARMUPS = (0, 2, 5, 10)
DELAY_NORM = 20.0


class QNetwork(nn.Module):
    """Same architecture as eval_window_sweep.py, required to load the checkpoints."""

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


def stressed_set(env):
    if env.scenario == "congestion":
        return {env.normalize_edge(*x) for x in CONGESTION_LINKS}
    if env.scenario == "unstable":
        return {env.normalize_edge(*x) for x in UNSTABLE_LINKS}
    if env.scenario == "dynamic":
        return env.get_dynamic_phase_sets()[0]
    return set()


def discrimination_ratio(env):
    """Mean delay instability on stressed links divided by the same on safe links.

    Above 1 the temporal factor flags danger; below 1 it points away from it.
    """
    stressed = stressed_set(env)
    if not stressed:
        return float("nan")
    hot, cold = [], []
    for u, v in EDGES:
        e = env.normalize_edge(u, v)
        history = list(env.delay_history[e])
        value = min(np.std(history) / DELAY_NORM, 1.0) if len(history) > 1 else 0.0
        (hot if e in stressed else cold).append(value)
    safe_mean = float(np.mean(cold))
    return float(np.mean(hot)) / safe_mean if safe_mean > 0 else float("nan")


def evaluate(model, window, scenario, warmup, seed, episodes):
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, risks, hops, ratios = [], [], [], [], []
    success = 0
    device = next(model.parameters()).device

    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=window)
        env.reset()
        for _ in range(warmup):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked_q = torch.full_like(q_values, -1e9)
            for a in valid_actions:
                masked_q[a] = q_values[a]
            state, _, done, _ = env.step(int(torch.argmax(masked_q).item()))

        if env.current_node == DESTINATION:
            success += 1
            metrics = env.calculate_path_metrics(env.visited)
            delays.append(metrics["total_delay"])
            losses.append(metrics["total_loss"])
            risks.append(metrics["total_risk"])
            hops.append(metrics["hop_count"])
            ratio = discrimination_ratio(env)
            if ratio == ratio:
                ratios.append(ratio)

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "success_rate": success / episodes * 100,
        "delay": mean(delays),
        "loss": mean(losses),
        "risk": mean(risks),
        "hop": mean(hops),
        "discrimination_ratio": mean(ratios),
    }


def discover_models(runs_per_window):
    """[(window, run_id, path)], keeping at most runs_per_window checkpoints each."""
    by_window = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "window*_run*.pth"))):
        match = re.search(r"window(\d+)_run(\d+)\.pth$", os.path.basename(path))
        if match:
            by_window[int(match.group(1))].append((int(match.group(2)), path))
    selected = []
    for window in sorted(by_window):
        for run_id, path in sorted(by_window[window])[:runs_per_window]:
            selected.append((window, run_id, path))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="dynamic",
                        choices=["normal", "congestion", "unstable", "dynamic"])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--runs-per-window", type=int, default=5,
                        help="checkpoints per window; window=3 has 10, the rest fewer")
    parser.add_argument("--warmups", type=int, nargs="+", default=list(WARMUPS))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = discover_models(args.runs_per_window)
    if not models:
        raise SystemExit("no window models found in " + MODEL_DIR)

    out_csv = os.path.join(OUT_DIR, "warmup_sweep_{}.csv".format(args.scenario))
    total = len(models) * len(args.warmups)
    print("{} models x {} warm-ups = {} evaluations x {} episodes ({})".format(
        len(models), len(args.warmups), total, args.episodes, args.scenario), flush=True)

    fields = ["scenario", "warmup", "window", "run_id", "success_rate",
              "delay", "loss", "risk", "hop", "discrimination_ratio"]
    started = time.time()
    index = 0

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for warmup in args.warmups:
            for window, run_id, path in models:
                index += 1
                probe = NetworkEnv(use_risk=True, scenario=args.scenario, risk_window=window)
                model = QNetwork(probe.state_size, probe.action_size).to(device)
                try:
                    model.load_state_dict(
                        torch.load(path, map_location=device, weights_only=False)
                    )
                except RuntimeError as exc:
                    print("  [skip] {}: {}".format(os.path.basename(path), exc), flush=True)
                    continue
                model.eval()

                result = evaluate(model, window, args.scenario, warmup,
                                  seed=run_id, episodes=args.episodes)
                row = {"scenario": args.scenario, "warmup": warmup,
                       "window": window, "run_id": run_id}
                row.update({k: "{:.6f}".format(v) for k, v in result.items()})
                writer.writerow(row)
                fh.flush()

                print("  [{:3}/{}] warmup={:<3} w={:<3} run{:02d} delay={:7.2f} "
                      "loss={:.5f} risk={:.5f} stressed/safe jitter={:6.2f}x "
                      "({:.1f} min)".format(
                          index, total, warmup, window, run_id, result["delay"],
                          result["loss"], result["risk"], result["discrimination_ratio"],
                          (time.time() - started) / 60), flush=True)

    print("\nwrote {} in {:.1f} min".format(out_csv, (time.time() - started) / 60),
          flush=True)


if __name__ == "__main__":
    main()
