"""Re-evaluate every trained temporal-window model under one window-independent yardstick.

No retraining: all 29 window models already exist in SDN_DQN_new_env/models.  Each is
replayed in FixedYardstickEnv, which feeds the policy the state distribution it was
trained on (its own risk_window) while also scoring each path with risk measured on a
fixed reference window.  "risk_own" reproduces the published number; "risk_ref" is the
comparable one.  At window=5 the two coincide by construction, which doubles as a
regression check on the yardstick.

The evaluation protocol mirrors eval_window_sweep.py: a fresh environment per episode,
ten warm-up QoS updates, then greedy argmax over valid actions.  The sample unit is a
trained model, matching the per-run result CSVs this re-analysis is compared against,
so each model is evaluated once with a seed derived from its run id.
"""

import argparse
import csv
import glob
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

from fixed_yardstick_env import REFERENCE_WINDOW, FixedYardstickEnv  # noqa: E402
from topology import DESTINATION  # noqa: E402

MODEL_DIR = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "models")
OUT_DIR = os.path.join(BASE_DIR, "review_results")
OUT_CSV = os.path.join(OUT_DIR, "window_fixed_yardstick_runs.csv")

EVAL_EPISODES = 200
ALL_SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
# Windows 1 and 5 were evaluated in every scenario; 3, 7 and 10 only in dynamic.
# Matching that layout keeps this re-analysis comparable to the existing run CSVs.
SCENARIOS_BY_WINDOW = {1: ALL_SCENARIOS, 5: ALL_SCENARIOS}
DEFAULT_SCENARIOS = ("dynamic",)


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


def discover_models():
    """[(window, run_id, path)] for every window{W}_run{NN}.pth checkpoint."""
    found = []
    for path in sorted(glob.glob(os.path.join(MODEL_DIR, "window*_run*.pth"))):
        match = re.search(r"window(\d+)_run(\d+)\.pth$", os.path.basename(path))
        if match:
            found.append((int(match.group(1)), int(match.group(2)), path))
    return sorted(found)


def evaluate(model, window, scenario, seed, episodes=EVAL_EPISODES):
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, risks_own, risks_ref, hops = [], [], [], [], []
    unstable_usages, congested_usages = [], []
    success = 0
    device = next(model.parameters()).device

    for _ in range(episodes):
        env = FixedYardstickEnv(
            use_risk=True, scenario=scenario,
            risk_window=window, ref_window=REFERENCE_WINDOW,
        )
        env.reset()
        for _ in range(10):
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
            risks_own.append(metrics["total_risk"])
            risks_ref.append(metrics["total_risk_ref"])
            hops.append(metrics["hop_count"])
            hop_count = max(metrics["hop_count"], 1)
            unstable_usages.append(metrics["unstable_count"] / hop_count * 100)
            congested_usages.append(metrics["congested_count"] / hop_count * 100)

    def mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "success_rate": success / episodes * 100,
        "delay": mean(delays),
        "loss": mean(losses),
        "risk_own": mean(risks_own),
        "risk_ref": mean(risks_ref),
        "hop": mean(hops),
        "unstable_usage": mean(unstable_usages),
        "congested_usage": mean(congested_usages),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios", nargs="+", choices=ALL_SCENARIOS, default=None,
        help="evaluate EVERY window in these scenarios instead of the published "
             "layout (windows 1 and 5 everywhere, 3/7/10 in dynamic only). "
             "Omit to reproduce the committed CSV exactly.")
    parser.add_argument(
        "--out", default=None,
        help="output CSV path; defaults to " + os.path.basename(OUT_CSV))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.out:
        out_csv = OUT_CSV
    elif os.path.dirname(args.out):
        out_csv = args.out
    else:
        out_csv = os.path.join(OUT_DIR, args.out)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = discover_models()
    if not models:
        raise SystemExit("no window models found in " + MODEL_DIR)

    if args.scenarios:
        jobs = [(w, run_id, path, scenario)
                for (w, run_id, path) in models
                for scenario in args.scenarios]
    else:
        jobs = [
            (w, run_id, path, scenario)
            for (w, run_id, path) in models
            for scenario in SCENARIOS_BY_WINDOW.get(w, DEFAULT_SCENARIOS)
        ]
    print(
        "{} models -> {} evaluations x {} episodes (reference window = {})".format(
            len(models), len(jobs), EVAL_EPISODES, REFERENCE_WINDOW
        ),
        flush=True,
    )

    fields = ["window", "run_id", "scenario", "success_rate", "delay", "loss",
              "risk_own", "risk_ref", "hop", "unstable_usage", "congested_usage"]
    started = time.time()

    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

        for index, (window, run_id, path, scenario) in enumerate(jobs, start=1):
            probe = FixedYardstickEnv(use_risk=True, scenario=scenario, risk_window=window)
            model = QNetwork(probe.state_size, probe.action_size).to(device)
            try:
                model.load_state_dict(
                    torch.load(path, map_location=device, weights_only=False)
                )
            except RuntimeError as exc:
                print("  [skip] {}: {}".format(os.path.basename(path), exc), flush=True)
                continue
            model.eval()

            result = evaluate(model, window, scenario, seed=run_id)
            row = {"window": window, "run_id": run_id, "scenario": scenario}
            row.update({k: "{:.6f}".format(v) for k, v in result.items()})
            writer.writerow(row)
            fh.flush()

            gap = result["risk_own"] - result["risk_ref"]
            elapsed = time.time() - started
            print(
                "  [{:2}/{}] w={:<3} run{:02d} {:11} delay={:7.2f} loss={:.5f} "
                "risk_own={:.5f} risk_ref={:.5f} gap={:+.5f}  ({:.1f} min)".format(
                    index, len(jobs), window, run_id, scenario,
                    result["delay"], result["loss"],
                    result["risk_own"], result["risk_ref"], gap, elapsed / 60,
                ),
                flush=True,
            )

    print("\nwrote {} in {:.1f} min".format(out_csv, (time.time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()
