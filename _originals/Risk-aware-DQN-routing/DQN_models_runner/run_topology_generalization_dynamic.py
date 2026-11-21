"""Dynamic-only unseen-link-configuration evaluation.

Run order:
1) python generate_unseen_topologies_dynamic.py
2) python run_topology_generalization_dynamic.py

Evaluation design:
- 25 nodes / 50 links fixed
- 20%, 40%, 60% rewiring ratios
- 10 unseen topologies per ratio
- Dynamic scenario only
- OSPF vs trained Risk-DQN
- No additional training
- 100 evaluation episodes per topology and method

The final summary is calculated across topologies, not across all episodes.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from network_env_generalization import GeneralizationNetworkEnv


MODEL_PATH = Path("risk_only_model_final.pth")
TOPOLOGY_DIR = Path("topologies_dynamic")
RESULT_DIR = Path("results")
SCENARIO = "dynamic"

EPISODES_PER_TOPOLOGY = 100
BASE_EVAL_SEED = 70000
WARMUP_STEPS = 10


class QNetwork(nn.Module):
    """Q-network architecture identical to the trained model."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
    ) -> None:
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

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)


def read_rewire_ratio(
    topology_path: Path,
) -> float:
    payload = json.loads(
        topology_path.read_text(
            encoding="utf-8"
        )
    )
    return float(
        payload["rewire_ratio_requested"]
    )


def select_dqn_action(
    model: QNetwork,
    state: np.ndarray,
    valid_actions: List[int],
    device: torch.device,
) -> Optional[int]:
    if not valid_actions:
        return None

    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)[0]

    valid_tensor = torch.as_tensor(
        valid_actions,
        dtype=torch.long,
        device=device,
    )

    masked_q = torch.full_like(
        q_values,
        -1e9,
    )
    masked_q[valid_tensor] = q_values[valid_tensor]

    return int(
        torch.argmax(masked_q).item()
    )


def warmup(
    env: GeneralizationNetworkEnv,
    steps: int = WARMUP_STEPS,
) -> None:
    for _ in range(steps):
        env.update_dynamic_qos()


def create_metric_store() -> Dict[str, List[float]]:
    return {
        "delay": [],
        "loss": [],
        "risk": [],
        "hop": [],
        "risky_link_usage": [],
    }


def append_metrics(
    store: Dict[str, List[float]],
    metrics: Dict[str, object],
) -> None:
    hop_count = max(
        int(metrics["hop_count"]),
        1,
    )

    # In the Dynamic scenario, unstable_count and congested_count
    # refer to the same active dangerous-link set.
    risky_count = max(
        int(metrics["unstable_count"]),
        int(metrics["congested_count"]),
    )

    store["delay"].append(
        float(metrics["total_delay"])
    )
    store["loss"].append(
        float(metrics["total_loss"])
    )
    store["risk"].append(
        float(metrics["total_risk"])
    )
    store["hop"].append(
        float(metrics["hop_count"])
    )
    store["risky_link_usage"].append(
        risky_count / hop_count * 100.0
    )


def summarize_topology(
    method: str,
    topology_path: Path,
    successes: int,
    metric_store: Dict[str, List[float]],
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "topology": topology_path.stem,
        "rewire_ratio": read_rewire_ratio(
            topology_path
        ),
        "scenario": SCENARIO,
        "method": method,
        "episodes": EPISODES_PER_TOPOLOGY,
        "success_rate": (
            successes
            / EPISODES_PER_TOPOLOGY
            * 100.0
        ),
    }

    for key, values in metric_store.items():
        result[key] = (
            float(np.mean(values))
            if values
            else float("nan")
        )
        result[f"{key}_episode_std"] = (
            float(np.std(values, ddof=1))
            if len(values) > 1
            else 0.0
        )

    return result


def evaluate_ospf(
    topology_path: Path,
) -> Dict[str, object]:
    metric_store = create_metric_store()
    successes = 0

    for episode in range(
        EPISODES_PER_TOPOLOGY
    ):
        seed = BASE_EVAL_SEED + episode

        env = GeneralizationNetworkEnv(
            topology_path,
            use_risk=False,
            scenario=SCENARIO,
            seed=seed,
        )

        env.reset()
        warmup(env)

        path = env.ospf_path(
            cost_type="bandwidth"
        )
        metrics = env.calculate_path_metrics(
            path
        )

        if (
            path
            and path[-1] == env.destination
        ):
            successes += 1
            append_metrics(
                metric_store,
                metrics,
            )

    return summarize_topology(
        method="OSPF",
        topology_path=topology_path,
        successes=successes,
        metric_store=metric_store,
    )


def evaluate_dqn(
    topology_path: Path,
    model: QNetwork,
    device: torch.device,
) -> Dict[str, object]:
    metric_store = create_metric_store()
    successes = 0

    for episode in range(
        EPISODES_PER_TOPOLOGY
    ):
        seed = BASE_EVAL_SEED + episode

        env = GeneralizationNetworkEnv(
            topology_path,
            use_risk=True,
            scenario=SCENARIO,
            seed=seed,
        )

        state = env.reset()
        warmup(env)
        state = env.get_state()
        done = False

        while not done:
            action = select_dqn_action(
                model=model,
                state=state,
                valid_actions=env.get_valid_actions(),
                device=device,
            )

            if action is None:
                break

            state, _, done, _ = env.step(
                action
            )

        metrics = env.calculate_path_metrics(
            env.visited
        )

        if (
            env.current_node
            == env.destination
        ):
            successes += 1
            append_metrics(
                metric_store,
                metrics,
            )

    return summarize_topology(
        method="Risk-DQN",
        topology_path=topology_path,
        successes=successes,
        metric_store=metric_store,
    )


def load_model(
    sample_topology: Path,
    device: torch.device,
) -> QNetwork:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    sample_env = GeneralizationNetworkEnv(
        sample_topology,
        scenario=SCENARIO,
        seed=0,
    )

    model = QNetwork(
        sample_env.state_size,
        sample_env.action_size,
    ).to(device)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model


def aggregate_across_topologies(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate using topology as the experimental unit."""
    metric_columns = [
        "success_rate",
        "delay",
        "loss",
        "risk",
        "hop",
        "risky_link_usage",
    ]

    summary = (
        raw_df
        .groupby(
            ["rewire_ratio", "method"],
            as_index=False,
        )[metric_columns]
        .agg(["mean", "std"])
    )

    # Flatten MultiIndex columns.
    summary.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part)
        ).strip("_")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    summary = summary.rename(
        columns={
            "rewire_ratio_": "rewire_ratio",
            "method_": "method",
        }
    )

    return summary


def save_results(
    rows: List[Dict[str, object]],
) -> None:
    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_path = (
        RESULT_DIR
        / "dynamic_generalization_by_topology.csv"
    )
    summary_path = (
        RESULT_DIR
        / "dynamic_generalization_by_ratio.csv"
    )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(
        raw_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary_df = aggregate_across_topologies(
        raw_df
    )
    summary_df.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("Saved:", raw_path)
    print("Saved:", summary_path)


def print_topology_rows(
    rows: List[Dict[str, object]],
) -> None:
    print("\n" + "=" * 122)
    print(
        "DYNAMIC GENERALIZATION RESULTS "
        "BY UNSEEN TOPOLOGY"
    )
    print("=" * 122)
    print(
        f"{'Topology':28s} | "
        f"{'Ratio':>6s} | "
        f"{'Method':9s} | "
        f"{'Success':>8s} | "
        f"{'Delay':>9s} | "
        f"{'Loss':>8s} | "
        f"{'Risk':>8s} | "
        f"{'Hop':>6s} | "
        f"{'Risky':>8s}"
    )
    print("-" * 122)

    for row in rows:
        print(
            f"{str(row['topology']):28s} | "
            f"{float(row['rewire_ratio']) * 100:5.0f}% | "
            f"{str(row['method']):9s} | "
            f"{float(row['success_rate']):7.2f}% | "
            f"{float(row['delay']):9.3f} | "
            f"{float(row['loss']):8.5f} | "
            f"{float(row['risk']):8.5f} | "
            f"{float(row['hop']):6.2f} | "
            f"{float(row['risky_link_usage']):7.2f}%"
        )

    print("-" * 122)


def print_ratio_summary(
    rows: List[Dict[str, object]],
) -> None:
    raw_df = pd.DataFrame(rows)
    summary_df = aggregate_across_topologies(
        raw_df
    )

    print("\n" + "=" * 120)
    print(
        "DYNAMIC GENERALIZATION SUMMARY "
        "(MEAN ± STD ACROSS TOPOLOGIES)"
    )
    print("=" * 120)
    print(
        f"{'Ratio':>6s} | "
        f"{'Method':9s} | "
        f"{'Success':>17s} | "
        f"{'Loss':>17s} | "
        f"{'Risk':>17s} | "
        f"{'Hop':>17s} | "
        f"{'Risky':>17s}"
    )
    print("-" * 120)

    for _, row in summary_df.iterrows():
        print(
            f"{float(row['rewire_ratio']) * 100:5.0f}% | "
            f"{str(row['method']):9s} | "
            f"{float(row['success_rate_mean']):7.2f}"
            f"±{float(row['success_rate_std']):6.2f} | "
            f"{float(row['loss_mean']):7.5f}"
            f"±{float(row['loss_std']):7.5f} | "
            f"{float(row['risk_mean']):7.5f}"
            f"±{float(row['risk_std']):7.5f} | "
            f"{float(row['hop_mean']):7.2f}"
            f"±{float(row['hop_std']):7.2f} | "
            f"{float(row['risky_link_usage_mean']):7.2f}"
            f"±{float(row['risky_link_usage_std']):7.2f}"
        )

    print("-" * 120)


def main() -> None:
    topology_files = sorted(
        TOPOLOGY_DIR.glob("*.json")
    )

    if not topology_files:
        raise FileNotFoundError(
            f"No topology JSON files found in "
            f"'{TOPOLOGY_DIR}'. "
            "Run generate_unseen_topologies_dynamic.py first."
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        topology_files[0],
        device,
    )

    print("Device:", device)
    print("Model:", MODEL_PATH)
    print("Scenario:", SCENARIO)
    print("Unseen topologies:", len(topology_files))
    print(
        "Episodes per topology/method:",
        EPISODES_PER_TOPOLOGY,
    )

    rows: List[Dict[str, object]] = []

    for topology_path in topology_files:
        print("\nEvaluating:", topology_path)

        rows.append(
            evaluate_ospf(
                topology_path
            )
        )
        rows.append(
            evaluate_dqn(
                topology_path,
                model,
                device,
            )
        )

    print_topology_rows(rows)
    print_ratio_summary(rows)
    save_results(rows)

    print(
        "\nInterpretation note: this experiment "
        "evaluates the trained policy on unseen "
        "link configurations with fixed node and "
        "link counts. It does not test node-count "
        "generalization."
    )


if __name__ == "__main__":
    main()
