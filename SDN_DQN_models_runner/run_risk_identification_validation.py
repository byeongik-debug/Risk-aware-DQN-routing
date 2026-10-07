# run_risk_identification_validation.py
# Validation experiment for the proposed risk score.
# This script does not use DQN and does not require retraining.
# It compares how accurately each current metric identifies injected dangerous links.

import os
import random
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from network_env_risk_validation import RiskValidationEnv


RESULT_DIR = "results"
SEEDS = list(range(10))
EPISODES_PER_SEED = 30
STEPS_PER_EPISODE = 30
TOP_RATIO = 0.20

METHODS = {
    "Proposed Risk": "risk",
    "Delay Only": "delay",
    "Loss Only": "loss",
    "Utilization Only": "utilization",
    "Simple QoS Score": "simple_qos_score",
}


def minmax(series):
    s_min = float(series.min())
    s_max = float(series.max())
    if abs(s_max - s_min) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def collect_validation_log():
    rows = []

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)

        for ep in range(EPISODES_PER_SEED):
            env_seed = seed * 10000 + ep
            env = RiskValidationEnv(seed=env_seed, danger_ratio=TOP_RATIO)

            for row in env.snapshot():
                row.update({"seed": seed, "episode": ep})
                rows.append(row)

            for _ in range(STEPS_PER_EPISODE):
                snapshot = env.step()
                for row in snapshot:
                    row.update({"seed": seed, "episode": ep})
                    rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULT_DIR, exist_ok=True)
    log_path = os.path.join(RESULT_DIR, "risk_identification_validation_log.csv")
    df.to_csv(log_path, index=False)
    print("Risk validation log saved:", log_path)
    return df


def evaluate_one_group(df_group, score_col, top_ratio=TOP_RATIO):
    n = len(df_group)
    k = max(1, int(np.ceil(n * top_ratio)))

    ranked = df_group.sort_values(score_col, ascending=False)
    pred_idx = set(ranked.head(k).index)
    actual_idx = set(df_group[df_group["danger_label"] == 1].index)

    tp = len(pred_idx & actual_idx)
    fp = len(pred_idx - actual_idx)
    fn = len(actual_idx - pred_idx)
    tn = n - tp - fp - fn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(n, 1)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def evaluate_detection(df):
    df = df.copy()

    # Simple QoS score is recalculated within each time slice to avoid global scale bias.
    df["simple_qos_score"] = 0.0
    for _, idx in df.groupby(["seed", "episode", "time"]).groups.items():
        part = df.loc[idx]
        score = (
            minmax(part["delay"]) +
            minmax(part["loss"]) +
            minmax(part["utilization"])
        ) / 3.0
        df.loc[idx, "simple_qos_score"] = score

    per_step_rows = []
    for (seed, ep, t), group in df.groupby(["seed", "episode", "time"]):
        for method, score_col in METHODS.items():
            metrics = evaluate_one_group(group, score_col, TOP_RATIO)
            metrics.update({
                "seed": seed,
                "episode": ep,
                "time": t,
                "method": method,
            })
            per_step_rows.append(metrics)

    per_step = pd.DataFrame(per_step_rows)

    summary = per_step.groupby("method").agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
    ).reset_index()

    # Keep method order stable.
    summary["order"] = summary["method"].map({m: i for i, m in enumerate(METHODS.keys())})
    summary = summary.sort_values("order").drop(columns=["order"])

    # Per danger type hit rate: among each injected danger category, how often selected by each method.
    danger_rows = []
    for (seed, ep, t), group in df.groupby(["seed", "episode", "time"]):
        n = len(group)
        k = max(1, int(np.ceil(n * TOP_RATIO)))
        for method, score_col in METHODS.items():
            pred_idx = set(group.sort_values(score_col, ascending=False).head(k).index)
            for danger_type, type_group in group[group["danger_label"] == 1].groupby("danger_type"):
                actual_idx = set(type_group.index)
                hit_rate = len(pred_idx & actual_idx) / max(len(actual_idx), 1)
                danger_rows.append({
                    "seed": seed,
                    "episode": ep,
                    "time": t,
                    "method": method,
                    "danger_type": danger_type,
                    "hit_rate": hit_rate,
                })

    by_type = pd.DataFrame(danger_rows).groupby(["method", "danger_type"]).agg(
        hit_rate_mean=("hit_rate", "mean"),
        hit_rate_std=("hit_rate", "std"),
    ).reset_index()

    os.makedirs(RESULT_DIR, exist_ok=True)
    per_step_path = os.path.join(RESULT_DIR, "risk_identification_per_step.csv")
    summary_path = os.path.join(RESULT_DIR, "risk_identification_summary.csv")
    by_type_path = os.path.join(RESULT_DIR, "risk_identification_by_danger_type.csv")

    per_step.to_csv(per_step_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_type.to_csv(by_type_path, index=False)

    print("Per-step detection metrics saved:", per_step_path)
    print("Summary saved:", summary_path)
    print("Danger-type hit rate saved:", by_type_path)

    return summary, by_type


def print_summary(summary):
    print("\n" + "=" * 120)
    print("RISK SCORE DANGEROUS LINK IDENTIFICATION VALIDATION")
    print("=" * 120)
    print(
        f"{'Method':18s} | {'Precision':>10s} | {'Recall':>10s} | "
        f"{'F1':>10s} | {'Accuracy':>10s}"
    )
    print("-" * 120)

    for _, r in summary.iterrows():
        print(
            f"{r['method']:18s} | "
            f"{r['precision_mean']:8.4f}±{r['precision_std']:.4f} | "
            f"{r['recall_mean']:8.4f}±{r['recall_std']:.4f} | "
            f"{r['f1_mean']:8.4f}±{r['f1_std']:.4f} | "
            f"{r['accuracy_mean']:8.4f}±{r['accuracy_std']:.4f}"
        )

    print("-" * 120)
    print("Prediction: top 20% links by each metric")
    print("Ground truth: injected dangerous links in the validation environment")


def save_graph(summary):
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary["f1_mean"], yerr=summary["f1_std"], capsize=4)
    plt.xticks(x, summary["method"], rotation=20, ha="right")
    plt.ylabel("F1-score")
    plt.title("Dangerous Link Identification F1-score")
    plt.tight_layout()
    path = os.path.join(RESULT_DIR, "risk_identification_f1_score.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Graph saved:", path)


def main():
    print("Risk identification validation started")
    print("Seeds:", SEEDS)
    print("Episodes per seed:", EPISODES_PER_SEED)
    print("Steps per episode:", STEPS_PER_EPISODE)
    print("Top ratio:", TOP_RATIO)

    df = collect_validation_log()
    summary, by_type = evaluate_detection(df)
    print_summary(summary)
    save_graph(summary)

    print("\nValidation complete. Check the results folder.")


if __name__ == "__main__":
    main()
