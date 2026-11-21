# run_risk_identification_validation_v2.py
# Final validation script for Temporal Risk dangerous-link identification.
#
# Improvements over the original version:
# 1) Reports mean ± std across SEEDS, not across all time steps.
# 2) Adds paired statistical tests between Proposed Risk and Simple QoS Score.
# 3) Keeps the original Top-20% ranking-based experiment.
# 4) Adds a fixed-threshold robustness experiment for danger ratios 10%, 20%, 30%.
#
# NOTE:
# - This script does NOT use DQN and does NOT require retraining.
# - It assumes RiskValidationEnv(seed=..., danger_ratio=...) and env.snapshot()/env.step()
#   return rows containing at least:
#   delay, loss, utilization, risk, danger_label, danger_type, time

import os
import random
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.stats import ttest_rel, wilcoxon
except ImportError as e:
    raise ImportError(
        "scipy is required for paired statistical tests. "
        "Install it with: pip install scipy"
    ) from e

from network_env_risk_validation import RiskValidationEnv

RESULT_DIR = "results"
SEEDS = list(range(10))
EPISODES_PER_SEED = 30
STEPS_PER_EPISODE = 30
TOP_RATIO = 0.20
DANGER_RATIOS = [0.10, 0.20, 0.30]
CALIBRATION_SEEDS = [0, 1, 2, 3, 4]
TEST_SEEDS = [5, 6, 7, 8, 9]

METHODS = {
    "Proposed Risk": "risk",
    "Delay Only": "delay",
    "Loss Only": "loss",
    "Utilization Only": "utilization",
    "Simple QoS Score": "simple_qos_score",
}

ROBUSTNESS_METHODS = {
    "Proposed Risk": "risk",
    "Simple QoS Score": "simple_qos_score",
}


def minmax(series):
    s_min = float(series.min())
    s_max = float(series.max())
    if abs(s_max - s_min) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


def add_simple_qos_score(df):
    df = df.copy()
    df["simple_qos_score"] = 0.0
    for _, idx in df.groupby(["seed", "episode", "time"]).groups.items():
        part = df.loc[idx]
        score = (
            minmax(part["delay"])
            + minmax(part["loss"])
            + minmax(part["utilization"])
        ) / 3.0
        df.loc[idx, "simple_qos_score"] = score
    return df


def calc_metrics(pred_idx, actual_idx, n):
    tp = len(pred_idx & actual_idx)
    fp = len(pred_idx - actual_idx)
    fn = len(actual_idx - pred_idx)
    tn = n - tp - fp - fn

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
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


def collect_validation_log(seeds, danger_ratio, file_tag,
                           episodes_per_seed=EPISODES_PER_SEED,
                           steps_per_episode=STEPS_PER_EPISODE):
    rows = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)

        for ep in range(episodes_per_seed):
            env_seed = seed * 10000 + ep
            env = RiskValidationEnv(seed=env_seed, danger_ratio=danger_ratio)

            for row in env.snapshot():
                row = dict(row)
                row.update({
                    "seed": seed,
                    "episode": ep,
                    "danger_ratio": danger_ratio,
                })
                rows.append(row)

            for _ in range(steps_per_episode):
                snapshot = env.step()
                for row in snapshot:
                    row = dict(row)
                    row.update({
                        "seed": seed,
                        "episode": ep,
                        "danger_ratio": danger_ratio,
                    })
                    rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f"risk_validation_log_{file_tag}.csv")
    df.to_csv(path, index=False)
    print("Validation log saved:", path)
    return df


def evaluate_one_group_topk(df_group, score_col, top_ratio=TOP_RATIO):
    n = len(df_group)
    k = max(1, int(np.ceil(n * top_ratio)))
    ranked = df_group.sort_values(score_col, ascending=False)
    pred_idx = set(ranked.head(k).index)
    actual_idx = set(df_group[df_group["danger_label"] == 1].index)
    return calc_metrics(pred_idx, actual_idx, n)


def evaluate_topk_detection(df):
    df = add_simple_qos_score(df)
    per_step_rows = []

    for (seed, ep, t), group in df.groupby(["seed", "episode", "time"]):
        for method, score_col in METHODS.items():
            metrics = evaluate_one_group_topk(group, score_col, TOP_RATIO)
            metrics.update({
                "seed": seed,
                "episode": ep,
                "time": t,
                "method": method,
            })
            per_step_rows.append(metrics)

    per_step = pd.DataFrame(per_step_rows)

    per_seed = (
        per_step.groupby(["seed", "method"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_seed.groupby("method")
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
    )

    order_map = {method: i for i, method in enumerate(METHODS.keys())}
    summary["order"] = summary["method"].map(order_map)
    summary = summary.sort_values("order").drop(columns=["order"]).reset_index(drop=True)

    danger_rows = []
    for (seed, ep, t), group in df.groupby(["seed", "episode", "time"]):
        n = len(group)
        k = max(1, int(np.ceil(n * TOP_RATIO)))
        for method, score_col in METHODS.items():
            pred_idx = set(group.sort_values(score_col, ascending=False).head(k).index)
            danger_only = group[group["danger_label"] == 1]
            for danger_type, type_group in danger_only.groupby("danger_type"):
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

    danger_df = pd.DataFrame(danger_rows)
    if not danger_df.empty:
        by_type_per_seed = (
            danger_df.groupby(["seed", "method", "danger_type"])
            .agg(hit_rate=("hit_rate", "mean"))
            .reset_index()
        )
        by_type = (
            by_type_per_seed.groupby(["method", "danger_type"])
            .agg(
                hit_rate_mean=("hit_rate", "mean"),
                hit_rate_std=("hit_rate", "std"),
            )
            .reset_index()
        )
    else:
        by_type = pd.DataFrame()

    os.makedirs(RESULT_DIR, exist_ok=True)
    per_step_path = os.path.join(RESULT_DIR, "risk_identification_top20_per_step.csv")
    per_seed_path = os.path.join(RESULT_DIR, "risk_identification_top20_per_seed.csv")
    summary_path = os.path.join(RESULT_DIR, "risk_identification_top20_summary.csv")
    by_type_path = os.path.join(RESULT_DIR, "risk_identification_top20_by_danger_type.csv")

    per_step.to_csv(per_step_path, index=False)
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)
    by_type.to_csv(by_type_path, index=False)

    print("Top-k per-step metrics saved:", per_step_path)
    print("Top-k per-seed metrics saved:", per_seed_path)
    print("Top-k seed-level summary saved:", summary_path)
    print("Danger-type hit-rate saved:", by_type_path)

    return df, per_step, per_seed, summary, by_type


def paired_statistical_tests(per_seed,
                             method_a="Proposed Risk",
                             method_b="Simple QoS Score"):
    a = (
        per_seed[per_seed["method"] == method_a][["seed", "f1"]]
        .rename(columns={"f1": "f1_a"})
    )
    b = (
        per_seed[per_seed["method"] == method_b][["seed", "f1"]]
        .rename(columns={"f1": "f1_b"})
    )

    paired = a.merge(b, on="seed", how="inner").sort_values("seed").reset_index(drop=True)
    paired["difference"] = paired["f1_a"] - paired["f1_b"]

    if len(paired) < 2:
        raise ValueError("At least two paired seeds are required.")

    t_stat, t_p = ttest_rel(paired["f1_a"], paired["f1_b"])

    try:
        w_stat, w_p = wilcoxon(
            paired["f1_a"],
            paired["f1_b"],
            zero_method="wilcox",
            alternative="two-sided",
        )
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    diff = paired["difference"].to_numpy()
    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    from scipy.stats import t as student_t
    se_diff = std_diff / np.sqrt(n)
    t_crit = student_t.ppf(0.975, df=n - 1)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff

    result = pd.DataFrame([{
        "method_a": method_a,
        "method_b": method_b,
        "n_pairs": n,
        "mean_f1_a": paired["f1_a"].mean(),
        "mean_f1_b": paired["f1_b"].mean(),
        "mean_paired_difference": mean_diff,
        "std_paired_difference": std_diff,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "paired_t_stat": t_stat,
        "paired_t_p": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
    }])

    paired_path = os.path.join(RESULT_DIR, "risk_identification_paired_seed_values.csv")
    test_path = os.path.join(RESULT_DIR, "risk_identification_statistical_tests.csv")
    paired.to_csv(paired_path, index=False)
    result.to_csv(test_path, index=False)

    print("\n" + "=" * 100)
    print("PAIRED STATISTICAL TEST")
    print("=" * 100)
    print(f"{method_a} vs. {method_b}")
    print("Paired seeds:", n)
    print(f"Mean F1: {method_a} = {paired['f1_a'].mean():.6f}")
    print(f"Mean F1: {method_b} = {paired['f1_b'].mean():.6f}")
    print(f"Mean paired difference = {mean_diff:.6f}")
    print(f"95% CI of paired difference = [{ci_low:.6f}, {ci_high:.6f}]")
    print(f"Paired t-test: t = {t_stat:.6f}, p = {t_p:.6f}")
    print(f"Wilcoxon: W = {w_stat:.6f}, p = {w_p:.6f}")
    print("=" * 100)

    print("Paired seed values saved:", paired_path)
    print("Statistical test results saved:", test_path)
    return paired, result


def calibrate_thresholds(df_calibration):
    df_calibration = add_simple_qos_score(df_calibration)
    thresholds = {}

    for method, score_col in ROBUSTNESS_METHODS.items():
        values = (
            df_calibration[score_col]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .to_numpy()
        )
        if len(values) == 0:
            raise ValueError(f"No valid calibration values for {method}")

        threshold = float(np.quantile(values, 1.0 - TOP_RATIO))
        thresholds[method] = threshold

    threshold_df = pd.DataFrame([
        {
            "method": method,
            "score_col": ROBUSTNESS_METHODS[method],
            "threshold": threshold,
            "calibration_top_ratio": TOP_RATIO,
        }
        for method, threshold in thresholds.items()
    ])

    path = os.path.join(RESULT_DIR, "risk_identification_fixed_thresholds.csv")
    threshold_df.to_csv(path, index=False)

    print("\nFixed thresholds:")
    for method, threshold in thresholds.items():
        print(f"  {method:20s}: {threshold:.6f}")
    print("Thresholds saved:", path)

    return thresholds


def evaluate_one_group_threshold(df_group, score_col, threshold):
    pred_idx = set(df_group[df_group[score_col] >= threshold].index)
    actual_idx = set(df_group[df_group["danger_label"] == 1].index)
    return calc_metrics(pred_idx, actual_idx, len(df_group))


def evaluate_fixed_threshold_ratio(df, thresholds, danger_ratio):
    df = add_simple_qos_score(df)
    per_step_rows = []

    for (seed, ep, t), group in df.groupby(["seed", "episode", "time"]):
        for method, score_col in ROBUSTNESS_METHODS.items():
            metrics = evaluate_one_group_threshold(group, score_col, thresholds[method])
            metrics.update({
                "seed": seed,
                "episode": ep,
                "time": t,
                "danger_ratio": danger_ratio,
                "method": method,
            })
            per_step_rows.append(metrics)

    per_step = pd.DataFrame(per_step_rows)

    per_seed = (
        per_step.groupby(["danger_ratio", "seed", "method"])
        .agg(
            precision=("precision", "mean"),
            recall=("recall", "mean"),
            f1=("f1", "mean"),
            accuracy=("accuracy", "mean"),
        )
        .reset_index()
    )

    summary = (
        per_seed.groupby(["danger_ratio", "method"])
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
        )
        .reset_index()
    )

    return per_step, per_seed, summary


def run_fixed_threshold_robustness():
    print("\n" + "=" * 100)
    print("FIXED-THRESHOLD ROBUSTNESS VALIDATION")
    print("=" * 100)

    calibration_df = collect_validation_log(
        seeds=CALIBRATION_SEEDS,
        danger_ratio=TOP_RATIO,
        file_tag="calibration_20pct",
    )

    thresholds = calibrate_thresholds(calibration_df)

    all_step = []
    all_seed = []
    all_summary = []

    for ratio in DANGER_RATIOS:
        print(f"\nEvaluating danger ratio {ratio:.0%} with fixed thresholds...")

        df = collect_validation_log(
            seeds=TEST_SEEDS,
            danger_ratio=ratio,
            file_tag=f"fixed_threshold_test_{int(ratio * 100)}pct",
        )

        per_step, per_seed, summary = evaluate_fixed_threshold_ratio(
            df,
            thresholds,
            ratio,
        )

        all_step.append(per_step)
        all_seed.append(per_seed)
        all_summary.append(summary)

    all_step = pd.concat(all_step, ignore_index=True)
    all_seed = pd.concat(all_seed, ignore_index=True)
    all_summary = pd.concat(all_summary, ignore_index=True)

    method_order = {method: i for i, method in enumerate(ROBUSTNESS_METHODS.keys())}
    all_summary["method_order"] = all_summary["method"].map(method_order)
    all_summary = (
        all_summary.sort_values(["danger_ratio", "method_order"])
        .drop(columns=["method_order"])
        .reset_index(drop=True)
    )

    per_step_path = os.path.join(RESULT_DIR, "risk_identification_fixed_threshold_per_step.csv")
    per_seed_path = os.path.join(RESULT_DIR, "risk_identification_fixed_threshold_per_seed.csv")
    summary_path = os.path.join(RESULT_DIR, "risk_identification_fixed_threshold_summary.csv")

    all_step.to_csv(per_step_path, index=False)
    all_seed.to_csv(per_seed_path, index=False)
    all_summary.to_csv(summary_path, index=False)

    print("Fixed-threshold per-step results saved:", per_step_path)
    print("Fixed-threshold per-seed results saved:", per_seed_path)
    print("Fixed-threshold summary saved:", summary_path)

    return thresholds, all_step, all_seed, all_summary


def print_topk_summary(summary):
    print("\n" + "=" * 120)
    print("TOP-20% RANKING-BASED DANGEROUS LINK IDENTIFICATION")
    print("Mean ± std are calculated across SEEDS.")
    print("=" * 120)
    print(
        f"{'Method':20s} | "
        f"{'Precision':>18s} | "
        f"{'Recall':>18s} | "
        f"{'F1':>18s} | "
        f"{'Accuracy':>18s}"
    )
    print("-" * 120)

    for _, r in summary.iterrows():
        print(
            f"{r['method']:20s} | "
            f"{r['precision_mean']:.4f}±{r['precision_std']:.4f} | "
            f"{r['recall_mean']:.4f}±{r['recall_std']:.4f} | "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f} | "
            f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
        )

    print("-" * 120)
    print("Prediction: top 20% links by each metric")
    print("Ground truth: injected dangerous links with danger_ratio = 20%")
    print(
        "Because predicted and actual counts are matched, "
        "Precision = Recall = F1 in this ranking experiment."
    )


def print_fixed_threshold_summary(summary):
    print("\n" + "=" * 135)
    print("FIXED-THRESHOLD ROBUSTNESS RESULTS (mean ± std across test seeds)")
    print("=" * 135)
    print(
        f"{'Danger':>8s} | "
        f"{'Method':20s} | "
        f"{'Precision':>18s} | "
        f"{'Recall':>18s} | "
        f"{'F1':>18s} | "
        f"{'Accuracy':>18s}"
    )
    print("-" * 135)

    for _, r in summary.iterrows():
        print(
            f"{r['danger_ratio'] * 100:7.0f}% | "
            f"{r['method']:20s} | "
            f"{r['precision_mean']:.4f}±{r['precision_std']:.4f} | "
            f"{r['recall_mean']:.4f}±{r['recall_std']:.4f} | "
            f"{r['f1_mean']:.4f}±{r['f1_std']:.4f} | "
            f"{r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}"
        )

    print("-" * 135)


def save_topk_graph(summary):
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.figure(figsize=(10, 5))
    x = np.arange(len(summary))
    plt.bar(x, summary["f1_mean"], yerr=summary["f1_std"], capsize=4)
    plt.xticks(x, summary["method"], rotation=20, ha="right")
    plt.ylabel("F1-score")
    plt.title("Dangerous Link Identification (Top-20% Ranking)")
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "risk_identification_top20_f1_seed_std.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Top-k graph saved:", path)


def save_fixed_threshold_graph(summary):
    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))

    for method in ROBUSTNESS_METHODS.keys():
        part = summary[summary["method"] == method].sort_values("danger_ratio")
        plt.errorbar(
            part["danger_ratio"] * 100.0,
            part["f1_mean"],
            yerr=part["f1_std"],
            marker="o",
            capsize=4,
            label=method,
        )

    plt.xlabel("Actual Dangerous Link Ratio (%)")
    plt.ylabel("F1-score")
    plt.title("Fixed-Threshold Robustness under Different Danger Ratios")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "risk_identification_fixed_threshold_f1.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Fixed-threshold graph saved:", path)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("=" * 100)
    print("TEMPORAL RISK IDENTIFICATION VALIDATION V2")
    print("=" * 100)
    print("All seeds:", SEEDS)
    print("Episodes per seed:", EPISODES_PER_SEED)
    print("Steps per episode:", STEPS_PER_EPISODE)
    print("Original Top-k ratio:", TOP_RATIO)

    print("\n[Experiment 1] Top-20% ranking validation")
    top20_df = collect_validation_log(
        seeds=SEEDS,
        danger_ratio=TOP_RATIO,
        file_tag="top20_all_seeds",
    )

    _, _, top20_per_seed, top20_summary, _ = evaluate_topk_detection(top20_df)
    print_topk_summary(top20_summary)
    paired_statistical_tests(
        top20_per_seed,
        method_a="Proposed Risk",
        method_b="Simple QoS Score",
    )
    save_topk_graph(top20_summary)

    print("\n[Experiment 2] Fixed-threshold robustness validation")
    _, _, _, fixed_summary = run_fixed_threshold_robustness()
    print_fixed_threshold_summary(fixed_summary)
    save_fixed_threshold_graph(fixed_summary)

    print("\n" + "=" * 100)
    print("Validation complete.")
    print("Check the 'results' folder for CSV files and figures.")
    print("=" * 100)


if __name__ == "__main__":
    main()