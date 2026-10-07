# run_policy.py
# Multi-Scenario Evaluation: OSPF vs Risk-Only DQN
# + Dynamic Risk Validation using env.qos actual risk values
# + Current Metric -> Future Dangerous Link Detection Accuracy

import os
import random
import csv
from collections import Counter

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from network_env import NetworkEnv
from topology import DESTINATION


SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EPISODES = 200
WARMUP_STEPS = 10
DEBUG_PATH = True
DYNAMIC_PATH_DEBUG_EPISODES = 20
PRINT_DYNAMIC_PATHS = True

RISK_ONLY_MODEL_PATH = "risk_only_model_final.pth"
RESULT_DIR = "results"

# Risk validation settings
RISK_VALIDATION_EPISODES = 200
FUTURE_WINDOW = 10
DYNAMIC_LOG_PATH = os.path.join(RESULT_DIR, "dynamic_link_log.csv")
RISK_SAMPLE_PATH = os.path.join(RESULT_DIR, "risk_future_exposure_samples.csv")
RISK_SUMMARY_PATH = os.path.join(RESULT_DIR, "risk_future_exposure_summary.csv")


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


def warmup_qos(env, steps=WARMUP_STEPS):
    for _ in range(steps):
        env.update_dynamic_qos()


def summarize_result(model_name, success_count, episodes, delays, losses, risks, hops, unstable_usages, congested_usages):
    if success_count == 0:
        return {
            "model": model_name,
            "success_rate": 0.0,
            "delay": 0.0,
            "loss": 0.0,
            "risk": 0.0,
            "hop": 0.0,
            "unstable_usage": 0.0,
            "congested_usage": 0.0,
        }

    return {
        "model": model_name,
        "success_rate": (success_count / episodes) * 100,
        "delay": float(np.mean(delays)),
        "loss": float(np.mean(losses)),
        "risk": float(np.mean(risks)),
        "hop": float(np.mean(hops)),
        "unstable_usage": float(np.mean(unstable_usages)),
        "congested_usage": float(np.mean(congested_usages)),
    }


def append_metrics(metrics, delays, losses, risks, hops, unstable_usages, congested_usages):
    delays.append(metrics["total_delay"])
    losses.append(metrics["total_loss"])
    risks.append(metrics["total_risk"])
    hops.append(metrics["hop_count"])

    hop_count = max(metrics["hop_count"], 1)
    unstable_usages.append((metrics["unstable_count"] / hop_count) * 100)
    congested_usages.append((metrics["congested_count"] / hop_count) * 100)


def evaluate_ospf(scenario, episodes=EPISODES):
    success_count = 0
    delays, losses, risks, hops = [], [], [], []
    unstable_usages, congested_usages = [], []
    dynamic_path_counter = Counter()

    for ep in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        warmup_qos(env)

        path = env.ospf_path(cost_type="bandwidth")
        metrics = env.calculate_path_metrics(path)

        if scenario == "dynamic":
            dynamic_path_counter[tuple(path)] += 1
            if PRINT_DYNAMIC_PATHS and ep < DYNAMIC_PATH_DEBUG_EPISODES:
                print(f"[DYNAMIC][OSPF] EP {ep + 1}")
                print("Path:", path)
                print(f"Hop={metrics['hop_count']} Risk={metrics['total_risk']:.5f}")
                print("-" * 50)

        if DEBUG_PATH and ep == 0:
            print("\n===================================")
            print(f"[{scenario.upper()}] OSPF PATH")
            print(path)
            print(
                f"Hop={metrics['hop_count']} "
                f"Risk={metrics['total_risk']:.5f} "
                f"Unstable={metrics['unstable_count']} "
                f"Congested={metrics['congested_count']}"
            )
            print("===================================\n")

        if path and path[-1] == DESTINATION:
            success_count += 1
            append_metrics(metrics, delays, losses, risks, hops, unstable_usages, congested_usages)

    if scenario == "dynamic" and dynamic_path_counter:
        print("\n[DYNAMIC][OSPF] PATH DISTRIBUTION TOP 10")
        for path_tuple, count in dynamic_path_counter.most_common(10):
            print(f"Count={count:3d} | Hop={len(path_tuple) - 1:2d} | Path={list(path_tuple)}")
        print()

    return summarize_result(
        "OSPF",
        success_count,
        episodes,
        delays,
        losses,
        risks,
        hops,
        unstable_usages,
        congested_usages,
    )


def select_dqn_action(model, state, valid_actions, device):
    if not valid_actions:
        return None

    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        q_values = model(state_tensor)[0]

    masked_q = torch.full_like(q_values, -1e9)
    for action in valid_actions:
        masked_q[action] = q_values[action]

    return int(torch.argmax(masked_q).item())


def evaluate_risk_only_dqn(model_path, scenario, method_name="Risk-Only-DQN", episodes=EPISODES):
    if not os.path.exists(model_path):
        print(f"[경고] 모델 파일 없음: {model_path}")
        return None

    base_env = NetworkEnv(use_risk=True, scenario=scenario)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    success_count = 0
    delays, losses, risks, hops = [], [], [], []
    unstable_usages, congested_usages = [], []
    dynamic_path_counter = Counter()

    for ep in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario)
        state = env.reset()
        warmup_qos(env)
        state = env.get_state()

        done = False

        while not done:
            valid_actions = env.get_valid_actions()
            action = select_dqn_action(model, state, valid_actions, device)

            if action is None:
                break

            next_state, _, done, _ = env.step(action)
            state = next_state

        metrics = env.calculate_path_metrics(env.visited)

        if scenario == "dynamic":
            dynamic_path_counter[tuple(env.visited)] += 1
            if PRINT_DYNAMIC_PATHS and ep < DYNAMIC_PATH_DEBUG_EPISODES:
                print(f"[DYNAMIC][{method_name}] EP {ep + 1}")
                print("Path:", env.visited)
                print(f"Hop={metrics['hop_count']} Risk={metrics['total_risk']:.5f}")
                print("-" * 50)

        if DEBUG_PATH and ep == 0:
            print("\n===================================")
            print(f"[{scenario.upper()}] {method_name} PATH")
            print(env.visited)
            print(
                f"Hop={metrics['hop_count']} "
                f"Risk={metrics['total_risk']:.5f} "
                f"Unstable={metrics['unstable_count']} "
                f"Congested={metrics['congested_count']}"
            )
            print("===================================\n")

        if env.current_node == DESTINATION:
            success_count += 1
            append_metrics(metrics, delays, losses, risks, hops, unstable_usages, congested_usages)

    if scenario == "dynamic" and dynamic_path_counter:
        print(f"\n[DYNAMIC][{method_name}] PATH DISTRIBUTION TOP 10")
        for path_tuple, count in dynamic_path_counter.most_common(10):
            print(f"Count={count:3d} | Hop={len(path_tuple) - 1:2d} | Path={list(path_tuple)}")
        print()

    return summarize_result(
        method_name,
        success_count,
        episodes,
        delays,
        losses,
        risks,
        hops,
        unstable_usages,
        congested_usages,
    )


def print_scenario_summary(scenario, results):
    print(f"\nScenario: {scenario.upper()}")
    print("-" * 95)
    print(
        f"{'Model':15s} | {'Success':>8s} | {'Delay':>9s} | {'Loss':>9s} | "
        f"{'Risk':>9s} | {'Hop':>5s} | {'Unstable':>9s} | {'Congested':>9s}"
    )
    print("-" * 95)

    for r in results:
        if r is None:
            continue

        print(
            f"{r['model']:15s} | "
            f"{r['success_rate']:7.2f}% | "
            f"{r['delay']:9.3f} | "
            f"{r['loss']:9.5f} | "
            f"{r['risk']:9.5f} | "
            f"{r['hop']:5.2f} | "
            f"{r['unstable_usage']:8.2f}% | "
            f"{r['congested_usage']:8.2f}%"
        )

    print("-" * 95)


def print_final_summary(all_results):
    print("\n" + "=" * 120)
    print("FINAL MULTI-SCENARIO SUMMARY")
    print("=" * 120)
    print(
        f"{'Scenario':12s} | {'Model':15s} | {'Success':>8s} | {'Delay':>10s} | "
        f"{'Loss':>10s} | {'Risk':>10s} | {'Hop':>6s} | {'Unstable':>9s} | {'Congested':>9s}"
    )
    print("-" * 120)

    for scenario, result_list in all_results:
        for r in result_list:
            print(
                f"{scenario:12s} | "
                f"{r['model']:15s} | "
                f"{r['success_rate']:7.2f}% | "
                f"{r['delay']:10.3f} | "
                f"{r['loss']:10.5f} | "
                f"{r['risk']:10.5f} | "
                f"{r['hop']:6.2f} | "
                f"{r['unstable_usage']:8.2f}% | "
                f"{r['congested_usage']:8.2f}%"
            )

    print("-" * 120)


def save_csv(all_results):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, "multi_scenario_summary.csv")

    rows = []
    for scenario, result_list in all_results:
        for r in result_list:
            row = {"scenario": scenario}
            row.update(r)
            rows.append(row)

    fieldnames = [
        "scenario",
        "model",
        "success_rate",
        "delay",
        "loss",
        "risk",
        "hop",
        "unstable_usage",
        "congested_usage",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("CSV 저장:", path)


def save_metric_graph(all_results, metric, title, ylabel, filename):
    os.makedirs(RESULT_DIR, exist_ok=True)

    scenarios = SCENARIOS
    methods = ["OSPF", "Risk-Only-DQN"]

    x = np.arange(len(scenarios))
    width = 0.35

    plt.figure(figsize=(10, 5))

    for i, method in enumerate(methods):
        values = []
        for scenario in scenarios:
            value = 0.0
            for sc, result_list in all_results:
                if sc != scenario:
                    continue
                matched = [r for r in result_list if r["model"] == method]
                if matched:
                    value = matched[0][metric]
            values.append(value)

        plt.bar(x + (i - 0.5) * width, values, width, label=method)

    plt.xticks(x, scenarios)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Scenario")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, filename)
    plt.savefig(path, dpi=300)
    plt.close()
    print("그래프 저장:", path)


def save_all_graphs(all_results):
    save_metric_graph(all_results, "success_rate", "Success Rate by Scenario", "Success Rate (%)", "success_rate_by_scenario.png")
    save_metric_graph(all_results, "delay", "Average Delay by Scenario", "Average Delay", "avg_delay_by_scenario.png")
    save_metric_graph(all_results, "loss", "Average Loss by Scenario", "Average Loss", "avg_loss_by_scenario.png")
    save_metric_graph(all_results, "risk", "Average Risk by Scenario", "Average Risk", "avg_risk_by_scenario.png")
    save_metric_graph(all_results, "unstable_usage", "Unstable Link Usage by Scenario", "Usage Ratio (%)", "unstable_usage_by_scenario.png")
    save_metric_graph(all_results, "congested_usage", "Congested Link Usage by Scenario", "Usage Ratio (%)", "congested_usage_by_scenario.png")


# ============================================================
# Risk validation section
# ============================================================

# Goal:
# Validate whether the proposed risk score at current time t identifies
# links that will be dangerous in the future window t+1 ~ t+H.
# This avoids same-time self-referential evaluation.

DETECTION_TOP_RATIO = 0.20
RISK_VALIDATION_EPISODES = 200
RISK_VALIDATION_STEPS = 60
FUTURE_WINDOW = 10

DYNAMIC_LOG_PATH = os.path.join(RESULT_DIR, "dynamic_link_log.csv")
FUTURE_DETECTION_STEP_PATH = os.path.join(RESULT_DIR, "risk_future_detection_accuracy_per_step.csv")
FUTURE_DETECTION_SUMMARY_PATH = os.path.join(RESULT_DIR, "risk_future_detection_accuracy_summary.csv")


def collect_dynamic_link_snapshot(env, episode, time_step, rows):
    """
    NetworkEnv stores actual per-link QoS values in env.qos.
    This records the real risk used in the model state.
    """
    for (u, v), q in env.qos.items():
        rows.append({
            "episode": int(episode),
            "time": int(time_step),
            "phase": int(getattr(env, "dynamic_phase", -1)),
            "src": int(u),
            "dst": int(v),
            "link": f"{int(u)}-{int(v)}",
            "risk": float(q.get("risk", 0.0)),
            "delay": float(q.get("delay", 0.0)),
            "loss": float(q.get("loss", 0.0)),
            "utilization": float(q.get("utilization", 0.0)),
            "congested": int(bool(q.get("is_congested", False))),
        })


def save_dynamic_link_log(episodes=RISK_VALIDATION_EPISODES, steps=RISK_VALIDATION_STEPS):
    os.makedirs(RESULT_DIR, exist_ok=True)
    rows = []

    for ep in range(episodes):
        env = NetworkEnv(use_risk=True, scenario="dynamic")
        env.reset()
        warmup_qos(env)

        collect_dynamic_link_snapshot(env, ep, 0, rows)

        for t in range(1, steps + FUTURE_WINDOW + 1):
            env.update_dynamic_qos()
            collect_dynamic_link_snapshot(env, ep, t, rows)

    df = pd.DataFrame(rows)
    df.to_csv(DYNAMIC_LOG_PATH, index=False, encoding="utf-8")
    print("Dynamic 링크 로그 저장:", DYNAMIC_LOG_PATH)
    return df


def exact_top_k_mask(values, top_ratio=DETECTION_TOP_RATIO):
    """Select exactly top-k rows by score."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=bool)

    k = max(1, int(np.ceil(n * top_ratio)))
    order = np.argsort(-np.asarray(values, dtype=float), kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def minmax_normalize(series):
    arr = np.asarray(series, dtype=float)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - mn) / (mx - mn)


def compute_detection_metrics(predicted, actual):
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)

    tp = int(np.logical_and(predicted, actual).sum())
    fp = int(np.logical_and(predicted, ~actual).sum())
    fn = int(np.logical_and(~predicted, actual).sum())
    tn = int(np.logical_and(~predicted, ~actual).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

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


def build_current_future_rows(df, future_window=FUTURE_WINDOW):
    """
    Build one row per current link state at time t.
    Prediction scores use current t values.
    Ground truth uses only future t+1 ~ t+H values.
    """
    required_cols = ["episode", "time", "link", "risk", "delay", "loss", "utilization", "congested"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column in dynamic link log: {col}")

    rows = []
    max_t_by_ep = df.groupby("episode")["time"].max().to_dict()

    grouped = df.set_index(["episode", "link", "time"]).sort_index()

    # Use only t where a full future window exists.
    current_df = df[df.apply(lambda r: r["time"] + future_window <= max_t_by_ep[r["episode"]], axis=1)].copy()

    for _, row in current_df.iterrows():
        ep = int(row["episode"])
        link = row["link"]
        t = int(row["time"])

        try:
            future = grouped.loc[(ep, link)].loc[t + 1:t + future_window]
        except KeyError:
            continue

        if future.empty:
            continue

        rows.append({
            "episode": ep,
            "time": t,
            "phase": int(row.get("phase", -1)),
            "src": int(row["src"]),
            "dst": int(row["dst"]),
            "link": link,
            "current_risk": float(row["risk"]),
            "current_delay": float(row["delay"]),
            "current_loss": float(row["loss"]),
            "current_utilization": float(row["utilization"]),
            "current_congested": int(row["congested"]),
            "future_avg_delay": float(future["delay"].mean()),
            "future_avg_loss": float(future["loss"].mean()),
            "future_avg_utilization": float(future["utilization"].mean()),
            "future_avg_risk": float(future["risk"].mean()),
            "future_congestion_ratio": float(future["congested"].mean()),
            "future_any_congested": int(future["congested"].max()),
        })

    return pd.DataFrame(rows)


def analyze_future_danger_detection(df, top_ratio=DETECTION_TOP_RATIO, future_window=FUTURE_WINDOW):
    """
    Current Metric -> Future Dangerous Link Detection.

    Prediction at time t:
      - Proposed Risk: top 20% current risk
      - Delay Only: top 20% current delay
      - Loss Only: top 20% current loss
      - Utilization Only: top 20% current utilization
      - Simple QoS Score: top 20% normalized current delay/loss/utilization

    Actual future dangerous links:
      - top 20% future average delay, OR
      - top 20% future average loss, OR
      - top 20% future average utilization, OR
      - top 20% future average risk, OR
      - future congestion occurs in t+1 ~ t+H.
    """
    sample_df = build_current_future_rows(df, future_window=future_window)

    if sample_df.empty:
        print("[경고] Future dangerous detection 샘플이 없습니다.")
        return sample_df, pd.DataFrame()

    methods = {
        "Proposed Risk": "current_risk",
        "Delay Only": "current_delay",
        "Loss Only": "current_loss",
        "Utilization Only": "current_utilization",
        "Simple QoS Score": "current_simple_qos_score",
    }

    per_step_rows = []

    for (ep, t), g in sample_df.groupby(["episode", "time"], sort=False):
        g = g.reset_index(drop=True).copy()

        g["current_simple_qos_score"] = (
            minmax_normalize(g["current_delay"]) +
            minmax_normalize(g["current_loss"]) +
            minmax_normalize(g["current_utilization"])
        ) / 3.0

        future_high_delay = exact_top_k_mask(g["future_avg_delay"].values, top_ratio)
        future_high_loss = exact_top_k_mask(g["future_avg_loss"].values, top_ratio)
        future_high_util = exact_top_k_mask(g["future_avg_utilization"].values, top_ratio)
        future_high_risk = exact_top_k_mask(g["future_avg_risk"].values, top_ratio)
        future_congested = g["future_any_congested"].astype(bool).values

        actual_future_danger = (
            future_high_delay |
            future_high_loss |
            future_high_util |
            future_high_risk |
            future_congested
        )

        actual_danger_ratio = float(actual_future_danger.mean())

        for method_name, score_col in methods.items():
            predicted = exact_top_k_mask(g[score_col].values, top_ratio)
            m = compute_detection_metrics(predicted, actual_future_danger)

            per_step_rows.append({
                "episode": int(ep),
                "time": int(t),
                "method": method_name,
                "actual_future_danger_ratio": actual_danger_ratio,
                "predicted_ratio": float(predicted.mean()),
                **m,
            })

    step_df = pd.DataFrame(per_step_rows)

    summary = step_df.groupby("method", sort=False).agg(
        steps=("method", "count"),
        precision=("precision", "mean"),
        precision_std=("precision", "std"),
        recall=("recall", "mean"),
        recall_std=("recall", "std"),
        f1=("f1", "mean"),
        f1_std=("f1", "std"),
        accuracy=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        actual_future_danger_ratio=("actual_future_danger_ratio", "mean"),
        predicted_ratio=("predicted_ratio", "mean"),
        tp=("tp", "sum"),
        fp=("fp", "sum"),
        fn=("fn", "sum"),
        tn=("tn", "sum"),
    ).reset_index()

    global_precision = []
    global_recall = []
    global_f1 = []

    for _, r in summary.iterrows():
        tp = int(r["tp"])
        fp = int(r["fp"])
        fn = int(r["fn"])
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0
        global_precision.append(p)
        global_recall.append(rec)
        global_f1.append(f1)

    summary["global_precision"] = global_precision
    summary["global_recall"] = global_recall
    summary["global_f1"] = global_f1

    method_order = ["Proposed Risk", "Delay Only", "Loss Only", "Utilization Only", "Simple QoS Score"]
    summary["_order"] = summary["method"].apply(lambda x: method_order.index(x) if x in method_order else 999)
    summary = summary.sort_values(["_order"]).drop(columns=["_order"])

    step_df.to_csv(FUTURE_DETECTION_STEP_PATH, index=False, encoding="utf-8")
    summary.to_csv(FUTURE_DETECTION_SUMMARY_PATH, index=False, encoding="utf-8")

    print("Future detection step별 결과 저장:", FUTURE_DETECTION_STEP_PATH)
    print("Future detection 요약 저장:", FUTURE_DETECTION_SUMMARY_PATH)

    return step_df, summary


def print_future_detection_summary(summary):
    if summary.empty:
        return

    print("\n" + "=" * 126)
    print("CURRENT METRIC -> FUTURE DANGEROUS LINK DETECTION ACCURACY")
    print("=" * 126)
    print(
        f"{'Method':18s} | {'Precision':>10s} | {'Recall':>10s} | {'F1':>10s} | "
        f"{'Accuracy':>10s} | {'G.Prec':>10s} | {'G.Rec':>10s} | {'G.F1':>10s}"
    )
    print("-" * 126)

    for _, r in summary.iterrows():
        print(
            f"{str(r['method']):18s} | "
            f"{float(r['precision']):10.4f} | "
            f"{float(r['recall']):10.4f} | "
            f"{float(r['f1']):10.4f} | "
            f"{float(r['accuracy']):10.4f} | "
            f"{float(r['global_precision']):10.4f} | "
            f"{float(r['global_recall']):10.4f} | "
            f"{float(r['global_f1']):10.4f}"
        )

    print("-" * 126)
    print(f"Prediction: top {DETECTION_TOP_RATIO:.0%} links by current metric at time t")
    print(f"Ground truth: future dangerous links from t+1 to t+{FUTURE_WINDOW}")
    print("Future dangerous = future high delay OR high loss OR high utilization OR high risk OR any future congestion")


def save_future_detection_graphs(summary):
    if summary.empty:
        return

    os.makedirs(RESULT_DIR, exist_ok=True)

    methods = summary["method"].astype(str).tolist()
    x = np.arange(len(methods))

    graph_specs = [
        ("precision", "Future Dangerous Link Detection Precision", "Precision", "future_detection_precision.png"),
        ("recall", "Future Dangerous Link Detection Recall", "Recall", "future_detection_recall.png"),
        ("f1", "Future Dangerous Link Detection F1-score", "F1-score", "future_detection_f1.png"),
        ("accuracy", "Future Dangerous Link Detection Accuracy", "Accuracy", "future_detection_accuracy.png"),
    ]

    for col, title, ylabel, filename in graph_specs:
        plt.figure(figsize=(9, 5))
        plt.bar(x, summary[col].values)
        plt.xticks(x, methods, rotation=20, ha="right")
        plt.title(title)
        plt.xlabel("Current Metric Used for Prediction")
        plt.ylabel(ylabel)
        plt.ylim(0, 1.05)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        path = os.path.join(RESULT_DIR, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print("그래프 저장:", path)


def run_dynamic_risk_validation():
    print("\nRisk score 검증 실험 시작: Current Metric -> Future Dangerous Link Detection")
    print(f"Top ratio: {DETECTION_TOP_RATIO:.0%}")
    print(f"Future window: {FUTURE_WINDOW} steps")
    print(f"Validation episodes: {RISK_VALIDATION_EPISODES}, steps per episode: {RISK_VALIDATION_STEPS}")

    df = save_dynamic_link_log(
        episodes=RISK_VALIDATION_EPISODES,
        steps=RISK_VALIDATION_STEPS,
    )

    if df["risk"].max() <= 0:
        print("[주의] env.qos에서 읽은 risk가 모두 0입니다. NetworkEnv 또는 risk_calculator를 확인하세요.")

    _, summary = analyze_future_danger_detection(
        df,
        top_ratio=DETECTION_TOP_RATIO,
        future_window=FUTURE_WINDOW,
    )
    print_future_detection_summary(summary)
    save_future_detection_graphs(summary)

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("실험 시작: Multi-Scenario OSPF vs Risk-Only DQN")
    print("Scenarios:", SCENARIOS)
    print("Episodes per scenario:", EPISODES)
    print("Warmup steps:", WARMUP_STEPS)
    print("Model path:", RISK_ONLY_MODEL_PATH)

    all_results = []

    for scenario in SCENARIOS:
        ospf = evaluate_ospf(scenario=scenario, episodes=EPISODES)
        risk_only_dqn = evaluate_risk_only_dqn(
            model_path=RISK_ONLY_MODEL_PATH,
            scenario=scenario,
            method_name="Risk-Only-DQN",
            episodes=EPISODES,
        )

        scenario_results = [r for r in [ospf, risk_only_dqn] if r is not None]
        print_scenario_summary(scenario, scenario_results)
        all_results.append((scenario, scenario_results))

    print_final_summary(all_results)
    save_csv(all_results)
    save_all_graphs(all_results)

    run_dynamic_risk_validation()

    print("\n실험 완료. results 폴더 확인")


if __name__ == "__main__":
    main()
