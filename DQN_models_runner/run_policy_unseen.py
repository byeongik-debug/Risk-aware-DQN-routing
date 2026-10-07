# run_policy.py
# Multi-Scenario Evaluation: OSPF vs Risk-Only DQN

import os
import random
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from network_env_unseen import NetworkEnv
from topology import DESTINATION


SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EPISODES = 200
WARMUP_STEPS = 10
DEBUG_PATH = True

RISK_ONLY_MODEL_PATH = "risk_only_model_final.pth"
RESULT_DIR = "results"

# 신경망 구조
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

    for ep in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        warmup_qos(env)

        path = env.ospf_path(cost_type="bandwidth")
        metrics = env.calculate_path_metrics(path)

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

    print("\n실험 완료. results 폴더 확인")


if __name__ == "__main__":
    main()
