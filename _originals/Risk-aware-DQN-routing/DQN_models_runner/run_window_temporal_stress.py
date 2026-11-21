# run_window_temporal_stress.py
# Window-size sensitivity under a dedicated temporal-stress environment
#
# Window 1, 5, 10을 각각 처음부터 학습하고 동일한 temporal-stress 환경에서 평가한다.
#
# Required:
#   network_env.py
#   network_env_temporal_stress.py
#   risk_calculator.py
#   topology.py
#
# Run:
#   python run_window_temporal_stress.py

from __future__ import annotations

import csv
import random
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import network_env
from network_env_temporal_stress import TemporalStressEnv
from topology import DESTINATION, NUM_NODES


WINDOW_SIZES = [1, 5, 10]
TRAIN_SEEDS = [0, 1, 2]
EPISODES = 8000

GAMMA = 0.99
LEARNING_RATE = 0.0003
BATCH_SIZE = 64
MEMORY_SIZE = 40000
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9994
TARGET_UPDATE = 50

EVAL_EPISODES = 200
WARMUP_STEPS = 10
EVAL_SEED_BASE = 200000

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_risk_window(window_size: int) -> None:
    network_env.RISK_WINDOW = int(window_size)


def build_action_mask(env: TemporalStressEnv) -> np.ndarray:
    mask = np.zeros(NUM_NODES, dtype=np.float32)

    for action in env.get_valid_actions():
        mask[action] = 1.0

    return mask


def warmup(env: TemporalStressEnv, steps: int = WARMUP_STEPS) -> None:
    for _ in range(steps):
        env.update_dynamic_qos()


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, next_mask) -> None:
        self.buffer.append(
            (state, action, reward, next_state, done, next_mask)
        )

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, next_masks = zip(*batch)

        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.as_tensor(dones, dtype=torch.float32),
            torch.as_tensor(np.asarray(next_masks), dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
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


class DQNAgent:
    def __init__(self, state_size: int, action_size: int, device) -> None:
        self.device = device
        self.policy_net = QNetwork(state_size, action_size).to(device)
        self.target_net = QNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=LEARNING_RATE,
        )
        self.memory = ReplayBuffer(MEMORY_SIZE)
        self.epsilon = EPSILON_START

    def select_action(
        self,
        state: np.ndarray,
        valid_actions: List[int],
        explore: bool = True,
    ) -> Optional[int]:
        if not valid_actions:
            return None

        if explore and random.random() < self.epsilon:
            return random.choice(valid_actions)

        state_tensor = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)[0]

        masked_q = torch.full_like(q_values, -1e9)
        valid_tensor = torch.as_tensor(
            valid_actions,
            dtype=torch.long,
            device=self.device,
        )
        masked_q[valid_tensor] = q_values[valid_tensor]

        return int(torch.argmax(masked_q).item())

    def train_step(self) -> float:
        if len(self.memory) < BATCH_SIZE:
            return 0.0

        states, actions, rewards, next_states, dones, next_masks = (
            self.memory.sample(BATCH_SIZE)
        )

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_masks = next_masks.to(self.device)

        current_q = self.policy_net(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            masked_next_q = next_q_values + (1.0 - next_masks) * -1e9
            next_q = masked_next_q.max(dim=1)[0]
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.HuberLoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=1.0,
        )
        self.optimizer.step()

        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            EPSILON_END,
            self.epsilon * EPSILON_DECAY,
        )


def train_one_model(window_size: int, train_seed: int, device) -> Path:
    set_risk_window(window_size)
    set_seed(train_seed)

    base_env = TemporalStressEnv()
    agent = DQNAgent(
        base_env.state_size,
        base_env.action_size,
        device,
    )

    recent_success = deque(maxlen=100)
    recent_reward = deque(maxlen=100)
    recent_volatile = deque(maxlen=100)

    print("\n" + "=" * 96)
    print(
        f"TEMPORAL-STRESS TRAINING | window={window_size} "
        f"| seed={train_seed} | episodes={EPISODES}"
    )
    print("=" * 96)

    for episode in range(1, EPISODES + 1):
        env = TemporalStressEnv()
        state = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(
                state,
                valid_actions,
                explore=True,
            )

            if action is None:
                break

            next_state, reward, done, info = env.step(action)
            next_mask = info.get(
                "next_valid_mask",
                build_action_mask(env),
            )

            agent.memory.push(
                state,
                action,
                reward,
                next_state,
                done,
                next_mask,
            )

            agent.train_step()
            state = next_state
            episode_reward += reward

        metrics = env.calculate_path_metrics(env.visited)
        recent_success.append(int(env.current_node == DESTINATION))
        recent_reward.append(episode_reward)
        recent_volatile.append(metrics["volatile_count"])

        agent.decay_epsilon()

        if episode % TARGET_UPDATE == 0:
            agent.update_target()

        if episode % 500 == 0:
            print(
                f"window={window_size:2d} | seed={train_seed:2d} | "
                f"ep={episode:5d} | "
                f"success={np.mean(recent_success) * 100:6.2f}% | "
                f"reward={np.mean(recent_reward):8.2f} | "
                f"volatile-links={np.mean(recent_volatile):6.2f} | "
                f"epsilon={agent.epsilon:.3f}"
            )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = (
        MODEL_DIR
        / f"temporal_stress_w{window_size}_seed{train_seed}.pth"
    )
    torch.save(agent.policy_net.state_dict(), model_path)
    print("Model saved:", model_path)

    return model_path


def evaluate(
    model_path: Path,
    window_size: int,
    train_seed: int,
    device,
) -> Dict[str, float]:
    set_risk_window(window_size)

    env_for_shape = TemporalStressEnv()
    model = QNetwork(
        env_for_shape.state_size,
        env_for_shape.action_size,
    ).to(device)

    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )
    model.eval()

    success = 0
    delays = []
    losses = []
    risks = []
    hops = []
    volatile_ratios = []
    volatile_path_usage = []
    representative_path = None

    for episode in range(EVAL_EPISODES):
        set_seed(EVAL_SEED_BASE + episode)

        env = TemporalStressEnv()
        state = env.reset()
        warmup(env)
        state = env.get_state()
        done = False

        while not done:
            valid_actions = env.get_valid_actions()

            if not valid_actions:
                break

            state_tensor = torch.as_tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state_tensor)[0]

            masked_q = torch.full_like(q_values, -1e9)
            valid_tensor = torch.as_tensor(
                valid_actions,
                dtype=torch.long,
                device=device,
            )
            masked_q[valid_tensor] = q_values[valid_tensor]
            action = int(torch.argmax(masked_q).item())

            state, _, done, _ = env.step(action)

        metrics = env.calculate_path_metrics(env.visited)

        if representative_path is None:
            representative_path = list(env.visited)

        if env.current_node == DESTINATION:
            success += 1
            delays.append(float(metrics["total_delay"]))
            losses.append(float(metrics["total_loss"]))
            risks.append(float(metrics["total_risk"]))
            hops.append(float(metrics["hop_count"]))

            hop_count = max(int(metrics["hop_count"]), 1)
            volatile_ratios.append(
                metrics["volatile_count"] / hop_count * 100.0
            )
            volatile_path_usage.append(
                float(metrics["used_volatile_path"])
            )

    def mean(values):
        return float(np.mean(values)) if values else float("nan")

    result = {
        "window": float(window_size),
        "train_seed": float(train_seed),
        "success_rate": success / EVAL_EPISODES * 100.0,
        "delay": mean(delays),
        "loss": mean(losses),
        "risk": mean(risks),
        "hop": mean(hops),
        "volatile_link_ratio": mean(volatile_ratios),
        "volatile_path_usage": mean(volatile_path_usage) * 100.0,
        "representative_path": str(representative_path),
    }

    print(
        f"\nEvaluation | window={window_size} | seed={train_seed} | "
        f"success={result['success_rate']:.2f}% | "
        f"delay={result['delay']:.3f} | "
        f"loss={result['loss']:.5f} | "
        f"risk={result['risk']:.5f} | "
        f"hop={result['hop']:.2f} | "
        f"volatile-link={result['volatile_link_ratio']:.2f}% | "
        f"volatile-path-use={result['volatile_path_usage']:.2f}%"
    )
    print("Representative path:", representative_path)

    return result


def summarize(rows: List[Dict[str, float]]):
    numeric_metrics = [
        "success_rate",
        "delay",
        "loss",
        "risk",
        "hop",
        "volatile_link_ratio",
        "volatile_path_usage",
    ]

    summary_rows = []

    for window_size in WINDOW_SIZES:
        subset = [
            row for row in rows
            if int(row["window"]) == window_size
        ]

        summary = {
            "window": window_size,
            "num_train_seeds": len(subset),
        }

        for metric in numeric_metrics:
            values = np.asarray(
                [float(row[metric]) for row in subset],
                dtype=float,
            )

            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        summary_rows.append(summary)

    return summary_rows


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", path)


def print_summary(rows: List[Dict[str, float]]) -> None:
    print("\n" + "=" * 142)
    print("TEMPORAL-STRESS WINDOW SENSITIVITY (MEAN ± STD ACROSS TRAINING SEEDS)")
    print("=" * 142)
    print(
        f"{'Window':>6s} | "
        f"{'Success':>17s} | "
        f"{'Delay':>19s} | "
        f"{'Loss':>19s} | "
        f"{'Risk':>19s} | "
        f"{'Hop':>17s} | "
        f"{'Volatile Link':>19s} | "
        f"{'Volatile Path Use':>20s}"
    )
    print("-" * 142)

    for row in rows:
        print(
            f"{int(row['window']):6d} | "
            f"{row['success_rate_mean']:7.2f}±{row['success_rate_std']:7.2f} | "
            f"{row['delay_mean']:8.3f}±{row['delay_std']:8.3f} | "
            f"{row['loss_mean']:8.5f}±{row['loss_std']:8.5f} | "
            f"{row['risk_mean']:8.5f}±{row['risk_std']:8.5f} | "
            f"{row['hop_mean']:7.2f}±{row['hop_std']:7.2f} | "
            f"{row['volatile_link_ratio_mean']:8.2f}±{row['volatile_link_ratio_std']:8.2f} | "
            f"{row['volatile_path_usage_mean']:8.2f}±{row['volatile_path_usage_std']:8.2f}"
        )

    print("-" * 142)


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Windows:", WINDOW_SIZES)
    print("Seeds:", TRAIN_SEEDS)
    print("Episodes per model:", EPISODES)
    print("Evaluation episodes:", EVAL_EPISODES)

    all_rows = []

    for window_size in WINDOW_SIZES:
        for train_seed in TRAIN_SEEDS:
            model_path = train_one_model(
                window_size,
                train_seed,
                device,
            )

            result = evaluate(
                model_path,
                window_size,
                train_seed,
                device,
            )
            all_rows.append(result)

    summary_rows = summarize(all_rows)

    save_csv(
        RESULT_DIR / "temporal_stress_by_seed.csv",
        all_rows,
    )
    save_csv(
        RESULT_DIR / "temporal_stress_summary.csv",
        summary_rows,
    )

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
