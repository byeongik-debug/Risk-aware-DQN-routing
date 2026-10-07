# network_env_risk_validation.py
# Risk score validation environment
# Purpose:
#   This file is NOT for DQN training.
#   It creates diverse dangerous-link patterns to verify whether the proposed
#   calculate_link_risk() score can identify dangerous links better than
#   single QoS metrics such as delay, loss, or utilization.
#
# Retraining note:
#   Do not replace your original network_env.py with this file for main DQN
#   experiments. Use this only for risk-index validation. No DQN retraining is
#   required when this file is used only by run_risk_identification_validation.py.

import random
from collections import deque

import numpy as np

from topology import EDGES, EDGE_TYPES
from risk_calculator import calculate_link_risk


RISK_WINDOW = 5

# Four different danger categories are intentionally separated.
# This prevents delay-only, loss-only, or utilization-only metrics from becoming
# identical to the proposed risk score.
DANGER_TYPES = [
    "delay_spike",        # high delay, relatively normal loss/utilization
    "loss_spike",         # high packet loss, relatively normal delay/utilization
    "congestion",         # high utilization/congestion
    "temporal_volatility", # large temporal variation with moderate averages
]


class RiskValidationEnv:
    def __init__(self, seed=None, danger_ratio=0.20):
        self.seed = seed
        self.danger_ratio = danger_ratio
        self.timestep = 0
        self.qos = {}
        self.delay_history = {}
        self.loss_history = {}
        self.danger_map = {}
        self.danger_type_map = {}

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        for edge in EDGES:
            e = self.normalize_edge(*edge)
            self.delay_history[e] = deque(maxlen=RISK_WINDOW)
            self.loss_history[e] = deque(maxlen=RISK_WINDOW)

        self.reset()

    def normalize_edge(self, u, v):
        return tuple(sorted((u, v)))

    def get_link_type(self, edge):
        e = self.normalize_edge(*edge)
        return EDGE_TYPES.get(e, EDGE_TYPES.get((edge[1], edge[0]), "normal"))

    def reset(self):
        self.timestep = 0
        self.qos = {}
        self.danger_map = {}
        self.danger_type_map = {}

        normalized_edges = [self.normalize_edge(*e) for e in EDGES]
        n_danger = max(1, int(len(normalized_edges) * self.danger_ratio))
        danger_edges = set(random.sample(normalized_edges, n_danger))

        # Assign danger types evenly across selected dangerous links.
        for i, e in enumerate(danger_edges):
            self.danger_map[e] = 1
            self.danger_type_map[e] = DANGER_TYPES[i % len(DANGER_TYPES)]

        for e in normalized_edges:
            if e not in self.danger_map:
                self.danger_map[e] = 0
                self.danger_type_map[e] = "normal"

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            link_type = self.get_link_type(e)

            if link_type == "backbone":
                delay = random.uniform(5, 10)
                loss = random.uniform(0.003, 0.015)
                bandwidth = random.uniform(800, 1000)
            elif link_type == "backup":
                delay = random.uniform(15, 25)
                loss = random.uniform(0.001, 0.010)
                bandwidth = random.uniform(400, 700)
            else:
                delay = random.uniform(10, 20)
                loss = random.uniform(0.005, 0.020)
                bandwidth = random.uniform(400, 800)

            capacity = bandwidth
            traffic = random.uniform(0.10, 0.35) * capacity
            utilization = traffic / max(capacity, 1e-6)
            is_congested = utilization >= 0.80

            self.delay_history[e].clear()
            self.loss_history[e].clear()
            self.delay_history[e].append(delay)
            self.loss_history[e].append(loss)

            self.qos[e] = {
                "delay": float(delay),
                "loss": float(loss),
                "bandwidth": float(bandwidth),
                "capacity": float(capacity),
                "traffic": float(traffic),
                "utilization": float(utilization),
                "is_congested": bool(is_congested),
                "risk": 0.0,
                "type": link_type,
                "danger_label": int(self.danger_map[e]),
                "danger_type": self.danger_type_map[e],
            }

            self.qos[e]["risk"] = calculate_link_risk(
                utilization=utilization,
                is_congested=is_congested,
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=link_type,
            )

        return self.snapshot()

    def step(self):
        self.timestep += 1

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]
            capacity = q["capacity"]
            base_type = q["type"]
            danger_type = q["danger_type"]

            # Normal background fluctuations.
            delay_noise = random.uniform(-1.0, 1.0)
            loss_noise = random.uniform(-0.002, 0.002)
            traffic_ratio = random.uniform(0.15, 0.45)
            risk_type = base_type

            # Inject separated danger mechanisms.
            if danger_type == "delay_spike":
                delay_noise += random.uniform(8.0, 18.0)
                loss_noise += random.uniform(-0.001, 0.004)
                traffic_ratio = random.uniform(0.25, 0.55)
                risk_type = "unstable"

            elif danger_type == "loss_spike":
                delay_noise += random.uniform(-1.0, 3.0)
                loss_noise += random.uniform(0.030, 0.090)
                traffic_ratio = random.uniform(0.25, 0.60)
                risk_type = "unstable"

            elif danger_type == "congestion":
                delay_noise += random.uniform(3.0, 8.0)
                loss_noise += random.uniform(0.005, 0.030)
                traffic_ratio = random.uniform(1.05, 1.50)
                risk_type = "congested"

            elif danger_type == "temporal_volatility":
                # Alternates between good and bad states. Average delay/loss may not
                # always be the highest, but temporal history should be unstable.
                sign = 1.0 if (self.timestep % 2 == 0) else -1.0
                delay_noise += sign * random.uniform(8.0, 16.0)
                loss_noise += sign * random.uniform(0.020, 0.060)
                traffic_ratio = random.uniform(0.35, 0.95)
                risk_type = "unstable"

            if base_type == "backbone":
                delay_noise *= 0.7
                loss_noise *= 0.7
                traffic_ratio *= 0.95
            elif base_type == "backup":
                delay_noise *= 0.8
                loss_noise *= 0.7
                traffic_ratio *= 0.8

            q["traffic"] = float(np.clip(traffic_ratio * capacity, 0.0, capacity * 1.5))
            q["utilization"] = float(np.clip(q["traffic"] / max(capacity, 1e-6), 0.0, 1.5))
            q["delay"] = float(np.clip(q["delay"] + delay_noise + q["utilization"] * 1.2, 3.0, 120.0))
            q["loss"] = float(np.clip(q["loss"] + loss_noise + max(0.0, q["utilization"] - 0.8) * 0.02, 0.0, 0.35))
            q["is_congested"] = bool(q["utilization"] >= 0.80)

            self.delay_history[e].append(q["delay"])
            self.loss_history[e].append(q["loss"])

            q["risk"] = float(calculate_link_risk(
                utilization=q["utilization"],
                is_congested=q["is_congested"],
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=risk_type,
            ))

        return self.snapshot()

    def snapshot(self):
        rows = []
        for (u, v), q in self.qos.items():
            rows.append({
                "time": self.timestep,
                "src": u,
                "dst": v,
                "delay": float(q["delay"]),
                "loss": float(q["loss"]),
                "utilization": float(q["utilization"]),
                "is_congested": int(q["is_congested"]),
                "risk": float(q["risk"]),
                "danger_label": int(q["danger_label"]),
                "danger_type": q["danger_type"],
            })
        return rows
