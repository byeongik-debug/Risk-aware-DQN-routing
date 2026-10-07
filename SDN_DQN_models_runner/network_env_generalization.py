"""Generalization environment for unseen fixed-size link configurations.

This environment keeps the trained DQN input/output dimensions unchanged:
- 25 nodes
- 50 ordered link-risk values
- current/destination/visited one-hot vectors

The topology itself is loaded from JSON.  Dynamic scenario sets are derived
from paths that exist in each loaded topology, so the scenario remains valid
after rewiring.
"""

from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from risk_calculator import calculate_link_risk

RISK_WINDOW = 5
MAX_STEPS = 35
DYNAMIC_PHASE_PER_EPISODE = True

Edge = Tuple[int, int]


def normalize_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def path_to_edges(path: Sequence[int]) -> List[Edge]:
    return [normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]


class GeneralizationNetworkEnv:
    def __init__(
        self,
        topology_path: str | Path,
        use_risk: bool = True,
        scenario: str = "normal",
        seed: Optional[int] = None,
    ) -> None:
        self.topology_path = Path(topology_path)
        payload = json.loads(self.topology_path.read_text(encoding="utf-8"))

        self.name = str(payload.get("name", self.topology_path.stem))
        self.num_nodes = int(payload["num_nodes"])
        self.source = int(payload["source"])
        self.destination = int(payload["destination"])

        records = sorted(payload["edges"], key=lambda x: int(x["slot"]))
        self.edges: List[Edge] = [normalize_edge(int(r["u"]), int(r["v"])) for r in records]
        self.edge_types: Dict[Edge, str] = {
            normalize_edge(int(r["u"]), int(r["v"])): str(r.get("type", "normal"))
            for r in records
        }

        if len(set(self.edges)) != len(self.edges):
            raise ValueError(f"Duplicate edges in topology: {self.topology_path}")
        if int(payload.get("num_edges", len(self.edges))) != len(self.edges):
            raise ValueError("num_edges metadata does not match edge records")

        self.use_risk = use_risk
        self.scenario = scenario
        self.seed = seed
        self.rng = random.Random(seed)
        self.timestep = 0
        self.dynamic_phase = 0

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(self.num_nodes))
        self.graph.add_edges_from(self.edges)
        if not nx.is_connected(self.graph):
            raise ValueError(f"Topology is disconnected: {self.topology_path}")

        self.qos: Dict[Edge, Dict[str, object]] = {}
        self.loss_history = {e: deque(maxlen=RISK_WINDOW) for e in self.edges}
        self.delay_history = {e: deque(maxlen=RISK_WINDOW) for e in self.edges}

        self.current_node = self.source
        self.visited: List[int] = []

        self.state_size = len(self.edges) + self.num_nodes * 3
        self.action_size = self.num_nodes

        self.scenario_paths = self._build_candidate_paths()
        self.congestion_links, self.unstable_links = self._build_static_danger_sets()
        self.dynamic_phases = self._build_dynamic_phases()

        self.init_qos()

    def _build_candidate_paths(self, max_paths: int = 6) -> List[List[int]]:
        paths: List[List[int]] = []
        try:
            generator = nx.shortest_simple_paths(self.graph, self.source, self.destination)
            for path in generator:
                if len(path) - 1 > 16:
                    break
                paths.append(path)
                if len(paths) >= max_paths:
                    break
        except nx.NetworkXNoPath as exc:
            raise ValueError("No source-destination path in topology") from exc

        if not paths:
            raise ValueError("No candidate paths found")
        return paths

    def _build_static_danger_sets(self) -> Tuple[set[Edge], set[Edge]]:
        # Use edges from several short candidate paths.  This mirrors the
        # original environment where multiple frequently selected routes can
        # become congested or unstable.
        congestion: set[Edge] = set()
        unstable: set[Edge] = set()
        for path in self.scenario_paths[:4]:
            edges = path_to_edges(path)
            congestion.update(edges)
            unstable.update(edges)
        return congestion, unstable

    def _build_dynamic_phases(self) -> Dict[int, Dict[str, set[Edge]]]:
        paths = self.scenario_paths
        while len(paths) < 3:
            paths.append(paths[-1])

        path_sets = [set(path_to_edges(p)) for p in paths[:3]]
        phases: Dict[int, Dict[str, set[Edge]]] = {}
        for i in range(3):
            support = path_sets[i]
            danger = set().union(*(path_sets[j] for j in range(3) if j != i))
            danger -= support
            phases[i] = {"danger": danger, "support": support}
        return phases

    def get_link_type(self, edge: Edge) -> str:
        return self.edge_types.get(normalize_edge(*edge), "normal")

    def init_qos(self) -> None:
        self.qos = {}
        for edge in self.edges:
            link_type = self.get_link_type(edge)

            if link_type == "backbone":
                delay = self.rng.uniform(5, 10)
                loss = self.rng.uniform(0.003, 0.015)
                bandwidth = self.rng.uniform(800, 1000)
            elif link_type == "backup":
                delay = self.rng.uniform(15, 25)
                loss = self.rng.uniform(0.001, 0.010)
                bandwidth = self.rng.uniform(400, 700)
            else:
                delay = self.rng.uniform(10, 20)
                loss = self.rng.uniform(0.005, 0.020)
                bandwidth = self.rng.uniform(400, 800)

            capacity = bandwidth
            traffic = self.rng.uniform(0.10, 0.35) * capacity
            utilization = traffic / max(capacity, 1e-6)
            is_congested = utilization >= 0.80

            self.loss_history[edge].clear()
            self.delay_history[edge].clear()
            self.loss_history[edge].append(loss)
            self.delay_history[edge].append(delay)

            risk = calculate_link_risk(
                utilization=utilization,
                is_congested=is_congested,
                delay_history=self.delay_history[edge],
                loss_history=self.loss_history[edge],
                link_type=link_type,
            )

            self.qos[edge] = {
                "delay": float(delay),
                "loss": float(loss),
                "bandwidth": float(bandwidth),
                "capacity": float(capacity),
                "traffic": float(traffic),
                "utilization": float(utilization),
                "is_congested": bool(is_congested),
                "risk": float(risk),
                "type": link_type,
            }

    def update_dynamic_qos(self) -> None:
        self.timestep += 1

        dynamic_danger: set[Edge] = set()
        dynamic_support: set[Edge] = set()
        if self.scenario == "dynamic":
            if not DYNAMIC_PHASE_PER_EPISODE:
                self.dynamic_phase = (self.timestep // 8) % len(self.dynamic_phases)
            dynamic_danger = self.dynamic_phases[self.dynamic_phase]["danger"]
            dynamic_support = self.dynamic_phases[self.dynamic_phase]["support"]

        for edge in self.edges:
            q = self.qos[edge]
            link_type = str(q["type"])
            capacity = float(q["capacity"])

            delay_noise = self.rng.uniform(-1.0, 1.0)
            loss_noise = self.rng.uniform(-0.002, 0.002)
            bw_noise = self.rng.uniform(-5.0, 15.0)
            traffic_ratio = self.rng.uniform(0.15, 0.45)
            risk_type = link_type

            if self.scenario == "congestion" and edge in self.congestion_links:
                traffic_ratio = self.rng.uniform(1.10, 1.55)
                delay_noise = self.rng.uniform(5.0, 10.0)
                loss_noise = self.rng.uniform(0.012, 0.050)
                bw_noise = self.rng.uniform(0.0, 30.0)
                risk_type = "congested"
            elif self.scenario == "unstable" and edge in self.unstable_links:
                traffic_ratio = self.rng.uniform(0.55, 1.10)
                delay_noise = self.rng.uniform(-5.0, 10.0)
                loss_noise = self.rng.uniform(-0.010, 0.055)
                bw_noise = self.rng.uniform(0.0, 25.0)
                risk_type = "unstable"
            elif self.scenario == "dynamic" and edge in dynamic_danger:
                traffic_ratio = self.rng.uniform(1.10, 1.55)
                delay_noise = self.rng.uniform(5.0, 10.0) + self.rng.uniform(-5.0, 10.0)
                loss_noise = self.rng.uniform(0.012, 0.050) + self.rng.uniform(-0.010, 0.055)
                bw_noise = self.rng.uniform(0.0, 30.0)
                risk_type = "unstable_congested"

            if link_type == "backbone":
                delay_noise *= 0.6
                loss_noise *= 0.6
                traffic_ratio *= 0.95
            elif link_type == "backup":
                delay_noise *= 0.75
                loss_noise *= 0.6
                traffic_ratio *= 0.75

            q["traffic"] = float(np.clip(traffic_ratio * capacity, 0.0, capacity * 1.5))
            q["utilization"] = float(np.clip(float(q["traffic"]) / max(capacity, 1e-6), 0.0, 1.5))
            q["delay"] = float(np.clip(float(q["delay"]) + delay_noise + float(q["utilization"]) * 2.0, 3.0, 120.0))
            q["loss"] = float(
                np.clip(
                    float(q["loss"]) + loss_noise + max(0.0, float(q["utilization"]) - 0.8) * 0.03,
                    0.0,
                    0.35,
                )
            )
            q["bandwidth"] = float(np.clip(float(q["bandwidth"]) + bw_noise, 50.0, 1000.0))
            q["is_congested"] = bool(float(q["utilization"]) >= 0.8)

            self.loss_history[edge].append(float(q["loss"]))
            self.delay_history[edge].append(float(q["delay"]))
            q["risk"] = calculate_link_risk(
                utilization=float(q["utilization"]),
                is_congested=bool(q["is_congested"]),
                delay_history=self.delay_history[edge],
                loss_history=self.loss_history[edge],
                link_type=risk_type,
            )

            if self.scenario in ("congestion", "unstable") and link_type == "backup":
                q["risk"] = float(np.clip(float(q["risk"]) * 0.60, 0.0, 1.0))
            if self.scenario == "dynamic" and edge in dynamic_support:
                q["risk"] = float(np.clip(float(q["risk"]) * 0.55, 0.0, 1.0))

    def reset(self) -> np.ndarray:
        self.current_node = self.source
        self.visited = [self.source]
        self.timestep = 0
        self.dynamic_phase = self.rng.choice(list(self.dynamic_phases)) if self.scenario == "dynamic" else 0
        self.init_qos()
        self.update_dynamic_qos()
        return self.get_state()

    def get_valid_actions(self, node: Optional[int] = None) -> List[int]:
        target = self.current_node if node is None else node
        return list(self.graph.neighbors(target))

    def shortest_distance_to_destination(self, node: int) -> int:
        try:
            return nx.shortest_path_length(self.graph, node, self.destination)
        except nx.NetworkXNoPath:
            return 999

    def get_state(self) -> np.ndarray:
        link_risks = [float(self.qos[e]["risk"]) for e in self.edges]

        curr = np.zeros(self.num_nodes, dtype=np.float32)
        curr[self.current_node] = 1.0
        dest = np.zeros(self.num_nodes, dtype=np.float32)
        dest[self.destination] = 1.0
        visited = np.zeros(self.num_nodes, dtype=np.float32)
        for node in self.visited:
            visited[node] = 1.0

        return np.concatenate([link_risks, curr, dest, visited]).astype(np.float32)

    def step(self, action: int):
        self.update_dynamic_qos()
        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            return self.get_state(), -10.0, True, {
                "reason": "invalid",
                "path": self.visited.copy(),
                "dynamic_phase": self.dynamic_phase,
                "next_valid_mask": np.zeros(self.num_nodes, dtype=np.float32),
            }

        edge = normalize_edge(self.current_node, action)
        q = self.qos[edge]
        prev_dist = self.shortest_distance_to_destination(self.current_node)
        next_dist = self.shortest_distance_to_destination(action)
        risk = float(q["risk"])

        if self.scenario == "normal":
            reward = -risk * 0.7 + (2.0 if next_dist < prev_dist else -2.0) - 0.10
        elif self.scenario == "congestion":
            reward = -risk * 10.0 + (0.7 if next_dist < prev_dist else -0.7) - 0.06
            if bool(q["is_congested"]):
                reward -= 4.0
        elif self.scenario == "unstable":
            reward = -risk * 10.0 + (0.7 if next_dist < prev_dist else -0.7) - 0.06
            if risk >= 0.22:
                reward -= 3.0
        elif self.scenario == "dynamic":
            reward = -risk * 22.0 + (0.35 if next_dist < prev_dist else -0.35) - 0.08
            if bool(q["is_congested"]):
                reward -= 5.0
            if risk >= 0.25:
                reward -= 5.0
        else:
            reward = -risk * 5.0 + (1.0 if next_dist < prev_dist else -1.0) - 0.05

        if action in self.visited:
            reward -= 8.0

        self.current_node = action
        self.visited.append(action)
        done = False

        if self.current_node == self.destination:
            metrics = self.calculate_path_metrics(self.visited)
            reward += 45.0 - metrics["total_risk"] * 6.0
            done = True
        elif len(self.visited) >= MAX_STEPS:
            reward -= 15.0
            done = True

        next_mask = np.zeros(self.num_nodes, dtype=np.float32)
        for node in self.get_valid_actions(self.current_node):
            next_mask[node] = 1.0

        return self.get_state(), float(reward), done, {
            "reason": "arrived" if self.current_node == self.destination else "continue",
            "path": self.visited.copy(),
            "dynamic_phase": self.dynamic_phase,
            "next_valid_mask": next_mask,
        }

    def calculate_path_metrics(self, path: Sequence[int]) -> Dict[str, float | int | List[int]]:
        total_delay = 0.0
        total_loss = 0.0
        total_risk = 0.0
        min_bandwidth = float("inf")
        unstable_count = 0
        congested_count = 0

        dynamic_danger = self.dynamic_phases[self.dynamic_phase]["danger"] if self.scenario == "dynamic" else set()
        hops = max(len(path) - 1, 1)

        for u, v in zip(path[:-1], path[1:]):
            edge = normalize_edge(u, v)
            if edge not in self.qos:
                continue
            q = self.qos[edge]
            total_delay += float(q["delay"])
            total_loss += float(q["loss"])
            total_risk += float(q["risk"])
            min_bandwidth = min(min_bandwidth, float(q["bandwidth"]))

            if self.scenario == "unstable" and edge in self.unstable_links:
                unstable_count += 1
            elif self.scenario == "congestion" and edge in self.congestion_links:
                congested_count += 1
            elif self.scenario == "dynamic" and edge in dynamic_danger:
                unstable_count += 1
                congested_count += 1

        return {
            "path": list(path),
            "hop_count": len(path) - 1,
            "total_delay": total_delay,
            "total_loss": total_loss / hops,
            "total_risk": total_risk / hops,
            "min_bandwidth": 0.0 if min_bandwidth == float("inf") else min_bandwidth,
            "unstable_count": unstable_count,
            "congested_count": congested_count,
            "dynamic_phase": self.dynamic_phase,
        }

    def ospf_path(self, cost_type: str = "bandwidth") -> List[int]:
        graph = nx.Graph()
        for edge in self.edges:
            q = self.qos[edge]
            if cost_type == "delay":
                cost = float(q["delay"])
            elif cost_type == "hop":
                cost = 1.0
            elif cost_type == "bandwidth":
                cost = 1000.0 / max(float(q["bandwidth"]), 1e-6)
            elif cost_type == "risk":
                cost = float(q["risk"])
            else:
                raise ValueError(f"Unsupported cost_type: {cost_type}")
            graph.add_edge(edge[0], edge[1], weight=cost)

        return nx.shortest_path(graph, self.source, self.destination, weight="weight")
