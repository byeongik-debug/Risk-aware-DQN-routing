# network_env.py
# 25-node Risk-Only DQN Environment
# 목표:
# Normal      : OSPF와 Risk-DQN 수치가 비슷
# Congestion  : OSPF는 bandwidth 기준으로 위험 링크 선택, Risk-DQN은 우회
# Unstable    : OSPF는 불안정 링크 사용률 높음, Risk-DQN은 안전 경로 우회
# Dynamic     : OSPF는 위험 구간 통과, Risk-DQN은 SAFE_PATH 쪽 우회

import random
from collections import deque

import networkx as nx
import numpy as np

from topology_unseen import NUM_NODES, SOURCE, DESTINATION, EDGES, EDGE_TYPES
from risk_calculator import calculate_link_risk

# window size 지정 -> 5개의 값을 봄
RISK_WINDOW = 5
MAX_STEPS = 35

# 혼잡 링크 구성 => 실험을 통한 경험적 근거를 통해 구현
CONGESTION_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]

# 불안정성 링크 구성 -> 실험을 통한 경험적 근거를 통해 구현
UNSTABLE_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]

# Dynamic만 수정:
# Risk-DQN이 타던 6-hop 경로
# 0 -> 1 -> 5 -> 11 -> 17 -> 22 -> 24
# 를 위험 후보에 포함
DYNAMIC_PHASES = {
    0: [(0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24)],
    1: [(0, 1), (1, 5), (5, 11), (11, 17), (17, 22)],
}

# safe 경로를 통해 risk 모델이 우회 가능한 경로 생성
SAFE_LINKS = [
    (4, 8),
    (8, 13),
    (13, 19),
    (19, 20),
    (20, 21),
    (21, 22),
    (22, 24),
]


class NetworkEnv:
    def __init__(self, use_risk=True, scenario="normal"):
        self.use_risk = use_risk
        self.scenario = scenario
        self.timestep = 0

        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(NUM_NODES))
        self.graph.add_edges_from(EDGES)

        self.qos = {}
        self.loss_history = {}
        self.delay_history = {}

        self.current_node = SOURCE
        self.visited = []

        self.state_size = len(EDGES) + NUM_NODES * 3
        self.action_size = NUM_NODES

        for edge in EDGES:
            e = self.normalize_edge(*edge)
            self.loss_history[e] = deque(maxlen=RISK_WINDOW)
            self.delay_history[e] = deque(maxlen=RISK_WINDOW)

        self.init_qos()

    def normalize_edge(self, u, v):
        return tuple(sorted((u, v)))

    def get_link_type(self, edge):
        e = self.normalize_edge(*edge)
        return EDGE_TYPES.get(
            e,
            EDGE_TYPES.get((edge[1], edge[0]), "normal")
        )

    def init_qos(self):
        self.qos = {}

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
            }

            self.loss_history[e].clear()
            self.delay_history[e].clear()
            self.loss_history[e].append(loss)
            self.delay_history[e].append(delay)

            self.qos[e]["risk"] = calculate_link_risk(
                utilization=utilization,
                is_congested=is_congested,
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=link_type,
            )

    def update_dynamic_qos(self):
        self.timestep += 1

        congestion_set = {
            self.normalize_edge(*x)
            for x in CONGESTION_LINKS
        }

        unstable_set = {
            self.normalize_edge(*x)
            for x in UNSTABLE_LINKS
        }

        safe_set = {
            self.normalize_edge(*x)
            for x in SAFE_LINKS
        }

        phase_keys = list(DYNAMIC_PHASES.keys())
        phase = phase_keys[(self.timestep // 5) % len(phase_keys)]

        dynamic_set = {
            self.normalize_edge(*x)
            for x in DYNAMIC_PHASES[phase]
        }

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]

            link_type = q["type"]
            capacity = q["capacity"]

            # 랜덤으로 값 생성
            delay_noise = random.uniform(-1.0, 1.0)
            loss_noise = random.uniform(-0.002, 0.002)
            bw_noise = random.uniform(-5.0, 15.0)
            traffic_ratio = random.uniform(0.15, 0.45)
            risk_type = link_type

            # 시나리오별 변수 생성
            if self.scenario == "congestion" and e in congestion_set:
                traffic_ratio = random.uniform(1.10, 1.55)
                delay_noise = random.uniform(5.0, 10.0)
                loss_noise = random.uniform(0.012, 0.050)
                bw_noise = random.uniform(0.0, 30.0)
                risk_type = "congested"

            elif self.scenario == "unstable" and e in unstable_set:
                traffic_ratio = random.uniform(0.55, 1.10)
                delay_noise = random.uniform(-5.0, 10.0)
                loss_noise = random.uniform(-0.010, 0.055)
                bw_noise = random.uniform(0.0, 25.0)
                risk_type = "unstable"

            elif self.scenario == "dynamic" and e in dynamic_set:
                traffic_ratio = random.uniform(1.10, 1.55)
                delay_noise = random.uniform(5.0, 10.0) + random.uniform(-5.0, 10.0)
                loss_noise = random.uniform(0.012, 0.050) + random.uniform(-0.010, 0.055)
                bw_noise = random.uniform(0.0, 30.0)
                risk_type = "unstable_congested"

            if link_type == "backbone":
                delay_noise *= 0.6
                loss_noise *= 0.6
                traffic_ratio *= 0.95

            # backup 링크는 backbone 링크에 혼잡/언스테이블 등 상황 발생시 우회 가능한 링크
            elif link_type == "backup":
                delay_noise *= 0.75
                loss_noise *= 0.6
                traffic_ratio *= 0.75

            q["traffic"] = float(
                np.clip(
                    traffic_ratio * capacity,
                    0.0,
                    capacity * 1.5
                )
            )

            q["utilization"] = float(
                np.clip(
                    q["traffic"] / max(capacity, 1e-6),
                    0.0,
                    1.5
                )
            )

            q["delay"] = float(
                np.clip(
                    q["delay"] + delay_noise + q["utilization"] * 2.0,
                    3.0,
                    120.0
                )
            )

            q["loss"] = float(
                np.clip(
                    q["loss"]
                    + loss_noise
                    + max(0.0, q["utilization"] - 0.8) * 0.03,
                    0.0,
                    0.35
                )
            )

            q["bandwidth"] = float(
                np.clip(
                    q["bandwidth"] + bw_noise,
                    50.0,
                    1000.0
                )
            )

            q["is_congested"] = q["utilization"] >= 0.8

            self.loss_history[e].append(q["loss"])
            self.delay_history[e].append(q["delay"])

            q["risk"] = calculate_link_risk(
                utilization=q["utilization"],
                is_congested=q["is_congested"],
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type=risk_type,
            )

            if self.scenario in ["congestion", "unstable", "dynamic"] and e in safe_set:
                q["risk"] = float(np.clip(q["risk"] * 0.35, 0.0, 1.0))

    def reset(self):
        self.current_node = SOURCE
        self.visited = [SOURCE]
        self.timestep = 0
        self.init_qos()
        self.update_dynamic_qos()
        return self.get_state()

    def get_valid_actions(self, node=None):
        target_node = self.current_node if node is None else node
        return list(self.graph.neighbors(target_node))

    def shortest_distance_to_destination(self, node):
        try:
            return nx.shortest_path_length(
                self.graph,
                node,
                DESTINATION
            )
        except nx.NetworkXNoPath:
            return 999

    def get_state(self):
        state = []

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            state.append(self.qos[e]["risk"])

        curr_onehot = np.zeros(NUM_NODES)
        curr_onehot[self.current_node] = 1.0

        dest_onehot = np.zeros(NUM_NODES)
        dest_onehot[DESTINATION] = 1.0

        visited_onehot = np.zeros(NUM_NODES)

        for node in self.visited:
            visited_onehot[node] = 1.0

        return np.concatenate(
            [
                state,
                curr_onehot,
                dest_onehot,
                visited_onehot
            ]
        ).astype(np.float32)

    def step(self, action):
        self.update_dynamic_qos()

        valid_actions = self.get_valid_actions()

        if action not in valid_actions:
            return (
                self.get_state(),
                -10.0,
                True,
                {
                    "reason": "invalid",
                    "path": self.visited.copy(),
                    "next_valid_mask": np.zeros(NUM_NODES, dtype=np.float32),
                }
            )

        e = self.normalize_edge(
            self.current_node,
            action
        )

        q = self.qos[e]

        prev_dist = self.shortest_distance_to_destination(
            self.current_node
        )

        next_dist = self.shortest_distance_to_destination(
            action
        )

        risk = q["risk"]
        reward = 0.0

        # 보상 구조 설계(시나리오별) -> 실험을 통해 현재 최적의 값 생성 완료

        if self.scenario == "normal":
            reward -= risk * 0.7
            reward += 2.0 if next_dist < prev_dist else -2.0
            reward -= 0.10

        elif self.scenario == "congestion":
            reward -= risk * 10.0

            if q.get("is_congested", False):
                reward -= 4.0

            reward += 0.7 if next_dist < prev_dist else -0.7
            reward -= 0.06

        elif self.scenario == "unstable":
            reward -= risk * 10.0

            if risk >= 0.22:
                reward -= 3.0

            reward += 0.7 if next_dist < prev_dist else -0.7
            reward -= 0.06
        
        # 다이나믹 상황 => 보상 구조 개편할 시 홉수와 risk trade off 관계 보임. 현재 값이 가장 좋다고 판단, 조정시 다른 값들이 무너지는 현상 발생
        #10홉 -> 9홉 risk 값은 2배 이상 상승 같은 안 좋은 구조로 바뀜
        elif self.scenario == "dynamic":
            reward -= risk * 25.0

            if q.get("is_congested", False):
                reward -= 6.0

            if risk >= 0.25:
                reward -= 6.0

            reward += 0.20 if next_dist < prev_dist else -0.20
            reward -= 0.07

        else:
            reward -= risk * 5.0
            reward += 1.0 if next_dist < prev_dist else -1.0
            reward -= 0.05

        if action in self.visited:
            reward -= 8.0

        self.current_node = action
        self.visited.append(action)

        done = False

        if self.current_node == DESTINATION:
            path_metrics = self.calculate_path_metrics(
                self.visited
            )

            reward += 45.0
            reward -= path_metrics["total_risk"] * 6.0

            done = True

        elif len(self.visited) >= MAX_STEPS:
            reward -= 15.0
            done = True

        next_valid = self.get_valid_actions(
            self.current_node
        )

        next_valid_mask = np.zeros(
            NUM_NODES,
            dtype=np.float32
        )

        for act in next_valid:
            next_valid_mask[act] = 1.0

        return (
            self.get_state(),
            reward,
            done,
            {
                "reason": "arrived" if self.current_node == DESTINATION else "continue",
                "path": self.visited.copy(),
                "delay": q["delay"],
                "loss": q["loss"],
                "bandwidth": q["bandwidth"],
                "risk": q["risk"],
                "link_type": q["type"],
                "traffic": q["traffic"],
                "utilization": q["utilization"],
                "is_congested": q["is_congested"],
                "next_valid_mask": next_valid_mask,
            }
        )

    def calculate_path_metrics(self, path):
        total_delay = 0.0
        total_loss = 0.0
        total_risk = 0.0
        min_bandwidth = float("inf")

        unstable_count = 0
        congested_count = 0

        unstable_set = {
            self.normalize_edge(*x)
            for x in UNSTABLE_LINKS
        }

        congestion_set = {
            self.normalize_edge(*x)
            for x in CONGESTION_LINKS
        }

        all_dynamic_set = set()

        for links in DYNAMIC_PHASES.values():
            for edge in links:
                all_dynamic_set.add(
                    self.normalize_edge(*edge)
                )

        hop_count = max(
            len(path) - 1,
            1
        )

        for u, v in zip(
            path[:-1],
            path[1:]
        ):
            e = self.normalize_edge(
                u,
                v
            )

            if e not in self.qos:
                continue

            q = self.qos[e]

            total_delay += q["delay"]
            total_loss += q["loss"]
            total_risk += q["risk"]

            min_bandwidth = min(
                min_bandwidth,
                q["bandwidth"]
            )

            if self.scenario == "unstable":
                if e in unstable_set:
                    unstable_count += 1

            elif self.scenario == "congestion":
                if e in congestion_set:
                    congested_count += 1

            elif self.scenario == "dynamic":
                if e in unstable_set or e in all_dynamic_set:
                    unstable_count += 1

                if e in congestion_set or e in all_dynamic_set:
                    congested_count += 1

        if min_bandwidth == float("inf"):
            min_bandwidth = 0.0

        return {
            "path": path,
            "hop_count": len(path) - 1,
            "total_delay": total_delay,
            "total_loss": total_loss / hop_count,
            "total_risk": total_risk / hop_count,
            "min_bandwidth": min_bandwidth,
            "unstable_count": unstable_count,
            "congested_count": congested_count,
        }

    def ospf_path(self, cost_type="bandwidth"):
        g = nx.Graph()

        for u, v in EDGES:
            e = self.normalize_edge(
                u,
                v
            )

            q = self.qos[e]

            if cost_type == "delay":
                cost = q["delay"]

            elif cost_type == "hop":
                cost = 1

            elif cost_type == "bandwidth":
                cost = 1000.0 / max(
                    q["bandwidth"],
                    1e-6
                )

            elif cost_type == "risk":
                cost = q["risk"]

            else:
                cost = q["delay"]

            g.add_edge(
                u,
                v,
                weight=cost
            )

        return nx.shortest_path(
            g,
            SOURCE,
            DESTINATION,
            weight="weight"
        )