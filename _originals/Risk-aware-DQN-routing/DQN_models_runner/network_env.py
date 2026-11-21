# network_env.py
# 25-node Risk-Only DQN Environment
#
# 핵심 수정 사항:
# - 기존 Dynamic은 SAFE_LINKS가 항상 낮은 risk를 받아 Risk-DQN이 하나의 고정 우회 경로로 수렴할 수 있었음.
# - 본 버전은 Dynamic 환경에서 episode마다 active phase를 선택하고,
#   phase별로 위험 경로(danger)와 상대적으로 안정적인 후보 경로(support)가 달라지도록 구성함.
# - 따라서 학습된 정책은 현재 관측된 risk state에 따라 서로 다른 우회 경로를 선택하도록 유도됨.
#
# 주의:
# - 이 파일로 바꾸면 기존 risk_only_model_final.pth는 환경 분포가 달라졌으므로 다시 학습해야 함.

import random
from collections import deque

import networkx as nx
import numpy as np

from topology import NUM_NODES, SOURCE, DESTINATION, EDGES, EDGE_TYPES
from risk_calculator import calculate_link_risk


RISK_WINDOW = 5
MAX_STEPS = 35

# Dynamic phase를 episode 단위로 유지한다.
# 한 episode 안에서 phase가 계속 바뀌면 이미 지나온 링크 때문에 "대응" 해석이 애매해질 수 있으므로,
# 평가에서는 phase별로 서로 다른 경로를 선택하는지를 확인하는 방식이 더 명확하다.
DYNAMIC_PHASE_PER_EPISODE = True


# 혼잡 링크 구성
CONGESTION_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]

# 불안정 링크 구성
UNSTABLE_LINKS = [
    (0, 4), (4, 9), (9, 14), (14, 20), (20, 23), (23, 24),
    (0, 1), (1, 5), (5, 10), (10, 15), (15, 20),
    (4, 5), (5, 11), (11, 17), (17, 22),
]


def path_to_edges(path):
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


# 후보 경로들
# A: 기존 Risk-DQN이 수렴하던 10-hop 저위험 우회 경로
PATH_A = path_to_edges([0, 1, 5, 4, 8, 13, 19, 20, 21, 22, 24])

# B: 상단/중간을 통과하는 짧은 경로
PATH_B = path_to_edges([0, 4, 9, 14, 20, 23, 24])

# C: 중앙 하단 경로
PATH_C = path_to_edges([0, 1, 5, 10, 15, 20, 23, 24])

# D: 기존 위험 후보였던 6-hop 계열 경로
PATH_D = path_to_edges([0, 1, 5, 11, 17, 22, 24])


# Dynamic phase 설계
# 각 phase마다 위험해지는 경로와 상대적으로 안정적인 후보 경로가 다르다.
# support는 "항상 안전한 경로"가 아니라 해당 phase에서만 risk가 완화되는 후보 경로다.
DYNAMIC_PHASES = {
    0: {
        "danger": PATH_B + PATH_C + PATH_D,
        "support": PATH_A,
    },
    1: {
        "danger": PATH_A + PATH_C,
        "support": PATH_B,
    },
    2: {
        "danger": PATH_A + PATH_B,
        "support": PATH_C,
    },
}


class NetworkEnv:
    def __init__(self, use_risk=True, scenario="normal"):
        self.use_risk = use_risk
        self.scenario = scenario
        self.timestep = 0
        self.dynamic_phase = 0

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

    def get_dynamic_phase_sets(self):
        phase_info = DYNAMIC_PHASES[self.dynamic_phase]

        danger_set = {
            self.normalize_edge(*edge)
            for edge in phase_info["danger"]
        }

        support_set = {
            self.normalize_edge(*edge)
            for edge in phase_info["support"]
        }

        return danger_set, support_set

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

        dynamic_danger_set = set()
        dynamic_support_set = set()

        if self.scenario == "dynamic":
            if not DYNAMIC_PHASE_PER_EPISODE:
                phase_keys = list(DYNAMIC_PHASES.keys())
                self.dynamic_phase = phase_keys[(self.timestep // 8) % len(phase_keys)]

            dynamic_danger_set, dynamic_support_set = self.get_dynamic_phase_sets()

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]

            link_type = q["type"]
            capacity = q["capacity"]

            # 기본 상태 변화
            delay_noise = random.uniform(-1.0, 1.0)
            loss_noise = random.uniform(-0.002, 0.002)
            bw_noise = random.uniform(-5.0, 15.0)
            traffic_ratio = random.uniform(0.15, 0.45)
            risk_type = link_type

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

            elif self.scenario == "dynamic" and e in dynamic_danger_set:
                # 해당 phase에서 위험해진 링크
                traffic_ratio = random.uniform(1.10, 1.55)
                delay_noise = random.uniform(5.0, 10.0) + random.uniform(-5.0, 10.0)
                loss_noise = random.uniform(0.012, 0.050) + random.uniform(-0.010, 0.055)
                bw_noise = random.uniform(0.0, 30.0)
                risk_type = "unstable_congested"

            if link_type == "backbone":
                delay_noise *= 0.6
                loss_noise *= 0.6
                traffic_ratio *= 0.95

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

            # Congestion / Unstable 시나리오에서는 기존 구조 유지:
            # backup 링크가 우회 경로 역할을 하도록 완만히 보정한다.
            if self.scenario in ["congestion", "unstable"] and link_type == "backup":
                q["risk"] = float(np.clip(q["risk"] * 0.60, 0.0, 1.0))

            # Dynamic 시나리오에서는 phase별 support 링크만 완만하게 보정한다.
            # 기존처럼 SAFE_LINKS 전체를 항상 0.35배 하지 않기 때문에 하나의 고정 경로로 수렴하는 문제를 줄인다.
            if self.scenario == "dynamic" and e in dynamic_support_set:
                q["risk"] = float(np.clip(q["risk"] * 0.55, 0.0, 1.0))

    def reset(self):
        self.current_node = SOURCE
        self.visited = [SOURCE]
        self.timestep = 0

        if self.scenario == "dynamic":
            self.dynamic_phase = random.choice(list(DYNAMIC_PHASES.keys()))
        else:
            self.dynamic_phase = 0

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
                    "dynamic_phase": self.dynamic_phase,
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

        elif self.scenario == "dynamic":
            # 기존 결과와 비슷한 risk 회피 성향은 유지하되,
            # 경로 길이 보상이 너무 약해서 무조건 긴 안전 경로로만 수렴하지 않도록 진행 보상을 조금 강화했다.
            reward -= risk * 22.0

            if q.get("is_congested", False):
                reward -= 5.0

            if risk >= 0.25:
                reward -= 5.0

            reward += 0.35 if next_dist < prev_dist else -0.35
            reward -= 0.08

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
                "dynamic_phase": self.dynamic_phase,
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

        dynamic_danger_set = set()
        if self.scenario == "dynamic":
            dynamic_danger_set, _ = self.get_dynamic_phase_sets()

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
                if e in dynamic_danger_set:
                    unstable_count += 1
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
            "dynamic_phase": self.dynamic_phase,
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
