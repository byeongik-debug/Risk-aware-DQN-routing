# network_env_temporal_stress.py
# Temporal-stress validation environment
#
# 목적:
# - 현재 utilization/congestion 수준은 유사하게 유지
# - 특정 짧은 경로의 delay/loss만 시간에 따라 크게 변동
# - window=1에서는 표준편차가 0이므로 변동 경로를 구분하기 어려움
# - window=5/10에서는 temporal instability가 Risk에 반영됨
#
# 기존 network_env.py, topology.py, risk_calculator.py와 같은 폴더에 둔다.

from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from network_env import NetworkEnv
from topology import EDGES
from risk_calculator import calculate_link_risk


def normalize_edge(u: int, v: int):
    return tuple(sorted((u, v)))


def path_to_edges(path: List[int]):
    return {
        normalize_edge(path[i], path[i + 1])
        for i in range(len(path) - 1)
    }


# 짧지만 시간 변동성이 큰 경로
VOLATILE_PATH = [0, 4, 9, 14, 20, 23, 24]
VOLATILE_EDGES = path_to_edges(VOLATILE_PATH)

# 길지만 안정적인 경로
STABLE_PATH = [0, 1, 5, 4, 8, 13, 19, 20, 21, 22, 24]
STABLE_EDGES = path_to_edges(STABLE_PATH)


class TemporalStressEnv(NetworkEnv):
    """
    현재 부하 수준은 비슷하지만 시간적 변동성만 다른 링크를 구성한다.

    Risk 계산 시 link_type은 모두 'normal'로 전달한다.
    따라서 window 간 차이는 delay/loss history의 표준편차에서 발생한다.
    """

    def __init__(self, use_risk: bool = True, scenario: str = "dynamic"):
        super().__init__(use_risk=use_risk, scenario="dynamic")

    def init_qos(self):
        self.qos = {}

        for u, v in EDGES:
            e = self.normalize_edge(u, v)

            # 모든 링크의 현재 부하 수준을 유사하게 설정
            capacity = 800.0
            utilization = 0.25
            traffic = utilization * capacity

            # 두 후보 경로의 현재 평균 QoS는 유사하게 시작
            if e in VOLATILE_EDGES:
                delay = 20.0
                loss = 0.010
            elif e in STABLE_EDGES:
                delay = 20.0
                loss = 0.010
            else:
                # 기타 경로는 약간 불리하게 설정해 두 후보 경로 비교를 명확화
                delay = 28.0
                loss = 0.018
                utilization = 0.45
                traffic = utilization * capacity

            self.qos[e] = {
                "delay": float(delay),
                "loss": float(loss),
                "bandwidth": float(capacity),
                "capacity": float(capacity),
                "traffic": float(traffic),
                "utilization": float(utilization),
                "is_congested": False,
                "risk": 0.0,
                "type": "normal",
            }

            self.loss_history[e].clear()
            self.delay_history[e].clear()
            self.loss_history[e].append(float(loss))
            self.delay_history[e].append(float(delay))

            self.qos[e]["risk"] = calculate_link_risk(
                utilization=utilization,
                is_congested=False,
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type="normal",
            )

    def update_dynamic_qos(self):
        self.timestep += 1

        # 큰 변동을 만들되 현재 평균 수준은 stable path와 유사하게 유지
        high_phase = (self.timestep % 2 == 0)

        for u, v in EDGES:
            e = self.normalize_edge(u, v)
            q = self.qos[e]

            if e in VOLATILE_EDGES:
                if high_phase:
                    delay = 48.0 + random.uniform(-2.0, 2.0)
                    loss = 0.065 + random.uniform(-0.003, 0.003)
                else:
                    delay = 8.0 + random.uniform(-1.0, 1.0)
                    loss = 0.002 + random.uniform(0.0, 0.002)

                utilization = 0.25

            elif e in STABLE_EDGES:
                delay = 20.0 + random.uniform(-0.5, 0.5)
                loss = 0.010 + random.uniform(-0.0005, 0.0005)
                utilization = 0.25

            else:
                delay = 28.0 + random.uniform(-1.0, 1.0)
                loss = 0.018 + random.uniform(-0.001, 0.001)
                utilization = 0.45

            q["delay"] = float(np.clip(delay, 1.0, 120.0))
            q["loss"] = float(np.clip(loss, 0.0, 0.35))
            q["utilization"] = float(utilization)
            q["traffic"] = float(utilization * q["capacity"])
            q["is_congested"] = False

            self.delay_history[e].append(q["delay"])
            self.loss_history[e].append(q["loss"])

            # link type 효과를 제거해 temporal history만 비교하도록 함
            q["risk"] = calculate_link_risk(
                utilization=q["utilization"],
                is_congested=False,
                delay_history=self.delay_history[e],
                loss_history=self.loss_history[e],
                link_type="normal",
            )

    def calculate_path_metrics(self, path):
        metrics = super().calculate_path_metrics(path)

        volatile_count = 0
        stable_count = 0

        for u, v in zip(path[:-1], path[1:]):
            e = self.normalize_edge(u, v)

            if e in VOLATILE_EDGES:
                volatile_count += 1
            if e in STABLE_EDGES:
                stable_count += 1

        metrics["unstable_count"] = volatile_count
        metrics["congested_count"] = 0
        metrics["volatile_count"] = volatile_count
        metrics["stable_count"] = stable_count
        metrics["used_volatile_path"] = int(volatile_count > 0)

        return metrics
