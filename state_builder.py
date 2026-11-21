# state_builder.py
# Compatibility helper.
# The final training/evaluation pipeline uses NetworkEnv.get_state().
# This file is kept only for old standalone tests.

import numpy as np

from topology import NUM_NODES, EDGES
from risk_calculator import calculate_link_risk


DIRECTED_EDGES = []
for u, v in EDGES:
    DIRECTED_EDGES.append((u, v))
    DIRECTED_EDGES.append((v, u))


def qos_to_service_class(delay, bandwidth, loss):
    score = 0

    if delay <= 25:
        score += 1
    if bandwidth >= 80:
        score += 1
    if loss <= 0.25:
        score += 1

    if score == 3:
        return 4
    if score == 2:
        return 3
    if score == 1:
        return 2
    return 1


def build_sample_network_state():
    qos_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)
    risk_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)

    link_info = {}
    link_risk_dict = {}

    for u, v in DIRECTED_EDGES:
        delay = np.random.uniform(1, 100)
        bandwidth = np.random.uniform(50, 100)
        loss = np.random.uniform(0.01, 1.0)

        capacity = bandwidth
        traffic = np.random.uniform(0.1, 1.2) * capacity
        utilization = traffic / max(capacity, 1e-6)
        is_congested = utilization >= 0.8

        delay_history = list(np.random.uniform(max(1, delay - 20), delay + 20, size=10))
        loss_history = list(np.random.uniform(0.01, max(loss, 0.02), size=10))

        link_type = "congested" if is_congested else "normal"
        service_class = qos_to_service_class(delay, bandwidth, loss)

        risk = calculate_link_risk(
            utilization=utilization,
            is_congested=is_congested,
            delay_history=delay_history,
            loss_history=loss_history,
            link_type=link_type,
        )

        qos_matrix[u][v] = service_class
        risk_matrix[u][v] = risk

        link_info[(u, v)] = {
            "delay": delay,
            "bandwidth": bandwidth,
            "loss": loss,
            "traffic": traffic,
            "utilization": utilization,
            "is_congested": is_congested,
            "service_class": service_class,
            "risk": risk,
            "link_type": link_type,
        }

        link_risk_dict[(u, v)] = risk

    return qos_matrix, risk_matrix, link_info, link_risk_dict


def build_baseline_state(qos_matrix):
    return qos_matrix.flatten() / 4.0


def build_risk_aware_state(qos_matrix, risk_matrix):
    qos_state = qos_matrix.flatten() / 4.0
    risk_state = risk_matrix.flatten()
    return np.concatenate([qos_state, risk_state]).astype(np.float32)
