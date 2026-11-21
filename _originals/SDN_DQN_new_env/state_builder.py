# state_builder.py
# 레거시 호환용 — 현재 파이프라인에서는 사용하지 않음
# 최종 학습/평가 파이프라인은 NetworkEnv.get_state()를 직접 사용
# 이 파일은 초기 개발 단계의 독립 테스트용으로만 보존

import numpy as np

from topology import NUM_NODES, EDGES
from risk_calculator import calculate_link_risk

# 모든 엣지를 양방향으로 풀어서 리스트화
DIRECTED_EDGES = []
for u, v in EDGES:
    DIRECTED_EDGES.append((u, v))
    DIRECTED_EDGES.append((v, u))


def qos_to_service_class(delay, bandwidth, loss):
    """QoS 세 지표를 1~4 서비스 등급으로 변환 (높을수록 좋음)
    조건을 충족하는 지표 수로 점수 계산 — 현재는 NetworkEnv가 직접 정규화해서 사용하므로 미사용"""
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
    """랜덤 QoS 값으로 샘플 네트워크 상태 행렬 생성
    실제 환경 대신 임의 값으로 테스트할 때 사용했던 함수"""
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
        is_congested = utilization >= 0.8  # 80% 이상이면 혼잡

        # 지연/손실 히스토리 샘플 생성 — 위험도 계산에 필요한 시계열 데이터
        delay_history = list(np.random.uniform(max(1, delay - 20), delay + 20, size=10))
        loss_history = list(np.random.uniform(0.01, max(loss, 0.02), size=10))

        link_type = "congested" if is_congested else "normal"
        service_class = qos_to_service_class(delay, bandwidth, loss)

        # 5요소 위험도 계산 (risk_calculator.py 호출)
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
    """QoS 행렬만으로 상태 벡터 구성 — NoRisk 베이스라인 초기 버전"""
    return qos_matrix.flatten() / 4.0  # 1~4 등급 → 0~1 정규화


def build_risk_aware_state(qos_matrix, risk_matrix):
    """QoS + 위험도를 합쳐 Risk-DQN 입력 벡터 구성 — 현재 버전은 NetworkEnv.get_state()로 대체됨"""
    qos_state = qos_matrix.flatten() / 4.0
    risk_state = risk_matrix.flatten()
    return np.concatenate([qos_state, risk_state]).astype(np.float32)