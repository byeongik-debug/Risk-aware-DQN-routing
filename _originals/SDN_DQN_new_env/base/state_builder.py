# state_builder.py
# 네트워크 링크의 QoS 정보를 수집해 에이전트 입력 상태 벡터를 구성하는 모듈
#
# 주요 역할:
#   - 각 링크의 지연/대역폭/손실 → 서비스 등급(1~4) 변환
#   - risk_calculator로 계산한 위험도와 결합 → Risk-DQN 입력 상태 생성
# 실제 환경에서는 NetworkEnv.get_state()가 이 역할을 수행

import numpy as np  # 수치 연산 라이브러리

from topology import NUM_NODES, EDGES       # 노드 수, 링크 목록
from risk_calculator import calculate_link_risk  # 링크 위험도 계산 함수

# EDGES는 단방향 정의 → 양방향으로 풀어서 실제 링크 방향 모두 포함
DIRECTED_EDGES = []
for u, v in EDGES:
    DIRECTED_EDGES.append((u, v))   # 정방향
    DIRECTED_EDGES.append((v, u))   # 역방향


def qos_to_service_class(delay, bandwidth, loss):
    # QoS 세 지표를 1~4 서비스 등급으로 변환 (높을수록 좋음)
    # 조건을 충족하는 지표 수로 점수 계산 — 현재는 NetworkEnv가 직접 정규화해서 사용하므로 미사용
    score = 0               # 충족 조건 카운터 초기화

    if delay <= 25:         # 지연 25ms 이하: 양호
        score += 1
    if bandwidth >= 80:     # 대역폭 80 이상: 양호
        score += 1
    if loss <= 0.25:        # 손실률 25% 이하: 양호
        score += 1

    if score == 3:          # 3개 조건 충족: 최상위 등급
        return 4
    if score == 2:          # 2개 조건 충족: 상위 등급
        return 3
    if score == 1:          # 1개 조건 충족: 중간 등급
        return 2
    return 1                # 0개 충족: 최하위 등급


def build_sample_network_state():
    # 랜덤 QoS 값으로 샘플 네트워크 상태 행렬 생성
    # 실제 환경 대신 임의 값으로 테스트할 때 사용했던 함수
    qos_matrix  = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)  # 서비스 등급 행렬 초기화
    risk_matrix = np.zeros((NUM_NODES, NUM_NODES), dtype=np.float32)  # 위험도 행렬 초기화

    link_info      = {}   # 링크별 상세 정보 딕셔너리
    link_risk_dict = {}   # 링크별 위험도 딕셔너리

    for u, v in DIRECTED_EDGES:                               # 모든 방향 링크에 대해
        delay     = np.random.uniform(1, 100)                 # 지연: 1~100ms 무작위
        bandwidth = np.random.uniform(50, 100)                # 대역폭: 50~100 무작위
        loss      = np.random.uniform(0.01, 1.0)             # 손실률: 1~100% 무작위

        capacity    = bandwidth                                         # 최대 용량 = 대역폭
        traffic     = np.random.uniform(0.1, 1.2) * capacity           # 실제 트래픽 (초과 가능)
        utilization = traffic / max(capacity, 1e-6)                     # 사용률 계산
        is_congested = utilization >= 0.8                               # 80% 이상이면 혼잡

        # 지연/손실 히스토리 샘플 생성 — 위험도 계산에 필요한 시계열 데이터
        delay_history = list(np.random.uniform(max(1, delay - 20), delay + 20, size=10))  # 지연 변동 시뮬레이션
        loss_history  = list(np.random.uniform(0.01, max(loss, 0.02), size=10))            # 손실 변동 시뮬레이션

        link_type     = "congested" if is_congested else "normal"       # 링크 상태 분류
        service_class = qos_to_service_class(delay, bandwidth, loss)    # 서비스 등급 변환

        # 5요소 위험도 계산 (risk_calculator.py 호출)
        risk = calculate_link_risk(
            utilization=utilization,      # 사용률
            is_congested=is_congested,    # 혼잡 여부
            delay_history=delay_history,  # 지연 히스토리
            loss_history=loss_history,    # 손실 히스토리
            link_type=link_type,          # 링크 유형
        )

        qos_matrix[u][v]  = service_class  # 서비스 등급 행렬에 저장
        risk_matrix[u][v] = risk           # 위험도 행렬에 저장

        link_info[(u, v)] = {              # 링크 상세 정보 저장
            "delay":        delay,
            "bandwidth":    bandwidth,
            "loss":         loss,
            "traffic":      traffic,
            "utilization":  utilization,
            "is_congested": is_congested,
            "service_class": service_class,
            "risk":         risk,
            "link_type":    link_type,
        }

        link_risk_dict[(u, v)] = risk      # 위험도만 별도 딕셔너리에 저장

    return qos_matrix, risk_matrix, link_info, link_risk_dict  # 4가지 결과 반환


def build_baseline_state(qos_matrix):
    # QoS 행렬만으로 상태 벡터 구성 — NoRisk 베이스라인 초기 버전
    return qos_matrix.flatten() / 4.0  # 1~4 등급 → 0~1 정규화 후 1차원 벡터 반환


def build_risk_aware_state(qos_matrix, risk_matrix):
    # QoS + 위험도를 합쳐 Risk-DQN 입력 벡터 구성 — 현재 버전은 NetworkEnv.get_state()로 대체됨
    qos_state  = qos_matrix.flatten() / 4.0   # QoS 등급 정규화
    risk_state = risk_matrix.flatten()         # 위험도 (이미 0~1)
    return np.concatenate([qos_state, risk_state]).astype(np.float32)  # 두 벡터 이어붙여 반환