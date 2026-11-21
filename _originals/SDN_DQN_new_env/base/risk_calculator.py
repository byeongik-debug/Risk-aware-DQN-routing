# risk_calculator.py
# 링크 하나의 위험도(risk score)를 0~1 사이 값으로 계산
# Risk-DQN의 핵심: 이 점수를 state에 포함해 학습

import numpy as np  # 표준편차 등 수치 계산에 사용

# 5개 위험 요소의 기본 가중치 (합계 = 1.0)
# util: 활용률, cong: 혼잡도, delay: 지연 변동성, loss: 손실 변동성, type: 링크 유형
DEFAULT_WEIGHTS = {"util": 0.35, "cong": 0.30, "delay": 0.15, "loss": 0.10, "type": 0.10}


def calculate_link_risk(
    utilization,        # 링크 사용률 (트래픽 / 최대 용량), 0~1
    is_congested,       # 혼잡 여부 (bool)
    delay_history,      # 최근 N 스텝간의 지연 기록 (시간적 변화 추적용)
    loss_history,       # 최근 N 스텝간의 손실률 기록
    link_type="normal"  # 링크 종류 (backbone / backup / normal 등)
):
    # 링크 하나의 시간적 위험도를 0~1로 계산해 반환
    # 5개 요소(활용률/혼잡도/순간지연변화/순간손실변화/링크유형)를 가중 합산

    # ======================================
    # 1. Utilization Risk (가중치 0.35)
    # 사용률이 높을수록 위험 — 링크가 거의 꽉 찼다는 신호
    # ======================================
    utilization_risk = min(max(utilization, 0.0), 1.0)  # 0~1 범위로 클리핑

    # ======================================
    # 2. Congestion Risk (가중치 0.30)
    # 현재 혼잡 상태면 즉각 1.0 부여 — 가장 직접적인 위험 신호
    # ======================================
    congestion_risk = 1.0 if is_congested else 0.0  # 혼잡이면 최대 위험

    # ======================================
    # 3. Temporal Delay Instability (가중치 0.15)
    # 지연의 표준편차 → 시간에 따라 지연이 얼마나 불안정한지
    # 20ms를 기준으로 정규화 (std=20이면 위험도 1.0)
    # QoS-Model은 이 시간적 패턴을 모름 → Risk-DQN의 핵심 차별점
    # ======================================
    delay_instability = (
        np.std(delay_history) / 20.0     # 표준편차를 20ms 기준으로 정규화
        if len(delay_history) > 1 else 0.0  # 기록이 1개 이하면 변동성 없음
    )
    delay_instability = min(delay_instability, 1.0)  # 1.0 초과 방지

    # ======================================
    # 4. Temporal Loss Instability (가중치 0.10)
    # 손실률의 표준편차 → 손실이 얼마나 불규칙하게 발생하는지
    # 0.03을 기준으로 정규화
    # ======================================
    loss_instability = (
        np.std(loss_history) / 0.03      # 표준편차를 0.03 기준으로 정규화
        if len(loss_history) > 1 else 0.0   # 기록이 1개 이하면 변동성 없음
    )
    loss_instability = min(loss_instability, 1.0)  # 1.0 초과 방지

    # ======================================
    # 5. Link Type Risk (가중치 0.10)
    # 링크 종류 자체가 가진 고유 위험도
    # unstable 링크는 구조적으로 불안정, backbone은 안정적
    # ======================================
    if link_type == "unstable_congested":
        link_type_risk = 1.0   # 혼잡+불안정: 최고 위험
    elif link_type == "unstable":
        link_type_risk = 0.9   # 불안정 링크
    elif link_type == "congested":
        link_type_risk = 0.8   # 혼잡 링크
    elif link_type == "backup":
        link_type_risk = 0.3   # 우회 링크: 평소엔 낮은 품질
    elif link_type == "backbone":
        link_type_risk = 0.05  # 백본: 가장 안정적
    else:
        link_type_risk = 0.15  # 일반 링크

    # ======================================
    # Final Risk Score: 5개 요소의 가중 합산
    # 합계 가중치 = 0.35+0.30+0.15+0.10+0.10 = 1.0
    # ======================================
    w = DEFAULT_WEIGHTS          # 기본 가중치 사용
    risk = (
        w["util"]  * utilization_risk +   # 활용률 기여분
        w["cong"]  * congestion_risk +    # 혼잡도 기여분
        w["delay"] * delay_instability +  # 지연 변동성 기여분
        w["loss"]  * loss_instability +   # 손실 변동성 기여분
        w["type"]  * link_type_risk       # 링크 유형 기여분
    )

    return float(np.clip(risk, 0.0, 1.0))  # 0~1 범위로 최종 클리핑 후 반환