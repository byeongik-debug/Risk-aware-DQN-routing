# risk_calculator.py

import numpy as np


def calculate_link_risk(
    utilization,
    is_congested,
    delay_history,
    loss_history,
    link_type="normal"
):

    # ======================================
    # 1. Utilization Risk
    # ======================================

    utilization_risk = min(max(utilization, 0.0), 1.0)

    # ======================================
    # 2. Congestion Risk
    # ======================================

    congestion_risk = 1.0 if is_congested else 0.0

    # ======================================
    # 3. Temporal Delay Instability
    # ======================================

    delay_instability = (
        np.std(delay_history) / 20.0
        if len(delay_history) > 1 else 0.0
    )

    delay_instability = min(delay_instability, 1.0)

    # ======================================
    # 4. Temporal Loss Instability
    # ======================================

    loss_instability = (
        np.std(loss_history) / 0.03
        if len(loss_history) > 1 else 0.0
    )

    loss_instability = min(loss_instability, 1.0)

    # ======================================
    # 5. Link Type Risk
    # ======================================

    if link_type == "unstable_congested":
        link_type_risk = 1.0

    elif link_type == "unstable":
        link_type_risk = 0.9

    elif link_type == "congested":
        link_type_risk = 0.8

    elif link_type == "backup":
        link_type_risk = 0.3

    elif link_type == "backbone":
        link_type_risk = 0.05

    else:
        link_type_risk = 0.15

    # ======================================
    # Final Risk
    # ======================================

    risk = (
        0.35 * utilization_risk +
        0.30 * congestion_risk +
        0.15 * delay_instability +
        0.10 * loss_instability +
        0.10 * link_type_risk
    )

    return float(np.clip(risk, 0.0, 1.0))