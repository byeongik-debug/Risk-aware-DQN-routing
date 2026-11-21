# eval_continuous_temporal.py
# Temporal Ablation 연속 평가 버전
#
# eval_temporal_ablation.py와 차이:
#   - 에피소드마다 env.reset()을 하지 않음
#   - QoS 히스토리가 경로 간에 누적됨 → 링크 상태가 시간에 따라 진화
#   - 매 스텝마다 env.update_dynamic_qos() 호출 → 실제 네트워크처럼 변화
#
# 목적: 더 현실적인 연속 트래픽 환경에서 window 크기 효과 측정

import os, csv
import numpy as np
import torch
import torch.nn as nn
from network_env import NetworkEnv
from topology import SOURCE, DESTINATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "risk_only_model_final.pth")
RESULT_DIR = os.path.join(BASE_DIR, "results")

SCENARIOS   = ["normal", "congestion", "unstable", "dynamic"]
EVAL_ROUTES = 200   # 완주한 경로 수 (실패는 카운트 안 함)
WARMUP      = 50    # 평가 전 QoS 워밍업 횟수 (eval_temporal_ablation의 10회보다 많음)
MAX_STEPS   = 35    # 한 경로당 최대 hop 수 (루프 방지)


class QNetwork(nn.Module):
    """risk_dqn_every_scenario.py와 동일한 구조"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_size),
        )
    def forward(self, x):
        return self.net(x)


def evaluate_continuous(model, device, scenario, risk_window, routes=EVAL_ROUTES):
    """환경 리셋 없이 연속으로 경로를 평가
    QoS 히스토리가 경로 간에 누적됨 → 시계열 위험도가 실제 변화를 반영"""
    env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=risk_window)
    env.reset()

    # 충분한 워밍업으로 히스토리 채우기 — window=5면 최소 5번 이상 필요
    for _ in range(WARMUP):
        env.update_dynamic_qos()

    delays, losses, hops = [], [], []
    success_count = 0
    attempt_count = 0

    while len(delays) < routes:
        # 위치와 방문 기록만 리셋 (QoS 히스토리는 유지)
        env.current_node = SOURCE
        env.visited = []
        state = env.get_state()
        done = False
        steps = 0
        attempt_count += 1

        while not done and steps < MAX_STEPS:
            # 매 스텝마다 QoS 업데이트 → 링크 상태가 한 경로 안에서도 진화
            env.update_dynamic_qos()

            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break

            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q = model(state_t)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid_actions:
                masked[a] = q[a]
            action = int(torch.argmax(masked).item())
            state, _, done, _ = env.step(action)
            steps += 1

        if env.current_node == DESTINATION:
            success_count += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])

        # 무한루프 방지: 요구 경로 수의 3배 시도해도 안 채워지면 중단
        if attempt_count > routes * 3:
            break

    return {
        "success_rate": success_count / attempt_count * 100 if attempt_count else 0,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
        "n":     len(delays),  # 실제 완주한 경로 수
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # window=5 기준으로 state_size 결정 → 모델 로드
    base_env = NetworkEnv(use_risk=True, scenario="normal", risk_window=5)
    model = QNetwork(base_env.state_size, base_env.action_size).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.eval()

    print("=" * 70)
    print("  Continuous Temporal Ablation")
    print(f"  워밍업 {WARMUP}회 후 {EVAL_ROUTES} routes 연속 평가 (QoS 누적 진화)")
    print("  조건 A: window=5  |  조건 B: window=1")
    print("=" * 70)

    results = []
    for scenario in SCENARIOS:
        print(f"\n[{scenario}]")
        for window in [5, 1]:
            r = evaluate_continuous(model, device, scenario, risk_window=window)
            r.update({"scenario": scenario, "window": window})
            results.append(r)
            print(f"  window={window} | Success={r['success_rate']:.1f}% | "
                  f"Delay={r['delay']:.1f} | Loss={r['loss']:.5f} | Hop={r['hop']:.2f}")

    sep = "-" * 70
    print("\n" + "=" * 70)
    print(f"  {'Scenario':12s} | {'Window':>6s} | {'Delay':>8s} | {'Loss':>8s} | {'Hop':>5s}")
    print(sep)
    for r in results:
        print(f"  {r['scenario']:12s} | {'w='+str(r['window']):>6s} | "
              f"{r['delay']:8.1f} | {r['loss']:8.5f} | {r['hop']:5.2f}")
    print(sep)

    # continuous_temporal_ablation.csv 저장 (show_results.py에서 직접 사용하지 않지만 참고용 보존)
    os.makedirs(RESULT_DIR, exist_ok=True)
    out = os.path.join(RESULT_DIR, "continuous_temporal_ablation.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario","window","success_rate","delay","loss","hop","n"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV 저장: {out}")


if __name__ == "__main__":
    main()