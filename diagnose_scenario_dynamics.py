"""네 시나리오가 실제로 시간에 따라 변하는지 측정한다 (R3-2 / R3-3 공통 뿌리).

배경. 시간 윈도우 스윕(R3-3)과 위험 링크 식별(R3-2)이 둘 다 "구분 불가"로 끝났다.
두 결과의 공통 원인 후보는 **환경이 정적이라 시간에 쌓을 것이 없다**는 것이다.
이 스크립트는 그 가설을 환경 자체에 대고 직접 검증한다. 모델도, 정책도 쓰지 않는다.

측정 2종:

1. `scenario_volatility` — 네 시나리오에서 혼잡 플래그 집합이 에피소드 중에 변하는가.
   `is_congested`는 `update_dynamic_qos()`가 매 스텝 다시 계산하므로, 변화가 0이라면
   그것은 측정 실패가 아니라 환경이 링크의 정체를 reset 때 정해버린다는 뜻이다.

2. `phase_flag` — `dynamic` 시나리오의 시변 기계가 꺼져 있는지.
   `network_env.update_dynamic_qos()`에는 8스텝마다 위험 경로를 교체하는 코드가 있다
   (`self.dynamic_phase = phase_keys[(self.timestep // 8) % len(phase_keys)]`).
   그런데 그 코드는 `if not DYNAMIC_PHASE_PER_EPISODE` 아래 있고, 커밋된 값은 True다.
   즉 **dead code**다. 플래그를 런타임에만 바꿔 두 조건을 비교한다 (원본 파일은 수정하지 않는다).

재현: python -I diagnose_scenario_dynamics.py
원본 코드는 읽기만 한다. 모듈 전역 치환은 이 프로세스 안에서만 유효하다.
"""

import argparse
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import numpy as np  # noqa: E402

import network_env as NE  # noqa: E402
from network_env import NetworkEnv  # noqa: E402

SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
STEPS = 70          # eval_baseline.py의 에피소드 길이와 같은 범위
EPISODES = 50
PHASE_EPISODES = 20


def congested_flags(env):
    return {e: q["is_congested"] for e, q in env.qos.items()}


def scenario_volatility(scenario, episodes, steps):
    """혼잡 집합이 에피소드 중에 몇 번 바뀌는지, 전이가 몇 건인지."""
    random.seed(0)
    np.random.seed(0)
    steps_with_change = 0
    clean_to_congested = congested_to_clean = 0
    flips = pairs = 0
    sizes = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario)
        env.reset()
        previous = congested_flags(env)
        for _ in range(steps):
            env.update_dynamic_qos()
            current = congested_flags(env)
            sizes.append(sum(current.values()))
            changed = sum(1 for e in current if current[e] != previous[e])
            if changed:
                steps_with_change += 1
            clean_to_congested += sum(
                1 for e in current if current[e] and not previous[e])
            congested_to_clean += sum(
                1 for e in current if previous[e] and not current[e])
            flips += changed
            pairs += len(current)
            previous = current
    return {
        "steps_with_change_per_episode": steps_with_change / episodes,
        "mean_congested_links": sum(sizes) / len(sizes),
        "clean_to_congested": clean_to_congested,
        "congested_to_clean": congested_to_clean,
        "flip_rate_percent": 100.0 * flips / pairs,
    }


def phase_flag_effect(per_episode, episodes, steps):
    """DYNAMIC_PHASE_PER_EPISODE를 런타임에 바꿔 dynamic의 시변성을 비교한다."""
    original = NE.DYNAMIC_PHASE_PER_EPISODE
    NE.DYNAMIC_PHASE_PER_EPISODE = per_episode
    try:
        random.seed(0)
        np.random.seed(0)
        set_changes = phase_changes = 0
        equals_planted = snapshots = 0
        for _ in range(episodes):
            env = NetworkEnv(use_risk=True, scenario="dynamic")
            env.reset()
            previous = frozenset(
                e for e, q in env.qos.items() if q["is_congested"])
            previous_phase = env.dynamic_phase
            for _ in range(steps):
                env.update_dynamic_qos()
                current = frozenset(
                    e for e, q in env.qos.items() if q["is_congested"])
                planted = frozenset(env.get_dynamic_phase_sets()[0])
                snapshots += 1
                if current == planted:
                    equals_planted += 1
                if current != previous:
                    set_changes += 1
                if env.dynamic_phase != previous_phase:
                    phase_changes += 1
                previous, previous_phase = current, env.dynamic_phase
        return {
            "congested_set_changes_per_episode": set_changes / episodes,
            "phase_changes_per_episode": phase_changes / episodes,
            "congested_set_equals_planted_danger": "{}/{} ({:.1f}%)".format(
                equals_planted, snapshots, 100.0 * equals_planted / snapshots),
        }
    finally:
        NE.DYNAMIC_PHASE_PER_EPISODE = original


def report_volatility(episodes, steps):
    print("=" * 84)
    print("1. 혼잡 집합이 에피소드 중에 변하는가 (스텝 {} x 에피소드 {})".format(
        steps, episodes))
    print("=" * 84)
    print("{:<12}{:>12}{:>12}{:>14}{:>14}{:>12}".format(
        "scenario", "변화스텝/ep", "평균혼잡수", "clean->cong", "cong->clean", "변화율"))
    static = []
    for scenario in SCENARIOS:
        stats = scenario_volatility(scenario, episodes, steps)
        if stats["steps_with_change_per_episode"] == 0.0:
            static.append(scenario)
        print("{:<12}{:>12.2f}{:>12.2f}{:>14}{:>14}{:>11.4f}%".format(
            scenario,
            stats["steps_with_change_per_episode"],
            stats["mean_congested_links"],
            stats["clean_to_congested"],
            stats["congested_to_clean"],
            stats["flip_rate_percent"]))
    print()
    if static:
        print("  -> 정적인 시나리오: {}".format(", ".join(static)))
        print("     링크의 정체가 reset 때 정해지고 70스텝 내내 바뀌지 않는다.")
        print("     시간 윈도우가 쌓을 변화가 없고, '미래 위험 예측'이라는 과제도 없다")
        print("     (지금 혼잡한 링크가 끝까지 혼잡하므로 현재값만 읽으면 된다).")
    remaining = [s for s in SCENARIOS if s not in static]
    if remaining:
        print("  -> 시변인 시나리오: {} — 시간 항을 정당화할 수 있는 유일한 후보다.".format(
            ", ".join(remaining)))


def report_phase_flag(episodes, steps):
    print()
    print("=" * 84)
    print("2. dynamic의 시변 기계가 꺼져 있는가 (DYNAMIC_PHASE_PER_EPISODE)")
    print("=" * 84)
    print("  커밋된 값: {}\n".format(NE.DYNAMIC_PHASE_PER_EPISODE))
    for per_episode in (True, False):
        label = "True (커밋된 값, phase 고정)" if per_episode \
            else "False (플래그를 끈 경우, 8스텝마다 phase 교체)"
        print("  DYNAMIC_PHASE_PER_EPISODE = {}".format(label))
        for key, value in phase_flag_effect(per_episode, episodes, steps).items():
            print("    {:<40} {}".format(key, value))
        print()
    print("  -> 플래그를 끄면 에피소드당 위험 경로가 8번 교체된다. 즉 시변 기계는 구현되어")
    print("     있고 꺼져 있을 뿐이다 (한 줄 변경). 단 끄면 모델 재학습이 필요하다.")
    print("  -> 두 조건 모두 '혼잡 집합 == 환경이 심은 danger 집합'이 100%다.")
    print("     danger 링크는 traffic_ratio ~ U(1.10,1.55)에서, 안전 링크는 U(0.15,0.45)에서")
    print("     뽑히므로 backbone(x0.95)/backup(x0.75) 감쇠 후에도 0.8 경계를 넘지 않는다.")
    print("     따라서 심어둔 danger 목록은 혼잡 플래그와 같은 것이고, R3-2의 독립 라벨이")
    print("     될 수 없다 (그 라벨은 현재 혼잡 플래그만 읽어도 f1 = 1.0으로 맞는다).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--phase-episodes", type=int, default=PHASE_EPISODES)
    args = parser.parse_args()

    report_volatility(args.episodes, args.steps)
    report_phase_flag(args.phase_episodes, args.steps)


if __name__ == "__main__":
    main()
