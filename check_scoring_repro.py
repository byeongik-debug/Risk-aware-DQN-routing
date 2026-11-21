"""`reeval_scoring_convention.py` A절 재현 실패(최대 1.12%)의 원인을 가린다.

증상이 비대칭이다. `qos/walk_final`은 네 시나리오 전부 +0.00%인데
`ospf_bw/plan_snapshot`만 −0.22 ~ +1.12%로 흔들린다. 그리고 그 폭이
`reeval_risk_dijkstra.py` A0절에서 측정한 표본 이동 폭(최대 1.13%)과 거의 같다.

가설: **난수 소비 차이로 에피소드 집합이 갈렸다.**
`reeval_scoring_convention.run_planner`는 (시나리오, 시드)마다 시드를 한 번 박고
200에피소드를 연속으로 돈다(`:152-155`). 그 루프는 계획 후 경로를 **주행**하고,
`step()`은 홉마다 `update_dynamic_qos()`로 50링크를 다시 추첨한다. 반면 커밋된
`ospf` 행을 만든 `reeval_qos_baseline_handicap.run_ospf`(`:249-263`)는 **주행하지
않는다**. 그러면 두 번째 에피소드부터 난수 상태가 갈리고 표본이 달라진다.
`qos`가 0.00%로 맞는 것도 같은 이유다 — `run_qos_post_warmup`도 주행하므로 소비가 같다.

`run_ospf`는 `run_planner`에서 주행만 뺀 것과 사실상 같은 루프다. 그래서 검증이 된다:

  예측 1  주행을 끈 변형은 커밋된 `ospf` 행과 **0.00%**로 맞는다.
  예측 2  주행을 켠 변형은 본 실행의 편차(−0.22 / −0.26 / +1.12 / +0.60%)를 재현한다.

둘 다 맞으면 A절 실패는 **규약의 문제가 아니라 스크립트 간 표본 차이**이고,
B·C·D절은 영향이 없다(한 에피소드·한 계획에서 세 규약을 같이 기록하므로 짝지어져 있다).
예측 1이 틀리면 원인이 다른 데 있다는 뜻이므로 그때는 B·C·D도 인용하면 안 된다.
"""

import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_qos_baseline_handicap as QBH  # noqa: E402
import reeval_scoring_convention as RSC  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from topology import DESTINATION  # noqa: E402

METRICS = ("delay", "published", "compounded")


def run_ospf_bw(scenario, seed, episodes, walk):
    """`QBH.run_ospf`와 같은 루프. walk=True면 계획대로 전진까지 한다."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(RSC.WARMUP):
            env.update_dynamic_qos()
        path = env.ospf_path(cost_type="bandwidth")
        if not path or path[-1] != DESTINATION:
            continue
        out.append(QBH.path_metrics(env, path))  # 채점은 계획 스냅샷에서
        if not walk:
            continue
        queue = list(path[1:])
        done = False
        while not done and queue:
            nxt = queue.pop(0)
            _, _, done, _ = env.step(nxt)
            if len(env.visited) >= MAX_STEPS:
                break
    return QBH.seed_mean(out)


def main():
    ref_rows = RSC.load_csv(RSC.REF_CSV, ("scenario", "variant"))
    print("A절 재현 실패의 원인 — 주행이 난수를 소비해 표본이 갈렸는가")
    print("커밋된 ospf 행 대비 편차, {}시드 x {}에피소드".format(
        len(RSC.SEEDS), RSC.EPISODES))
    print()
    print("  {:<12}{:<10}{:>12}{:>12}{:>12}".format(
        "시나리오", "변형", *[m[:10] for m in METRICS]))

    worst = {False: 0.0, True: 0.0}
    for scenario in RSC.SCENARIOS:
        for walk in (False, True):
            per_seed = [run_ospf_bw(scenario, s, RSC.EPISODES, walk)
                        for s in RSC.SEEDS]
            line = "  {:<12}{:<10}".format(scenario, "주행O" if walk else "주행X")
            for metric in METRICS:
                mine = float(np.mean([r[metric] for r in per_seed]))
                theirs = float(np.mean(
                    RSC.ref_pick(ref_rows, scenario, "ospf", metric)))
                dev = (mine - theirs) / theirs * 100 if theirs else 0.0
                worst[walk] = max(worst[walk], abs(dev))
                line += "{:>12}".format("{:+.3f}%".format(dev))
            print(line)

    print()
    print("  주행X 최대 편차 {:.3f}%   주행O 최대 편차 {:.3f}%".format(
        worst[False], worst[True]))
    print()
    if worst[False] <= RSC.REPRO_TOL_PCT < worst[True]:
        print("  판정: 가설 지지. 주행을 끄면 커밋된 행과 맞고, 켜면 어긋난다.")
        print("  -> A절 실패는 규약 문제가 아니라 스크립트 간 표본 차이다.")
        print("     B·C·D절은 한 에피소드·한 계획에서 세 규약을 같이 기록하므로 무관하다.")
    elif worst[False] > RSC.REPRO_TOL_PCT:
        print("  판정: 가설 반증. 주행을 꺼도 맞지 않는다 — 원인이 다른 데 있다.")
        print("  -> B·C·D절도 인용하면 안 된다. 다음으로 볼 것: 워밍업 횟수,")
        print("     use_risk 플래그, path_metrics 구현, 커밋 CSV의 생성 시점.")
    else:
        print("  판정: 애매. 주행을 켜도 허용 오차 안이다 — 본 실행의 편차와 안 맞는다.")


if __name__ == "__main__":
    main()
