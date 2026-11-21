"""본 실험(고정 토폴로지)의 Risk-DQN 경로에도 루프가 있는지 센다.

왜. R3-4에서 재배선 60% 실패가 전부 2-주기 왕복이었다
(`diagnose_generalization_failure.py`). 원인은 환경이 방문 노드를 마스킹하지 않는
것인데(`get_valid_actions`가 이웃 전체를 반환), **모든** 환경이 같다 —
`SDN_DQN_new_env/network_env.py:264`, `review_response/network_env.py:337`,
`network_env_unseen.py:285`, `network_env_generalization.py:279`. 전부 `MAX_STEPS = 35`.

그래서 본 실험 표의 홉·손실도 루프로 부풀어 있을 수 있다. `reeval_loss_conventions.py`는
`loss_conventions(env, env.visited)`로 보행을 그대로 받으므로, 경로에 루프가 있으면
`zip(path[:-1], path[1:])`이 같은 간선을 두 번 센다.

다만 본 실험은 **학습에 쓴 고정 토폴로지**다. 정책이 익숙한 그래프에서는 주기에 안
갇힐 수 있으므로, 구조적 위험이 있다는 것만으로 결론을 낼 수 없다. 측정한다.

재는 것:
    성공률, 실패 모드(step_limit / no_action / invalid), 성공 경로의 재방문 분포,
    경로 길이 대 최단거리.

평가 루프는 `reeval_loss_conventions.run_dqn_seed`를 그대로 베꼈다 — 같은 시드,
같은 워밍업 위치, 같은 마스킹 argmax. 거기에 통계 수집만 더했다. 원본과 기존
스크립트는 건드리지 않는다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 diagnose_main_experiment_loops.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 diagnose_main_experiment_loops.py \
        --seeds 0 1 2 3 4 --episodes 200
"""

import argparse
import os
import random
import sys
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import reeval_loss_conventions as RLC  # noqa: E402
from network_env import NetworkEnv  # noqa: E402
from topology import DESTINATION  # noqa: E402


def run_episode(model, scenario, device):
    """한 에피소드. (실패 모드 또는 'arrived', 상세)."""
    env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RLC.RISK_WINDOW)
    env.reset()
    for _ in range(RLC.WARMUP):
        env.update_dynamic_qos()
    state = env.get_state()

    source = env.current_node
    try:
        shortest = nx.shortest_path_length(env.graph, source, DESTINATION)
    except nx.NetworkXNoPath:
        return "unreachable", {}

    done = False
    reason = "step_limit"
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            reason = "no_action"
            break
        with torch.no_grad():
            q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
        masked = torch.full_like(q_values, -1e9)
        for a in valid:
            masked[a] = q_values[a]
        state, _, done, info = env.step(int(torch.argmax(masked).item()))
        if isinstance(info, dict) and info.get("reason") == "invalid":
            reason = "invalid"
            break

    path = list(env.visited)
    if env.current_node == DESTINATION:
        reason = "arrived"
    return reason, {
        "path": path,
        "hop": max(len(path) - 1, 0),
        "unique": len(set(path)),
        "revisits": len(path) - len(set(path)),
        "source": source,
        "shortest": shortest,
    }


def count_paths(method, scenario, seed, episodes, model, device):
    """한 방법이 내놓는 서로 다른 경로를 센다.

    평가 루프는 `reeval_loss_conventions.py`의 것을 그대로 쓴다 — OSPF는 워밍업 후
    대역폭 Dijkstra(`run_ospf_seed`), QoS는 t=0 일회성 계획을 워밍업 뒤에 실행
    (`run_qos_seed`), Risk-DQN은 홉마다 greedy.
    """
    np.random.seed(seed)
    random.seed(seed)
    paths = Counter()
    for _ in range(episodes):
        if method == "OSPF":
            env = NetworkEnv(use_risk=False, scenario=scenario)
            env.reset()
            for _ in range(RLC.WARMUP):
                env.update_dynamic_qos()
            path = env.ospf_path(cost_type="bandwidth")
            if not path or path[-1] != DESTINATION:
                continue
            paths[tuple(path)] += 1
        elif method == "QoS":
            env = NetworkEnv(use_risk=False, scenario=scenario)
            env.reset()
            planned = RLC.qos_plan_full_path(env)
            if not planned or planned[-1] != DESTINATION:
                continue
            for _ in range(RLC.WARMUP):
                env.update_dynamic_qos()
            for next_node in planned[1:]:
                _, _, done, _ = env.step(next_node)
                if done:
                    break
            if env.current_node == DESTINATION:
                paths[tuple(env.visited)] += 1
        else:
            reason, detail = run_episode(model, scenario, device)
            if reason == "arrived":
                paths[tuple(detail["path"])] += 1
    return paths


def report_methods(scenarios, seeds, episodes, model, device):
    """세 방법의 경로 다양성을 나란히 놓는다.

    왜. Risk-DQN이 시나리오마다 경로를 하나만 내놓는다면 "위험을 보고 적응한다"는
    서술이 성립하지 않는다. 그런데 그게 문제인지는 **기준선과 비교해야** 안다 —
    OSPF도 하나만 내놓으면 과제 자체가 경로 하나로 풀리는 것이고, OSPF가 여럿을
    내놓는데 Risk-DQN이 하나면 제안 방법이 덜 적응하는 것이다.
    """
    print()
    print("=" * 76)
    print("방법별 서로 다른 경로 개수 (출발 {} -> 목적지 {}, 고정 단일 쌍)".format(
        0, DESTINATION))
    print("=" * 76)
    print("  {:<12}{:>6}{:>10}{:>10}{:>12}".format(
        "시나리오", "시드", "OSPF", "QoS", "Risk-DQN"))
    for scenario in scenarios:
        for seed in seeds:
            counts = {}
            for method in ("OSPF", "QoS", "Risk-DQN"):
                counts[method] = len(count_paths(
                    method, scenario, seed, episodes, model, device))
            print("  {:<12}{:>6}{:>10}{:>10}{:>12}".format(
                scenario, seed, counts["OSPF"], counts["QoS"],
                counts["Risk-DQN"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", nargs="*", default=list(RLC.SCENARIOS))
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=RLC.EVAL_EPISODES)
    parser.add_argument("--trace", action="store_true",
                        help="루프가 있는 성공 경로 예시를 출력한다")
    parser.add_argument("--paths", action="store_true",
                        help="시나리오별로 서로 다른 경로가 몇 개 나오는지 센다")
    parser.add_argument("--compare-methods", action="store_true",
                        help="OSPF·QoS·Risk-DQN의 경로 다양성을 나란히 센다")
    args = parser.parse_args()

    device = torch.device("cpu")
    model = RLC.load_risk_model()

    if args.compare_methods:
        report_methods(args.scenarios, args.seeds, args.episodes, model, device)
        return

    print("model={}  MAX_STEPS=35  seeds={}  episodes={}\n".format(
        os.path.basename(RLC.RISK_MODEL), args.seeds, args.episodes))

    print("  {:<12}{:>5}{:>9}{:>12}{:>13}{:>10}{:>10}{:>11}".format(
        "시나리오", "시드", "성공률", "실패모드", "루프 포함", "평균 홉",
        "최단거리", "루프시 재방문"))

    for scenario in args.scenarios:
        for seed in args.seeds:
            np.random.seed(seed)
            random.seed(seed)
            counts = Counter()
            succeeded = []
            example = None
            for _ in range(args.episodes):
                reason, detail = run_episode(model, scenario, device)
                counts[reason] += 1
                if reason == "arrived":
                    succeeded.append(detail)
                    if detail["revisits"] and example is None:
                        example = detail

            total = sum(counts.values())
            modes = ",".join("{}={}".format(r, c) for r, c in counts.most_common()
                             if r != "arrived") or "-"
            looped = [d for d in succeeded if d["revisits"]]
            hop = (sum(d["hop"] for d in succeeded) / len(succeeded)
                   if succeeded else float("nan"))
            short = (sum(d["shortest"] for d in succeeded) / len(succeeded)
                     if succeeded else float("nan"))
            revisit = (sum(d["revisits"] for d in looped) / len(looped)
                       if looped else 0.0)
            print("  {:<12}{:>5}{:>8.0f}%{:>12}{:>9}/{:<3}{:>10.2f}{:>10.2f}"
                  "{:>11.2f}".format(
                      scenario, seed, counts["arrived"] / total * 100, modes,
                      len(looped), len(succeeded), hop, short, revisit))

            if args.trace and example:
                print("      루프 예시: {} -> {} (최단 {}홉): {}".format(
                    example["source"], DESTINATION, example["shortest"],
                    example["path"]))

            if args.paths:
                # 홉 수가 전 에피소드에서 같아도 경로가 같다는 뜻은 아니다.
                # 실제 노드 열을 세서 정책이 경로 하나만 내놓는지 확인한다.
                distinct = Counter(tuple(d["path"]) for d in succeeded)
                print("      서로 다른 경로 {}개 / 성공 {}건".format(
                    len(distinct), len(succeeded)))
                for path, count in distinct.most_common(3):
                    print("        {:>4}회  {}".format(count, list(path)))


if __name__ == "__main__":
    main()
