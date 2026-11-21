"""재배선 60%에서 동결 모델이 왜 실패하는지 실제 경로를 추적한다.

왜 필요한가. `FINDINGS_topology_generalization.md`에 실패 원인을 "유효 행동 마스킹
안에서 길을 잃는다"고 적었지만 그건 측정한 게 아니라 추측이었다. 환경을 읽어보니
`get_valid_actions`는 이웃 **전체**를 돌려준다 — 방문한 노드를 마스킹하지 않는다
(`network_env_generalization.py:279-281`). 재방문은 보상 −8로 벌점만 받고 허용된다.
그리고 `MAX_STEPS = 35`에서 강제 종료된다.

따라서 가능한 실패 모드는 이것들이다:

    step_limit    35스텝까지 못 도달 (루프 추정)
    no_action     select_dqn_action이 None (이웃 없음 — 고립 노드)
    invalid       선택이 유효 행동이 아님 (환경이 -10으로 즉시 종료)
    unreachable   그래프상 목적지 도달 경로 자체가 없음

이 스크립트는 실패 에피소드마다 어느 모드인지, 경로 길이·고유 노드 수·재방문 횟수를
기록한다. 루프라면 고유 노드 수가 경로 길이보다 훨씬 적게 나온다.

OSPF가 같은 토폴로지에서 100% 성공하므로 `unreachable`은 배제될 것으로 본다 —
그걸 확인하는 것도 이 스크립트의 목적이다 (nx.has_path로 직접 검사한다).

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 diagnose_generalization_failure.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 diagnose_generalization_failure.py \
        --topologies unseen_dynamic_r60_01 --episodes 5 --trace
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import networkx as nx  # noqa: E402
import torch  # noqa: E402

import run_topology_generalization_dynamic as RTG  # noqa: E402
from network_env_generalization import MAX_STEPS, GeneralizationNetworkEnv  # noqa: E402

FROZEN_MODEL = Path(BASE_DIR) / ".." / "SDN_DQN_new_env" / "risk_only_model_final.pth"
MERGED_MODEL = Path(BASE_DIR) / "risk_only_model_final.pth"
TOPOLOGY_DIR = Path(BASE_DIR) / "topologies_dynamic"
MODEL_CHOICES = {"frozen": FROZEN_MODEL, "merged": MERGED_MODEL}

# 동결 모델이 실패했던 토폴로지들 (analyze_topology_generalization.py 결과).
DEFAULT_TOPOLOGIES = (
    "unseen_dynamic_r60_01",   # 성공률 0%
    "unseen_dynamic_r60_09",   # 39%
    "unseen_dynamic_r60_08",   # 72%
    "unseen_dynamic_r40_08",   # 82%
)


def choose(model, env, state, device, no_revisit):
    """행동 선택. no_revisit이면 미방문 이웃이 있을 때 그 중에서만 고른다.

    모델 가중치는 건드리지 않는다 — 같은 Q값에서 후보 집합만 좁힌다. 이게 실패를
    없애면 60% 실패는 학습된 정책의 한계가 아니라 **추론 시점 디코딩 artifact**다.
    원본 평가(select_dqn_action + get_valid_actions)는 이웃 전체를 후보로 두고
    방문 노드를 거르지 않으므로 결정론적 argmax가 2-주기에 갇힐 수 있다.
    """
    neighbors = env.get_valid_actions()
    candidates = neighbors
    if no_revisit:
        unvisited = [n for n in neighbors if n not in env.visited]
        if unvisited:
            candidates = unvisited
    return RTG.select_dqn_action(
        model=model, state=state, valid_actions=candidates, device=device)


def run_episode(topology_path, seed, model, device, trace=False, no_revisit=False):
    """한 에피소드를 돌리고 (결과, 상세) 를 돌려준다."""
    env = GeneralizationNetworkEnv(
        topology_path, use_risk=True, scenario=RTG.SCENARIO, seed=seed)
    env.reset()
    for _ in range(RTG.WARMUP_STEPS):
        env.update_dynamic_qos()

    # 그래프상 도달 가능한가부터 확인한다 (정책과 무관한 실패를 분리).
    reachable = nx.has_path(env.graph, env.current_node, env.destination)
    if not reachable:
        return "unreachable", {"path": list(env.visited), "reachable": False}

    state = env.get_state()
    done = False
    reason = "step_limit"
    while not done:
        action = choose(model, env, state, device, no_revisit)
        if action is None:
            reason = "no_action"
            break
        state, _, done, info = env.step(action)
        if info.get("reason") == "invalid":
            reason = "invalid"
            break

    path = list(env.visited)
    if env.current_node == env.destination:
        reason = "arrived"

    detail = {
        "path": path,
        "steps": len(path),
        "unique": len(set(path)),
        "revisits": len(path) - len(set(path)),
        "source": path[0] if path else None,
        "destination": env.destination,
        "reachable": True,
        "shortest": nx.shortest_path_length(env.graph, path[0], env.destination)
        if path else None,
    }
    if trace:
        detail["trace"] = path
    return reason, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES), default="frozen")
    parser.add_argument("--topologies", nargs="*", default=list(DEFAULT_TOPOLOGIES))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--trace", action="store_true",
                        help="실패 에피소드의 경로를 그대로 출력한다")
    parser.add_argument("--no-revisit", action="store_true",
                        help="미방문 이웃이 있으면 재방문을 배제한다 (디코딩 변형)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RTG.MODEL_PATH = MODEL_CHOICES[args.model].resolve()

    topology_files = [TOPOLOGY_DIR / "{}.json".format(name) for name in args.topologies]
    missing = [str(p) for p in topology_files if not p.exists()]
    if missing:
        raise SystemExit("topology not found: {}".format(", ".join(missing)))

    model = RTG.load_model(topology_files[0], device)
    print("model={} ({})  MAX_STEPS={}  episodes={}  no_revisit={}\n".format(
        args.model, RTG.MODEL_PATH.name, MAX_STEPS, args.episodes, args.no_revisit))

    for topology_path in topology_files:
        counts = Counter()
        failed = []
        succeeded = []
        for episode in range(args.episodes):
            reason, detail = run_episode(
                topology_path, RTG.BASE_EVAL_SEED + episode, model, device,
                args.trace, args.no_revisit)
            counts[reason] += 1
            if reason == "arrived":
                succeeded.append(detail)
            else:
                failed.append((episode, reason, detail))

        total = sum(counts.values())
        print("{}  성공 {}/{} ({:.0f}%)".format(
            topology_path.stem, counts["arrived"], total,
            counts["arrived"] / total * 100))
        for reason, count in counts.most_common():
            if reason != "arrived":
                print("    실패 모드 {:<12} {:>4}회".format(reason, count))

        if succeeded:
            # 성공 경로에 루프가 있으면 calculate_path_metrics가 같은 간선을 두 번
            # 세므로 홉·손실 합이 부푼다. 그래서 성공 쪽도 재방문을 센다.
            looped = [d for d in succeeded if d["revisits"]]
            steps = [d["steps"] for d in succeeded]
            uniq = [d["unique"] for d in succeeded]
            short = [d["shortest"] for d in succeeded if d.get("shortest")]
            print("    성공 경로: 길이 평균 {:.2f} / 고유 노드 {:.2f} / "
                  "최단거리 {:.2f}홉".format(
                      sum(steps) / len(steps), sum(uniq) / len(uniq),
                      sum(short) / len(short) if short else float("nan")))
            print("    성공 경로 중 루프 포함 {}/{} ({:.0f}%), 루프 건 평균 재방문 "
                  "{:.2f}회".format(
                      len(looped), len(succeeded),
                      len(looped) / len(succeeded) * 100,
                      sum(d["revisits"] for d in looped) / len(looped)
                      if looped else 0.0))

        if failed:
            steps = [d["steps"] for _, _, d in failed if "steps" in d]
            uniq = [d["unique"] for _, _, d in failed if "unique" in d]
            revisits = [d["revisits"] for _, _, d in failed if "revisits" in d]
            shortest = [d["shortest"] for _, _, d in failed if d.get("shortest")]
            if steps:
                print("    실패 경로: 길이 평균 {:.1f} / 고유 노드 {:.1f} / "
                      "재방문 {:.1f}회 / 최단거리 {:.1f}홉".format(
                          sum(steps) / len(steps), sum(uniq) / len(uniq),
                          sum(revisits) / len(revisits),
                          sum(shortest) / len(shortest) if shortest else float("nan")))
            if args.trace:
                episode, reason, detail = failed[0]
                print("    예시(ep {}, {}): {} -> {}".format(
                    episode, reason, detail["source"], detail["destination"]))
                print("      {}".format(detail["path"]))
        print()


if __name__ == "__main__":
    main()
