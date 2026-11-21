"""R3-4 일반화를 손실·risk 규약까지 바로잡아 재평가한다.

문제. `network_env_generalization.py:398-399`가 경로 지표를 홉당 평균으로 낸다:

    "total_loss": total_loss / hops,
    "total_risk": total_risk / hops,

`FINDINGS_loss_metric.md`에서 확인한 것과 같은 정의 문제다. 그런데 R3-4에서는 홉 과다가
본 실험보다 훨씬 심하다 — 미지 토폴로지 재배선 20%에서 Risk-DQN 7.73홉 vs OSPF 4.56홉
(+69%). 홉당 평균은 긴 경로를 유리하게 보이게 하므로, 일반화 주장의 손실·risk 우위가
얼마나 희석 덕인지 분리해야 한다.

방법. `GeneralizationNetworkEnv`를 상속해 `calculate_path_metrics`만 감싼다. 기존 키는
손대지 않고(원본 수치가 그대로 재현된다) 세 키를 **추가로** 반환한다:

    loss_published   mean(p_i)            현행 보고값 = total_loss와 동일
    loss_path_sum    sum(p_i)             delay가 이미 보고되는 방식과 같은 양
    loss_compounded  1 - prod(1 - p_i)    경로 어딘가에서 패킷이 버려질 확률
    risk_path_sum    sum(risk_i)          risk의 경로 합

오버라이드는 난수를 소비하지 않으므로 궤적이 원본과 동일하다. `loss_published`가
`total_loss`와 일치하는지를 스크립트가 매 에피소드 assert로 검증한다 — 일치하지 않으면
나머지 규약도 의미가 없다.

평가 프로토콜은 `run_topology_generalization_dynamic.py`를 그대로 복제한다
(시드 = 70000 + episode, 워밍업 10스텝, 유효 행동 마스킹 후 argmax).
모델은 동결 기준(`../SDN_DQN_new_env/risk_only_model_final.pth`)을 기본으로 쓴다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_generalization_conventions.py
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import torch  # noqa: E402

import run_topology_generalization_dynamic as RTG  # noqa: E402
from network_env_generalization import GeneralizationNetworkEnv, normalize_edge  # noqa: E402

FROZEN_MODEL = Path(BASE_DIR) / ".." / "SDN_DQN_new_env" / "risk_only_model_final.pth"
MERGED_MODEL = Path(BASE_DIR) / "risk_only_model_final.pth"
TOPOLOGY_DIR = Path(BASE_DIR) / "topologies_dynamic"
OUT_DIR = Path(BASE_DIR) / "review_results"
OUT_CSV = OUT_DIR / "generalization_conventions.csv"

MODEL_CHOICES = {"frozen": FROZEN_MODEL, "merged": MERGED_MODEL}
BASE_EVAL_SEED = RTG.BASE_EVAL_SEED
WARMUP_STEPS = RTG.WARMUP_STEPS
SCENARIO = RTG.SCENARIO
CONVENTIONS = ("loss_published", "loss_path_sum", "loss_compounded", "risk_path_sum")


class ConventionEnv(GeneralizationNetworkEnv):
    """경로 지표에 손실·risk 규약 3종을 덧붙인다. 기존 키는 건드리지 않는다."""

    def calculate_path_metrics(self, path):
        metrics = super().calculate_path_metrics(path)

        rates, risks = [], []
        for u, v in zip(path[:-1], path[1:]):
            edge = normalize_edge(u, v)
            if edge not in self.qos:
                continue
            rates.append(float(self.qos[edge]["loss"]))
            risks.append(float(self.qos[edge]["risk"]))

        hops = max(len(path) - 1, 1)
        survival = 1.0
        for rate in rates:
            survival *= max(0.0, 1.0 - rate)

        metrics["loss_published"] = sum(rates) / hops
        metrics["loss_path_sum"] = sum(rates)
        metrics["loss_compounded"] = 1.0 - survival
        metrics["risk_path_sum"] = sum(risks)
        return metrics


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose(model, env, state, device, no_revisit):
    """행동 선택. no_revisit이면 미방문 이웃이 있을 때 그 중에서만 고른다.

    원본 평가는 이웃 전체를 후보로 두고 방문 노드를 거르지 않는다
    (`network_env_generalization.py:279-281`). 그래서 결정론적 argmax가 2-주기에
    갇히면 MAX_STEPS=35까지 왕복하다 실패한다 — `diagnose_generalization_failure.py`가
    재배선 60%의 실패 100%가 이 모드임을 확인했다. 가중치는 건드리지 않고 후보
    집합만 좁히는 변형이며, 품질까지 유지되는지 보려고 여기에 넣었다.
    """
    neighbors = env.get_valid_actions()
    candidates = neighbors
    if no_revisit:
        unvisited = [n for n in neighbors if n not in env.visited]
        if unvisited:
            candidates = unvisited
    return RTG.select_dqn_action(
        model=model, state=state, valid_actions=candidates, device=device)


def run_episode(topology_path, seed, model, device, no_revisit=False):
    """한 에피소드. model=None이면 OSPF 최단경로를 쓴다."""
    # 원본과 동일: OSPF는 use_risk=False로 생성한다.
    env = ConventionEnv(topology_path, use_risk=model is not None,
                        scenario=SCENARIO, seed=seed)
    env.reset()
    for _ in range(WARMUP_STEPS):
        env.update_dynamic_qos()

    if model is None:
        path = env.ospf_path(cost_type="bandwidth")
        if not path or path[-1] != env.destination:
            return None
        return env.calculate_path_metrics(path)

    state = env.get_state()
    done = False
    while not done:
        action = choose(model, env, state, device, no_revisit)
        if action is None:
            break
        state, _, done, _ = env.step(action)

    if env.current_node != env.destination:
        return None
    return env.calculate_path_metrics(env.visited)


def evaluate(topology_path, model, device, episodes, no_revisit=False):
    store = {key: [] for key in CONVENTIONS}
    store["hop"] = []
    store["delay"] = []
    successes = 0
    for episode in range(episodes):
        metrics = run_episode(
            topology_path, BASE_EVAL_SEED + episode, model, device, no_revisit)
        if metrics is None:
            continue
        successes += 1
        # 추가한 규약이 원본 정의를 재현하는지 검증한다.
        assert abs(metrics["loss_published"] - metrics["total_loss"]) < 1e-9, (
            "loss_published != total_loss; override drifted from the original")
        for key in CONVENTIONS:
            store[key].append(float(metrics[key]))
        store["hop"].append(float(metrics["hop_count"]))
        # delay는 원본에서 이미 경로 합이다(정규화 없음). 디코딩을 고치면
        # 경로가 길어지므로 delay가 나빠질 수 있다 — 품질 주장에 필요하다.
        store["delay"].append(float(metrics["total_delay"]))

    result = {"episodes": episodes, "success_rate": successes / episodes * 100.0}
    for key, values in store.items():
        result[key] = float(np.mean(values)) if values else float("nan")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES) + ["both"], default="frozen")
    parser.add_argument("--episodes", type=int, default=RTG.EPISODES_PER_TOPOLOGY)
    parser.add_argument("--out", default=str(OUT_CSV))
    parser.add_argument("--no-revisit", action="store_true",
                        help="미방문 이웃이 있으면 재방문을 배제한다 (디코딩 변형)")
    args = parser.parse_args()

    topology_files = sorted(TOPOLOGY_DIR.glob("*.json"))
    if not topology_files:
        raise SystemExit("no topology JSON in {}".format(TOPOLOGY_DIR))

    labels = list(MODEL_CHOICES) if args.model == "both" else [args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device={} topologies={} episodes={} no_revisit={}".format(
        device, len(topology_files), args.episodes, args.no_revisit), flush=True)

    rows = []
    started = time.time()

    for topology_path in topology_files:
        result = evaluate(topology_path, None, device, args.episodes)
        result.update({
            "topology": topology_path.stem,
            "rewire_ratio": RTG.read_rewire_ratio(topology_path),
            "method": "OSPF", "model_label": "-", "model_md5": "-",
        })
        rows.append(result)
        print("  {:24} OSPF             success={:6.2f}% hop={:6.3f} "
              "pub={:.5f} sum={:.5f} comp={:.5f}  ({:.1f} min)".format(
                  result["topology"], result["success_rate"], result["hop"],
                  result["loss_published"], result["loss_path_sum"],
                  result["loss_compounded"], (time.time() - started) / 60), flush=True)

    for label in labels:
        path = MODEL_CHOICES[label].resolve()
        RTG.MODEL_PATH = path
        digest = md5(path)
        model = RTG.load_model(topology_files[0], device)
        for topology_path in topology_files:
            result = evaluate(topology_path, model, device, args.episodes,
                              args.no_revisit)
            result.update({
                "topology": topology_path.stem,
                "rewire_ratio": RTG.read_rewire_ratio(topology_path),
                "method": "Risk-DQN[{}]".format(label),
                "model_label": label, "model_md5": digest,
            })
            rows.append(result)
            print("  {:24} {:16} success={:6.2f}% hop={:6.3f} "
                  "pub={:.5f} sum={:.5f} comp={:.5f}  ({:.1f} min)".format(
                      result["topology"], result["method"], result["success_rate"],
                      result["hop"], result["loss_published"], result["loss_path_sum"],
                      result["loss_compounded"], (time.time() - started) / 60), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["topology", "rewire_ratio", "method", "model_label", "model_md5",
              "episodes", "success_rate", "hop", "delay"] + list(CONVENTIONS)
    out_path = Path(args.out)
    if not out_path.is_absolute() and not out_path.parent.name:
        out_path = OUT_DIR / out_path
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    print("\nwrote {} ({} rows) in {:.1f} min".format(
        out_path, len(rows), (time.time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()
