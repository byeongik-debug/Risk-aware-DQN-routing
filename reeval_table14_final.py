"""표 14 최종 측정 — 체크포인트 2개 x 디코딩 2개를 같은 시드로 한 번에 돌린다.

왜 또 돌리나. 표 14를 정직하게 채우려면 한 CSV 안에 네 가지가 같이 있어야 한다.

1. **본표와 같은 체크포인트.** 원고 표 14는 `merged`(review_response 사본)로 만들어
   졌고, 표 10/11/12는 `frozen`(../SDN_DQN_new_env/)을 쓴다. 두 파일은 재저장이
   아니라 별개 학습본이다(동일 원소 5.93%). 같은 이름이 두 모델을 가리키면 안 된다.
2. **고친 디코딩.** 원 평가는 `get_valid_actions`가 이웃 전체를 돌려주고 방문 노드를
   거르지 않아서(`network_env_generalization.py:279-281`) 결정론적 argmax가 2-주기에
   갇힌다. 재배선 60%에서 `frozen`의 실패 100%가 이 모드다. 미방문 이웃이 있을 때만
   후보를 좁히면(가중치는 건드리지 않는다) 실패가 사라진다.
3. **지연.** 표 14의 요구된 빈칸이다. 원 평가 CSV에는 있지만 규약 CSV에는 없었다.
4. **위험 링크 적중수.** 표 14의 Risky 열은 `적중수 / 홉 x 100`이다. 홉이 분모이므로
   보고된 감소율에 홉 희석이 섞여 있다. 분해해서 보려면 적중수 자체가 필요한데
   어느 CSV에도 없었다. 규약 CSV에는 Risky 열 자체가 없다.

그래서 이 스크립트는 OSPF + {frozen, merged} x {원 디코딩, 재방문 배제} = 5개 arm을
같은 토폴로지·같은 평가 시드(`BASE_EVAL_SEED + i`)로 돌리고, 지표를 한 CSV에 모은다.
arm마다 체크포인트 md5와 디코딩 이름을 행에 적는다 — 출처가 CSV 안에 남아야 한다.

검증 단정 둘을 매 에피소드 걸었다.
  - `loss_published == total_loss`: 덧붙인 규약이 원본 정의를 재현하는지.
  - `risky_link_usage == risky_count / hop x 100`: 표 14 Risky 열의 정의를 재현하는지
    (`run_topology_generalization_dynamic.py:142-168`).

원본 파일은 수정하지 않는다. `RTG.MODEL_PATH`만 이 프로세스 안에서 치환한다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 reeval_table14_final.py
    (-I 는 못 쓴다 — 이 환경엘 conda site 설정을 막아서 pandas import가 죽는다)
    (30 토폴로지 x 100 에피소드 x 5 arm = 15,000 에피소드, 25분 안팎)
"""

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import torch  # noqa: E402

import run_topology_generalization_dynamic as RTG  # noqa: E402
from network_env_generalization import (  # noqa: E402
    GeneralizationNetworkEnv,
    normalize_edge,
)

FROZEN_MODEL = Path(BASE_DIR) / ".." / "SDN_DQN_new_env" / "risk_only_model_final.pth"
MERGED_MODEL = Path(BASE_DIR) / "risk_only_model_final.pth"
TOPOLOGY_DIR = Path(BASE_DIR) / "topologies_dynamic"
OUT_DIR = Path(BASE_DIR) / "review_results"
OUT_CSV = OUT_DIR / "table14_final_runs.csv"

MODEL_CHOICES = {"frozen": FROZEN_MODEL, "merged": MERGED_MODEL}
BASE_EVAL_SEED = RTG.BASE_EVAL_SEED
WARMUP_STEPS = RTG.WARMUP_STEPS
SCENARIO = RTG.SCENARIO

METRICS = ("delay", "hop", "loss_published", "loss_path_sum", "loss_compounded",
           "risk_published", "risk_path_sum", "risky_count", "risky_link_usage")
FIELDS = (["topology", "rewire_ratio", "method", "model_label", "model_md5",
           "decoding", "episodes", "success_rate"]
          + list(METRICS) + ["delay_episode_std", "hop_episode_std",
                             "risky_link_usage_episode_std"])


class Table14Env(GeneralizationNetworkEnv):
    """경로 지표에 손실·risk 규약 3종과 위험 적중수를 덧붙인다.

    기존 키는 건드리지 않는다. 오버라이드는 난수를 소비하지 않으므로 궤적이 원본과
    같다 — 같은 시드면 같은 경로가 나온다.
    """

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

        # dynamic에서 unstable_count와 congested_count는 같은 활성 위험 집합을
        # 가리킨다 (run_topology_generalization_dynamic.py:148-152).
        risky_count = max(int(metrics["unstable_count"]),
                          int(metrics["congested_count"]))

        metrics["loss_published"] = sum(rates) / hops
        metrics["loss_path_sum"] = sum(rates)
        metrics["loss_compounded"] = 1.0 - survival
        metrics["risk_published"] = float(metrics["total_risk"])
        metrics["risk_path_sum"] = sum(risks)
        metrics["risky_count"] = float(risky_count)
        metrics["risky_link_usage"] = risky_count / hops * 100.0
        metrics["delay"] = float(metrics["total_delay"])
        metrics["hop"] = float(metrics["hop_count"])
        return metrics


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose(model, env, state, device, no_revisit):
    """행동 선택. no_revisit이면 미방문 이웃이 있을 때 그 중에서만 고른다.

    가중치는 건드리지 않는다 — 같은 Q값에서 후보 집합만 좁힌다.
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
    """한 에피소드. model=None이면 OSPF 최단경로(bandwidth 비용)를 쓴다."""
    env = Table14Env(topology_path, use_risk=model is not None,
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
    store = {key: [] for key in METRICS}
    successes = 0
    for episode in range(episodes):
        metrics = run_episode(topology_path, BASE_EVAL_SEED + episode,
                              model, device, no_revisit)
        if metrics is None:
            continue
        successes += 1
        # 덧붙인 규약이 원본 정의를 재현하는지 매 에피소드 검증한다.
        assert abs(metrics["loss_published"] - metrics["total_loss"]) < 1e-9, (
            "loss_published != total_loss; 오버라이드가 원본에서 벗어났다")
        hops = max(int(metrics["hop_count"]), 1)
        expect = metrics["risky_count"] / hops * 100.0
        assert abs(metrics["risky_link_usage"] - expect) < 1e-9, (
            "risky_link_usage가 적중수/홉 정의와 다르다")
        for key in METRICS:
            store[key].append(float(metrics[key]))

    result = {"episodes": episodes,
              "success_rate": successes / episodes * 100.0}
    for key, values in store.items():
        result[key] = statistics.fmean(values) if values else float("nan")
    for key in ("delay", "hop", "risky_link_usage"):
        values = store[key]
        result[key + "_episode_std"] = (statistics.stdev(values)
                                        if len(values) > 1 else float("nan"))
    return result


def arms(choice):
    """[(method, 라벨, 경로 또는 None, 디코딩 이름, no_revisit)]"""
    plan = [("OSPF", "-", None, "-", False)]
    labels = list(MODEL_CHOICES) if choice == "both" else [choice]
    for label in labels:
        path = MODEL_CHOICES[label].resolve()
        if not path.exists():
            raise SystemExit("model not found: {}".format(path))
        for decoding, no_revisit in (("original", False), ("no_revisit", True)):
            plan.append(("Risk-DQN[{}/{}]".format(label, decoding),
                         label, path, decoding, no_revisit))
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CHOICES) + ["both"],
                        default="both")
    parser.add_argument("--episodes", type=int,
                        default=RTG.EPISODES_PER_TOPOLOGY)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    topology_files = sorted(TOPOLOGY_DIR.glob("*.json"))
    if not topology_files:
        raise SystemExit("no topology JSON in {}".format(TOPOLOGY_DIR))

    plan = arms(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device={} topologies={} episodes={} arms={}".format(
        device, len(topology_files), args.episodes, len(plan)), flush=True)
    for method, label, path, decoding, _ in plan:
        print("  {:28} model={} decoding={}".format(
            method, md5(path) if path else "-", decoding), flush=True)

    rows = []
    started = time.time()
    for method, label, path, decoding, no_revisit in plan:
        model = None
        digest = "-"
        if path is not None:
            RTG.MODEL_PATH = path
            digest = md5(path)
            model = RTG.load_model(topology_files[0], device)
        for topology_path in topology_files:
            result = evaluate(topology_path, model, device, args.episodes,
                              no_revisit)
            result.update({
                "topology": topology_path.stem,
                "rewire_ratio": RTG.read_rewire_ratio(topology_path),
                "method": method, "model_label": label,
                "model_md5": digest, "decoding": decoding,
            })
            rows.append(result)
            print("  {:22} {:26} succ={:6.2f}% delay={:7.2f} hop={:6.3f} "
                  "cnt={:5.3f} risky={:6.2f}% comp={:.5f}  ({:.1f} min)".format(
                      result["topology"], method, result["success_rate"],
                      result["delay"], result["hop"], result["risky_count"],
                      result["risky_link_usage"], result["loss_compounded"],
                      (time.time() - started) / 60), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    if not out_path.is_absolute() and not out_path.parent.name:
        out_path = OUT_DIR / out_path
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in FIELDS})
    print("\nwrote {} ({} rows) in {:.1f} min".format(
        out_path, len(rows), (time.time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()
