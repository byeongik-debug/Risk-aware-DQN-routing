"""R3-4 규모/구성 일반화 재평가 — 동결 모델을 명시적으로 지정해서 돌린다.

문제. `run_topology_generalization_dynamic.py`는 모델 경로가 상대경로다
(`MODEL_PATH = Path("risk_only_model_final.pth")`, 34행). 그래서 그 스크립트를
`review_response/`에서 실행하면 **그 디렉터리의 사본**을 집는다. 그런데 그 사본은
친구 repo(`Risk-aware-DQN-routing/DQN_models_runner/`)에서 2026-11-10 병합 때 들어온
것이고, 논문 본표의 평가 스크립트들(`SDN_DQN_new_env/eval_baseline.py` 등)이 쓰는
`SDN_DQN_new_env/risk_only_model_final.pth`(2026-07-11)와 **다른 모델**이다:

    review_response/   md5 770d17d9...  936293 B  Nov 10 22:47
    SDN_DQN_new_env/   md5 20e74c39...  935868 B  Jul 11 14:22

구조는 같아서(입력 125, 키 10개) 둘 다 조용히 로드된다 — 그래서 위험하다. 가중치는
별개로 학습된 것이다: 동일 원소가 232,985개 중 13,813개(5.93%)뿐이고 최대 차이가
가중치 규모와 맞먹는다(상대 최대 1.02). 재저장이 아니다.

이 스크립트가 하는 일:

1. `MODEL_PATH`를 동결 기준(`../SDN_DQN_new_env/...`)으로 **런타임 치환**한다.
   원본 파일은 수정하지 않는다.
2. `--model both`(기본)이면 **두 모델을 같은 토폴로지·같은 시드로 평가**한다. 병합
   함정이 R3-4 결론을 바꿨는지가 그래야 측정된다. OSPF는 모델과 무관하므로 한 번만 돈다.
3. 출력을 `review_results/`로 보낸다. 원본 `results/`를 덮지 않는다.
4. 각 행에 쓰인 모델의 md5를 기록한다. 출처가 CSV 안에 남아야 재현 가능하다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_topology_generalization.py
"""

import argparse
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import torch  # noqa: E402

import run_topology_generalization_dynamic as RTG  # noqa: E402

FROZEN_MODEL = Path(BASE_DIR) / ".." / "SDN_DQN_new_env" / "risk_only_model_final.pth"
MERGED_MODEL = Path(BASE_DIR) / "risk_only_model_final.pth"
TOPOLOGY_DIR = Path(BASE_DIR) / "topologies_dynamic"
OUT_DIR = Path(BASE_DIR) / "review_results"
OUT_CSV = OUT_DIR / "topology_generalization_runs.csv"

MODEL_CHOICES = {"frozen": FROZEN_MODEL, "merged": MERGED_MODEL}


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_models(choice):
    """[(label, path)] — 평가할 모델 목록."""
    labels = list(MODEL_CHOICES) if choice == "both" else [choice]
    chosen = []
    for label in labels:
        path = MODEL_CHOICES[label].resolve()
        if not path.exists():
            raise SystemExit("model not found: {}".format(path))
        chosen.append((label, path))
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=sorted(MODEL_CHOICES) + ["both"], default="both",
        help="frozen = SDN_DQN_new_env 기준(본표와 동일), "
             "merged = review_response 사본(친구 repo 유래), both = 둘 다 비교")
    parser.add_argument("--episodes", type=int, default=RTG.EPISODES_PER_TOPOLOGY)
    parser.add_argument("--out", default=str(OUT_CSV))
    args = parser.parse_args()

    topology_files = sorted(TOPOLOGY_DIR.glob("*.json"))
    if not topology_files:
        raise SystemExit("no topology JSON in {}".format(TOPOLOGY_DIR))

    models = resolve_models(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 원본 모듈 전역을 이 프로세스 안에서만 치환한다.
    RTG.TOPOLOGY_DIR = TOPOLOGY_DIR
    RTG.EPISODES_PER_TOPOLOGY = args.episodes

    print("device={} topologies={} episodes/topology={}".format(
        device, len(topology_files), args.episodes), flush=True)
    for label, path in models:
        print("  model[{}] {}  md5={}".format(label, path, md5(path)), flush=True)

    rows = []
    started = time.time()

    # OSPF는 모델과 무관하므로 토폴로지당 한 번만 평가한다.
    for index, topology_path in enumerate(topology_files, start=1):
        row = RTG.evaluate_ospf(topology_path)
        row["model_label"] = "-"
        row["model_md5"] = "-"
        rows.append(row)
        print("  [{:2}/{}] {:24} OSPF      success={:6.2f}% delay={:7.2f} "
              "loss={:.5f} hop={:.3f}  ({:.1f} min)".format(
                  index, len(topology_files), row["topology"], row["success_rate"],
                  row["delay"],
                  row["loss"],
                  row["hop"],
                  (time.time() - started) / 60), flush=True)

    for label, path in models:
        RTG.MODEL_PATH = path
        digest = md5(path)
        model = RTG.load_model(topology_files[0], device)
        for index, topology_path in enumerate(topology_files, start=1):
            row = RTG.evaluate_dqn(topology_path, model, device)
            row["method"] = "Risk-DQN[{}]".format(label)
            row["model_label"] = label
            row["model_md5"] = digest
            rows.append(row)
            print("  [{:2}/{}] {:24} {:14} success={:6.2f}% delay={:7.2f} "
                  "loss={:.5f} hop={:.3f}  ({:.1f} min)".format(
                      index, len(topology_files), row["topology"], row["method"],
                      row["success_rate"],
                      row["delay"],
                      row["loss"],
                      row["hop"],
                      (time.time() - started) / 60), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    out_path = Path(args.out)
    if not out_path.is_absolute() and not out_path.parent.name:
        out_path = OUT_DIR / out_path
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("\nwrote {} ({} rows) in {:.1f} min".format(
        out_path, len(rows), (time.time() - started) / 60), flush=True)


if __name__ == "__main__":
    main()
