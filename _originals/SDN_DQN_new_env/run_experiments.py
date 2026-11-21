# run_experiments.py
# Risk-DQN 10회 반복 학습·평가 후 평균 출력
# 목적: 학습 결과의 통계적 안정성 확인 (1회 결과는 랜덤 시드에 따라 편차 있음)
# 실행 순서: risk_dqn_every_scenario.py → run_policy.py → 결과 수집 → 반복

import os
import csv
import subprocess
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "risk_only_model_final.pth")
CSV_PATH = os.path.join(BASE_DIR, "results", "multi_scenario_summary.csv")
RUNS = 10  # 반복 학습 횟수


def run_script(script_name):
    """지정된 스크립트를 서브프로세스로 실행, 리턴 코드 반환"""
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, script_name)],
        cwd=BASE_DIR,
    )
    return result.returncode


def read_csv():
    """run_policy.py가 생성한 multi_scenario_summary.csv 읽기"""
    rows = []
    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    all_runs = []

    for i in range(1, RUNS + 1):
        print(f"\n{'='*50}")
        print(f"  RUN {i} / {RUNS}")
        print(f"{'='*50}")

        # 1단계: 학습 — 새 모델 파일 생성
        print("[1/2] 학습 중...")
        rc = run_script("risk_dqn_every_scenario.py")
        if rc != 0:
            print(f"학습 실패 (run {i}), 건너뜀")
            continue

        # 2단계: 평가 — 학습된 모델로 4개 시나리오 평가 후 CSV 저장
        print("[2/2] 평가 중...")
        rc = run_script("run_policy.py")
        if rc != 0:
            print(f"평가 실패 (run {i}), 건너뜀")
            continue

        rows = read_csv()
        all_runs.append(rows)

        # 이번 회차 결과 개별 저장 (나중에 회차별 비교 가능)
        run_csv_path = os.path.join(BASE_DIR, "results", f"run_{i:02d}_summary.csv")
        with open(run_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Run {i} 완료. 개별 결과 저장: run_{i:02d}_summary.csv")

        # 모델 삭제 → 다음 회차에서 새로 학습
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print("모델 삭제 완료.")

    if not all_runs:
        print("수집된 결과 없음.")
        return

    # 10회 평균 계산
    metrics = ["success_rate", "delay", "loss", "risk", "hop", "unstable_usage", "congested_usage"]
    accum = defaultdict(lambda: defaultdict(float))
    count = defaultdict(int)

    for run_rows in all_runs:
        for row in run_rows:
            key = (row["scenario"], row["model"])
            for m in metrics:
                accum[key][m] += float(row[m])
            count[key] += 1

    print(f"\n{'='*110}")
    print(f"  {RUNS}회 평균 결과")
    print(f"{'='*110}")
    print(
        f"{'Scenario':12s} | {'Model':15s} | {'Success':>8s} | {'Delay':>10s} | "
        f"{'Loss':>10s} | {'Risk':>10s} | {'Hop':>6s} | {'Unstable':>9s} | {'Congested':>9s}"
    )
    print("-" * 110)

    avg_results = []
    for key in sorted(accum.keys()):
        scenario, model = key
        n = count[key]
        avg = {m: accum[key][m] / n for m in metrics}
        avg_results.append((scenario, model, avg))

        print(
            f"{scenario:12s} | {model:15s} | "
            f"{avg['success_rate']:7.2f}% | "
            f"{avg['delay']:10.3f} | "
            f"{avg['loss']:10.5f} | "
            f"{avg['risk']:10.5f} | "
            f"{avg['hop']:6.2f} | "
            f"{avg['unstable_usage']:8.2f}% | "
            f"{avg['congested_usage']:8.2f}%"
        )

    # avg_10runs_summary.csv 저장 — 통계 분석에 사용
    out_path = os.path.join(BASE_DIR, "results", "avg_10runs_summary.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["scenario", "model"] + metrics
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scenario, model, avg in avg_results:
            row = {"scenario": scenario, "model": model}
            row.update({m: avg[m] for m in metrics})
            writer.writerow(row)

    print(f"\n평균 결과 저장: {out_path}")


if __name__ == "__main__":
    main()