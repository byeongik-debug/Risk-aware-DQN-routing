# run_ablation_training.py
# Temporal Ablation(Exp 2)용 학습 자동화 스크립트
#
# 실행할 것:
#   window=5 run 1~5 : Proposed 모델 (temporal risk 포함 학습)
#   window=1 run 4~5 : NoTemporal 모델 (run 1~3은 이미 존재)
#
# 사용법: python run_ablation_training.py
# 순서대로 실행되며 각 학습마다 평가 후 CSV 저장

import subprocess
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run(window, run_id):
    cmd = [sys.executable, "train_window_dqn.py", str(window), str(run_id)]
    print(f"\n{'='*60}")
    print(f"  학습 시작: window={window} run={run_id}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        print(f"[오류] window={window} run={run_id} 학습 실패 (종료 코드 {result.returncode})")
    else:
        print(f"[완료] window={window} run={run_id}")

def main():
    tasks = []

    # window=5: 5회 학습 (Proposed — temporal risk 포함)
    for run_id in range(1, 6):
        tasks.append((5, run_id))

    # window=1: run 4, 5 추가 (run 1~3은 이미 완료)
    for run_id in range(4, 6):
        tasks.append((1, run_id))

    print(f"총 {len(tasks)}회 학습 예정:")
    for w, r in tasks:
        csv_check = os.path.join(BASE_DIR, "results", f"window{w}_normal_run{r:02d}_result.csv")
        status = "이미 완료" if os.path.exists(csv_check) else "학습 필요"
        print(f"  window={w} run={r:02d}  → {status}")

    print()
    for w, run_id in tasks:
        csv_check = os.path.join(BASE_DIR, "results", f"window{w}_normal_run{run_id:02d}_result.csv")
        if os.path.exists(csv_check):
            print(f"[건너뜀] window={w} run={run_id} (결과 CSV 이미 존재)")
            continue
        run(w, run_id)

    print("\n" + "="*60)
    print("  모든 학습 완료. 이제 아래 명령어로 결과 확인:")
    print("    python eval_ablation.py")
    print("    python show_results.py")
    print("="*60)

if __name__ == "__main__":
    main()