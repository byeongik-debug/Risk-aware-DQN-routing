# run_exp3_remaining.py
# Exp 3 Window Sweep 나머지 학습 자동화
#
# window=7: run 9, 10 (2회 추가)
# window=10: run 2~10 (9회 추가)
#
# 완료 후: python eval_window_sweep.py

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
        print(f"[오류] window={window} run={run_id} 종료 코드 {result.returncode}")
    else:
        print(f"[완료] window={window} run={run_id}")

def main():
    tasks = []

    # window=7: 8회 이미 존재 → 5회 기준이면 추가 불필요, 건너뜀
    # window=10: run 2~5 (4회 추가)
    for run_id in range(2, 6):
        tasks.append((10, run_id))

    print(f"총 {len(tasks)}회 학습 예정:")
    for w, r in tasks:
        csv_check = os.path.join(BASE_DIR, "results", f"window{w}_dynamic_run{r:02d}_result.csv")
        status = "이미 완료" if os.path.exists(csv_check) else "학습 필요"
        print(f"  window={w} run={r:02d}  → {status}")

    print()
    for w, run_id in tasks:
        csv_check = os.path.join(BASE_DIR, "results", f"window{w}_dynamic_run{run_id:02d}_result.csv")
        if os.path.exists(csv_check):
            print(f"[건너뜀] window={w} run={run_id}")
            continue
        run(w, run_id)

    print("\n" + "="*60)
    print("  완료. 이제 실행:")
    print("    python eval_window_sweep.py")
    print("    python show_results.py")
    print("="*60)

if __name__ == "__main__":
    main()
