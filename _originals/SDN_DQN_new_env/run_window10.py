# run_window10.py
# window=10 전용 실행 스크립트
# auto_pipeline.py가 window=3,7,10을 함께 다루지만,
# window=10만 추가로 돌릴 때 이 파일을 독립적으로 실행
#
# 실행 순서:
#   1) train_window_dqn.py 10 {run_id} → window=10으로 학습 (10회 반복)
#   2) eval_window_sweep.py → 결과 취합 및 CSV 저장

import io, os, sys, subprocess
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Windows에서 한글 출력 깨짐 방지
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run(script, *args, label=""):
    """서브프로세스로 스크립트 실행, stdout을 실시간으로 출력"""
    cmd = [sys.executable, "-u", os.path.join(BASE_DIR, script)] + [str(a) for a in args]
    if label:
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
    process = subprocess.Popen(
        cmd, cwd=BASE_DIR,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    if process.returncode != 0:
        print(f"[오류] {script} 실패 (code={process.returncode})")
        return False
    return True


def main():
    start = datetime.now()
    print(f"Window=10 실험 시작: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # window=10으로 10회 반복 학습 → windowl10_dynamic_run01~10_result.csv 생성
    for run_id in range(1, 11):
        run("train_window_dqn.py", 10, run_id,
            label=f"window=10 run {run_id}/10")

    # 10회 결과 취합해 window_sweep_summary.csv 업데이트
    run("eval_window_sweep.py", label="eval_window_sweep")

    elapsed = datetime.now() - start
    h, rem = divmod(int(elapsed.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    print(f"\n완료. 소요: {h}시간 {m}분 {s}초")


if __name__ == "__main__":
    main()