# auto_pipeline.py
# 전체 실험 파이프라인 자동 실행 스크립트
#
# 실행 전 전제 조건:
#   risk_only_model_final.pth 가 존재해야 함 (risk_dqn_every_scenario.py로 학습)
#
# 실행 순서:
#   Step 0: eval_baseline.py    → baseline_reference.csv 생성 (Risk-DQN 기준값)
#   실험 2: train_window_dqn.py (window=1, 3 runs) → eval_ablation.py
#   실험 3: train_window_dqn.py (window=3,7,10, 각 10 runs) → eval_window_sweep.py
#
# 모든 출력은 results/pipeline_log_YYYY-MM-DD_HH-MM.txt에 동시 저장

import io
import os
import sys
import subprocess
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
LOG_FILE = os.path.join(RESULT_DIR, f"pipeline_log_{timestamp}.txt")


class Tee:
    """stdout을 터미널과 로그 파일에 동시에 쓰는 래퍼"""
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def run(script, *args, label=""):
    """지정 스크립트를 서브프로세스로 실행, 실시간 출력
    실패 시 False 반환 (파이프라인 계속 실행할지 호출부에서 결정)"""
    cmd = [sys.executable, os.path.join(BASE_DIR, script)] + [str(a) for a in args]
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
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
        print(f"[오류] {script} {args} 실패 (returncode={process.returncode})")
        return False
    return True


def main():
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    start_time = datetime.now()
    print(f"파이프라인 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"로그 저장: {LOG_FILE}\n")

    # 모델 파일 존재 확인 (risk_dqn_every_scenario.py로 미리 학습 필요)
    model_path = os.path.join(BASE_DIR, "risk_only_model_final.pth")
    if not os.path.exists(model_path):
        print(f"[오류] 베이스라인 모델 없음: {model_path}")
        print("먼저 risk_dqn_every_scenario.py 를 실행해서 모델을 학습하세요.")
        sys.stdout = tee.terminal
        tee.close()
        return

    # ─── Step 0: 베이스라인 평가 ─────────────────────────────────────────────
    # baseline_reference.csv 생성 → 이후 모든 실험의 기준으로 사용
    print("\n" + "#" * 60)
    print("  Step 0: 베이스라인 모델 평가 (4 scenarios → 기준 저장)")
    print("#" * 60)

    ok = run("eval_baseline.py", label="[Baseline] 기준 평가")
    if not ok:
        print("[오류] 베이스라인 평가 실패. 파이프라인 중단.")
        sys.stdout = tee.terminal
        tee.close()
        return

    # ─── 실험 2: Temporal Ablation ──────────────────────────────────────────
    # window=1로 3번 반복 학습 → 학습 불안정성 완화를 위해 평균 사용
    print("\n" + "#" * 60)
    print("  실험 2: Temporal Ablation (window=1, 3 runs)")
    print("#" * 60)

    ablation_runs = 3
    for run_id in range(1, ablation_runs + 1):
        ok = run(
            "train_window_dqn.py", 1, run_id,
            label=f"[Ablation] window=1 run {run_id}/{ablation_runs} 학습"
        )
        if not ok:
            print(f"[경고] Ablation run {run_id} 실패, 계속 진행")

    run("eval_ablation.py", label="[Ablation] 결과 평가")

    # ─── 실험 3: Window Sweep ─────────────────────────────────────────────────
    # window=3,7,10을 각 10회 학습 → 학습 변동성 반영한 평균 사용
    print("\n" + "#" * 60)
    print("  실험 3: Window Sweep (window=3,7,10 × 10 runs, dynamic only)")
    print("#" * 60)

    sweep_windows = [3, 7, 10]
    sweep_runs = 10
    total = len(sweep_windows) * sweep_runs
    done_count = 0

    for window in sweep_windows:
        for run_id in range(1, sweep_runs + 1):
            done_count += 1
            ok = run(
                "train_window_dqn.py", window, run_id,
                label=f"[Sweep] window={window} run {run_id}/{sweep_runs}  ({done_count}/{total})"
            )
            if not ok:
                print(f"[경고] window={window} run={run_id} 실패, 계속 진행")

    run("eval_window_sweep.py", label="[Sweep] 결과 평가")

    # ─── 완료 ─────────────────────────────────────────────────────────────────
    end_time = datetime.now()
    elapsed = end_time - start_time
    hours, rem = divmod(int(elapsed.total_seconds()), 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 60)
    print("  파이프라인 완료")
    print(f"  시작: {start_time.strftime('%H:%M:%S')}")
    print(f"  종료: {end_time.strftime('%H:%M:%S')}")
    print(f"  소요: {hours}시간 {minutes}분 {seconds}초")
    print(f"  로그: {LOG_FILE}")
    print("=" * 60)

    sys.stdout = tee.terminal
    tee.close()


if __name__ == "__main__":
    main()