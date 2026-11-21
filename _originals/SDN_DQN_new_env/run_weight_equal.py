import subprocess, sys, os

RESULT_DIR = "results"

for run_id in range(1, 6):
    path = os.path.join(RESULT_DIR, f"weight_equal_dynamic_run{run_id:02d}_result.csv")
    if os.path.exists(path):
        print(f"[SKIP] equal run{run_id} (이미 완료)")
        continue
    print(f"\n===== [W-Equal] run{run_id} =====")
    subprocess.run([sys.executable, "train_weight_compare.py", "equal", str(run_id)])