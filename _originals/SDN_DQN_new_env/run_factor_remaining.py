import subprocess, sys, os

RESULT_DIR = "results"
factors = ["util", "cong", "delay_inst", "loss_inst", "link_type"]

for f in factors:
    for run_id in range(1, 6):
        path = os.path.join(RESULT_DIR, f"factor_{f}_dynamic_run{run_id:02d}_result.csv")
        if os.path.exists(path):
            print(f"[SKIP] {f} run{run_id} (이미 완료)")
            continue
        print(f"\n===== [{f}] run{run_id} =====")
        subprocess.run([sys.executable, "train_factor_ablation.py", f, str(run_id)])
