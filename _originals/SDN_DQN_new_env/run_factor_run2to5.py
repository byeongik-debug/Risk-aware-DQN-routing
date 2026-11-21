import subprocess, sys

factors = ["util", "cong", "delay_inst", "loss_inst", "link_type"]

for f in factors:
    for run_id in range(2, 6):
        print(f"\n===== [{f}] run{run_id} =====")
        subprocess.run([sys.executable, "train_factor_ablation.py", f, str(run_id)])