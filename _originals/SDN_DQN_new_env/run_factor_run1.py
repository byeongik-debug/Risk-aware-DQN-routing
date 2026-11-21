import subprocess, sys

factors = ["util", "cong", "delay_inst", "loss_inst", "link_type"]
for f in factors:
    subprocess.run([sys.executable, "train_factor_ablation.py", f, "1"])