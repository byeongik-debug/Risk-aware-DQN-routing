import subprocess, sys, os

RESULT_DIR = "results"

# equal run05 + temporal run01~05
runs = [
    ("equal",    5),
    ("temporal", 1),
    ("temporal", 2),
    ("temporal", 3),
    ("temporal", 4),
    ("temporal", 5),
]

for config, run_id in runs:
    path = os.path.join(RESULT_DIR, f"weight_{config}_dynamic_run{run_id:02d}_result.csv")
    if os.path.exists(path):
        print(f"[SKIP] {config} run{run_id} (이미 완료)")
        continue
    print(f"\n===== [{config}] run{run_id} =====")
    subprocess.run([sys.executable, "train_weight_compare.py", config, str(run_id)])