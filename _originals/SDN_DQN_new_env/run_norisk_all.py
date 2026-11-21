import subprocess, sys, os, csv
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))

# 1~5 순서대로 학습
for run_id in range(1, 6):
    print(f"\n{'='*60}")
    print(f"  NoRisk Run {run_id}/5")
    print(f"{'='*60}")
    subprocess.run([sys.executable, "train_norisk_dqn.py", str(run_id)], cwd=BASE)

# 결과 집계 → norisk_reference.csv (dynamic 행만 갱신)
print("\n\n===== 결과 집계 중 =====")
delays, losses, hops = [], [], []
for run_id in range(1, 6):
    path = os.path.join(BASE, "results", f"norisk_dynamic_run{run_id:02d}_result.csv")
    with open(path, encoding="utf-8") as f:
        row = list(csv.DictReader(f))[0]
        delays.append(float(row["delay"]))
        losses.append(float(row["loss"]))
        hops.append(float(row["hop"]))

dm  = float(np.mean(delays));  ds = float(np.std(delays,  ddof=1))
lm  = float(np.mean(losses));  ls = float(np.std(losses,  ddof=1))
hm  = float(np.mean(hops));    hs = float(np.std(hops,    ddof=1))

print(f"\n[Dynamic 최종]")
print(f"  Delay : {dm:.1f} ± {ds:.1f} ms")
print(f"  Loss  : {lm:.5f} ± {ls:.5f}")
print(f"  Hop   : {hm:.2f} ± {hs:.2f}")

# norisk_reference.csv의 dynamic 행 업데이트
ref_path = os.path.join(BASE, "results", "norisk_reference.csv")
rows = []
with open(ref_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    for row in reader:
        if row["scenario"] == "dynamic":
            row["delay"]     = dm
            row["delay_std"] = ds
            row["loss"]      = lm
            row["loss_std"]  = ls
            row["hop"]       = hm
            row["hop_std"]   = hs
        rows.append(row)

with open(ref_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)

print(f"\nnorisk_reference.csv 업데이트 완료")
print("===== 완료 =====")