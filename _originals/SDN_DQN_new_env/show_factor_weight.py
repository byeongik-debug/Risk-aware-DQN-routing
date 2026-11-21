import sys, os, csv, glob
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# 고정 컬럼 너비
WM  = 10   # Model
WD  = 13   # Delay value  "XXX.X ±  XX.X"
WVD = 16   # vs delay     "+XX.X (+XX.X%)"
WL  = 17   # Loss value   "0.XXXXX ± 0.XXXXX"
WVL = 18   # vs loss      "-0.XXXXX (-XX.X%)"
SEP_W = 2 + WM + 3 + WD + 3 + WVD + 3 + WL + 3 + WVL   # = 88

def hline(): print("=" * SEP_W)
def dline(): print("-" * SEP_W)

def header(vs_label="vs Full"):
    print(f"  {'Model':<{WM}} | {'Delay (ms)':>{WD}} | {vs_label:>{WVD}} | {'Loss':>{WL}} | {vs_label:>{WVL}}")

def row(name, dm, ds, lm, ls, ref_d=None, ref_l=None):
    d_str = f"{dm:6.1f} ± {ds:4.1f}"
    l_str = f"{lm:.5f} ± {ls:.5f}"
    if ref_d is None:
        dv, lv = "(baseline)", "(baseline)"
    else:
        dd = (dm - ref_d) / ref_d * 100
        dl = (lm - ref_l) / ref_l * 100
        dv = f"{dm - ref_d:+.1f} ({dd:+.1f}%)"
        lv = f"{lm - ref_l:+.5f} ({dl:+.1f}%)"
    print(f"  {name:<{WM}} | {d_str:>{WD}} | {dv:>{WVD}} | {l_str:>{WL}} | {lv:>{WVL}}")

def load_runs(prefix, scenario="dynamic"):
    rows = []
    for f in sorted(glob.glob(os.path.join(BASE, f"{prefix}_{scenario}_run*.csv"))):
        with open(f, encoding="utf-8") as fp:
            r = list(csv.DictReader(fp))[0]
            rows.append({"delay": float(r["delay"]), "loss": float(r["loss"])})
    return rows

def mean_std(rows):
    d = [r["delay"] for r in rows]
    l = [r["loss"]  for r in rows]
    return np.mean(d), np.std(d, ddof=1), np.mean(l), np.std(l, ddof=1), len(rows)

ORDER = ["Full", "-Util", "-Cong", "-ΔDelay", "-ΔLoss", "-Type"]


# ── 1. Factor Ablation — Dynamic ─────────────────────────────────────────
ablation_rows = {}
with open(os.path.join(BASE, "factor_ablation_summary.csv"), encoding="utf-8") as fp:
    for r in csv.DictReader(fp):
        ablation_rows[r["model"]] = r

fd = float(ablation_rows["Full"]["delay"])
fl = float(ablation_rows["Full"]["loss"])

print()
hline()
print("  Factor Ablation — Dynamic Scenario  (mean ± std, n=5)")
hline()
header()
dline()
for name in ORDER:
    r = ablation_rows[name]
    dm, ds = float(r["delay"]), float(r["delay_std"])
    lm, ls = float(r["loss"]),  float(r["loss_std"])
    row(name, dm, ds, lm, ls, None if name == "Full" else fd, None if name == "Full" else fl)
hline()


# ── 2. Factor Ablation — Fixed Scenarios ─────────────────────────────────
fixed_rows = {}
with open(os.path.join(BASE, "factor_ablation_fixed_scenarios.csv"), encoding="utf-8") as fp:
    for r in csv.DictReader(fp):
        sc = r["scenario"]
        if sc not in fixed_rows:
            fixed_rows[sc] = {}
        fixed_rows[sc][r["model"]] = r

for scenario, label in [("normal", "Normal"), ("congestion", "Congestion"), ("unstable", "Unstable")]:
    rows = fixed_rows.get(scenario, {})
    if not rows:
        continue
    fd2 = float(rows["Full"]["delay"])
    fl2 = float(rows["Full"]["loss"])
    print()
    hline()
    print(f"  Factor Ablation — {label} Scenario  (mean ± std, n=5)")
    hline()
    header()
    dline()
    for name in ORDER:
        if name not in rows:
            continue
        r = rows[name]
        dm, ds = float(r["delay"]), float(r["delay_std"])
        lm, ls = float(r["loss"]),  float(r["loss_std"])
        row(name, dm, ds, lm, ls, None if name == "Full" else fd2, None if name == "Full" else fl2)
    hline()


# ── 3. Weight Comparison — Dynamic ───────────────────────────────────────
weight_configs = [
    ("W-Current",  "window5"),
    ("W-Equal",    "weight_equal"),
    ("W-Temporal", "weight_temporal"),
]

wresults = []
for label, prefix in weight_configs:
    runs = load_runs(prefix, "dynamic")
    if not runs:
        continue
    dm, ds, lm, ls, n = mean_std(runs)
    wresults.append((label, dm, ds, lm, ls, n))

print()
hline()
print("  Weight Comparison — Dynamic Scenario  (mean ± std, n=5)")
hline()
header("vs Current")
dline()
if wresults:
    bd, bl = wresults[0][1], wresults[0][3]
    for label, dm, ds, lm, ls, n in wresults:
        row(label, dm, ds, lm, ls, None if label == "W-Current" else bd, None if label == "W-Current" else bl)
hline()
print()