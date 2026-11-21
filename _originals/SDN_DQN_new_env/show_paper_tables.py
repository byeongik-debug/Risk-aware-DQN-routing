import csv, os, sys
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

def g(row, key):
    try: return float(row[key])
    except: return 0.0


# ── 표 3 컬럼 너비 ────────────────────────────────────────────────────────────
WS  = 11   # Scenario
WA  = 10   # Algorithm
WD  = 14   # Delay   "594.2 ±  4.6"
WL  = 22   # Loss    "0.30005 ± 0.00221"
WR  = 17   # Risk    "0.1957 ± 0.0003"
WH  =  5   # Hop
SEP3 = 2 + WS + 3 + WA + 3 + WD + 3 + WL + 3 + WR + 3 + WH

def hline3(): print("=" * SEP3)
def dline3(): print("-" * SEP3)

def row3(sc, alg, d, ds, l, ls, rk, rks, hop):
    d_str = f"{d:5.1f} ± {ds:4.1f}"
    l_str = f"{l:.5f} ± {ls:.5f}"
    r_str = f"{rk:.4f} ± {rks:.4f}" if rk > 0 else "N/A"
    print(f"  {sc:<{WS}} | {alg:<{WA}} | {d_str:>{WD}} | {l_str:>{WL}} | {r_str:>{WR}} | {hop:>{WH}.2f}")


# ── 표 5 컬럼 너비 ────────────────────────────────────────────────────────────
WM   = 10   # Model
WDV  = 14   # Delay value   "400.1 ± 35.2"
WVD  = 16   # vs Full delay "+31.3 (+8.5%)"
WLV  = 22   # Loss value    "0.07953 ± 0.00352"
WVL  = 18   # vs Full loss  "-0.00835 (-10.5%)"
SEP5 = 2 + WM + 3 + WDV + 3 + WVD + 3 + WLV + 3 + WVL

def hline5(): print("=" * SEP5)
def dline5(): print("-" * SEP5)

def row5(name, dm, ds, lm, ls, ref_d=None, ref_l=None):
    d_str = f"{dm:5.1f} ± {ds:4.1f}"
    l_str = f"{lm:.5f} ± {ls:.5f}"
    if ref_d is None:
        dv, lv = "(baseline)", "(baseline)"
    else:
        dd = (dm - ref_d) / ref_d * 100
        dl = (lm - ref_l) / ref_l * 100
        dv = f"{dm - ref_d:+.1f} ({dd:+.1f}%)"
        lv = f"{lm - ref_l:+.5f} ({dl:+.1f}%)"
    print(f"  {name:<{WM}} | {d_str:>{WDV}} | {dv:>{WVD}} | {l_str:>{WLV}} | {lv:>{WVL}}")


# ── 표 3 출력 ────────────────────────────────────────────────────────────────
ospf = {}
with open(os.path.join(BASE, "exp1_reference.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["model"] == "OSPF":
            ospf[r["scenario"]] = r

risk = {}
with open(os.path.join(BASE, "baseline_reference.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        risk[r["scenario"]] = r

SCENARIOS = [
    ("normal",     "Normal"),
    ("congestion", "Congestion"),
    ("unstable",   "Unstable"),
    ("dynamic",    "Dynamic"),
]

print()
hline3()
print("  Table 3: Scenario Performance Comparison  (mean +- std, n=5)")
hline3()
print(f"  {'Scenario':<{WS}} | {'Algorithm':<{WA}} | {'Delay (ms)':>{WD}} | {'Loss':>{WL}} | {'Risk':>{WR}} | {'Hop':>{WH}}")
dline3()

for sc, label in SCENARIOS:
    o = ospf.get(sc, {})
    r = risk.get(sc, {})
    row3(label, "OSPF",
         g(o,"delay"), g(o,"delay_std"),
         g(o,"loss"),  g(o,"loss_std"),
         0, 0, g(o,"hop"))
    row3("", "Risk-DQN",
         g(r,"delay"), g(r,"delay_std"),
         g(r,"loss"),  g(r,"loss_std"),
         g(r,"risk"),  g(r,"risk_std"), g(r,"hop"))
    dline3()

hline3()


# ── 표 5 출력 ────────────────────────────────────────────────────────────────
norisk = {}
with open(os.path.join(BASE, "norisk_reference.csv"), encoding="utf-8") as f:
    for r in csv.DictReader(f):
        norisk[r["scenario"]] = r

# Dynamic 시나리오만
nr = norisk["dynamic"]
pr = risk["dynamic"]   # Proposed = Risk-DQN (baseline_reference)

# 홉 컬럼 추가를 위해 표5 전용 너비 재정의
WM2  = 12
WDV2 = 14
WVD2 = 18
WLV2 = 22
WVL2 = 18
WH2  =  5
SEP5 = 2 + WM2 + 3 + WDV2 + 3 + WVD2 + 3 + WLV2 + 3 + WVL2 + 3 + WH2

def hline5(): print("=" * SEP5)
def dline5(): print("-" * SEP5)

def row5b(name, dm, ds, lm, ls, hop, ref_d=None, ref_l=None):
    d_str = f"{dm:5.1f} ± {ds:4.1f}"
    l_str = f"{lm:.5f} ± {ls:.5f}"
    if ref_d is None:
        dv, lv = "(baseline)", "(baseline)"
    else:
        dd = (dm - ref_d) / ref_d * 100
        dl = (lm - ref_l) / ref_l * 100
        dv = f"{dm - ref_d:+.1f} ({dd:+.1f}%)"
        lv = f"{lm - ref_l:+.5f} ({dl:+.1f}%)"
    print(f"  {name:<{WM2}} | {d_str:>{WDV2}} | {dv:>{WVD2}} | {l_str:>{WLV2}} | {lv:>{WVL2}} | {hop:>{WH2}.2f}")

pd = g(pr,"delay"); pl = g(pr,"loss")

print()
hline5()
print("  Table 5: Ablation Study on Temporal Risk — Dynamic Scenario  (mean +- std, n=5)")
hline5()
print(f"  {'Model':<{WM2}} | {'Delay (ms)':>{WDV2}} | {'vs Proposed':>{WVD2}} | {'Loss':>{WLV2}} | {'vs Proposed':>{WVL2}} | {'Hop':>{WH2}}")
dline5()

row5b("Proposed",    pd, g(pr,"delay_std"), pl, g(pr,"loss_std"), g(pr,"hop"))
row5b("NoRisk-DQN",  g(nr,"delay"), g(nr,"delay_std"),
                     g(nr,"loss"),  g(nr,"loss_std"), g(nr,"hop"), pd, pl)

hline5()
print()