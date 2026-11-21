import csv, os

RESULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exp4_nowarmup.csv")
SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]

rows = {}
with open(RESULT_CSV, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows[(r["model"], r["scenario"])] = r

sep = "-" * 88
print("=" * 88)
print("  Exp 4 (No Warm-up): Risk-DQN vs NoRisk-DQN")
print("=" * 88)
print(f"  {'Scenario':12s} | {'Model':>10s} | {'Delay (mean±std)':>18s} | {'Loss (mean±std)':>22s} | {'Hop':>5s}")
print(sep)

for sc in SCENARIOS:
    r  = rows[("Risk-DQN", sc)]
    nr = rows[("NoRisk",   sc)]
    print(f"  {sc:12s} | {'Risk-DQN':>10s} | {float(r['delay']):6.1f} ± {float(r['delay_std']):4.1f} | "
          f"{float(r['loss']):.5f} ± {float(r['loss_std']):.5f} | {float(r['hop']):5.2f}")
    dd = float(nr['delay']) - float(r['delay'])
    dl = float(nr['loss'])  - float(r['loss'])
    dd_pct = dd / float(r['delay']) * 100 if float(r['delay']) else 0
    dl_pct = dl / float(r['loss'])  * 100 if float(r['loss'])  else 0
    print(f"  {sc:12s} | {'NoRisk':>10s} | {float(nr['delay']):6.1f} ± {float(nr['delay_std']):4.1f} | "
          f"{float(nr['loss']):.5f} ± {float(nr['loss_std']):.5f} | {float(nr['hop']):5.2f}  "
          f"(Δdelay={dd:+.1f}/{dd_pct:+.1f}%, Δloss={dl:+.5f}/{dl_pct:+.1f}%)")
    print(sep)