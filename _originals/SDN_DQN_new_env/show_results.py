# show_results.py
# 4개 실험 결과를 콘솔에 출력하는 통합 뷰어
# Exp 1: 3개 모델 모두 seed 5개 mean ± std 표시
# 각 실험 CSV를 읽어서 Risk-DQN 베이스라인 대비 diff를 함께 표시
#
# 실행 전 필요한 CSV 파일:
#   results/baseline_reference.csv   (eval_baseline.py 실행)
#   results/exp1_reference.csv        (eval_qos.py 실행)
#   results/temporal_ablation_same_model.csv (eval_temporal_ablation.py 실행)
#   results/window{n}_dynamic_run{i}.csv     (train_window_dqn.py 실행)
#   results/norisk_reference.csv      (eval_norisk.py 실행)

import csv, os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(BASE, "results")

def sep(n=58): print("-" * n)
def title(s): print("\n" + "=" * 70); print("  " + s); print("=" * 70)


def _get(row, key, default=0.0):
    """CSV 행에서 값 읽기 — 컬럼이 없으면 default 반환 (하위 호환)"""
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


# 베이스라인 먼저 로드 (모든 섹션에서 기준으로 사용)
baseline = {}
with open(os.path.join(R, "baseline_reference.csv")) as f:
    for r in csv.DictReader(f):
        baseline[r["scenario"]] = {
            "delay":     _get(r, "delay"),
            "delay_std": _get(r, "delay_std"),
            "loss":      _get(r, "loss"),
            "loss_std":  _get(r, "loss_std"),
            "hop":       _get(r, "hop"),
        }

# ── Baseline ──────────────────────────────────────────────────────────
# Risk-DQN(window=5, seed 5개) 기준값을 가장 먼저 출력
title("Baseline: Risk-DQN (window=5, 5 seeds)")
print(f"  {'Scenario':12s} | {'Success':>8s} | {'Delay (mean±std)':>18s} | {'Loss (mean±std)':>22s} | {'Hop':>5s}")
sep(80)
with open(os.path.join(R, "baseline_reference.csv")) as f:
    for r in csv.DictReader(f):
        d_s  = _get(r, "delay_std")
        l_s  = _get(r, "loss_std")
        d_str = f"{_get(r,'delay'):6.1f} ± {d_s:4.1f}"
        l_str = f"{_get(r,'loss'):7.5f} ± {l_s:.5f}"
        print(f"  {r['scenario']:12s} | {_get(r,'success_rate'):7.1f}% | {d_str:>18s} | {l_str:>22s} | {_get(r,'hop'):5.2f}")
sep(80)

# ── Exp 1: OSPF vs QoS-Model vs Risk-DQN ─────────────────────────────
# 세 모델 모두 seed 5개 mean ± std 표시
# diff는 Risk-DQN 평균값 대비 계산
exp1_csv = os.path.join(R, "exp1_reference.csv")
if os.path.exists(exp1_csv):
    title("Exp 1: OSPF vs QoS-Model vs Risk-DQN  (5 seeds, mean ± std)")
    W = 24  # diff 컬럼 너비
    HDR1 = (f"  {'Scenario':12s} | {'Model':>10s} | {'Delay (mean±std)':>18s} | "
            f"{'vs Risk-DQN':>{W}s} | {'Loss (mean±std)':>22s} | {'vs Risk-DQN':>{W}s}")
    print(HDR1)
    SEP1 = "-" * len(HDR1)
    print(SEP1)

    exp1 = {}
    with open(exp1_csv) as f:
        for r in csv.DictReader(f):
            exp1.setdefault(r["scenario"], {})[r["model"]] = r

    for sc in ["normal", "congestion", "unstable", "dynamic"]:
        b = baseline[sc]
        # Risk-DQN — baseline에서 가져옴
        bd_str = f"{b['delay']:6.1f} ± {b['delay_std']:4.1f}"
        bl_str = f"{b['loss']:7.5f} ± {b['loss_std']:.5f}"
        print(f"  {sc:12s} | {'Risk-DQN':>10s} | {bd_str:>18s} | {'(baseline)':>{W}s} | {bl_str:>22s} | {'(baseline)':>{W}s}")

        for model, label in [("QoS-DQN", "QoS-Model"), ("OSPF", "OSPF")]:
            if sc not in exp1 or model not in exp1[sc]:
                continue
            r = exp1[sc][model]
            d,  d_s = _get(r, "delay"),     _get(r, "delay_std")
            l,  l_s = _get(r, "loss"),      _get(r, "loss_std")
            # mean 대비 diff (±std는 diff 계산에 포함하지 않음 — 평균값만 비교)
            dd = f"{d - b['delay']:+.1f} ({(d - b['delay'])/b['delay']*100:+.1f}%)"
            dl = f"{l - b['loss']:+.5f} ({(l - b['loss'])/b['loss']*100:+.1f}%)"
            d_str = f"{d:6.1f} ± {d_s:4.1f}"
            l_str = f"{l:7.5f} ± {l_s:.5f}"
            print(f"  {sc:12s} | {label:>10s} | {d_str:>18s} | {dd:>{W}s} | {l_str:>22s} | {dl:>{W}s}")
        print(SEP1)
else:
    title("Exp 1: OSPF vs QoS-Model vs Risk-DQN")
    print("\n[결과 없음] python eval_qos.py 실행하세요.")

# ── Exp 2: Temporal Ablation ──────────────────────────────────────────
# Proposed(w=5)와 NoTemporal(w=1)을 각각 처음부터 별도 학습 후 비교
# 같은 모델로 평가 window만 바꾸는 방식이 아님 — 학습 자체가 다른 조건
ablation_csv = os.path.join(R, "ablation_summary.csv")
if os.path.exists(ablation_csv):
    title("Exp 2: Temporal Ablation  Proposed(w=5) vs NoTemporal(w=1)  (각 5회 학습, mean ± std)")
    W2 = 24
    HDR2 = (f"  {'Scenario':12s} | {'Model':>16s} | {'Delay (mean±std)':>18s} | "
            f"{'vs Proposed':>{W2}s} | {'Loss (mean±std)':>22s} | {'vs Proposed':>{W2}s}")
    SEP2 = "-" * len(HDR2)
    print(HDR2)
    print(SEP2)

    ab_rows = list(csv.DictReader(open(ablation_csv)))
    w5_ab = {r["scenario"]: r for r in ab_rows if r["window"] == "5"}

    for r in ab_rows:
        sc = r["scenario"]
        w  = r["window"]
        d,   d_s = _get(r, "delay"), _get(r, "delay_std")
        l,   l_s = _get(r, "loss"),  _get(r, "loss_std")
        n        = int(_get(r, "n_runs"))
        d_str = f"{d:6.1f} ± {d_s:4.1f}"
        l_str = f"{l:7.5f} ± {l_s:.5f}"
        if w == "5":
            label = "Proposed(w=5)"
            print(f"  {sc:12s} | {label:>16s} | {d_str:>18s} | {'(baseline)':>{W2}s} | {l_str:>22s} | {'(baseline)':>{W2}s}  n={n}")
        else:
            label = "NoTemporal(w=1)"
            ref = w5_ab.get(sc)
            if ref:
                rd, rl = _get(ref, "delay"), _get(ref, "loss")
                dd = f"{d - rd:+.1f} ({(d - rd)/rd*100:+.1f}%)"
                dl = f"{l - rl:+.5f} ({(l - rl)/rl*100:+.1f}%)"
            else:
                dd = dl = "N/A"
            print(f"  {sc:12s} | {label:>16s} | {d_str:>18s} | {dd:>{W2}s} | {l_str:>22s} | {dl:>{W2}s}  n={n}")
            print(SEP2)
    print(SEP2)
else:
    title("Exp 2: Temporal Ablation  Proposed(w=5) vs NoTemporal(w=1)")
    print("\n[결과 없음]")
    print("  1. python train_window_dqn.py 5 1  ~  python train_window_dqn.py 5 5")
    print("  2. python train_window_dqn.py 1 4  (run=4,5 추가)")
    print("  3. python eval_ablation.py")

# ── Exp 3: Window Sweep ───────────────────────────────────────────────
# w=5: 5개 시드 mean ± std / w=3,7,10: 10회 학습 run mean ± std
title("Exp 3: Window Sweep vs Baseline (dynamic only, mean ± std)")
W3 = 24
HDR3 = (f"  {'Window':>8s} | {'Delay (mean±std)':>18s} | {'vs w=5':>{W3}s} | "
        f"{'Loss (mean±std)':>22s} | {'vs w=5':>{W3}s} | {'N':>4s}")
SEP3 = "-" * len(HDR3)
print(HDR3)
print(SEP3)

sweep_csv = os.path.join(R, "window_sweep_summary.csv")
if os.path.exists(sweep_csv):
    sweep_rows = {int(r["window"]): r for r in csv.DictReader(open(sweep_csv))}
    if 5 in sweep_rows:
        r5 = sweep_rows[5]
        base_d, base_d_s = _get(r5, "delay"), _get(r5, "delay_std")
        base_l, base_l_s = _get(r5, "loss"),  _get(r5, "loss_std")
        n5 = int(_get(r5, "n_runs"))
        d_str = f"{base_d:6.1f} ± {base_d_s:4.1f}"
        l_str = f"{base_l:7.5f} ± {base_l_s:.5f}"
        print(f"  {'w=5':>8s} | {d_str:>18s} | {'(baseline)':>{W3}s} | {l_str:>22s} | {'(baseline)':>{W3}s} | {n5:>4d}")
        for w in [3, 7, 10]:
            if w not in sweep_rows:
                continue
            rw = sweep_rows[w]
            d, d_s = _get(rw, "delay"), _get(rw, "delay_std")
            l, l_s = _get(rw, "loss"),  _get(rw, "loss_std")
            n = int(_get(rw, "n_runs"))
            dd = f"{d - base_d:+.1f} ({(d - base_d)/base_d*100:+.1f}%)"
            dl = f"{l - base_l:+.5f} ({(l - base_l)/base_l*100:+.1f}%)"
            d_str = f"{d:6.1f} ± {d_s:4.1f}"
            l_str = f"{l:7.5f} ± {l_s:.5f}"
            print(f"  {'w='+str(w):>8s} | {d_str:>18s} | {dd:>{W3}s} | {l_str:>22s} | {dl:>{W3}s} | {n:>4d}")
else:
    print("  [결과 없음] eval_window_sweep.py 실행하세요.")
print(SEP3)

# ── Exp 4: NoRisk vs Risk-DQN ─────────────────────────────────────────
# 두 모델 모두 5개 시드 mean ± std 표시
norisk_csv = os.path.join(R, "norisk_reference.csv")
if os.path.exists(norisk_csv):
    title("Exp 4: NoRisk-DQN vs Risk-DQN (5 seeds, mean ± std)")
    W4 = 24
    HDR4 = (f"  {'Scenario':12s} | {'Model':>10s} | {'Delay (mean±std)':>18s} | "
            f"{'vs Risk-DQN':>{W4}s} | {'Loss (mean±std)':>22s} | {'vs Risk-DQN':>{W4}s}")
    SEP4 = "-" * len(HDR4)
    print(HDR4)
    print(SEP4)
    norisk = {}
    with open(norisk_csv) as f:
        for r in csv.DictReader(f):
            norisk[r["scenario"]] = r
    for sc in ["normal", "congestion", "unstable", "dynamic"]:
        b = baseline[sc]
        bd_str = f"{b['delay']:6.1f} ± {b['delay_std']:4.1f}"
        bl_str = f"{b['loss']:7.5f} ± {b['loss_std']:.5f}"
        print(f"  {sc:12s} | {'Risk-DQN':>10s} | {bd_str:>18s} | {'(baseline)':>{W4}s} | {bl_str:>22s} | {'(baseline)':>{W4}s}")
        if sc in norisk:
            nr = norisk[sc]
            nd,  nd_s = _get(nr, "delay"), _get(nr, "delay_std")
            nl,  nl_s = _get(nr, "loss"),  _get(nr, "loss_std")
            dd = f"{nd - b['delay']:+.1f} ({(nd - b['delay'])/b['delay']*100:+.1f}%)"
            dl = f"{nl - b['loss']:+.5f} ({(nl - b['loss'])/b['loss']*100:+.1f}%)"
            nd_str = f"{nd:6.1f} ± {nd_s:4.1f}"
            nl_str = f"{nl:7.5f} ± {nl_s:.5f}"
            print(f"  {sc:12s} | {'NoRisk':>10s} | {nd_str:>18s} | {dd:>{W4}s} | {nl_str:>22s} | {dl:>{W4}s}")
        print(SEP4)
else:
    print("\n[NoRisk 결과 없음] train_norisk_dqn.py -> eval_norisk.py 순으로 실행하세요.")