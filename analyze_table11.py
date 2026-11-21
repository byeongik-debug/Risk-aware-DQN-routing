# -*- coding: utf-8 -*-
"""Table 11 (learning-based models, Dynamic) vs the committed logs.

Two jobs
--------
1. Fill the two blank cells. The manuscript leaves QoS-Model's Hop and the
   `vs QoS-Model` Hop delta as [O]. Both are in eval_qos.py's output
   (exp1_reference.csv), which already supplies that row's Delay and Loss.

2. Check the cells that are NOT blank. Table 11's three model rows each carry
   a standard deviation, a Loss and a Hop. Those nine values are matched here
   against the only committed files that could have produced them. Whatever
   matches to the printed precision pins the row's provenance; whatever does
   not is then a transcription error in a row whose provenance is pinned.

Why the provenance check is the whole point
-------------------------------------------
A delay mean on its own could come from anywhere. A delay mean sitting next to
a standard deviation that reproduces a committed file to four significant
figures cannot: that pair identifies the file. So if the sd, the Loss and the
Hop all match one file and the delay does not, the delay is wrong -- it is not
evidence of some other, better evaluation.

The variance in these rows does not mean the same thing in each row
-------------------------------------------------------------------
    eval_baseline.py:15,20,22  one frozen model, SEEDS=[0..4], 200 episodes
    eval_norisk.py:13,18,19    one frozen model, SEEDS=[0..4], 200 episodes
    eval_qos.py:18,22          one frozen model, SEEDS=[0..4], 200 episodes
    run_norisk_all.py:13,33-46 OVERWRITES norisk_reference.csv's dynamic row
                               with the aggregate of 5 SEPARATELY TRAINED runs

So in the dynamic row the proposed and QoS sd's are evaluation-seed noise on a
single model, while NoRisk's sd is across-model training noise. They are not
interchangeable, and the asymmetry flatters the single-model rows: their sd is
missing the training variance that the ablation study measures at 15-22 ms
(FINDINGS_table10_f1.md 11.1/14.6). This script therefore reports the test
three ways -- as published, with the proposed sd inflated to NoRisk's, and
model-to-model with five independently trained runs on both sides.

Usage:
    python -X utf8 analyze_table11.py
"""
import csv
import io
import os

import review_stats as RS

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.join(os.path.dirname(HERE), "SDN_DQN_new_env", "results")
OUT = os.path.join(HERE, "review_results", "table11_learning_models.csv")
SCEN = "dynamic"

# Table 11 exactly as the manuscript prints it (p.11).
PUB = {
    "Proposed":     {"delay": 348.7, "delay_sd": 1.5, "loss": 8.4,
                     "loss_sd": 0.1, "hop": 7.7},
    "NoRisk-DQN":   {"delay": 427.2, "delay_sd": 3.9, "loss": 9.0,
                     "loss_sd": 0.2, "hop": 7.0},
    "QoS-Model":    {"delay": 403.7, "delay_sd": 2.4, "loss": 12.9,
                     "loss_sd": 0.2, "hop": None},   # [O] blank
}
PUB_VS = {
    ("NoRisk-DQN", "delay"): -18.4, ("NoRisk-DQN", "loss"): -6.7,
    ("NoRisk-DQN", "hop"): +0.7,
    ("QoS-Model", "delay"): -13.6, ("QoS-Model", "loss"): -34.9,
    ("QoS-Model", "hop"): None,                      # [O] blank
}


def rows(name):
    with io.open(os.path.join(ORIG, name), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def pick(name, **where):
    """The one row of a reference CSV matching every key=value in `where`."""
    hit = [r for r in rows(name)
           if all(r.get(k) == v for k, v in where.items())]
    assert len(hit) == 1, (name, where, len(hit))
    return hit[0]


def g(row, key, default=None):
    v = row.get(key)
    return default if v in (None, "") else float(v)


def welch_stats(m1, s1, n1, m2, s2, n2):
    """Welch's t-test from summary statistics. Returns (diff, t, df, p)."""
    v1, v2 = s1 * s1 / n1, s2 * s2 / n2
    se = (v1 + v2) ** 0.5
    if se == 0.0:
        return m1 - m2, float("inf"), 0.0, 0.0
    t = (m1 - m2) / se
    df = (v1 + v2) ** 2 / (v1 * v1 / (n1 - 1) + v2 * v2 / (n2 - 1))
    return m1 - m2, t, df, RS.t_two_sided_p(t, df)


def line(ch="-", n=78):
    print(ch * n)


# ---------------------------------------------------------------- 1. 출처 확인
print("=" * 78)
print("  1. 표 11의 세 행은 어느 파일에서 왔나 (± / Loss / Hop 대조)")
print("=" * 78)

prop = pick("baseline_reference.csv", scenario=SCEN)
nori = pick("norisk_reference.csv", scenario=SCEN)
qos = pick("exp1_reference.csv", model="QoS-DQN", scenario=SCEN)
SRC = {"Proposed": ("baseline_reference.csv", prop),
       "NoRisk-DQN": ("norisk_reference.csv", nori),
       "QoS-Model": ("exp1_reference.csv [QoS-DQN]", qos)}

MEAS = {}
for name, (src, r) in SRC.items():
    MEAS[name] = {"delay": g(r, "delay"), "delay_sd": g(r, "delay_std", 0.0),
                  "loss": g(r, "loss") * 100.0,
                  "loss_sd": g(r, "loss_std", 0.0) * 100.0,
                  "hop": g(r, "hop"), "hop_sd": g(r, "hop_std", 0.0),
                  "src": src}

print("%-11s %-9s %10s %10s %10s   %s"
      % ("행", "열", "원고", "측정", "차이", "판정"))
line()
mismatch, rounding = [], []
for name in ("Proposed", "NoRisk-DQN", "QoS-Model"):
    for col, prec in (("delay", 1), ("delay_sd", 1), ("loss", 1),
                      ("loss_sd", 1), ("hop", 1)):
        p, m = PUB[name][col], MEAS[name][col]
        if p is None:
            print("%-11s %-9s %10s %10.3f %10s   빈칸 -> 채움"
                  % (name, col, "[O]", m, "-"))
            continue
        tol = 0.5 * 10 ** (-prec)
        if abs(round(m, prec) - p) < 1e-9:
            note = "일치"
        elif abs(m - p) <= 2 * tol:
            note = "반올림 차이"
            rounding.append((name, col, p, m))
        else:
            note = "** 불일치 **"
            mismatch.append((name, col, p, m))
        print("%-11s %-9s %10.1f %10.3f %+10.3f   %s"
              % (name, col, p, m, m - p, note))
    print("%-11s %-9s  <- %s" % ("", "출처", MEAS[name]["src"]))
    line()

print("요약: 14개 비교(빈칸 1개 제외) 중 불일치 %d개, 반올림 차이 %d개"
      % (len(mismatch), len(rounding)))
for name, col, p, m in mismatch:
    print("   불일치     %s / %s : 원고 %.1f, 측정 %.4f" % (name, col, p, m))
for name, col, p, m in rounding:
    print("   반올림     %s / %s : 원고 %.1f, 측정 %.4f (%.1f 로 반올림된다)"
          % (name, col, p, m, round(m, 1)))

# -------------------------------------------- 2. NoRisk 행은 런별로도 확인된다
print()
print("=" * 78)
print("  2. NoRisk 행 교차검증 — 런별 파일 5개를 직접 합산")
print("=" * 78)
nr_runs = []
for i in range(1, 6):
    r = pick("norisk_dynamic_run%02d_result.csv" % i, run_id=str(i))
    nr_runs.append(g(r, "delay"))
    print("  run%02d  delay %10.4f   loss %.6f   hop %.3f"
          % (i, g(r, "delay"), g(r, "loss"), g(r, "hop")))
m_nr, s_nr = RS.mean_std(nr_runs)
print("  ------")
print("  5런 평균 %.6f ± %.6f" % (m_nr, s_nr))
print("  norisk_reference.csv  %.6f ± %.6f"
      % (MEAS["NoRisk-DQN"]["delay"], MEAS["NoRisk-DQN"]["delay_sd"]))
assert abs(m_nr - MEAS["NoRisk-DQN"]["delay"]) < 1e-6, "런 합산이 파일과 다르다"
assert abs(s_nr - MEAS["NoRisk-DQN"]["delay_sd"]) < 1e-6, "sd가 파일과 다르다"
print("  -> 일치. 이 행의 지연은 %.1f ms 이며, 원고의 %.1f ms 는 어느 런에도 없다"
      % (m_nr, PUB["NoRisk-DQN"]["delay"]))
print("     (런 범위 %.1f ~ %.1f ms)" % (min(nr_runs), max(nr_runs)))
assert not (min(nr_runs) <= PUB["NoRisk-DQN"]["delay"] <= max(nr_runs))

# ------------------------------------------------------- 3. vs 행 재계산 + 빈칸
print()
print("=" * 78)
print("  3. vs 행 재계산 (원고 규약: (제안 - 기타) / 기타, 음수 = 제안이 낮다)")
print("=" * 78)
print("%-12s %-7s %10s %10s   %s" % ("비교", "열", "원고", "재계산", "비고"))
line()
recalc = {}
for other in ("NoRisk-DQN", "QoS-Model"):
    for col in ("delay", "loss", "hop"):
        o, p = MEAS[other][col], MEAS["Proposed"][col]
        if col == "hop":
            val, unit = p - o, "홉"
        else:
            val, unit = (p - o) / o * 100.0, "%"
        recalc[(other, col)] = val
        pub = PUB_VS[(other, col)]
        if pub is None:
            note = "빈칸 -> %+.2f %s" % (val, unit)
            print("%-12s %-7s %10s %+10.2f   %s"
                  % (other, col, "[O]", val, note))
            continue
        flip = (pub < 0) != (val < 0)
        if flip:
            note = "** 부호 역전 **"
        elif abs(val - pub) < 0.5:
            note = "일치"
        else:
            note = "차이 %+.1f%s" % (val - pub, unit)
        print("%-12s %-7s %+10.1f %+10.2f   %s" % (other, col, pub, val, note))
    line()

# ------------------------------------------------- 4. 제안 대 NoRisk 통계 검정
print()
print("=" * 78)
print("  4. 제안 대 NoRisk (dynamic 지연) — 세 가지 방식")
print("=" * 78)
mp, sp = MEAS["Proposed"]["delay"], MEAS["Proposed"]["delay_sd"]
mn, sn = MEAS["NoRisk-DQN"]["delay"], MEAS["NoRisk-DQN"]["delay_sd"]

print("(ㄱ) 발표된 ± 그대로 (제안 sd 는 평가시드 노이즈뿐)")
d, t, df, p = welch_stats(mp, sp, 5, mn, sn, 5)
print("     제안 %.2f ± %.2f  대  NoRisk %.2f ± %.2f" % (mp, sp, mn, sn))
print("     차이 %+.2f ms   t=%.3f  df=%.2f  p=%s" % (d, t, df, RS.fmt_p(p)))

print()
print("(ㄴ) 민감도 — 제안 sd 를 NoRisk 수준(%.2f)으로 올리면" % sn)
d2, t2, df2, p2 = welch_stats(mp, sn, 5, mn, sn, 5)
print("     차이 %+.2f ms   t=%.3f  df=%.2f  p=%s" % (d2, t2, df2, RS.fmt_p(p2)))

print()
print("(ㄷ) 모델 대 모델 — 양쪽 다 독립 학습 5런")
TRAINED = []
ab = [r for r in rows("ablation_summary.csv")
      if r["scenario"] == SCEN and r["window"] == "5"]
assert len(ab) == 1
TRAINED.append(("Exp2 Proposed(w=5)", g(ab[0], "delay"),
                g(ab[0], "delay_std"), int(g(ab[0], "n_runs"))))
full_runs = []
for i in range(1, 6):
    pth = os.path.join(HERE, "review_results",
                       "factor_full_dynamic_run%02d_result.csv" % i)
    if not os.path.exists(pth):
        continue
    with io.open(pth, encoding="utf-8-sig") as f:
        full_runs.append(g(list(csv.DictReader(f))[0], "delay"))
if len(full_runs) >= 2:
    m_f, s_f = RS.mean_std(full_runs)
    TRAINED.append(("재학습 Full (11.1)", m_f, s_f, len(full_runs)))
for label, m, s, n in TRAINED:
    d3, t3, df3, p3 = welch_stats(m, s, n, m_nr, s_nr, len(nr_runs))
    print("     %-20s %7.2f ± %5.2f (n=%d)  대  NoRisk %7.2f ± %4.2f (n=%d)"
          % (label, m, s, n, m_nr, s_nr, len(nr_runs)))
    print("     %-20s 차이 %+.2f ms  t=%.3f  df=%.2f  p=%s"
          % ("", d3, t3, df3, RS.fmt_p(p3)))

print()
print("  세 방식 전부 부호가 같다: dynamic 에서 위험항을 완전히 제거한 모델이")
print("  제안 기법보다 지연이 낮다. 원고 표 11 의 -18.4% 는 제안이 더 낮다는")
print("  주장이므로 방향이 반대다.")

# ----------------------------------------------------------------- 5. 표 저장
print()
print("=" * 78)
print("  5. 표 11 — 측정값으로 다시 쓴 형태")
print("=" * 78)
print("%-12s %18s %14s %14s" % ("Model", "Delay (ms)", "Loss (%)", "Hop"))
line()
for name in ("Proposed", "NoRisk-DQN", "QoS-Model"):
    m = MEAS[name]
    print("%-12s %10.1f ± %-5.1f %8.2f ± %-3.2f %7.3f ± %-5.3f"
          % (name, m["delay"], m["delay_sd"], m["loss"], m["loss_sd"],
             m["hop"], m["hop_sd"]))
line()
for other in ("NoRisk-DQN", "QoS-Model"):
    print("%-12s %+15.1f%% %+12.1f%% %+14.3f"
          % ("vs " + other, recalc[(other, "delay")],
             recalc[(other, "loss")], recalc[(other, "hop")]))

if not os.path.isdir(os.path.dirname(OUT)):
    os.makedirs(os.path.dirname(OUT))
with io.open(OUT, "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row", "source_file", "delay", "delay_sd", "loss_pct",
                "loss_pct_sd", "hop", "hop_sd", "pub_delay", "pub_delay_sd",
                "pub_loss_pct", "pub_hop", "variance_meaning"])
    VAR = {"Proposed": "eval seeds on 1 frozen model",
           "NoRisk-DQN": "5 independently trained models",
           "QoS-Model": "eval seeds on 1 frozen model"}
    for name in ("Proposed", "NoRisk-DQN", "QoS-Model"):
        m = MEAS[name]
        w.writerow([name, m["src"], m["delay"], m["delay_sd"], m["loss"],
                    m["loss_sd"], m["hop"], m["hop_sd"],
                    PUB[name]["delay"], PUB[name]["delay_sd"],
                    PUB[name]["loss"], PUB[name]["hop"], VAR[name]])
    for other in ("NoRisk-DQN", "QoS-Model"):
        w.writerow(["vs " + other, "recomputed",
                    recalc[(other, "delay")], "", recalc[(other, "loss")], "",
                    recalc[(other, "hop")], "",
                    PUB_VS[(other, "delay")], "",
                    PUB_VS[(other, "loss")], PUB_VS[(other, "hop")],
                    "(other - proposed) / other, hop as difference"])
print()
print("저장: %s" % OUT)
