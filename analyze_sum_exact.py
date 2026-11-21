# -*- coding: utf-8 -*-
"""23.9.6.10 의 '합 버전'·'홉당 버전'을 **근사 없이** 다시 계산한다.

23.9.6.10 은 이렇게 적어 두고 한계로 남겼다:
  "합 버전은 mean(loss_i) × mean(hop_i) 로, 에피소드별
   mean(loss_i × hop_i) 가 아니다(CSV 에 에피소드별 값이 없다 —
   Jensen 격차). 정확히 하려면 측정 스크립트에 loss_sum 열을 넣고
   재측정해야 한다."
`measure_episode_metrics.py` 가 그 열을 만들었으므로 여기서 메운다.

하는 것:
1) 모델별로 에피소드 평균을 다시 낸다 — 근사(mean×mean)와 정확
   (mean of product) 둘 다. 격차를 수치로 보여 준다.
2) 2×2 대비(코사인/노름/상호작용)를 **정확한 종단간 합**과 **정확한
   홉당 지연**으로 계산하고 n=10/5/3 Holm 을 매긴다.
3) 23.9.6.10 표의 근사값과 **나란히** 찍어 결론이 바뀌는지 본다.

족 주의: 23.9.6.10 은 손실 계열·지연 계열을 **각각 족 15** 로 보정했다.
여기서는 변형 수가 다르므로 **족을 따로 세고 표에 적는다**. 같은 대비의
p 를 족을 섞어 비교하면 안 된다(23.9.6.6).

출력: review_results/sum_exact.csv
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402
sys.path.append(HERE)
import review_stats as RS  # noqa: E402
import reanalyze_cluster_n as RC  # noqa: E402
import analyze_axis_factorial as AX  # noqa: E402

OUT_DIR = OC.OUT_DIR
EM_CSV = os.path.join(OUT_DIR, "episode_metrics.csv")
CELLS = (AX.A, AX.B, AX.C, AX.D)


def read_episodes():
    """조건 -> 모델 -> 에피소드 행 리스트."""
    out = {}
    with io.open(EM_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = out.setdefault(r["condition"], {}).setdefault(r["model"], [])
            d.append({
                "delay": float(r["delay"]),
                "loss_perhop": float(r["loss_perhop"]),
                "loss_sum": float(r["loss_sum"]),
                "hop": float(r["hop"]),
                "hop_count": float(r["hop_count"]),
                "has_rev": int(r["has_rev"]),
            })
    return out


def agg(rows):
    """모델 하나의 에피소드 행에서 필요한 집계값 전부."""
    n = float(len(rows))
    mn = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    return {
        "delay": mn("delay"),
        "loss_perhop": mn("loss_perhop"),
        "hop": mn("hop"),
        # 정확: 에피소드별 곱의 평균
        "loss_sum_exact": mn("loss_sum"),
        "delay_perhop_exact": sum(r["delay"] / r["hop_count"]
                                  for r in rows) / n,
        # 근사: 평균의 곱 / 평균의 몫 (23.9.6.10 이 쓴 것)
        "loss_sum_approx": mn("loss_perhop") * mn("hop"),
        "delay_perhop_approx": (mn("delay") / mn("hop")
                                if mn("hop") else float("nan")),
        "rev_pct": 100.0 * mn("has_rev"),
        "n_ep": int(n),
    }


def contrasts(data, key):
    """2×2 대비 셋을 주어진 집계 열로. AX 와 **같은 정의**.

    축약도 AX 와 같아야 한다(`analyze_axis_factorial.py:155-157`):
    n=5 는 중복군(`RC.DUP_GROUP`), n=3 은 모델 클래스(`RC.model_class`).
    앞/뒤 5개로 naive 분할하면 p 를 23.9.6.10 표와 나란히 못 놓는다.
    """
    a, b, c, d = CELLS
    models = sorted(set(data[a]) & set(data[b]) & set(data[c]) & set(data[d]))
    out = {}
    for lab in ("코사인", "노름", "상호작용"):
        pm = {}
        for m in models:
            va, vb = data[a][m][key], data[b][m][key]
            vc, vd = data[c][m][key], data[d][m][key]
            if lab == "코사인":
                pm[m] = (vc + vd) / 2.0 - (va + vb) / 2.0
            elif lab == "노름":
                pm[m] = (vb + vd) / 2.0 - (va + vc) / 2.0
            else:
                pm[m] = (vd - vc) - (vb - va)
        out[lab] = pm
    return out, models


def main():
    raw = read_episodes()
    missing = [c for c in CELLS if c not in raw]
    if missing:
        raise SystemExit("2×2 칸이 빈다: %s (measure_episode_metrics.py 먼저)"
                         % ", ".join(missing))
    data = dict((c, dict((m, agg(rs)) for m, rs in raw[c].items()))
                for c in raw)
    bar = "=" * 86
    print("")
    print(bar)
    print("[0] 입력 — 에피소드 단위 측정")
    print("-" * 86)
    for c in sorted(raw):
        ms = sorted(raw[c])
        tot = sum(len(raw[c][m]) for m in ms)
        rv = sum(data[c][m]["rev_pct"] for m in ms) / len(ms)
        print("  %-20s 모델 %2d · 에피소드 행 %5d · 재방문 %.2f%%"
              % (c, len(ms), tot, rv))

    print("")
    print("[1] 근사 대 정확 — Jensen 격차를 실제로 쟀다")
    print("-" * 86)
    print("  23.9.6.10 은 mean(loss)×mean(hop) 를 썼다. 정확한 값은")
    print("  mean(loss_i × hop_i) 다. 아래가 그 차이다(모델 평균).")
    print("%-20s %12s %12s %10s %8s"
          % ("조건", "합 정확", "합 근사", "격차", "격차%"))
    for c in sorted(raw, key=lambda x: (x != "intact", x)):
        ms = sorted(raw[c])
        ex = sum(data[c][m]["loss_sum_exact"] for m in ms) / len(ms)
        ap = sum(data[c][m]["loss_sum_approx"] for m in ms) / len(ms)
        print("%-20s %12.5f %12.5f %10.5f %7.2f%%"
              % (c, ex, ap, ex - ap, 100.0 * (ex - ap) / ap if ap else 0.0))
    print("")
    print("%-20s %12s %12s %10s %8s"
          % ("조건", "홉당지연 정확", "홉당지연 근사", "격차", "격차%"))
    for c in sorted(raw, key=lambda x: (x != "intact", x)):
        ms = sorted(raw[c])
        ex = sum(data[c][m]["delay_perhop_exact"] for m in ms) / len(ms)
        ap = sum(data[c][m]["delay_perhop_approx"] for m in ms) / len(ms)
        print("%-20s %12.4f %12.4f %10.4f %7.2f%%"
              % (c, ex, ap, ex - ap, 100.0 * (ex - ap) / ap if ap else 0.0))

    VARIANTS = (
        ("손실 홉당평균(원본)", "loss_perhop"),
        ("손실 종단간합 **정확**", "loss_sum_exact"),
        ("손실 종단간합 근사", "loss_sum_approx"),
        ("지연 합(원본)", "delay"),
        ("지연 홉당 **정확**", "delay_perhop_exact"),
        ("지연 홉당 근사", "delay_perhop_approx"),
    )
    print("")
    print("[2] 2×2 대비 — 정확/근사 나란히 (Holm 족 %d)"
          % (len(VARIANTS) * 3))
    print("-" * 86)
    rows_out = []
    RC.duplication_report("rank_vs_value.csv", AX.A)
    pack = {}
    for lab, key in VARIANTS:
        cs, models = contrasts(data, key)
        for cl in ("코사인", "노름", "상호작용"):
            pack[(lab, cl)] = cs[cl]
    for nlab, mode in (("n=10", None), ("n=5", "dup"), ("n=3", "cls")):
        keys, ps, stats = [], [], {}
        for lab, _k in VARIANTS:
            for cl in ("코사인", "노름", "상호작용"):
                pm = pack[(lab, cl)]
                if mode is None:
                    use = [pm[m] for m in sorted(pm)]
                elif mode == "dup":
                    use = RC.collapse(pm, lambda nm: RC.DUP_GROUP[nm])
                else:
                    use = RC.collapse(pm, RC.model_class)
                mu, _sd, t, _d, p = RC.one_sample(use)
                keys.append((lab, cl))
                ps.append(p)
                stats[(lab, cl)] = (mu, t, p, len(use))
        hol = RS.holm_bonferroni(ps)
        print("")
        print("  [%s]  %-26s %12s %9s %9s"
              % (nlab, "변형 / 대비 (축약: %s)"
                 % {None: "없음", "dup": "중복군", "cls": "모델클래스"}[mode],
                 "추정", "t", "p_holm"))
        for (lab, cl), ph in zip(keys, hol):
            mu, t, _p, nn = stats[(lab, cl)]
            mark = "**" if ph < 0.05 else "  "
            print("  %s %-26s %12.4f %9.3f %9s (n=%d)"
                  % (mark, "%s / %s" % (lab, cl), mu, t, RS.fmt_p(ph), nn))
            rows_out.append([nlab, lab, cl, "%.6f" % mu, "%.4f" % t,
                             "%.6f" % ph, len(VARIANTS) * 3, nn])

    print("")
    print("[3] 모델별 **부호 일관성** — 피해와 홉 단축이 같이 가는가")
    print("-" * 86)
    print("  왜: 23.9.6.10 은 홉 보정을 회귀로 했다. 그 전에 홉 단축이")
    print("  애초에 **모든** 모델에서 일어나는지 본다. 피해가 10/10 인데")
    print("  홉 단축이 그보다 적으면 홉은 피해의 필요 경로가 아니다.")
    SKEYS = (("홉", "hop", -1), ("지연", "delay", +1),
             ("손실 홉당", "loss_perhop", +1),
             ("손실 종단간합", "loss_sum_exact", +1))
    pmv, models = {}, []
    for _lab, key, _s in SKEYS:
        cs, models = contrasts(data, key)
        pmv[key] = cs["코사인"]
    print("%-20s %9s %9s %10s %11s"
          % ("model", "d홉", "d지연", "d손실", "d손실합"))
    for m in models:
        print("%-20s %+9.3f %+9.2f %+10.5f %+11.5f"
              % (m, pmv["hop"][m], pmv["delay"][m],
                 pmv["loss_perhop"][m], pmv["loss_sum_exact"][m]))
    print("")
    for lab, key, sgn in SKEYS:
        n_ok = sum(1 for m in models if sgn * pmv[key][m] > 0)
        print("  %-14s 기대부호 %-10s : **%d/%d**"
              % (lab, "음수(단축)" if sgn < 0 else "양수(피해)",
                 n_ok, len(models)))
        rows_out.append(["sign", lab, "코사인", "%d" % n_ok, "",
                         "", len(VARIANTS) * 3, len(models)])
    odd = [m for m in models if pmv["hop"][m] > 0]
    for m in odd:
        print("  예외 %s: 홉 %+.3f (**길어졌다**) 인데 지연 %+.2f"
              % (m, pmv["hop"][m], pmv["delay"][m]))
    if models:
        best = max(models, key=lambda x: pmv["delay"][x])
        print("  지연 피해 최대 모델 %s (%+.2f) — 홉 예외와 %s"
              % (best, pmv["delay"][best],
                 "**같다**" if best in odd else "다르다"))

    path = os.path.join(OUT_DIR, "sum_exact.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "variant", "contrast", "estimate", "t",
                    "p_holm", "holm_family", "n"])
        for r in rows_out:
            w.writerow(r)
    print("")
    print("  주의: 이 표의 족은 %d 이고 23.9.6.10 의 표는 손실/지연 각 15"
          % (len(VARIANTS) * 3))
    print("  이다. 같은 대비의 p 를 족을 섞어 비교하면 안 된다(23.9.6.6).")
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
