# -*- coding: utf-8 -*-
"""23.9.6.10 — 손실 코사인 주효과가 '경로 단축'의 산술 부산물인지 검정.

왜 필요한가
-----------
`network_env.py:453` 은 손실을 **홉당 평균**으로 낸다
(`total_loss / hop_count`). 지연은 **합**이다(`total_delay`). 그래서
경로 길이가 바뀌면 두 지표가 **반대 방향**으로 끌린다:
  · 경로가 짧아지면 지연(합)은 내려간다
  · 경로가 짧아지면 손실(홉당 평균)은 분모가 작아져 올라갈 수 있다
메모리에 적힌 바로 그 함정이다(손실은 긴 경로에 유리, r2=0.88).

23.9.6.2 의 2x2 에서 코사인 주효과는 이렇게 나왔다:
  손실 +0.0526 (n=5 p_holm 0.0077 — n=5 에서 유일하게 서는 근거)
  홉   -0.6000 (경로가 짧아졌다)
부호 조합이 홉 산술 효과와 똑같다. 그러므로 보정 전에는 손실
주효과를 '위험 정보 손상'의 근거로 쓸 수 없다.

무엇을 하는가
------------
기울기 b (= 홉 1개당 손실 변화)를 서로 다른 세 방법으로 재고,
손실에서 홉 몫을 뺀 뒤 같은 대비를 다시 검정한다.
  (A) 대응차 회귀 : 모든 (모델, 조건!=intact) 쌍의 (d홉, d손실)
  (B) 순수 배율 통제: scale184 / scale054 만. 블록 방향을
      전혀 안 건드리고 크기만 바꾸므로 '정보는 그대로인데 홉이
      움직였을 때' 손실이 얼마 따라오는지를 준다.
  (C) 무해 조건만 : rankonly / pow50 / pow33 / pow400.
그리고 반대쪽 괄호로 합 버전(손실 x 홉)에도 같은 대비를 건다.
홉당 평균과 합은 길이에 대해 반대로 편향되므로, 양쪽에서 같은
부호로 유의하면 정규화 방향 어느 쪽으로도 설명이 안 된다.

한계를 먼저 적는다
-----------------
- 합 버전은 에피소드 평균의 곱이다. CSV 에 에피소드별 값이 없어
  mean(loss_i * hop_i) 를 복원할 수 없다(Jensen 격차). 그래서 합
  버전은 괄호(bracket)용 근사이고 단독 근거로 쓰지 않는다.
  정확한 합이 필요하면 측정 스크립트에 loss_sum 열을 넣고 재측정해야
  한다.
- (B) 로 보정하면 통제 조건 자신의 손실 효과는 정의상 0 근처가
  된다(순환). 코사인 대비에 대해서는 순환이 아니다.
- b 는 조건 수준 관계라 모델 안 이질성을 흡수하지 못한다. 세 추정이
  갈리면 셋 다 적고 가장 보수적인 쪽으로 읽는다.

출력: review_results/loss_hop_confound.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import review_stats as RS                            # noqa: E402
import reanalyze_cluster_n as RC                     # noqa: E402
import analyze_perturb_vs_damage as AP               # noqa: E402
import analyze_axis_factorial as AX                  # noqa: E402

OUT_DIR = RC.OUT_DIR
CELLS = (AX.A, AX.B, AX.C, AX.D)
CTRL = ("risk_scale184", "risk_scale054")
HARMLESS = ("risk_rankonly", "risk_rankpow50", "risk_rankpow33",
            "risk_rankpow400")


def ols(xs, ys):
    """기울기 b, 절편, 피어슨 r. n<3 이면 nan."""
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    xm = sum(xs) / n
    ym = sum(ys) / n
    sxx = sum((x - xm) ** 2 for x in xs)
    syy = sum((y - ym) ** 2 for y in ys)
    sxy = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    if sxx <= 1e-18 or syy <= 1e-18:
        return float("nan"), float("nan"), float("nan")
    b = sxy / sxx
    return b, ym - b * xm, sxy / (sxx * syy) ** 0.5


def paired_slope(data, models, conds, met="loss", base=AX.A):
    """(d홉, d지표) 대응차에서 기울기. 지연도 같은 방식으로 재야 한다 —
    손실만 보정하고 지연은 원본으로 쓰면 '불리한 지표만 보정했다' 는
    반론이 그대로 선다. 지연은 합이라 b 가 크게 양수로 나와야 맞다."""
    dh, dl = [], []
    for nm in models:
        for c in conds:
            if c not in data[nm] or base not in data[nm]:
                continue
            dh.append(data[nm][c]["hop"] - data[nm][base]["hop"])
            dl.append(data[nm][c][met] - data[nm][base][met])
    return ols(dh, dl) + (len(dh),)


def main():
    bar = "=" * 92
    RC.duplication_report("rank_vs_value.csv", AX.A)
    data, note = AP.merge_damage()
    models = sorted(n for n in data if all(c in data[n] for c in CELLS))
    print(bar)
    print("[0] 자료")
    print("-" * 92)
    for ln in note:
        print("  " + ln)
    print("  2x2 네 칸이 다 있는 모델 %d개" % len(models))
    if not models:
        raise SystemExit("2x2 가 비었다")
    allc = sorted(set(c for nm in models for c in data[nm] if c != AX.A))
    print("  intact 외 조건 %d개: %s" % (len(allc), ", ".join(allc)))

    print("")
    print(bar)
    print("[1] 손실은 홉당 평균이다 — 홉과 얼마나 묶여 있나")
    print("-" * 92)
    print("  정의: network_env.py:453  total_loss / hop_count  (지연은 합)")
    slopes = {"loss": {}, "delay": {}}
    sets = (("(A) 대응차 전체", tuple(allc)),
            ("(B) 순수 배율 통제", CTRL),
            ("(C) 무해 조건만", HARMLESS))
    for met in ("loss", "delay"):
        print("")
        print("  [%s]" % met)
        for label, conds in sets:
            use = tuple(c for c in conds if c in allc)
            b, _a, r, n = paired_slope(data, models, use, met)
            slopes[met][label] = b
            print("  %-18s b = %10.5f /홉 · r = %7.4f · r^2 = %6.4f"
                  " · n = %3d"
                  % (label, b, r, r * r if r == r else float("nan"), n))
        print("    조건: %s" % ", ".join(use))
    print("")
    print("  손실 b 가 음수면 '경로가 짧아질 때 손실이 오른다' 는 뜻이고,")
    print("  그게 바로 23.9.6.2 코사인 대비의 부호 조합이다.")
    print("  지연 b 는 합이라 양수여야 한다 — 그러면 경로 단축이 지연")
    print("  효과를 **깎고** 있었다는 뜻이고 보정하면 더 커진다.")
    print("  (A) 는 피해 조건까지 넣어 기울기를 맞추므로 검정 대상 효과를")
    print("  홉 항으로 **빨아들인다**. 그래서 (A) 는 최선 추정이 아니라")
    print("  **과보정 상한**이다. 비순환 통제는 (B)/(C) 다.")

    print("")
    print(bar)
    print("[2] 코사인 대비에서 홉이 설명하는 몫")
    print("-" * 92)
    cos_fn = dict(AX.CONTRASTS)["코사인 주효과"]
    dh = [cos_fn({c: data[nm][c]["hop"] for c in CELLS}) for nm in models]
    mh = sum(dh) / len(dh)
    obs = {}
    for met in ("loss", "delay"):
        dl = [cos_fn({c: data[nm][c][met] for c in CELLS})
              for nm in models]
        obs[met] = sum(dl) / len(dl)
    print("  코사인 대비 평균:  d홉 = %+.4f · d손실 = %+.6f"
          " · d지연 = %+.4f" % (mh, obs["loss"], obs["delay"]))
    for met in ("loss", "delay"):
        print("")
        print("  [%s] 관측 %+.6f" % (met, obs[met]))
        for label in sorted(slopes[met]):
            pred = slopes[met][label] * mh
            frac = (100.0 * pred / obs[met]
                    if abs(obs[met]) > 1e-12 else float("nan"))
            print("  %-18s 홉 예측 = %+10.6f  (관측의 %+7.1f%%)"
                  % (label, pred, frac))
    print("")
    print("  100% 에 가까우면 그 효과는 전부 경로 단축의 산술 결과다.")
    print("  0% 에 가까우면 홉과 무관하다. **음수**면 홉이 효과를")
    print("  가리고 있었다는 뜻이다(보정하면 커진다).")
    ml = obs["loss"]

    print("")
    print(bar)
    print("[3] 홉 보정 후 · 합 버전으로 같은 2x2 대비 재검정")
    print("-" * 92)

    def mk_adj(met, bb):
        return lambda nm, c: (data[nm][c][met] - bb * data[nm][c]["hop"])

    def mk_per_hop(met):
        return lambda nm, c: (data[nm][c][met] / data[nm][c]["hop"]
                              if data[nm][c]["hop"] > 1e-9
                              else float("nan"))

    def mk_sum(met):
        return lambda nm, c: data[nm][c][met] * data[nm][c]["hop"]

    groups = []
    for met in ("loss", "delay"):
        vs = [("%s 원본" % met, (lambda m: lambda nm, c: data[nm][c][m])(met))]
        for label in ("(A) 대응차 전체", "(B) 순수 배율 통제",
                      "(C) 무해 조건만"):
            b = slopes[met][label]
            if b != b:
                continue
            vs.append(("%s-%s" % (met, label.split()[0]), mk_adj(met, b)))
        if met == "loss":
            # 홉당 평균의 반대쪽 괄호 = 합. 근사(평균의 곱)임을 docstring
            # 에 적었다.
            vs.append(("loss x hop(근사)", mk_sum("loss")))
        else:
            # 지연은 이미 합이므로 반대쪽 괄호는 홉당 평균이다.
            vs.append(("delay/hop(홉당)", mk_per_hop("delay")))
        groups.append(("%s 계열" % met, vs))

    allrows = []
    for gname, variants in groups:
        rows, p1s, p5s, p3s = [], [], [], []
        for vname, getv in variants:
            for clabel, fn in AX.CONTRASTS:
                per_model = {}
                for nm in models:
                    per_model[nm] = fn({c: getv(nm, c) for c in CELLS})
                vals = [per_model[nm] for nm in models]
                m1, s1, t1, _d, p1 = RC.one_sample(vals)
                dup = RC.collapse(per_model, lambda n: RC.DUP_GROUP[n])
                m5, _s, t5, _d, p5 = RC.one_sample(dup)
                cls = RC.collapse(per_model, RC.model_class)
                m3, _s, t3, _d, p3 = RC.one_sample(cls)
                rows.append([vname, clabel, m1, s1, t1, len(vals), m5,
                             t5, len(dup), m3, t3, len(cls)])
                p1s.append(p1)
                p5s.append(p5)
                p3s.append(p3)
        h1 = RS.holm_bonferroni(p1s)
        h5 = RS.holm_bonferroni(p5s)
        h3 = RS.holm_bonferroni(p3s)
        print("")
        print("  [%s]  Holm 족 %d — 손실 계열과 지연 계열은 **다른 족**"
              "이다." % (gname, len(rows)))
        print("%-18s %-12s %11s %8s %8s | %8s %8s | %8s %8s"
              % ("변형", "대비", "추정", "t(n=10)", "p_holm",
                 "t(n=5)", "p_holm", "t(n=3)", "p_holm"))
        for i, r in enumerate(rows):
            (vn, cl, m1, _s1, t1, _n1, _m5, t5, _n5, _m3, t3, _n3) = r
            print("%-18s %-12s %11.6f %8.3f %8s | %8.3f %8s | %8.3f %8s"
                  % (vn, cl, m1, t1, RS.fmt_p(h1[i]), t5,
                     RS.fmt_p(h5[i]), t3, RS.fmt_p(h3[i])))
            allrows.append([gname] + r + [h1[i], h5[i], h3[i]])
    print("")
    print("  23.9.6.2 의 axis_factorial 표는 족 9 로 보정한 값이라 같은")
    print("  대비의 p 가 여기와 다르다 — 23.9.6.6 의 '족을 섞지 말라' 와")
    print("  같은 사안이다. 인용할 때 어느 족인지 함께 적어야 한다.")

    print("")
    print(bar)
    print("[4] 읽는 법 — 사전 기록")
    print("-" * 92)
    print("  · 보정 후에도 코사인 주효과가 n=5 에서 서고, 합 버전에서도")
    print("    같은 부호로 서면 → 홉 정규화로 설명 안 된다. 23.9.6 의")
    print("    '코사인이 피해를 가른다' 를 손실로 인용해도 된다.")
    print("  · 보정하면 죽는다면 → 손실 근거는 철회하고, 23.9.6 결론은")
    print("    n=10 지연(0.0026)만 남는다. 그건 23.9.5 보수 기준을")
    print("    못 넘으므로 '측정 못 했다' 로 적어야 한다.")
    print("  · 참고: 지연 코사인 대비는 +78.93 인데 홉은 -0.6000 이다.")
    print("    경로가 짧아졌는데 지연 합이 올랐으므로 지연 효과는")
    print("    홉 때문에 부풀려진 게 아니라 가려져 있었다.")
    print(bar)

    path = os.path.join(OUT_DIR, "loss_hop_confound.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "slope_estimate", "b_per_hop"])
        for met in ("loss", "delay"):
            for label in sorted(slopes[met]):
                w.writerow([met, label, "%.8f" % slopes[met][label]])
        w.writerow([])
        w.writerow(["cosine_contrast_mean_dhop", "%.6f" % mh])
        w.writerow(["cosine_contrast_mean_dloss", "%.8f" % ml])
        w.writerow(["cosine_contrast_mean_ddelay", "%.6f" % obs["delay"]])
        w.writerow([])
        w.writerow(["holm_family", "variant", "contrast", "est_n10",
                    "sd_n10", "t_n10", "n10", "p_holm_n10", "est_n5",
                    "t_n5", "n5", "p_holm_n5", "est_n3", "t_n3", "n3",
                    "p_holm_n3"])
        for r in allrows:
            (gn, vn, cl, m1, s1, t1, n1, m5, t5, n5, m3, t3, n3,
             a1, a5, a3) = r
            w.writerow([gn, vn, cl, "%.8f" % m1, "%.8f" % s1,
                        "%.4f" % t1, n1, "%.6f" % a1, "%.8f" % m5,
                        "%.4f" % t5, n5, "%.6f" % a5, "%.8f" % m3,
                        "%.4f" % t3, n3, "%.6f" % a3])
    print("")
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
