# -*- coding: utf-8 -*-
"""23.9.6.2 판정 — 코사인×노름 2x2 를 주효과와 상호작용으로 분해한다.

왜 필요한가
-----------
`pow05` 는 rho=1.0000 인데 피해를 낸다. 그런데 그 조건은 코사인(0.7662)과
노름비(1.8409)가 **동시에** 움직이므로 원인을 못 가른다. 축을 하나씩만
움직인 조건을 더 돌렸고(`rankpow05n` 코사인만, `scale184` 노름만), 그
결과 네 조건이 **직교하는 2x2** 를 이룬다:

            노름비 1.0000        노름비 1.8409
  코사인 1.0000   intact            risk_scale184
  코사인 0.7662   risk_rankpow05n   risk_rankpow05

그러면 '분리력(간극)' 같은 서술 통계 대신 **대비(contrast)** 로 직접
분해할 수 있다:
  코사인 주효과 = (C+D)/2 - (A+B)/2
  노름  주효과 = (B+D)/2 - (A+C)/2
  상호작용     = (D-C) - (B-A)
상호작용이 0 이 아니면 "축 하나가 원인"이라는 말 자체가 성립하지 않는다.

한계를 먼저 적는다
-----------------
- 네 칸이 **한 실행에서 같이** 나온 게 아니다. A/B/C 는 `RV_TAG=axis`,
  D 는 `RV_TAG=pow2` 실행이다. 조건 수준 시드 독립성은 따로 쟀다
  (네 원 조건 36칸 재현, 최대 차 0.049 ms = 로그 반올림 폭)
  — 그 전제가 깨지면 이 분해도 깨진다.
- 모델 10개는 2팔x5시드이고 intact 롤아웃이 겹치는 묶음이 있다
  (23.9.5). 그래서 n=10/5/3 세 수준을 전부 낸다. **보수 기준은 n=5** 다.
- `scale054`(코사인 1.0000, 노름 0.5432)는 2x2 밖이다. 노름 축의
  **아래쪽** 보조 팔로 따로 낸다. `scale184` 는 칸의 24.51%가 참 블록
  관측 구간 밖이므로(23.9.6 [7] 표) 노름 주효과는 `scale054` 와
  **같이** 읽어야 한다.
- 대비 셋은 서로 직교하지만 세 지표(지연·손실·홉)는 아니다. Holm 족은
  3x3=9 로 잡는다.

출력: review_results/axis_factorial.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import review_stats as RS                            # noqa: E402
import reanalyze_cluster_n as RC                     # noqa: E402
import analyze_perturb_vs_damage as AP               # noqa: E402

OUT_DIR = RC.OUT_DIR
METRICS = RC.METRICS
A = "intact"
B = "risk_scale184"
C = "risk_rankpow05n"
D = "risk_rankpow05"
AUX = "risk_scale054"
GEOM = {A: (1.0000, 1.0000), B: (1.0000, 1.8409),
        C: (0.7662, 1.0000), D: (0.7662, 1.8409),
        AUX: (1.0000, 0.5432)}

CONTRASTS = (
    ("코사인 주효과", lambda v: (v[C] + v[D]) / 2.0 - (v[A] + v[B]) / 2.0),
    ("노름 주효과", lambda v: (v[B] + v[D]) / 2.0 - (v[A] + v[C]) / 2.0),
    ("상호작용", lambda v: (v[D] - v[C]) - (v[B] - v[A])),
)


def main():
    RC.duplication_report("rank_vs_value.csv", A)
    data, note = AP.merge_damage()
    bar = "=" * 92
    print(bar)
    print("[0] 자료 합치기")
    print("-" * 92)
    for ln in note:
        print("  " + ln)

    need = (A, B, C, D)
    models = sorted(n for n in data if all(c in data[n] for c in need))
    missing = sorted(set(data) - set(models))
    print("  2x2 네 칸이 다 있는 모델 %d개" % len(models))
    if missing:
        print("  빠진 모델 %d개: %s" % (len(missing), ", ".join(missing)))
        for nm in missing:
            print("    %s 에 없는 조건: %s"
                  % (nm, ",".join(c for c in need if c not in data[nm])))
    print("")
    print(bar)
    print("[1] '무해' 는 '정책이 그대로' 를 뜻하지 않는다 — 따로 센다")
    print("-" * 92)
    print("  지연이 안 나빠진 것과 정책이 안 바뀐 것은 다른 얘기다.")
    print("  조건별로 (지연,손실,홉) 세쌍이 intact 와 **달라진 모델 수**와,")
    print("  10개 모델이 만드는 **서로 다른 세쌍 개수**를 센다. 뒤 숫자가")
    print("  intact 보다 작아지면 그 조건은 서로 다른 정책들을 **한 곳으로")
    print("  모은다**(= 큰 정책 변화인데 품질 지표로는 안 보인다).")
    print("")
    allm = sorted(data.keys())

    def triple(nm, cond):
        return tuple(round(data[nm][cond][m], 4) for m in METRICS)

    base_tr = set(triple(nm, A) for nm in allm)
    print("  %-20s %8s %10s %12s"
          % ("조건", "바뀐 모델", "다른 세쌍", "홉 평균"))
    print("  %-20s %8s %10d %12.4f"
          % (A, "-", len(base_tr),
             sum(data[nm][A]["hop"] for nm in allm) / len(allm)))
    conds = []
    for nm in allm:
        for c in data[nm]:
            if c != A and c not in conds:
                conds.append(c)
    for c in sorted(conds):
        have = [nm for nm in allm if c in data[nm]]
        if not have:
            continue
        chg = sum(1 for nm in have if triple(nm, c) != triple(nm, A))
        uniq = len(set(triple(nm, c) for nm in have))
        hop = sum(data[nm][c]["hop"] for nm in have) / len(have)
        print("  %-20s %5d/%-3d %10d %12.4f"
              % (c, chg, len(have), uniq, hop))
    print("")
    print("  주의: intact 의 '다른 세쌍' 개수가 기준선이다. 그보다 작으면")
    print("  정책 수렴, 크면 정책 분산이다. 둘 다 '무해'(지연 p_holm)와")
    print("  **독립**이므로 두 숫자를 같이 인용해야 한다.")

    if not models:
        raise SystemExit("2x2 를 채운 모델이 없다 — axis 실행이 끝나야 한다")

    print("")
    print(bar)
    print("[2] 2x2 칸 평균 (모델 %d개, 지표별)" % len(models))
    print("-" * 92)
    for met in METRICS:
        cell = {c: sum(data[n][c][met] for n in models) / len(models)
                for c in need}
        print("  %s" % met)
        print("    %-14s %14s %14s" % ("", "노름 1.0000", "노름 1.8409"))
        print("    %-14s %14.4f %14.4f"
              % ("코사인 1.0000", cell[A], cell[B]))
        print("    %-14s %14.4f %14.4f"
              % ("코사인 0.7662", cell[C], cell[D]))
        if AUX in data[models[0]]:
            aux = sum(data[n][AUX][met] for n in models) / len(models)
            print("    (보조) 코사인 1.0000 · 노름 0.5432 = %.4f" % aux)

    rows, p1s, p5s, p3s = [], [], [], []
    for met in METRICS:
        for label, fn in CONTRASTS:
            per_model = {}
            for nm in models:
                v = {c: data[nm][c][met] for c in need}
                per_model[nm] = fn(v)
            vals = [per_model[nm] for nm in models]
            m1, s1, t1, _d, p1 = RC.one_sample(vals)
            dup = RC.collapse(per_model, lambda n: RC.DUP_GROUP[n])
            m5, s5, t5, _d, p5 = RC.one_sample(dup)
            cls = RC.collapse(per_model, RC.model_class)
            m3, s3, t3, _d, p3 = RC.one_sample(cls)
            nuniq = len(set(round(x, 6) for x in vals))
            rows.append([met, label, m1, s1, t1, len(vals),
                         m5, t5, len(dup), m3, t3, len(cls), nuniq])
            p1s.append(p1)
            p5s.append(p5)
            p3s.append(p3)
    h1 = RS.holm_bonferroni(p1s)
    h5 = RS.holm_bonferroni(p5s)
    h3 = RS.holm_bonferroni(p3s)

    print("")
    print(bar)
    print("[3] 대비 분해 (Holm 족 %d · 세 수준 각각 따로 보정)" % len(rows))
    print("-" * 92)
    print("%-6s %-12s %10s %9s %8s | %9s %8s | %9s %8s | %s"
          % ("지표", "대비", "추정", "t(n=%d)" % len(models), "p_holm",
             "t(n=5)", "p_holm", "t(n=3)", "p_holm", "다름"))
    for i, r in enumerate(rows):
        (met, label, m1, _s1, t1, n1, _m5, t5, _n5,
         _m3, t3, _n3, nuniq) = r
        print("%-6s %-12s %10.4f %9.3f %8s | %9.3f %8s | %9.3f %8s | %d/%d"
              % (met, label, m1, t1, RS.fmt_p(h1[i]), t5, RS.fmt_p(h5[i]),
                 t3, RS.fmt_p(h3[i]), nuniq, n1))

    print("")
    print("  읽는 법 (사전 기록, 23.9.6.2):")
    print("   · 코사인만 유의 → 피해 원인은 블록 안 대비의 **방향**.")
    print("   · 노름만 유의 → 크기. 단 scale184 는 24.51%가 범위 밖이라")
    print("     scale054 보조 팔이 같은 방향이어야 선다.")
    print("   · 상호작용이 유의 → '축 하나가 원인'이라는 말이 성립 안 한다.")
    print("     그때는 주효과를 단독으로 인용하면 안 된다.")
    print("  보수 기준은 n=5 다(23.9.5).")

    path = os.path.join(OUT_DIR, "axis_factorial.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "contrast", "est_n10", "sd_n10", "t_n10",
                    "n10", "p_holm_n10", "est_n5", "t_n5", "n5",
                    "p_holm_n5", "est_n3", "t_n3", "n3", "p_holm_n3",
                    "n_distinct"])
        for i, r in enumerate(rows):
            (met, label, m1, s1, t1, n1, m5, t5, n5, m3, t3, n3, nu) = r
            w.writerow([met, label, "%.6f" % m1, "%.6f" % s1, "%.4f" % t1,
                        n1, "%.6f" % h1[i], "%.6f" % m5, "%.4f" % t5, n5,
                        "%.6f" % h5[i], "%.6f" % m3, "%.4f" % t3, n3,
                        "%.6f" % h3[i], nu])
        w.writerow([])
        w.writerow(["cell_means"])
        w.writerow(["metric"] + list(need) + [AUX, "cosine_norm_of_each"])
        for met in METRICS:
            cells = [sum(data[n][c][met] for n in models) / len(models)
                     for c in need]
            aux = (sum(data[n][AUX][met] for n in models) / len(models)
                   if AUX in data[models[0]] else "")
            w.writerow([met] + ["%.6f" % v for v in cells]
                       + ["%.6f" % aux if aux != "" else "",
                          ";".join("%s=%.4f/%.4f" % (c, GEOM[c][0],
                                                     GEOM[c][1])
                                   for c in list(need) + [AUX])])
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
