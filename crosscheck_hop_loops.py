# -*- coding: utf-8 -*-
"""재방문 측정(`revisit_rate.csv`)을 axis/pow2 측정과 **대조**하고,
코사인 대비를 **고리 뺀 길이**로 다시 계산한다.

왜 필요한가. 23.9.6.9 의 판정에서 홉은 **하중을 받는 숫자**다 —
코사인 대비 d홉 = −0.600 이 "경로가 짧아졌다"로 읽히고, 23.9.6.10 은
그 홉으로 손실(홉당 평균)을 보정했다. 그런데 `network_env.py:367-368`
은 재방문을 **금지하지 않고 보상 −8.0 만** 깎고 `get_valid_actions`
(:264-266)는 이웃 전부를 돌려준다. 그래서 `hop_count` 에 **고리**가
섞일 수 있고, 섞였으면 "짧아졌다" 가 실제로는 "고리가 줄었다" 일
수 있다. 이 스크립트가 그 둘을 가른다.

세 가지를 한다:
1) 홉 평균을 axis/pow2 CSV 와 모델별로 대조한다(같은 평가시드·같은
   에피소드 수라 **일치해야** 한다. 불일치면 롤아웃 규약이 다르다).
2) 2×2 코사인/노름/상호작용 대비를 `hop` 과 `loopfree` **양쪽**으로
   계산해 차이를 본다. 차이가 작으면 23.9.6.9·23.9.6.10 의 홉 해석이
   고리와 무관하다.
3) 최단경로 일치율을 피해량과 나란히 놓는다. 이건 **품질이 아니라
   탐색 폭** 지표다 — 일치율이 높으면 위험 정보를 덜 쓰고 최단경로로
   붙은 것이다. 메커니즘은 단정하지 않고 관측만 적는다.

출력: review_results/hop_loop_crosscheck.csv
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
import analyze_axis_factorial as AX  # noqa: E402
import analyze_perturb_vs_damage as AP  # noqa: E402

OUT_DIR = OC.OUT_DIR
REV_CSV = os.path.join(OUT_DIR, "revisit_rate.csv")


def read_revisit():
    """조건 -> 모델 -> 측정값."""
    out = {}
    with io.open(REV_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = out.setdefault(r["condition"], {})
            d[r["model"]] = {
                "hop": float(r["hop"]),
                "loopfree": float(r["loopfree"]),
                "revisit": float(r["revisit"]),
                "has_rev": float(r["has_rev_pct"]),
                "is_short": float(r["is_short_pct"]),
                "sp_len": float(r["sp_len"]),
            }
    return out


def contrast(rev, key, cells):
    """대비 셋을 주어진 열(key)로 계산. AX 와 같은 정의."""
    a, b, c, d = cells
    models = sorted(set(rev[a]) & set(rev[b]) & set(rev[c]) & set(rev[d]))
    out = {}
    for lab in ("코사인", "노름", "상호작용"):
        vals = []
        for m in models:
            va = rev[a][m][key]
            vb = rev[b][m][key]
            vc = rev[c][m][key]
            vd = rev[d][m][key]
            if lab == "코사인":
                vals.append((vc + vd) / 2.0 - (va + vb) / 2.0)
            elif lab == "노름":
                vals.append((vb + vd) / 2.0 - (va + vc) / 2.0)
            else:
                vals.append((vd - vc) - (vb - va))
        mu, sd = RS.mean_std(vals)
        n = len(vals)
        t = mu / (sd / (n ** 0.5)) if sd > 0 else float("nan")
        out[lab] = (mu, t, RS.t_two_sided_p(t, n - 1), n)
    return out, models


def main():
    rev = read_revisit()
    data, _note = AP.merge_damage()
    bar = "=" * 86
    print("")
    print(bar)
    print("[1] 홉 평균 대조 — 재방문 측정 대 axis/pow2 측정")
    print("-" * 86)
    print("  같은 평가시드(%d)·같은 에피소드 수라 **일치해야** 한다."
          % OC.EVAL_SEED)
    print("%-20s %10s %10s %10s %s"
          % ("조건", "재방문측정", "기존측정", "차이", "판정"))
    worst = 0.0
    rowsx = []
    for c in sorted(rev):
        ms = [m for m in rev[c] if m in data and c in data[m]]
        if not ms:
            print("%-20s %10.3f %10s %10s %s"
                  % (c, sum(rev[c][m]["hop"] for m in rev[c]) / len(rev[c]),
                     "-", "-", "기존 측정 없음"))
            continue
        a = sum(rev[c][m]["hop"] for m in ms) / len(ms)
        b = sum(data[m][c]["hop"] for m in ms) / len(ms)
        dmax = max(abs(rev[c][m]["hop"] - data[m][c]["hop"]) for m in ms)
        worst = max(worst, dmax)
        print("%-20s %10.3f %10.3f %10.4f %s"
              % (c, a, b, a - b,
                 "일치" if dmax < 5e-3 else "불일치(모델최대 %.4f)" % dmax))
        rowsx.append((c, a, b, dmax))
    print("")
    print("  모델 수준 최대 불일치 = %.6f  (%s)"
          % (worst, "규약 동일" if worst < 5e-3 else "**규약 다름 — 확인**"))

    print("")
    print("[2] 고리(재방문) 실측")
    print("-" * 86)
    print("%-20s %10s %10s %10s %10s %10s"
          % ("조건", "홉", "고리뺀길이", "고리분", "재방문%", "최단경로%"))
    for c in sorted(rev, key=lambda x: (x != "intact", x)):
        ms = sorted(rev[c])
        h = sum(rev[c][m]["hop"] for m in ms) / len(ms)
        lf = sum(rev[c][m]["loopfree"] for m in ms) / len(ms)
        hr = sum(rev[c][m]["has_rev"] for m in ms) / len(ms)
        sh = sum(rev[c][m]["is_short"] for m in ms) / len(ms)
        print("%-20s %10.3f %10.3f %10.4f %9.1f%% %9.1f%%"
              % (c, h, lf, h - lf, hr, sh))

    cells = (AX.A, AX.B, AX.C, AX.D)
    have = all(c in rev for c in cells)
    print("")
    print("[3] 2×2 대비를 홉 / 고리뺀길이 **양쪽**으로")
    print("-" * 86)
    if not have:
        print("  2×2 네 칸이 재방문 측정에 다 없다 — 건너뜀.")
    else:
        ch, models = contrast(rev, "hop", cells)
        cl, _ = contrast(rev, "loopfree", cells)
        print("  모델 %d개. 홉 해석이 고리 제거분이면 두 열이 **벌어진다**."
              % len(models))
        print("%-12s %12s %12s %12s %s"
              % ("대비", "홉", "고리뺀길이", "차이", "p(홉, 족1)"))
        for lab in ("코사인", "노름", "상호작용"):
            mh, th, ph, _ = ch[lab]
            ml, _tl, _pl, _ = cl[lab]
            print("%-12s %+12.4f %+12.4f %+12.4f %s(t=%.3f)"
                  % (lab, mh, ml, ml - mh, RS.fmt_p(ph), th))
        print("")
        print("  읽는 법: '차이' 가 0 에 가까우면 23.9.6.9 의 d홉 과")
        print("  23.9.6.10 의 홉 보정은 **고리와 무관**하다.")

    print("")
    print("[4] 최단경로 일치율과 피해량 (조건 수준 · 기술 통계)")
    print("-" * 86)
    print("  일치율은 품질이 아니라 **탐색 폭** 지표다(100%=위험 정보 미사용).")
    print("%-20s %10s %12s %12s"
          % ("조건", "최단경로%", "지연차이", "손실차이"))
    xs, ys = [], []
    for c in sorted(rev, key=lambda x: (x != "intact", x)):
        ms = sorted(rev[c])
        sh = sum(rev[c][m]["is_short"] for m in ms) / len(ms)
        mm = [m for m in ms if m in data and c in data[m]]
        if c == "intact" or not mm:
            dd = ll = 0.0
        else:
            dd = sum(data[m][c]["delay"] - data[m]["intact"]["delay"]
                     for m in mm) / len(mm)
            ll = sum(data[m][c]["loss"] - data[m]["intact"]["loss"]
                     for m in mm) / len(mm)
        print("%-20s %9.1f%% %12.2f %12.4f" % (c, sh, dd, ll))
        if c != "intact":
            xs.append(sh)
            ys.append(dd)
    if len(xs) >= 3:
        print("")
        print("  최단경로%% 대 지연차이 rho = %+.4f (조건 n=%d, 검정 아님)"
              % (AP.spearman(xs, ys), len(xs)))

    path = os.path.join(OUT_DIR, "hop_loop_crosscheck.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "model", "hop_revisit", "hop_prev",
                    "abs_diff", "loopfree", "loop_part", "has_rev_pct",
                    "is_short_pct", "sp_len"])
        for c in sorted(rev):
            for m in sorted(rev[c]):
                r = rev[c][m]
                hp = (data[m][c]["hop"]
                      if m in data and c in data[m] else float("nan"))
                w.writerow([c, m, "%.4f" % r["hop"],
                            "" if hp != hp else "%.4f" % hp,
                            "" if hp != hp else "%.6f" % abs(r["hop"] - hp),
                            "%.4f" % r["loopfree"],
                            "%.6f" % (r["hop"] - r["loopfree"]),
                            "%.4f" % r["has_rev"], "%.4f" % r["is_short"],
                            "%.2f" % r["sp_len"]])
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
