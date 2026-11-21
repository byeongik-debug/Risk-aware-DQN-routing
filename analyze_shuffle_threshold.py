# -*- coding: utf-8 -*-
"""위험 벡터를 **몇 칸까지** 섞어도 되는가 — 내구 문턱을 구간으로 잡는다.

왜 필요한가
-----------
`analyze_damage_magnitude.py` [4] 는 `risk_shuf08`·`risk_shuf16` 의 피해가
**유의하지 않다**고 찍는다(p_holm 1.0000). 그런데 "유의하지 않다" 는
"0 이다" 가 아니다 — 검정력이 모자랐을 수도 있다. 그래서 그걸로
"16칸까지는 괜찮다" 라고 쓰면 null 을 증거로 쓰는 오류가 된다.

여기서는 **p 를 새로 만들지 않는다**. 대신 모델 대응차의 **95% 신뢰
구간**을 찍어서 "피해가 많아도 X ms 를 넘지 않는다" 는 상한을 준다.
그 상한이 다음 k 점의 하한보다 낮으면 문턱이 두 점 사이에 있다고
**구간으로** 말할 수 있다. 검정 족을 새로 만들지 않으니
23.9.6 의 Holm 족 혼용 문제도 안 생긴다(유의 표시는
`damage_magnitude.csv` [4] 의 **족 30** 값을 그대로 인용한다).

한계 — 이 축이 **무엇의** 용량축인지
------------------------------------
`SHUFK` 는 k 가 커지면 cos_dev·rho_dev·l1·건드린칸이 **같이** 커진다
(`shuffle_geometry.csv`: rho_dev/cos_dev 가 2.08~2.11 의 거의 일정한
비). 그래서 k 는 "섞은 양" 의 용량축이고 **"코사인을 고정한 채의 순위
교란" 의 용량축이 아니다**. 이 파일의 결과로 순위 가설을 지지하거나
반박할 수 없다. 그건 23.9.6.16 [5b] 의 짝지은 비교가 할 일이다.

그리고 피해가 0 인 구간이 "정책이 안 바뀌어서" 인지는 따로 봐야 한다.
`decision_flip.csv` 를 같이 읽어 같은 k 에서 결정이 몇 % 바뀌는지를
나란히 찍는다 — 결정이 바뀌는데도 피해가 0 이면 그건 **여분**이다.

출력: review_results/shuffle_threshold.csv
환경변수: DM_IN(기본 episode_metrics_all.csv — 섞기 조건이 들어 있어야 한다)
"""
import csv
import io
import math
import os
import sys

os.environ.setdefault("DM_IN", "episode_metrics_all.csv")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import analyze_damage_magnitude as DM  # noqa: E402
import occlude_state_blocks as OC  # noqa: E402
import review_stats as RS  # noqa: E402

sys.path.append(HERE)

ALPHA = 0.05
FLIP_CSV = os.path.join(OC.OUT_DIR, "decision_flip.csv")
DAMAGE_CSV = os.path.join(OC.OUT_DIR, "damage_magnitude.csv")
GEOM_CSV = os.path.join(OC.OUT_DIR, "shuffle_geometry.csv")
OUT_CSV = os.path.join(OC.OUT_DIR, "shuffle_threshold.csv")


def paired_ci(diffs, alpha=ALPHA):
    """대응차의 평균·표준오차·양측 신뢰구간. n<2 면 nan."""
    n = len(diffs)
    if n < 2:
        return (float("nan"),) * 4
    mu = sum(diffs) / n
    var = sum((v - mu) ** 2 for v in diffs) / (n - 1)
    se = math.sqrt(var / n)
    half = RS.t_critical(n - 1, alpha) * se
    return mu, se, mu - half, mu + half


def read_two_col(path, want_block, kcol, vcols):
    """블록이 여러 개 섞인 CSV 에서 한 블록만 dict 로 뽑는다."""
    out = {}
    if not os.path.exists(path):
        return out
    with io.open(path, encoding="utf-8") as fh:
        hdr = None
        for row in csv.reader(fh):
            if not row:
                hdr = None
                continue
            if row[0] == "block":
                hdr = row
                continue
            if hdr is None or len(row) != len(hdr):
                continue
            r = dict(zip(hdr, row))
            if r.get("block") != want_block:
                continue
            key = r.get(kcol)
            if key is None:
                continue
            out.setdefault(key, []).append(
                dict((c, r.get(c)) for c in vcols))
    return out


# 문턱 전용 k 격자 — `DM.KPTS` 에 **중간점만** 더한다. DM 쪽을 건드리지
# 않는 이유: `analyze_damage_magnitude.py` 의 `[4]` Holm 족(**30**)과
# `[5c]` 용량-반응(**6점**, rho 0.9029 · t 30.533)이 FINDINGS 23.9.6.16
# 에 **이미 인용돼 있다**. 거기에 점을 더하면 그 둘이 조용히 바뀐다
# (23.9.6.16 [F] 가 바로 그 사고다: 조건이 늘어 족이 18→30 이 되자
# `pow05n` 의 p_holm 이 0.0319→0.0532 로 유의성을 잃었다).
# 이 파일은 p 를 **만들지 않고** CI 만 찍으므로 격자를 늘려도 다중성이
# 안 생긴다. 그래서 중간점은 **여기서만** 본다.
MIDK = (("risk_shuf20", 20),)
KPTS_T = tuple(sorted(tuple(DM.KPTS) + MIDK, key=lambda ck: ck[1]))


def main():
    data, _nbad = DM.read_ref()
    avail = [(c, k) for c, k in KPTS_T if c in data]
    bar = "=" * 92
    print(bar)
    print("섞기 내구 문턱 — p 를 새로 만들지 않고 **구간**으로 말한다")
    print("  입력 %s · 점 %d개" % (os.environ["DM_IN"], len(avail)))
    print(bar)
    if len(avail) < 3:
        raise SystemExit("섞기 조건이 없다. DM_IN 을 확인할 것.")

    models = sorted(set.intersection(*[set(data[c]) for c, _ in avail]))
    print("  공통 모델 %d개" % len(models))
    per = {}
    for c, k in avail:
        per[c] = dict((m, DM.agg(data[c][m])) for m in models)

    # [1] 결정 전환율 — decision_flip.csv 에서 끌어온다
    flip = {}
    fr = read_two_col(FLIP_CSV, "per_model", "condition",
                      ("flip", "loc_argmin", "loc_inv", "glob_cos"))
    for c, rows in fr.items():
        agg2 = {}
        for key in ("flip", "loc_argmin", "loc_inv", "glob_cos"):
            vs = [float(r[key]) for r in rows if r.get(key)]
            agg2[key] = sum(vs) / len(vs) if vs else float("nan")
        flip[c] = agg2

    # [2] DM [4] 의 유의 표시 — 족 30 을 그대로 인용
    pholm = {}
    dr = read_two_col(DAMAGE_CSV, "cond_vs_intact", "a",
                      ("b", "p_holm", "holm_family"))
    for c, rows in dr.items():
        for r in rows:
            if r.get("b") and "지연 초과" in r["b"]:
                try:
                    pholm[c] = (float(r["p_holm"]), r.get("holm_family"))
                except (TypeError, ValueError):
                    pass

    print("")
    print("[1] k 별 피해 — **95% 신뢰구간**(새 검정 아님)")
    print("-" * 92)
    print("  대응은 모델 수준이다(23.9.6.14: 에피소드 번호로 짝짓지 않음).")
    print("  유의 표시는 damage_magnitude.csv [4] 의 **족 %s** 인용."
          % (list(pholm.values())[0][1] if pholm else "?"))
    rows_out = []
    for slab, skey in DM.SCALES:
        print("")
        print("  %s" % slab)
        print("    %-18s %4s %10s %9s %20s %8s %10s"
              % ("조건", "k", "추정", "SE", "95% 구간", "부호",
                 "p_holm[4]"))
        base = [per["intact"][m][skey] for m in models]
        for c, k in avail:
            if c == "intact":
                continue
            d = [per[c][m][skey] - per["intact"][m][skey] for m in models]
            mu, se, lo, hi = paired_ci(d)
            npos = sum(1 for v in d if v > 0)
            pv = pholm.get(c)
            print("    %-18s %4d %10.4f %9.4f  [%+8.3f, %+8.3f] %5d/%d %10s"
                  % (c, k, mu, se, lo, hi, npos, len(d),
                     RS.fmt_p(pv[0]) if pv else "-"))
            rows_out.append(["ci_vs_intact", c, k, skey, "%.6f" % mu,
                             "%.6f" % se, "%.6f" % lo, "%.6f" % hi,
                             npos, len(d),
                             ("%.6f" % pv[0]) if pv else "",
                             (pv[1] or "") if pv else ""])
        print("    (intact 평균 %.4f)" % (sum(base) / len(base)))

    print("")
    print("[2] 문턱이 어디에 있는가 — 구간이 **겹치지 않는** 첫 자리")
    print("-" * 92)
    print("  '유의하지 않다' 는 '0 이다' 가 아니다. 그래서 상한을 읽는다:")
    print("  낮은 k 의 **상한**이 높은 k 의 **하한**보다 작으면 문턱이")
    print("  그 둘 사이에 있다고 구간으로 말할 수 있다.")
    for slab, skey in DM.SCALES:
        pts = []
        for c, k in avail:
            if c == "intact":
                continue
            d = [per[c][m][skey] - per["intact"][m][skey] for m in models]
            mu, se, lo, hi = paired_ci(d)
            pts.append((k, c, mu, lo, hi))
        print("")
        print("  %s" % slab)
        found = False
        for i in range(len(pts) - 1):
            k1, c1, _m1, _l1, h1 = pts[i]
            k2, c2, _m2, l2, _h2 = pts[i + 1]
            mark = "  <= 여기서 갈린다" if (h1 < l2 and not found) else ""
            if h1 < l2 and not found:
                found = True
            print("    k=%2d 상한 %+9.3f  vs  k=%2d 하한 %+9.3f%s"
                  % (k1, h1, k2, l2, mark))
        if not found:
            print("    → 인접 구간이 전부 겹친다. 문턱을 구간으로 못 집는다.")

    print("")
    print("[3] 피해 0 인 구간에서 **정책은 바뀌는가**")
    print("-" * 92)
    print("  바뀌는데도 피해가 0 이면 그 칸들은 **여분**이라는 뜻이다.")
    print("  결정 전환은 공유 배터리에서 잰 값이다(decision_flip.csv).")
    if not flip:
        print("  → decision_flip.csv 가 없다. measure_decision_flip.py 를"
              " 먼저 돌릴 것.")
    else:
        print("  %-18s %4s %9s %11s %9s %10s %12s"
              % ("조건", "k", "flip", "loc_argmin", "loc_inv", "glob_cos",
                 "피해(초과)"))
        for c, k in avail:
            f = flip.get(c)
            if not f:
                continue
            if c == "intact":
                ex = 0.0
            else:
                ex = sum(per[c][m]["excess"] - per["intact"][m]["excess"]
                         for m in models) / len(models)
            print("  %-18s %4d %8.2f%% %10.2f%% %9.4f %10.4f %12.3f"
                  % (c, k, 100 * f["flip"], 100 * f["loc_argmin"],
                     f["loc_inv"], f["glob_cos"], ex))
            rows_out.append(["flip_vs_damage", c, k, "excess",
                             "%.6f" % ex, "", "", "", "", len(models),
                             "", ""])
            rows_out.append(["flip_vs_damage_geom", c, k, "flip",
                             "%.6f" % f["flip"],
                             "%.6f" % f["loc_argmin"],
                             "%.6f" % f["loc_inv"],
                             "%.6f" % f["glob_cos"], "", "", "", ""])

    print("")
    print("[4] 이 축이 섞는 **양** 의 축임을 다시 적어 둔다")
    print("-" * 92)
    if os.path.exists(GEOM_CSV):
        print("  %-18s %9s %9s %9s %9s %9s"
              % ("조건", "cos_dev", "rho_dev", "노름비", "l1", "건드린칸%"))
        with io.open(GEOM_CSV, encoding="utf-8") as fh:
            g = dict((r["condition"], r) for r in csv.DictReader(fh))
        for c, k in avail:
            r = g.get(c)
            if not r:
                continue
            print("  %-18s %9s %9s %9s %9s %9s"
                  % (c, r["cos_dev"], r["rho_dev"], r["norm_ratio"],
                     r["mean_abs_diff"], r["moved_pct"]))
        print("  → 네 열이 k 와 **같이** 움직인다. 그래서 [1]~[3] 은")
        print("    '섞은 양' 의 용량-반응이고, '코사인 고정 순위 교란' 의")
        print("    용량-반응이 **아니다**. 순위 가설은 짝지은 비교가 할 일.")
    else:
        print("  shuffle_geometry.csv 가 없다 — 기하 열을 못 붙인다.")

    with io.open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "condition", "k", "scale", "estimate_or_v1",
                    "se_or_v2", "ci_lo_or_v3", "ci_hi_or_v4", "n_pos",
                    "n", "p_holm_cited", "holm_family_cited"])
        w.writerows(rows_out)
    print("")
    print("저장: %s (%d행)" % (OUT_CSV, len(rows_out)))
    print(bar)


if __name__ == "__main__":
    main()
