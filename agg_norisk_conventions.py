# -*- coding: utf-8 -*-
"""`review_results/norisk_conventions.csv` 를 읽어 표 11 의 손실 비교를 판정한다.

롤아웃은 `reeval_norisk_conventions.py` 가 이미 했다. 이 스크립트는 읽기만 한다.

왜 검정을 셋으로 나누는가. 두 팔의 ± 정의가 다르다.
  제안  = 동결 모델 1개 × 시드 5개  -> 산포에 **평가 표본 잡음**이 들어 있다
  NoRisk = 학습 모델 5개 × 시드 5개 -> 모델별로 시드를 평균하면 그 잡음이 빠진다
표 11 은 이 비대칭을 그대로 인쇄하므로 (B) 에서 재현하되, 비교의 참값은
같은 시드끼리 맞대는 (C) 공통난수 쌍체에 있다. (D) 는 NoRisk 다섯 모델이
서로 다른 정책인지부터 확인한다 — 같으면 그 ± 는 학습 분산이 아니다.

출력: review_results/norisk_conventions_summary.csv
"""

import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import review_stats as RS  # noqa: E402

RD = os.path.join(HERE, "review_results")
SRC = os.path.join(RD, "norisk_conventions.csv")
OUT = os.path.join(RD, "norisk_conventions_summary.csv")

CONV = ("published", "path_sum", "compounded")
NUM = ("hop", "delay") + CONV

PUB = {
    "NoRisk": {"delay": 332.85556682983463, "published": 0.09063384250192018,
               "hop": 7.0, "sd_delay": 3.8940751750168716,
               "sd_published": 0.0021549761533751956},
    "Proposed": {"delay": 350.124972, "published": 0.083550, "hop": 7.677},
}


def load():
    with io.open(SRC, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in NUM + ("reach_pct",):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    return rows


def paired_t(diffs):
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    mu = float(d.mean())
    sd = float(d.std(ddof=1))
    if sd == 0.0:
        return mu, float("inf") if mu else 0.0, n - 1, 0.0 if mu else 1.0
    t = mu / (sd / np.sqrt(n))
    return mu, float(t), n - 1, RS.t_two_sided_p(float(t), n - 1)


def main():
    rows = load()
    out = []

    def add(block, **kw):
        rec = {"block": block, "arm": "", "model": "", "metric": "",
               "stat": "", "value": "", "note": ""}
        rec.update(kw)
        out.append(rec)

    prop = [r for r in rows if r["arm"] == "Proposed"]
    nor = [r for r in rows if r["arm"] == "NoRisk"]
    models = sorted({r["model"] for r in nor})
    seeds = sorted({r["seed"] for r in rows})
    print("원자료 %d행 — 제안 %d, NoRisk %d (모델 %d개 × 시드 %d개)"
          % (len(rows), len(prop), len(nor), len(models), len(seeds)))

    def pvals(metric):
        return [r[metric] for r in prop]

    def nvals_modelmean(metric):
        return [float(np.mean([r[metric] for r in nor if r["model"] == m]))
                for m in models]

    def nvals_all(metric):
        return [r[metric] for r in nor]

    bad = [r for r in rows if r["reach_pct"] < 100.0]
    print("도달률 100%% 아닌 블록: %d개" % len(bad))
    for r in bad:
        print("   %s %s seed=%d reach=%.2f%%"
              % (r["arm"], r["model"], r["seed"], r["reach_pct"]))
    add("A_reach", metric="reach_pct", stat="blocks_below_100",
        value="%d" % len(bad), note="전체 %d블록" % len(rows))

    # ---------------- A. 발표값 재현 ----------------
    print()
    print("=" * 86)
    print("A. 발표값 재현 (표 11 과 같은 ± 정의로)")
    print("=" * 86)
    print("  %-9s %-10s %14s %12s %14s %9s"
          % ("팔", "지표", "측정 평균", "측정 sd", "발표", "편차"))
    for arm, getter, nlab in (("Proposed", pvals, "시드 5개"),
                              ("NoRisk", nvals_modelmean, "모델 5개")):
        for k in ("delay", "published", "hop"):
            xs = getter(k)
            mu, sd = RS.mean_std(xs)
            pub = PUB[arm][k]
            dev = (mu - pub) / pub * 100.0 if pub else float("nan")
            print("  %-9s %-10s %14.6f %12.6f %14.6f %+8.3f%%  (%s)"
                  % (arm, k, mu, sd, pub, dev, nlab))
            add("A_reproduce", arm=arm, metric=k, stat="mean",
                value="%.10f" % mu, note=nlab)
            add("A_reproduce", arm=arm, metric=k, stat="sd",
                value="%.10f" % sd, note=nlab)
            add("A_reproduce", arm=arm, metric=k, stat="dev_pct_vs_published",
                value="%.6f" % dev, note="발표 %.8f" % pub)

    # ---------------- D. NoRisk 다섯 모델은 서로 다른 정책인가 ----------------
    print()
    print("=" * 86)
    print("D. NoRisk 다섯 '독립 학습' 모델이 서로 다른 정책인가")
    print("=" * 86)
    bykey = {(r["model"], r["seed"]): r for r in nor}
    ident_pairs = 0
    total_pairs = 0
    print("  %-11s %-11s %10s %14s"
          % ("모델1", "모델2", "동일 시드", "최대 지연차(ms)"))
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            m1, m2 = models[i], models[j]
            same = 0
            mx = 0.0
            for s in seeds:
                a, b = bykey[(m1, s)], bykey[(m2, s)]
                d = max(abs(a[k] - b[k]) for k in NUM)
                mx = max(mx, abs(a["delay"] - b["delay"]))
                if d == 0.0:
                    same += 1
            total_pairs += 1
            if same == len(seeds):
                ident_pairs += 1
            print("  %-11s %-11s %6d/%-3d %14.6f"
                  % (m1.replace(".pth", ""), m2.replace(".pth", ""),
                     same, len(seeds), mx))
            add("D_model_identity", model="%s|%s" % (m1, m2),
                stat="identical_seeds", value="%d/%d" % (same, len(seeds)),
                note="max_delay_diff=%.6f" % mx)
    print("  -> 열 쌍 중 **%d/%d 쌍이 다섯 시드 전부 비트 동일**"
          % (ident_pairs, total_pairs))
    uniq = set()
    for m in models:
        uniq.add(tuple(round(bykey[(m, s)]["delay"], 10) for s in seeds))
    print("  -> 서로 다른 값벡터 **%d개** (모델 %d개 중)"
          % (len(uniq), len(models)))
    add("D_model_identity", stat="identical_pairs",
        value="%d/%d" % (ident_pairs, total_pairs))
    add("D_model_identity", stat="distinct_value_vectors",
        value="%d/%d" % (len(uniq), len(models)))

    hops = {r["hop"] for r in nor}
    print("  -> NoRisk 홉 값 집합: %s (블록 %d개)"
          % (sorted(hops), len(nor)))
    add("D_model_identity", metric="hop", stat="distinct_block_means",
        value=";".join("%.6f" % h for h in sorted(hops)))

    # ---------------- B. 표 11 정의 그대로의 비교 ----------------
    print()
    print("=" * 86)
    print("B. 표 11 의 ± 정의 그대로 (제안 시드5 대 NoRisk 모델5) — Welch")
    print("=" * 86)
    hdr = "  %-11s %13s %13s %12s %11s %10s %10s"
    print(hdr % ("규약", "제안", "NoRisk", "차", "제안우위%", "p", "p_holm"))
    tests = []
    for c in CONV:
        xp, xn = pvals(c), nvals_modelmean(c)
        diff, t, df, p = RS.welch(xp, xn)
        mp, sp = RS.mean_std(xp)
        mn, sn = RS.mean_std(xn)
        tests.append([c, mp, sp, mn, sn, diff, p])
    holm = RS.holm_bonferroni([x[6] for x in tests])
    for (c, mp, sp, mn, sn, diff, p), hp in zip(tests, holm):
        ph = hp[1] if isinstance(hp, (list, tuple)) else hp
        adv = -(mp - mn) / mn * 100.0
        print(hdr % (c, "%.6f" % mp, "%.6f" % mn, "%+.6f" % diff,
                     "%+.2f%%" % adv, RS.fmt_p(p), RS.fmt_p(ph)))
        add("B_table11_def", metric=c, stat="prop_mean", value="%.10f" % mp)
        add("B_table11_def", metric=c, stat="prop_sd", value="%.10f" % sp)
        add("B_table11_def", metric=c, stat="norisk_mean", value="%.10f" % mn)
        add("B_table11_def", metric=c, stat="norisk_sd", value="%.10f" % sn)
        add("B_table11_def", metric=c, stat="adv_pct", value="%.6f" % adv)
        add("B_table11_def", metric=c, stat="p", value="%.8f" % p)
        add("B_table11_def", metric=c, stat="p_holm", value="%.8f" % ph)

    for k in ("delay", "hop"):
        xp, xn = pvals(k), nvals_modelmean(k)
        diff, t, df, p = RS.welch(xp, xn)
        mp, _ = RS.mean_std(xp)
        mn, _ = RS.mean_std(xn)
        print("  %-11s %13.6f %13.6f %+12.6f %+10.2f%% %10s"
              % (k, mp, mn, diff, (mp - mn) / mn * 100.0, RS.fmt_p(p)))
        add("B_table11_def", metric=k, stat="prop_mean", value="%.10f" % mp)
        add("B_table11_def", metric=k, stat="norisk_mean", value="%.10f" % mn)
        add("B_table11_def", metric=k, stat="p", value="%.8f" % p)

    # ---------------- C. 공통난수 쌍체 ----------------
    print()
    print("=" * 86)
    print("C. 같은 시드끼리 맞댄 쌍체 (공통난수, n=5, df=4)")
    print("=" * 86)
    print(hdr % ("규약", "평균차", "", "sd(차)", "제안우위%", "p", "p_holm"))
    ptests = []
    for c in CONV + ("delay", "hop"):
        ds = []
        for s in seeds:
            a = [r[c] for r in prop if r["seed"] == s][0]
            b = float(np.mean([bykey[(m, s)][c] for m in models]))
            ds.append(a - b)
        mu, t, df, p = paired_t(ds)
        base = float(np.mean(nvals_all(c)))
        ptests.append([c, mu, float(np.std(ds, ddof=1)), base, p])
    sel = [x for x in ptests if x[0] in CONV]
    holm = RS.holm_bonferroni([x[4] for x in sel])
    hmap = {}
    for (c, mu, sd, base, p), hp in zip(sel, holm):
        hmap[c] = hp[1] if isinstance(hp, (list, tuple)) else hp
    for c, mu, sd, base, p in ptests:
        ph = hmap.get(c)
        adv = -mu / base * 100.0
        print(hdr % (c, "%+.6f" % mu, "", "%.6f" % sd,
                     "%+.2f%%" % adv, RS.fmt_p(p),
                     RS.fmt_p(ph) if ph is not None else "-"))
        add("C_paired_seed", metric=c, stat="mean_diff", value="%.10f" % mu)
        add("C_paired_seed", metric=c, stat="sd_diff", value="%.10f" % sd)
        add("C_paired_seed", metric=c, stat="adv_pct", value="%.6f" % adv)
        add("C_paired_seed", metric=c, stat="p", value="%.8f" % p)
        if ph is not None:
            add("C_paired_seed", metric=c, stat="p_holm", value="%.8f" % ph)

    # ---------------- E. 희석 분해 ----------------
    print()
    print("=" * 86)
    print("E. 홉당 평균이 비교에 얹는 몫 (NoRisk 홉이 상수 7 이라 정확히 선다)")
    print("=" * 86)
    hp_, hn_ = float(np.mean(pvals("hop"))), float(np.mean(nvals_all("hop")))
    rp = float(np.mean(pvals("published"))) / float(np.mean(nvals_all("published")))
    rs = float(np.mean(pvals("path_sum"))) / float(np.mean(nvals_all("path_sum")))
    fac = hn_ / hp_
    print("  홉:  제안 %.6f   NoRisk %.6f   비 NoRisk/제안 = %.6f" % (hp_, hn_, fac))
    print("  비:  published %.6f   path_sum %.6f" % (rp, rs))
    print("  예측 published 비 = path_sum 비 x 홉비 = %.6f x %.6f = %.6f"
          % (rs, fac, rs * fac))
    print("  실측 published 비 = %.6f   (잔차 %+.6f)" % (rp, rp - rs * fac))
    print("  -> 홉당 평균이 제안에 얹어 주는 공짜 몫 = %+.2f%%"
          % ((1.0 - fac) * 100.0))
    for k, v in (("hop_prop", hp_), ("hop_norisk", hn_), ("hop_ratio", fac),
                 ("ratio_published", rp), ("ratio_path_sum", rs),
                 ("predicted_ratio_published", rs * fac),
                 ("residual", rp - rs * fac),
                 ("free_gift_pct", (1.0 - fac) * 100.0)):
        add("E_dilution", stat=k, value="%.10f" % v)

    # ---------------- F. 발표값으로 역산한 교차검증 ----------------
    print()
    print("=" * 86)
    print("F. 발표 숫자만으로 역산한 값과의 대조")
    print("=" * 86)
    print("  NoRisk 홉이 상수 7 이면 path_sum = published x 7 이 **정확히** 성립한다.")
    for m in models:
        xs = [bykey[(m, s)] for s in seeds]
        pub_ = float(np.mean([r["published"] for r in xs]))
        sum_ = float(np.mean([r["path_sum"] for r in xs]))
        print("    %-18s published x 7 = %.8f   실측 path_sum = %.8f   차 %+.2e"
              % (m.replace(".pth", ""), pub_ * 7.0, sum_, pub_ * 7.0 - sum_))
    back = PUB["NoRisk"]["published"] * 7.0
    print("  발표 NoRisk published %.8f x 7 = **%.8f** (= 발표 기준 종단간 합)"
          % (PUB["NoRisk"]["published"], back))
    print("  발표 제안 path_sum (loss_conventions_exact.csv) = 0.64930700")
    print("  -> 발표 숫자만으로도 제안이 종단간 합에서 %+.2f%% 로 **불리**하다"
          % ((0.649307 - back) / back * 100.0))
    add("F_backsolve", stat="norisk_published_x7", value="%.10f" % back)
    add("F_backsolve", stat="prop_path_sum_committed", value="0.6493070000")
    add("F_backsolve", stat="prop_disadvantage_pct",
        value="%.6f" % ((0.649307 - back) / back * 100.0))

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["block", "arm", "model", "metric",
                                           "stat", "value", "note"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print()
    print("기록: %s (%d행)" % (OUT, len(out)))


if __name__ == "__main__":
    main()
