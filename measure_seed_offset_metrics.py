# -*- coding: utf-8 -*-
u"""표 10 각주가 "지연·홉" 이라고 선언한 범위를 측정으로 확인한다.

왜. `DRAFT_4_6_2_table10.md` (5) 의 각주는 "표 10 의 **지연·홉** 절대값은
평가 난수 표본 `EVAL_SEED = 20261114` 하나에서 읽은 값" 이라고 쓰고
"다른 평가 표본에서는 **열 전체가 함께 약 1%** 움직인다" 고 주장하는데,
그 각주가 적은 숫자는 **전부 지연**이다(360.84→356.57, +4.27 ms, +1.20%,
그리고 정책 셋의 +1.398 / +1.156 / +0.468%). 홉이 같이 움직이는지는
재지 않았다. FINDINGS 26.10.10 에서 **규약** 편향은 홉에 없다는 것이
나왔으므로(+0.2173%, p_holm 0.3617), 각주의 홉 선언도 검증 없이 두면
안 된다. 이 스크립트는 같은 15개 추첨으로 **홉·손실**에 같은 계산을 한다.

재학습·재채점 없음. 커밋된 CSV 만 읽는다.

입력(모두 커밋됨)
  review_results/full_seed_offset.csv            Full-r01 / Full-r05 × 15 추첨
  review_results/eval_protocol_noise.csv         nov 규약 3팔 × 15 추첨
  review_results/table10_factor_runs.csv         표 10 인쇄값(평가 표본 1개)
  review_results/path_identity_all_classes.csv   정책 동일성 근거
출력
  review_results/seed_offset_metrics.csv
"""
import csv
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
sys.path.insert(0, HERE)
from review_stats import t_two_sided_p, holm_bonferroni  # noqa: E402

OUT = os.path.join(RES, "seed_offset_metrics.csv")

# 15개 추첨이 다 있는 다섯 라벨 → (표 10 팔, 런)
LABELS = [
    (u"Full-r01", u"Full", u"1"),
    (u"Full-r05", u"Full", u"5"),
    (u"-Cong-r03", u"-Cong", u"3"),
    (u"-Util-r03", u"-Util", u"3"),
    (u"-dLoss-r03", u"-dLoss", u"3"),
]
METRICS = [(u"delay", u"ms"), (u"hop", u"hops"), (u"loss", u"loss/hop")]


def mean_sd(xs):
    n = len(xs)
    m = sum(xs) / float(n)
    if n < 2:
        return m, 0.0
    v = sum((x - m) ** 2 for x in xs) / float(n - 1)
    return m, math.sqrt(v)


def one_sample_t(xs, mu):
    u"""xs 의 평균이 mu 와 다른가 (양측). 반환 (t, p, df)."""
    n = len(xs)
    m, sd = mean_sd(xs)
    if sd == 0.0:
        return float("nan"), 1.0, n - 1
    t = (m - mu) / (sd / math.sqrt(n))
    return t, t_two_sided_p(t, n - 1), n - 1


def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / float(n)
    my = sum(ys) / float(n)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def read_draws():
    u"""라벨 → 지표 → 추첨순 값 리스트. 두 CSV 의 추첨 씨앗이 같은지도 본다."""
    draws = {}
    seeds = {}
    p = os.path.join(RES, "full_seed_offset.csv")
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        lab = row["label"]
        d = int(row["draw"])
        draws.setdefault(lab, {}).setdefault("_draw", {})[d] = {
            m: float(row[m]) for m, _u in METRICS}
        seeds.setdefault(lab, {})[d] = int(row["seed"])
    p = os.path.join(RES, "eval_protocol_noise.csv")
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        if row["protocol"] != "nov":
            continue
        lab = row["label"]
        d = int(row["draw"])
        draws.setdefault(lab, {}).setdefault("_draw", {})[d] = {
            m: float(row[m]) for m, _u in METRICS}
        seeds.setdefault(lab, {})[d] = int(row["seed"])
    # 씨앗 정렬 확인 — 다르면 라벨 간 쌍대 비교가 성립하지 않는다
    ref = None
    for lab in sorted(seeds):
        v = [seeds[lab][d] for d in sorted(seeds[lab])]
        if ref is None:
            ref = v
        assert v == ref, (u"추첨 씨앗 불일치", lab)
    out = {}
    for lab, d in draws.items():
        dd = d["_draw"]
        ks = sorted(dd)
        out[lab] = {m: [dd[k][m] for k in ks] for m, _u in METRICS}
    return out, ref


def read_printed():
    u"""표 10 인쇄값(평가 표본 1개). (팔, 런) → 지표 → 값."""
    p = os.path.join(RES, "table10_factor_runs.csv")
    pr = {}
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        try:
            vals = {m: float(row[m]) for m, _u in METRICS}
        except (TypeError, ValueError):
            continue  # -VD 행은 열이 밀려 있다 — 이 분석에 안 쓴다
        pr[(row["arm"], row["run"])] = vals
    return pr


def policy_classes():
    u"""모델 → 정책 클래스 id. 표 10 Full 런 1~4 가 한 정책인지 확인용."""
    p = os.path.join(RES, "path_identity_all_classes.csv")
    cls = {}
    for row in csv.DictReader(io.open(p, encoding="utf-8")):
        for mdl in row["models"].split():
            cls[mdl] = int(row["policy_id"])
    return cls


def main():
    draws, seeds = read_draws()
    printed = read_printed()
    cls = policy_classes()
    rows = []

    print(u"추첨 씨앗 %d개 공통: %s … %s" % (len(seeds), seeds[0], seeds[-1]))

    # ---- 0. 정책 동일성: 다섯 라벨 중 서로 다른 정책은 몇 개인가 ----------
    sig = {}
    for lab, _a, _r in LABELS:
        key = tuple(round(v, 9) for m, _u in METRICS for v in draws[lab][m])
        sig.setdefault(key, []).append(lab)
    print(u"\n[0] 다섯 라벨의 **세 지표 전부**가 비트 동일한 묶음:")
    for i, (_k, labs) in enumerate(sorted(sig.items(), key=lambda kv: -len(kv[1])), 1):
        ids = sorted(set(cls.get(l, -1) for l in labs))
        print(u"    묶음%d (%d개, 경로동일성 정책 id %s): %s"
              % (i, len(labs), ids, u" = ".join(sorted(labs))))
        rows.append({
            "scope": "policy_identity", "label": u"|".join(sorted(labs)),
            "metric": "all3", "unit": "", "n": len(labs),
            "printed": "", "draw_mean": "", "draw_sd": "", "diff": "",
            "diff_pct": "", "z": "", "t": "", "p": "", "p_holm": "",
            "min_pct": "", "max_pct": "", "note": u"policy_id=%s" % ids,
        })
    distinct = [sorted(labs)[0] for _k, labs in sig.items()]
    full_cls = sorted(set(cls.get(u"Full-r%02d" % r, -1) for r in (1, 2, 3, 4)))
    print(u"    표 10 Full 런 1~4 의 정책 id: %s (하나면 팔 재구성 근거 성립)"
          % full_cls)

    # ---- 1. 라벨별 치우침: 인쇄값 대 15추첨 평균 ---------------------------
    print(u"\n[1] 라벨별 치우침 (인쇄값 − 15추첨 평균)")
    print(u"  %-11s %-6s %10s %10s %9s %9s %8s %8s"
          % (u"라벨", u"지표", u"인쇄값", u"추첨평균", u"차", u"차%", u"z", u"p"))
    tests = []
    for lab, arm, run in LABELS:
        for m, unit in METRICS:
            xs = draws[lab][m]
            pv = printed[(arm, run)][m]
            dm, dsd = mean_sd(xs)
            diff = pv - dm
            pct = 100.0 * diff / dm
            z = diff / dsd if dsd else float("nan")
            t, p, _df = one_sample_t(xs, pv)
            pcts = [100.0 * (x / dm - 1.0) for x in xs]
            r = {
                "scope": "label", "label": lab, "metric": m, "unit": unit,
                "n": len(xs), "printed": "%.6f" % pv,
                "draw_mean": "%.6f" % dm, "draw_sd": "%.6f" % dsd,
                "diff": "%+.6f" % diff, "diff_pct": "%+.4f" % pct,
                "z": "%+.4f" % z, "t": "%+.4f" % t, "p": "%.8f" % p,
                "p_holm": "", "min_pct": "%+.4f" % min(pcts),
                "max_pct": "%+.4f" % max(pcts), "note": u"arm=%s run=%s" % (arm, run),
            }
            rows.append(r)
            if lab in distinct:
                tests.append(r)
            print(u"  %-11s %-6s %10.4f %10.4f %+9.4f %+8.4f%% %+8.3f %8.5f"
                  % (lab, m, pv, dm, diff, pct, z, p))

    # Holm: 서로 다른 정책만 센다(중복 라벨은 같은 검정을 또 세는 것)
    hp = holm_bonferroni([float(r["p"]) for r in tests])
    for r, q in zip(tests, hp):
        r["p_holm"] = "%.8f" % q
    print(u"  Holm 족 = 서로 다른 정책 %d개 × 지표 3 = %d개 검정"
          % (len(distinct), len(tests)))

    # ---- 2. 표 10 Full 행(팔 평균) 재구성 ---------------------------------
    # 런 1~4 는 Full-r01 과 한 정책(위 [0] 에서 확인), 런 5 는 Full-r05.
    assert len(full_cls) == 1, (u"Full 런 1~4 가 한 정책이 아니다", full_cls)
    print(u"\n[2] 표 10 `Full` 행 = 런 5개 평균. 추첨별로 같은 가중(4:1)으로 재구성")
    print(u"  %-6s %10s %10s %9s %9s %8s %8s"
          % (u"지표", u"인쇄평균", u"추첨평균", u"차", u"차%", u"t", u"p"))
    arm_tests = []
    for m, unit in METRICS:
        pv = sum(printed[(u"Full", str(r))][m] for r in range(1, 6)) / 5.0
        per_draw = [(4.0 * a + b) / 5.0
                    for a, b in zip(draws[u"Full-r01"][m], draws[u"Full-r05"][m])]
        dm, dsd = mean_sd(per_draw)
        diff = pv - dm
        pct = 100.0 * diff / dm
        t, p, _df = one_sample_t(per_draw, pv)
        pcts = [100.0 * (x / dm - 1.0) for x in per_draw]
        r = {
            "scope": "arm_full", "label": u"Full", "metric": m, "unit": unit,
            "n": len(per_draw), "printed": "%.6f" % pv,
            "draw_mean": "%.6f" % dm, "draw_sd": "%.6f" % dsd,
            "diff": "%+.6f" % diff, "diff_pct": "%+.4f" % pct,
            "z": "%+.4f" % (diff / dsd if dsd else float("nan")),
            "t": "%+.4f" % t, "p": "%.8f" % p, "p_holm": "",
            "min_pct": "%+.4f" % min(pcts), "max_pct": "%+.4f" % max(pcts),
            "note": u"4x(policy of run1-4) + 1x(run5), /5",
        }
        rows.append(r)
        arm_tests.append(r)
        print(u"  %-6s %10.4f %10.4f %+9.4f %+8.4f%% %+8.3f %8.5f"
              % (m, pv, dm, diff, pct, t, p))
    hp = holm_bonferroni([float(r["p"]) for r in arm_tests])
    for r, q in zip(arm_tests, hp):
        r["p_holm"] = "%.8f" % q

    # ---- 3. 정책 셋 평균 치우침 (각주가 쓰는 +1.007% 꼴) ------------------
    print(u"\n[3] 서로 다른 정책 %d개의 치우침 평균 — 각주가 인용하는 꼴"
          % len(distinct))
    for m, unit in METRICS:
        ps = []
        for lab in distinct:
            arm, run = [(a, r) for l, a, r in LABELS if l == lab][0]
            dm, _sd = mean_sd(draws[lab][m])
            ps.append(100.0 * (printed[(arm, run)][m] - dm) / dm)
        mm, sd = mean_sd(ps)
        rows.append({
            "scope": "policy_set_mean", "label": u"|".join(distinct),
            "metric": m, "unit": unit, "n": len(ps), "printed": "",
            "draw_mean": "", "draw_sd": "", "diff": "",
            "diff_pct": "%+.4f" % mm, "z": "", "t": "", "p": "", "p_holm": "",
            "min_pct": "%+.4f" % min(ps), "max_pct": "%+.4f" % max(ps),
            "note": u"sd=%.4f; " % sd + u" / ".join(u"%+.4f" % v for v in ps),
        })
        print(u"  %-6s 평균 %+8.4f%%  (개별 %s)"
              % (m, mm, u" / ".join(u"%+.4f" % v for v in ps)))

    # ---- 4. "열 전체가 함께 움직이는가" ----------------------------------
    # 추첨을 바꾸면 모든 정책이 같은 방향으로 가는가, 그 공통 폭은 얼마인가.
    print(u"\n[4] 추첨 간 **공통 이동** (각 정책의 자기 15추첨 평균 대비 %)")
    print(u"  %-6s %10s %10s %10s %10s %8s"
          % (u"지표", u"공통sd%", u"공통범위%", u"공통몫", u"부호일치", u"평균|r|"))
    for m, unit in METRICS:
        devs = {}
        for lab in distinct:
            dm, _sd = mean_sd(draws[lab][m])
            devs[lab] = [100.0 * (x / dm - 1.0) for x in draws[lab][m]]
        nd = len(seeds)
        common = [sum(devs[l][i] for l in distinct) / float(len(distinct))
                  for i in range(nd)]
        _cm, csd = mean_sd(common)
        crange = max(common) - min(common)
        # 공통 몫 = 추첨 평균의 분산 / 정책별 분산의 평균
        within = sum(mean_sd(devs[l])[1] ** 2 for l in distinct) / float(len(distinct))
        share = (csd ** 2) / within if within > 0 else float("nan")
        agree = sum(1 for i in range(nd)
                    if len(set(1 if devs[l][i] > 0 else (0 if devs[l][i] == 0 else -1)
                               for l in distinct)) == 1)
        rs = []
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                rs.append(pearson(devs[distinct[i]], devs[distinct[j]]))
        ar = sum(abs(v) for v in rs) / float(len(rs)) if rs else float("nan")
        rows.append({
            "scope": "draw_common", "label": u"|".join(distinct), "metric": m,
            "unit": unit, "n": nd, "printed": "", "draw_mean": "",
            "draw_sd": "%.6f" % csd, "diff": "", "diff_pct": "",
            "z": "%.4f" % share, "t": "", "p": "", "p_holm": "",
            "min_pct": "%+.4f" % min(common), "max_pct": "%+.4f" % max(common),
            "note": (u"range=%.4f; sign_agree=%d/%d; mean_abs_r=%.4f; r=%s"
                     % (crange, agree, nd, ar,
                        u",".join(u"%+.3f" % v for v in rs))),
        })
        print(u"  %-6s %10.4f %10.4f %10.4f %10s %8.4f"
              % (m, csd, crange, share, u"%d/%d" % (agree, nd), ar))
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                rows.append({
                    "scope": "policy_corr",
                    "label": u"%s~%s" % (distinct[i], distinct[j]),
                    "metric": m, "unit": unit, "n": nd, "printed": "",
                    "draw_mean": "", "draw_sd": "", "diff": "",
                    "diff_pct": "", "z": "%+.4f" % pearson(devs[distinct[i]],
                                                           devs[distinct[j]]),
                    "t": "", "p": "", "p_holm": "", "min_pct": "",
                    "max_pct": "", "note": u"r across 15 draws of dev%",
                })

    # ---- 4b. 설정 간 차이에 남는 잔류 -------------------------------------
    # 각주의 마지막 문장("설정 간 차이에서는 치우침이 대부분 지워진다 —
    # 잔류 0.15 / 3.08 / 3.23 ms")이 재는 양. 지연에서만 적혀 있었다.
    print(u"\n[4b] 쌍별 잔류 = (인쇄 차) − (추첨평균 차)")
    print(u"  %-6s %-23s %11s %11s %9s %9s %9s"
          % (u"지표", u"쌍", u"인쇄차", u"추첨평균차", u"잔류", u"잔류/|차|%", u"추첨sd"))
    for m, unit in METRICS:
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                a, b = distinct[i], distinct[j]
                arm_a, run_a = [(ar, r) for l, ar, r in LABELS if l == a][0]
                arm_b, run_b = [(ar, r) for l, ar, r in LABELS if l == b][0]
                pd_ = printed[(arm_a, run_a)][m] - printed[(arm_b, run_b)][m]
                da = mean_sd(draws[a][m])[0]
                db = mean_sd(draws[b][m])[0]
                dd = da - db
                res = pd_ - dd
                pergap = 100.0 * res / abs(pd_) if pd_ else float("nan")
                pdiff = [x - y for x, y in zip(draws[a][m], draws[b][m])]
                _pm, psd = mean_sd(pdiff)
                rows.append({
                    "scope": "pair_residual", "label": u"%s~%s" % (a, b),
                    "metric": m, "unit": unit, "n": len(pdiff),
                    "printed": "%+.6f" % pd_, "draw_mean": "%+.6f" % dd,
                    "draw_sd": "%.6f" % psd, "diff": "%+.6f" % res,
                    "diff_pct": "%+.4f" % pergap,
                    "z": "%+.4f" % (res / psd if psd else float("nan")),
                    "t": "", "p": "", "p_holm": "", "min_pct": "",
                    "max_pct": "", "note": u"residual of the arm-to-arm gap",
                })
                print(u"  %-6s %-23s %+11.4f %+11.4f %+9.4f %+9.2f%% %9.4f"
                      % (m, u"%s~%s" % (a, b), pd_, dd, res, pergap, psd))

    # ---- 5. 못 재는 것 ----------------------------------------------------
    print(u"\n[5] 도달률은 이 자료로 못 가른다 — 두 CSV 의 nov 행 `reach_pct` 가 "
          u"전부 100.0000 이다(추첨 변동 0).")

    cols = ["scope", "label", "metric", "unit", "n", "printed", "draw_mean",
            "draw_sd", "diff", "diff_pct", "z", "t", "p", "p_holm",
            "min_pct", "max_pct", "note"]
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(u"\n저장 %s (%d행)" % (os.path.relpath(OUT, HERE), len(rows)))


if __name__ == "__main__":
    main()
