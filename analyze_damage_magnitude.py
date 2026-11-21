# -*- coding: utf-8 -*-
"""23.9.6.4 의 미해결 — "코사인은 피해 유무만 맞추고 **양**은 못 맞춘다".

왜 막혀 있었나
--------------
피해를 ms 로만 적었는데, 그 ms 가 큰지 작은지 말할 **분모**가 없었다.
`measure_episode_metrics.py` 가 에피소드마다 `ospf_path(cost_type=...)`
네 가지를 **정책이 실제로 걸은 그 환경·도착 그 시점**에서 뽑아
`episode_metrics_ref.csv` 에 남겼으므로, 이제 두 가지가 생긴다.

1) **분모**: `ref_delay_delay`(그 환경의 지연 최적)와
   `ref_bandwidth_delay`(논문의 모든 평가 스크립트가 쓰는 OSPF 기준선)
   사이가 "그 환경에서 실현된 여유" 다. 피해를 그 비율로 적을 수 있다.
2) **공변량**: 23.9.6.14 에서 조건 간 환경 대응이 깨진 걸 쟀다(지문
   59/60 불일치). 잃은 것은 공통난수 분산감소였다. 기준 경로는 **같은
   env 인스턴스·같은 틱**에서 계산되므로 `delay - ref_delay_delay` 는
   환경 수준 성분이 상쇄된 **안쪽 차이**다. 그 분산감소를 되찾는지 잰다.

한계를 먼저 적는다
-----------------
- 기준 경로는 **도착 틱**에서 뽑혔다. `step()` 첫 줄이
  `update_dynamic_qos()` 이고 그 함수의 228-229 가 누산이므로
  (23.9.6.14), 정책 경로가 길면 기준 경로도 **더 늦은 환경**에서
  재진다. 즉 `ref_*_delay` 는 홉 수에 오염돼 있다. [2] 절에서 그
  오염을 r 로 **직접 재고**, 초과값이 홉 매개 성분까지 같이 지운다는
  것을 명시한다. 초과값은 "환경 수준 제거" 가 아니라 "그 시점 그
  환경에서의 경로 품질 초과" 로만 읽어야 한다.
- 조건 간 비교는 **모델 수준 대응**만 쓴다. 23.9.6.14 가 금지한
  "에피소드 번호로 조건을 짝짓기" 는 하지 않는다 — 조건마다 먼저
  에피소드를 평균해 모델 값을 만들고, 그 다음 모델로 짝짓는다.
- [5] 의 상관은 조건 **6개**(또는 기존 열로 10개)짜리다. 검정력이
  사실상 없다. 서술 통계로만 읽어야 하고 p 를 붙이지 않는다.
- **비율 척도에서 Jensen 격차가 또 나왔다.** 여유(분모)가 에피소드마다
  변하므로 에피소드별 비율의 평균은 꼬리에 끌려간다(축소판에서 `frozen`
  1.417 대 평균끼리의 비 0.929). 검정에는 **평균끼리의 비**만 쓰고,
  에피소드별 평균은 [3] 에 꼬리 비중과 함께 서술로만 남긴다.
- Holm 족 **네 개**가 이 파일 안에 있다. 절대 섞어 인용하지 말 것:
  [4] 6조건 × 3척도 = **18** · [5b] 쌍 × 3척도(쌍 3개면 **9**) ·
  [5c] 3척도 = **3** · [6] 3척도 × 3대비 = **9**.
  23.9.6.10 의 15, 23.9.6.13 의 18 과도 **다른 족**이다. 인용 시 병기.

출력: review_results/damage_magnitude.csv
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
import analyze_perturb_vs_damage as AP  # noqa: E402

OUT_DIR = OC.OUT_DIR
# 축소판으로 전 구간을 미리 통과시킬 수 있게 입출력을 뺀다.
REF_CSV = os.path.join(OUT_DIR,
                       os.environ.get("DM_IN", "episode_metrics_ref.csv"))
OLD_CSV = os.path.join(OUT_DIR, "episode_metrics.csv")
OUT_CSV = os.environ.get("DM_OUT", "damage_magnitude.csv")
BASE = ("delay", "loss_perhop", "loss_sum", "hop", "hop_count", "has_rev")
REFS = ("delay", "hop", "bandwidth", "risk")
# 척도 세 가지. 모두 **지연**이다(손실은 기준 경로가 손실최적이 아니라
# 분모를 못 만든다 — ospf_path 에 loss 비용이 없다. [1] 에 서술만 한다).
SCALES = (("지연 원값", "delay"),
          ("지연 초과(최적대비)", "excess"),
          ("지연 여유분율", "frac_rom"))
# 용량-반응 축. `SHUFK` 는 50칸 중 k칸만 재배치하므로 값 다중집합을
# **정확히** 보존한다 — 노름비 1.000000, 주변분포 동일(측정:
# shuffle_geometry.csv). k=0 은 `intact`, k=50 은 `risk_dimshuffle` 이
# 그대로 된다.
# **주의(2026-11-14 수정)**: 전에 이 자리에 "k 는 순위 교란의 양만
# 움직인다" 고 적었는데 그건 **틀렸다**. shuffle_geometry.csv 를 재 보니
# k 가 오르면 cos_dev·rho_dev·l1·건드린칸이 **같이** 오른다(rho_dev/
# cos_dev 가 2.08~2.11 의 거의 일정한 비). 그래서 [5c] 의 용량-반응은
# "섞은 양" 의 용량-반응이고 **"코사인 고정 순위 교란" 의 것이 아니다**.
# 순위 가설은 [5b] 의 짝지은 비교만 다룰 수 있다.
KPTS = (("intact", 0), ("risk_shuf08", 8), ("risk_shuf16", 16),
        ("risk_shuf25", 25), ("risk_shuf32", 32),
        ("risk_dimshuffle", 50))


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    sxx = sum((a - mx) ** 2 for a in xs)
    syy = sum((b - my) ** 2 for b in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / (sxx * syy) ** 0.5


def fmean(vals):
    """nan 을 **빼고** 평균. 에피소드가 적으면 홉이 상수라 상관계수가
    nan 이 되는데, 그게 평균 전체를 오염시키면 안 된다."""
    ok = [v for v in vals if v == v]
    if not ok:
        return float("nan"), 0
    return sum(ok) / len(ok), len(ok)


def sd(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return (sum((v - m) ** 2 for v in vals) / (n - 1)) ** 0.5


def read_ref():
    """조건 -> 모델 -> 에피소드 행. 기준 열까지."""
    out = {}
    nbad = 0
    with io.open(REF_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = {}
            for c in BASE:
                d[c] = float(r[c])
            for ct in REFS:
                for sfx in ("delay", "lossph", "losssum", "hop"):
                    nm = "ref_%s_%s" % (ct, sfx)
                    d[nm] = float(r[nm])
            d["episode"] = int(r["episode"])
            d["excess"] = d["delay"] - d["ref_delay_delay"]
            room = d["ref_bandwidth_delay"] - d["ref_delay_delay"]
            d["room"] = room
            if room > 1e-9:
                d["frac"] = d["excess"] / room
            else:
                d["frac"] = float("nan")
                nbad += 1
            out.setdefault(r["condition"], {}).setdefault(
                r["model"], []).append(d)
    return out, nbad


def verify_overlap():
    """겹치는 9열이 기존 CSV 와 **정확히** 같은지. Dijkstra 가 RNG 를
    안 쓴다는 주장의 유일한 검증이다. 하나라도 다르면 기준 열 전부
    못 쓴다."""
    if not os.path.exists(OLD_CSV):
        return (0, 0, ["기존 CSV 가 없어 검증 생략"])
    cols = ("delay", "loss_perhop", "loss_sum", "hop", "hop_count",
            "has_rev")
    old = {}
    with io.open(OLD_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            old[(r["condition"], r["model"], r["episode"])] = r
    ncmp = nbad = 0
    notes = []
    newconds = set()
    with io.open(REF_CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            k = (r["condition"], r["model"], r["episode"])
            if k not in old:
                newconds.add(r["condition"])
                continue
            o = old[k]
            for c in cols:
                ncmp += 1
                if o[c] != r[c]:
                    nbad += 1
                    if len(notes) < 6:
                        notes.append("불일치 %s %s" % (str(k), c))
    notes.append("기존 CSV 에 없던 조건(검증 제외): %s"
                 % (", ".join(sorted(newconds)) or "없음"))
    return ncmp, nbad, notes


def agg(rows):
    n = float(len(rows))
    fr = [r["frac"] for r in rows if r["frac"] == r["frac"]]
    out = {
        "n_ep": int(n),
        "delay": sum(r["delay"] for r in rows) / n,
        "excess": sum(r["excess"] for r in rows) / n,
        # 에피소드별 비율의 평균. 분모(여유)가 에피소드마다 변하고
        # 작아질 수 있어 꼬리가 두껍다 — **검정에는 쓰지 않는다**.
        "frac_ep": (sum(fr) / len(fr)) if fr else float("nan"),
        "frac_tail": 100.0 * sum(1 for v in fr if v > 3.0) / max(len(fr), 1),
        "room_min": min(r["room"] for r in rows),
        "hop": sum(r["hop"] for r in rows) / n,
        "loss_sum": sum(r["loss_sum"] for r in rows) / n,
        "room": sum(r["room"] for r in rows) / n,
        "sd_delay": sd([r["delay"] for r in rows]),
        "sd_excess": sd([r["excess"] for r in rows]),
        "r_delay_ref": pearson([r["delay"] for r in rows],
                               [r["ref_delay_delay"] for r in rows]),
        "r_ref_hop": pearson([r["ref_delay_delay"] for r in rows],
                             [r["hop"] for r in rows]),
    }
    for ct in REFS:
        out["ref_%s" % ct] = sum(r["ref_%s_delay" % ct] for r in rows) / n
        out["refhop_%s" % ct] = sum(r["ref_%s_hop" % ct] for r in rows) / n
    # **검정용** 비율: 평균끼리의 비. 분모가 모델 하나당 하나뿐이라
    # 꼬리가 없다. 에피소드별 비율의 평균(`frac_ep`)과 크게 다를 수
    # 있고, 그 차이 자체가 Jensen 격차다 — [3] 에 나란히 찍는다.
    rm = out["ref_bandwidth"] - out["ref_delay"]
    out["frac_rom"] = ((out["delay"] - out["ref_delay"]) / rm
                       if rm > 1e-9 else float("nan"))
    # 정책 경로가 기준 경로와 **같은지**: 지연 6자리 일치 + 홉 일치.
    for ct in REFS:
        hit = sum(1 for r in rows
                  if abs(r["delay"] - r["ref_%s_delay" % ct]) < 1e-6
                  and abs(r["hop"] - r["ref_%s_hop" % ct]) < 1e-9)
        out["same_%s" % ct] = 100.0 * hit / n
    return out


def main():
    bar = "=" * 92
    print("")
    print(bar)
    print("23.9.6.4 의 '양' 질문 — 분모를 넣고 다시 본다")
    print(bar)

    print("")
    print("[0] 검증 — 기준 열을 붙인 롤아웃이 **비트 단위로 같은가**")
    print("-" * 92)
    print("  nx.shortest_path 는 결정론적이라 RNG 를 안 쓴다. 그러면")
    print("  겹치는 9열이 episode_metrics.csv 와 정확히 같아야 한다.")
    ncmp, nbad, notes = verify_overlap()
    print("  비교한 셀 %d개 · **불일치 %d개**" % (ncmp, nbad))
    for ln in notes:
        print("    " + ln)
    if nbad:
        raise SystemExit("롤아웃이 달라졌다 — 기준 열을 쓸 수 없다.")

    raw, nbadroom = read_ref()
    data = dict((c, dict((m, agg(rs)) for m, rs in raw[c].items()))
                for c in raw)
    conds = sorted(raw, key=lambda x: (x != "intact", x))
    print("")
    print("  조건 %d · 모델 %d · 분모<=0 에피소드 %d"
          % (len(conds), len(data[conds[0]]), nbadroom))

    print("")
    print("[1] 기준 경로 풍경 — 그 환경에서 무엇이 달성 가능했나")
    print("-" * 92)
    print("  '여유' = bandwidth 기준선 − 지연최적. 논문이 OSPF 라고")
    print("  부르는 비교군이 bandwidth 비용이므로 그게 천장이다.")
    print("%-20s %9s %9s %9s %9s %9s %9s"
          % ("조건", "정책", "최적", "hop비용", "OSPF(bw)", "risk", "여유"))
    for c in conds:
        ms = sorted(data[c])
        g = lambda k: sum(data[c][m][k] for m in ms) / len(ms)  # noqa: E731
        print("%-20s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f"
              % (c, g("delay"), g("ref_delay"), g("ref_hop"),
                 g("ref_bandwidth"), g("ref_risk"), g("room")))
    print("")
    print("  정책 경로가 기준 경로와 **동일**한 비율(지연 6자리+홉 일치)")
    print("%-20s %9s %9s %9s %9s %9s"
          % ("조건", "=최적", "=hop", "=OSPF", "=risk", "정책홉"))
    for c in conds:
        ms = sorted(data[c])
        g = lambda k: sum(data[c][m][k] for m in ms) / len(ms)  # noqa: E731
        print("%-20s %8.1f%% %8.1f%% %8.1f%% %8.1f%% %9.3f"
              % (c, g("same_delay"), g("same_hop"), g("same_bandwidth"),
                 g("same_risk"), g("hop")))
    print("")
    print("  손실은 분모를 못 만든다 — ospf_path 에 loss 비용이 없고")
    print("  bandwidth 기준 경로의 손실은 clip 상한(0.35/홉)에 붙어")
    print("  포화한다. 그래서 척도 셋은 **지연만** 다룬다.")

    print("")
    print("[2] 기준 경로가 **홉에 오염**됐는가 — 쓰기 전에 잰다")
    print("-" * 92)
    print("  기준 경로는 도착 틱에서 뽑혔고 누산은 매 스텝 돈다")
    print("  (23.9.6.14). 경로가 길면 기준도 더 늦은 환경에서 재진다.")
    print("%-20s %13s %13s %10s %10s"
          % ("조건", "r(정책,기준)", "r(기준,홉)", "SD 원값", "SD 초과"))
    for c in conds:
        ms = sorted(data[c])
        g = lambda k: sum(data[c][m][k] for m in ms) / len(ms)  # noqa: E731
        r1, k1 = fmean([data[c][m]["r_delay_ref"] for m in ms])
        r2, k2 = fmean([data[c][m]["r_ref_hop"] for m in ms])
        print("%-20s %+9.4f(%d) %+9.4f(%d) %10.3f %10.3f"
              % (c, r1, k1, r2, k2, g("sd_delay"), g("sd_excess")))
    allm = [(c, m) for c in conds for m in sorted(data[c])]
    s1 = sum(data[c][m]["sd_delay"] for c, m in allm) / len(allm)
    s2 = sum(data[c][m]["sd_excess"] for c, m in allm) / len(allm)
    print("")
    print("  전체 평균 SD: 원값 %.3f → 초과 %.3f (%+.1f%%)"
          % (s1, s2, 100.0 * (s2 - s1) / s1))
    if s2 < s1:
        print("  → 환경 수준 성분이 실제로 빠진다. 23.9.6.14 가 잃었다고")
        print("    적은 공통난수 분산감소를 **일부** 되찾는다.")
    else:
        print("  → 분산이 **늘었다**. 기준 경로는 공변량으로 쓸모가 없다.")

    print("")
    print("[3] 척도별 조건 평균")
    print("-" * 92)
    print("  여유분율 두 가지를 **나란히** 찍는다. `평균비`는 평균끼리의")
    print("  비(검정에 쓰는 것), `에피비`는 에피소드별 비율의 평균이다.")
    print("  분모가 에피소드마다 변해 꼬리가 두꺼우므로 둘이 벌어진다.")
    print("%-20s %10s %10s %9s %9s %7s %9s %7s"
          % ("조건", "원값(ms)", "초과(ms)", "평균비", "에피비",
             ">3%", "최소여유", "홉"))
    for c in conds:
        ms = sorted(data[c])
        g = lambda k: sum(data[c][m][k] for m in ms) / len(ms)  # noqa: E731
        print("%-20s %10.3f %10.3f %9.5f %9.5f %6.1f%% %9.3f %7.3f"
              % (c, g("delay"), g("excess"), g("frac_rom"), g("frac_ep"),
                 g("frac_tail"), min(data[c][m]["room_min"] for m in ms),
                 g("hop")))

    rows_out = []
    print("")
    print("[4] 피해 양 — intact 대비 **모델 대응차** (Holm 족 %d)"
          % ((len(conds) - 1) * len(SCALES)))
    print("-" * 92)
    print("  에피소드 번호로 짝짓지 않는다(23.9.6.14 금지). 조건마다")
    print("  모델별 에피소드 평균을 먼저 내고 모델로 짝짓는다.")
    base = "intact"
    dmg = {}
    keys, ps, stats = [], [], {}
    for c in conds:
        if c == base:
            continue
        for slab, skey in SCALES:
            pm = {}
            for m in sorted(set(data[c]) & set(data[base])):
                pm[m] = data[c][m][skey] - data[base][m][skey]
            mu, _sd, t, _d, p = RC.one_sample([pm[m] for m in sorted(pm)])
            dmg.setdefault(c, {})[skey] = mu
            keys.append((c, slab))
            ps.append(p)
            stats[(c, slab)] = (mu, t, len(pm))
    hol = RS.holm_bonferroni(ps)
    print("%-22s %-20s %12s %9s %10s"
          % ("조건", "척도", "추정", "t", "p_holm"))
    for (c, slab), ph in zip(keys, hol):
        mu, t, nn = stats[(c, slab)]
        mark = "**" if ph < 0.05 else "  "
        print("%s %-20s %-20s %12.5f %9.3f %10s"
              % (mark, c, slab, mu, t, RS.fmt_p(ph)))
        rows_out.append(["cond_vs_intact", c, slab, "%.6f" % mu,
                         "%.4f" % t, "%.6f" % ph,
                         (len(conds) - 1) * len(SCALES), nn])

    print("")
    print("[5] 같은 코사인에서 양이 갈리는 문제 — 척도가 메우는가")
    print("-" * 92)
    geo = {}
    with io.open(os.path.join(OUT_DIR, "perturb_vs_damage.csv"),
                 encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["metric"] == "delay":
                geo[r["condition"]] = dict(
                    l1=float(r["l1"]), norm_dev=float(r["norm_dev"]),
                    cos_dev=float(r["cos_dev"]), rho_dev=float(r["rho_dev"]),
                    diff=float(r["diff_n10"]))
    trio = [c for c in ("risk_rankpow05n", "risk_rankpow05", "risk_frozen",
                        "risk_dimshuffle") if c in dmg and c in geo]
    print("  코사인 편차가 거의 같은 조건들을 나란히 둔다.")
    print("%-20s %8s %8s %8s %10s %10s %10s"
          % ("조건", "cos_dev", "norm", "rho_dev", "원값", "초과", "여유분율"))
    for c in trio:
        print("%-20s %8.4f %8.4f %8.4f %10.3f %10.3f %10.5f"
              % (c, geo[c]["cos_dev"], geo[c]["norm_dev"],
                 geo[c]["rho_dev"], dmg[c]["delay"], dmg[c]["excess"],
                 dmg[c]["frac_rom"]))
    for slab, skey in SCALES:
        vs = [dmg[c][skey] for c in trio]
        if min(abs(v) for v in vs) > 1e-12:
            print("  %-20s 최대/최소 = %.2f배"
                  % (slab, max(vs) / min(vs)))
    print("")
    print("  기하 열과 피해의 순위상관(조건 %d개 — **검정력 없음**, 서술)"
          % len(dmg))
    both = sorted(set(dmg) & set(geo))
    print("%-20s %10s %10s %10s %10s"
          % ("척도", "l1", "norm_dev", "cos_dev", "rho_dev"))
    for slab, skey in SCALES:
        ys = [dmg[c][skey] for c in both]
        cols = []
        for g in ("l1", "norm_dev", "cos_dev", "rho_dev"):
            rr = AP.spearman([geo[c][g] for c in both], ys)
            cols.append(rr)
            rows_out.append(["rho_geom", slab, g, "%.4f" % rr, "", "",
                             0, len(both)])
        print("%-20s %+10.4f %+10.4f %+10.4f %+10.4f"
              % (slab, cols[0], cols[1], cols[2], cols[3]))
    print("")
    print("  기존 열만으로 조건 %d개(23.9.6.4 와 같은 집합):" % len(geo))
    allc = sorted(geo)
    ys = [geo[c]["diff"] for c in allc]
    for g in ("l1", "norm_dev", "cos_dev", "rho_dev"):
        rr = AP.spearman([geo[c][g] for c in allc], ys)
        print("    rho(%s, 지연피해) = %+.4f" % (g, rr))
        rows_out.append(["rho_geom_n10", "diff_n10", g, "%.4f" % rr,
                         "", "", 0, len(allc)])

    print("")
    print("[5b] **짝지은 비교** — 기하가 맞춰진 쌍에서 무엇이 남는가")
    print("-" * 92)
    print("  [5] 의 조건 상관은 n 이 작아 못 쓴다. 대신 cos_dev 와")
    print("  norm_dev 가 **거의 같은** 쌍을 골라 모델 대응차로 직접")
    print("  검정한다. 이건 모델 10개짜리라 검정력이 있다.")
    print("  주의: 짝은 **모델 수준**이다(23.9.6.14).")
    PAIRS = (
        # **설계된** 쌍. `frozen` 쌍은 공여 모델 값을 넣어 주변분포까지
        # 바뀌는 교란이 남아 있었다. `shuf25` 는 값 다중집합을 정확히
        # 보존하므로 노름비·주변분포가 `pow05n` 과 **같고** 코사인도
        # 0.2306 대 0.2338 로 맞다. 남는 차이는 rho_dev 뿐이다.
        # 그리고 남은 교란 둘(l1 0.1358<0.2685, 건드린 칸 46.8%<100%)은
        # **둘 다 `shuf25` 쪽이 더 약한 교란**이라, 피해가 더 크면
        # 주변분포·크기 논거로는 설명이 안 된다(23.9.6.14 와 같은 구조).
        ("risk_shuf25", "risk_rankpow05n",
         "cos 0.2306/0.2338 · 노름비 1.0000/1.0000 · 주변분포 동일"
         " → rho_dev 0.4867/0.0000 (교란 l1·건드린칸 모두 불리)"),
        ("risk_frozen", "risk_rankpow05n",
         "cos 0.2364/0.2338 · norm 0.0005/0.0000 → rho_dev 0.7505/0.0000"),
        ("risk_dimshuffle", "risk_frozen",
         "rho_dev 0.9965/0.7505 · cos 0.4748/0.2364"),
    )
    kk, pp, st = [], [], {}
    for hi, lo, _why in PAIRS:
        if hi not in data or lo not in data:
            continue
        for slab, skey in SCALES:
            pm = {}
            for m in sorted(set(data[hi]) & set(data[lo])):
                pm[m] = data[hi][m][skey] - data[lo][m][skey]
            mu, _sd, t, _d, p = RC.one_sample([pm[m] for m in sorted(pm)])
            kk.append((hi, lo, slab))
            pp.append(p)
            st[(hi, lo, slab)] = (mu, t, len(pm),
                                  sum(1 for v in pm.values() if v > 0))
    hh = RS.holm_bonferroni(pp) if pp else []
    npair = sum(1 for h, lo2, _w in PAIRS if h in data and lo2 in data)
    print("  Holm 족 %d (쌍 %d × 척도 %d)"
          % (len(pp), npair, len(SCALES)))
    for hi, lo, why in PAIRS:
        if hi not in data or lo not in data:
            continue  # 제목만 찍히는 유령 쌍을 막는다
        print("")
        print("  %s − %s" % (hi, lo))
        print("    %s" % why)
        for (a, b, slab), ph in zip(kk, hh):
            if (a, b) != (hi, lo):
                continue
            mu, t, nn, npos = st[(a, b, slab)]
            mark = "**" if ph < 0.05 else "  "
            print("  %s   %-22s %12.5f  t %7.3f  %10s  부호 %d/%d"
                  % (mark, slab, mu, t, RS.fmt_p(ph), npos, nn))
            rows_out.append(["pair", "%s-%s" % (a, b), slab, "%.6f" % mu,
                             "%.4f" % t, "%.6f" % ph, len(pp), nn])

    print("")
    print("[5c] **용량-반응** — 순위 교란의 **양**에 피해가 따라오는가")
    print("-" * 92)
    print("  [5b] 는 한 점(k=25)이다. 섞기 족은 k 를 올려도 값 다중집합이")
    print("  그대로라 노름비 1.0000·주변분포가 고정되고 **순위 교란만**")
    print("  늘어난다. 그래서 k 가 용량 축이 된다(k=0 intact, k=50")
    print("  dimshuffle 이 양 끝). 검정은 모델마다 Spearman(k, 피해) 를")
    print("  내고 그 값들에 일표본 t — 조건 간 n=6 상관이 아니라 **모델")
    print("  수준 n=10** 이다(23.9.6.14 가 금지하는 짝짓기를 피한다).")
    avail = [(c, k) for c, k in KPTS if c in data]
    print("  점 %d개: %s" % (len(avail), ", ".join(
        "%s(k=%d)" % (c, k) for c, k in avail)))
    if len(avail) < 4:
        print("  → 4점 미만이라 생략(섞기 롤아웃이 아직 없다).")
    else:
        models = sorted(set.intersection(*[set(data[c]) for c, _ in avail]))
        print("  공통 모델 %d개" % len(models))
        print("")
        print("  곡선(공통 모델 평균 피해):")
        for slab, skey in SCALES:
            cur = []
            for c, kv in avail:
                mu, _nn = fmean([data[c][m][skey] for m in models])
                cur.append("k%-2d %9.4f" % (kv, mu))
            print("    %-22s %s" % (slab, " · ".join(cur)))
        print("")
        lab2, pp2, st2 = [], [], {}
        for slab, skey in SCALES:
            rh = []
            for m in models:
                pr = [(float(kv), data[c][m][skey]) for c, kv in avail
                      if data[c][m][skey] == data[c][m][skey]]
                if len(pr) < 4:
                    continue
                rh.append(AP.spearman([a for a, _ in pr],
                                      [b for _, b in pr]))
            if not rh:
                continue
            mu, _sd, t, _df, pv = RC.one_sample(rh)
            lab2.append(slab)
            pp2.append(pv)
            st2[slab] = (mu, t, len(rh), sum(1 for v in rh if v > 0))
        hh2 = RS.holm_bonferroni(pp2) if pp2 else []
        print("  Holm 족 %d (척도 %d) — [5b] 족과 **다른 족**이다"
              % (len(pp2), len(SCALES)))
        print("   %-22s %10s %9s %11s %8s"
              % ("척도", "평균 rho", "t", "p_holm", "부호"))
        for slab, ph in zip(lab2, hh2):
            mu, t, nn, npos = st2[slab]
            mark = "**" if ph < 0.05 else "  "
            print("%s %-22s %10.4f %9.3f %11s  %d/%d"
                  % (mark, slab, mu, t, RS.fmt_p(ph), npos, nn))
            rows_out.append(["dose", "k_curve", slab, "%.6f" % mu,
                             "%.4f" % t, "%.6f" % ph, len(pp2), nn])
        print("")
        print("  한계: rho 는 **단조성**만 본다. 기울기 크기는 곡선에서")
        print("  읽어야 하고, k 간격이 균등하지 않다(0·8·16·25·32·50).")

    print("")
    print("[6] 2×2 대비를 세 척도로 다시 — 척도가 판정을 바꾸는가")
    print("-" * 92)
    print("  Holm 족 %d (3척도 × 3대비). 23.9.6.10 의 15, 23.9.6.13 의"
          % (len(SCALES) * 3))
    print("  18 과 **다른 족**이다.")
    RC.duplication_report("rank_vs_value.csv", AX.A)
    cells = (AX.A, AX.B, AX.C, AX.D)
    miss = [c for c in cells if c not in data]
    if miss:
        raise SystemExit("2×2 칸이 빈다: %s" % ", ".join(miss))
    mods = sorted(set.intersection(*[set(data[c]) for c in cells]))
    pack = {}
    for slab, skey in SCALES:
        for cl in ("코사인", "노름", "상호작용"):
            pm = {}
            for m in mods:
                va = data[cells[0]][m][skey]
                vb = data[cells[1]][m][skey]
                vc = data[cells[2]][m][skey]
                vd = data[cells[3]][m][skey]
                if cl == "코사인":
                    pm[m] = (vc + vd) / 2.0 - (va + vb) / 2.0
                elif cl == "노름":
                    pm[m] = (vb + vd) / 2.0 - (va + vc) / 2.0
                else:
                    pm[m] = (vd - vc) - (vb - va)
            pack[(slab, cl)] = pm
    for nlab, mode in (("n=10", None), ("n=5", "dup"), ("n=3", "cls")):
        kk, pp, st = [], [], {}
        for slab, _s in SCALES:
            for cl in ("코사인", "노름", "상호작용"):
                pm = pack[(slab, cl)]
                if mode is None:
                    use = [pm[m] for m in sorted(pm)]
                elif mode == "dup":
                    use = RC.collapse(pm, lambda nm: RC.DUP_GROUP[nm])
                else:
                    use = RC.collapse(pm, RC.model_class)
                mu, _sd, t, _d, p = RC.one_sample(use)
                kk.append((slab, cl))
                pp.append(p)
                st[(slab, cl)] = (mu, t, len(use))
        hh = RS.holm_bonferroni(pp)
        print("")
        print("  [%s] %-28s %12s %9s %10s"
              % (nlab, "척도 / 대비", "추정", "t", "p_holm"))
        for (slab, cl), ph in zip(kk, hh):
            mu, t, nn = st[(slab, cl)]
            mark = "**" if ph < 0.05 else "  "
            print("  %s %-28s %12.5f %9.3f %10s (n=%d)"
                  % (mark, "%s / %s" % (slab, cl), mu, t, RS.fmt_p(ph), nn))
            rows_out.append([nlab, slab, cl, "%.6f" % mu, "%.4f" % t,
                             "%.6f" % ph, len(SCALES) * 3, nn])

    path = os.path.join(OUT_DIR, OUT_CSV)
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "a", "b", "estimate", "t", "p_holm",
                    "holm_family", "n"])
        for r in rows_out:
            w.writerow(r)
    print("")
    print("저장: %s (%d행)" % (path, len(rows_out)))
    print(bar)


if __name__ == "__main__":
    main()
