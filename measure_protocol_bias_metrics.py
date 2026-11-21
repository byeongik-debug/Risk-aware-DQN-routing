# -*- coding: utf-8 -*-
u"""7월 규약 대 11월 규약의 **계통편향을 지표 셋 전부**에서 잰다.

왜 쓰는가
---------
FINDINGS 26.10.4/26.10.8 은 같은 `.pth` 25개를 두 평가 규약으로 채점해
지연의 계통편향 **+1.249%** 를 얻었고, 26.10.11 은 한계로 이렇게 적어
두었다 — "계통 +1.249% 분석은 그대로 지연 하나다. **홉·손실의 계통편향은
안 재었다**." 그런데 그 측정의 산출 `review_results/july_nov_protocol.csv`
에는 `july_hop/nov_hop`, `july_loss/nov_loss` 가 **이미 같이** 들어 있다.
재학습도 재채점도 필요 없이 닫히는 한계라 여기서 닫는다.

같이 갚는 것
------------
지연 쪽 집계값(평균 +1.249 · 절대평균 1.385 · sd 1.108 · 22/25 · 이항 p)
은 FINDINGS 산문에만 있고 **CSV 로 저장된 적이 없다**(32.5절이 말하는
"산출물 미보존"). 이 스크립트가 지연을 포함해 세 지표를 같은 형식으로
저장하므로 그 값들도 산출물에 남는다.

못 하는 것
----------
`july_reach` 는 있는데 **`nov_reach` 열이 없다** → 도달률 계통편향은 이
CSV 로 못 잰다. 추정하지 않고 결과에 그대로 적는다.

출력: `review_results/july_nov_protocol_bias.csv`
"""
import csv
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import review_stats as RS                                  # noqa: E402

SRC = os.path.join(HERE, "review_results", "july_nov_protocol.csv")
OUT = os.path.join(HERE, "review_results", "july_nov_protocol_bias.csv")

METRICS = [
    (u"delay", "july_delay", "nov_delay", u"ms"),
    (u"hop", "july_hop", "nov_hop", u"홉"),
    (u"loss", "july_loss", "nov_loss", u"유실률"),
]


def binom_two_sided(k, n):
    u"""부호검정: 양수 k개가 n개 중 나올 정확 이항 양측 p (p0=0.5)."""
    k = max(k, n - k)
    tail = sum(math.factorial(n) // (math.factorial(i) * math.factorial(n - i))
               for i in range(k, n + 1))
    return min(1.0, 2.0 * tail / float(2 ** n))


def paired_t(diffs):
    u"""차이에 대한 일표본 t (= 쌍체 t). review_stats 의 t 분포를 쓴다."""
    n = len(diffs)
    m = sum(diffs) / n
    sd = math.sqrt(sum((d - m) ** 2 for d in diffs) / (n - 1))
    if sd == 0.0:
        return m, sd, float("inf") if m else 0.0, 0.0 if m else 1.0
    t = m / (sd / math.sqrt(n))
    return m, sd, t, RS.t_two_sided_p(t, n - 1)


def summarize(name, pairs, unit):
    u"""pairs: [(july, nov)] → 집계 한 줄 (dict)."""
    d = [b - a for a, b in pairs]
    pct = [100.0 * (b - a) / a for a, b in pairs if a]
    m, sd, t, p_t = paired_t(d)
    mp, sdp, _tp, _pp = paired_t(pct)
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    return {
        "metric": name,
        "unit": unit,
        "n": len(d),
        "mean_diff": "%.6f" % m,
        "sd_diff": "%.6f" % sd,
        "mean_pct": "%.4f" % mp,
        "sd_pct": "%.4f" % sdp,
        "abs_mean_pct": "%.4f" % (sum(abs(x) for x in pct) / len(pct)),
        "min_pct": "%.4f" % min(pct),
        "max_pct": "%.4f" % max(pct),
        "n_pos": pos,
        "n_neg": neg,
        "n_zero": len(d) - pos - neg,
        "sign_p": "%.8f" % binom_two_sided(pos, len(d)),
        "paired_t": "%.4f" % t,
        "paired_p": "%.8f" % p_t,
    }


def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def ols(xs, ys):
    u"""최소제곱 (기울기, 절편)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return b, my - b * mx


def hop_control_rows(rows):
    u"""홉을 통제해도 지연·손실 치우침이 남는가.

    두 가지 반론을 같이 막는다. (가) 손실은 **홉당 평균**이므로 분모가
    줄면 올라간다 → `r(홉%, 손실%)` 과 홉을 곱한 지표로 본다. (나) 지연은
    경로가 길어지면 늘어난다 → 홉 변화를 설명변수로 회귀해 **절편**(홉
    변화 0 일 때 남는 치우침)과 그 보정값의 쌍체 검정을 본다.

    `print` 만 하지 않고 **행으로 돌려준다** — 산출물에 안 남으면 나중에
    문서의 숫자를 다시 받을 길이 없다(32.5절의 "산출물 미보존").
    """
    def pct(cj, cn):
        return [100.0 * (float(r[cn]) - float(r[cj])) / float(r[cj])
                for r in rows]

    hp = pct("july_hop", "nov_hop")
    lp = pct("july_loss", "nov_loss")
    dp = pct("july_delay", "nov_delay")

    res = []
    for nm, xs, ys in ((u"r_hop_loss", hp, lp), (u"r_hop_delay", hp, dp),
                       (u"r_loss_delay", lp, dp)):
        res.append({"scope": "hop_control", "arm": u"(전체)", "metric": nm,
                    "unit": u"상관", "n": len(rows),
                    "pearson_r": "%+.4f" % pearson(xs, ys)})

    ps = []
    for nm, ys in ((u"delay_given_hop", dp), (u"loss_given_hop", lp)):
        b, a0 = ols(hp, ys)
        adj = [y - b * x for x, y in zip(hp, ys)]
        m, sd, t, p = paired_t(adj)
        ps.append(p)
        res.append({"scope": "hop_control", "arm": u"(전체)", "metric": nm,
                    "unit": u"%", "n": len(adj),
                    "mean_pct": "%+.4f" % m, "sd_pct": "%.4f" % sd,
                    "min_pct": "%+.4f" % min(adj),
                    "max_pct": "%+.4f" % max(adj),
                    "n_pos": sum(1 for x in adj if x > 0),
                    "n_neg": sum(1 for x in adj if x < 0), "n_zero": 0,
                    "sign_p": "%.8f" % binom_two_sided(
                        sum(1 for x in adj if x > 0), len(adj)),
                    "paired_t": "%.4f" % t, "paired_p": "%.8f" % p,
                    "slope_vs_hop": "%+.4f" % b,
                    "pearson_r": "%+.4f" % pearson(hp, ys)})
        assert abs(a0 - m) < 1e-9, (a0, m)   # 절편 = 보정값 평균
    holm = RS.holm_bonferroni(ps)
    for s, h in zip([r for r in res if r["metric"].endswith("_given_hop")],
                    holm):
        s["paired_p_holm"] = "%.8f" % (h[1] if isinstance(h, tuple) else h)
    return res


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8")))
    print(u"원본 %s — %d행" % (os.path.basename(SRC), len(rows)))
    assert len(rows) == 25, len(rows)

    # 저장된 delay_pct 와 재계산이 같은지부터 확인한다(원본 신뢰 점검).
    worst = max(abs(float(r["delay_pct"])
                    - 100.0 * (float(r["nov_delay"]) - float(r["july_delay"]))
                    / float(r["july_delay"])) for r in rows)
    print(u"  저장된 delay_pct 와 재계산 최대차 %.6f %%p (반올림 한계 0.00005)"
          % worst)

    out = []
    for name, cj, cn, unit in METRICS:
        pairs = [(float(r[cj]), float(r[cn])) for r in rows]
        s = summarize(name, pairs, unit)
        s["scope"] = "all"
        s["arm"] = "(전체)"
        out.append(s)

    # 손실 열은 **홉당 평균**이다. 홉이 안 움직였는데 손실이 움직였다면
    # "분모가 줄어서 올라간 것" 이라는 반론이 먼저 나온다. 홉을 곱해
    # 분모를 없앤 대리 지표(경로 총량)로 같이 재서 그 반론을 막는다.
    prod = [(float(r["july_loss"]) * float(r["july_hop"]),
             float(r["nov_loss"]) * float(r["nov_hop"])) for r in rows]
    s = summarize(u"loss_x_hop", prod, u"유실률×홉")
    s["scope"] = "all"
    s["arm"] = u"(전체)"
    out.append(s)

    ps = [float(s["paired_p"]) for s in out]
    holm = RS.holm_bonferroni(ps)
    for s, h in zip(out, holm):
        s["paired_p_holm"] = "%.8f" % (h[1] if isinstance(h, tuple) else h)

    # 팔별로도 나눠 둔다 — 치우침이 한 팔에서 오는지 보려면 필요하다.
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        for name, cj, cn, unit in METRICS:
            pairs = [(float(r[cj]), float(r[cn])) for r in sub]
            s = summarize(name, pairs, unit)
            s["scope"] = "arm"
            s["arm"] = a
            s["paired_p_holm"] = ""
            out.append(s)

    cols = ["scope", "arm", "metric", "unit", "n", "mean_diff", "sd_diff",
            "mean_pct", "sd_pct", "abs_mean_pct", "min_pct", "max_pct",
            "n_pos", "n_neg", "n_zero", "sign_p", "paired_t", "paired_p",
            "paired_p_holm", "slope_vs_hop", "pearson_r"]
    out += hop_control_rows(rows)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for s in out:
            w.writerow(s)

    print()
    print(u"=== 전체 25개 모델 — 지표별 계통편향 (11월 − 7월) ===")
    print(u"  %-6s %12s %10s %8s %9s %10s %10s"
          % (u"지표", u"평균차", u"평균%", u"양수", u"부호p", u"쌍체p",
             u"쌍체p_holm"))
    for s in out:
        if s["scope"] != "all":
            continue
        print(u"  %-6s %12s %9s%% %5s/%-2s %9s %10s %10s"
              % (s["metric"], s["mean_diff"], s["mean_pct"], s["n_pos"],
                 s["n"], RS.fmt_p(float(s["sign_p"])),
                 RS.fmt_p(float(s["paired_p"])),
                 RS.fmt_p(float(s["paired_p_holm"]))))
    print()
    print(u"  범위: " + u" · ".join(
        u"%s %s~%s%%" % (s["metric"], s["min_pct"], s["max_pct"])
        for s in out if s["scope"] == "all"))
    print(u"  절대평균%%: " + u" · ".join(
        u"%s %s" % (s["metric"], s["abs_mean_pct"])
        for s in out if s["scope"] == "all"))
    print()
    for s in out:
        if s["scope"] != "hop_control":
            continue
        if s["metric"].startswith("r_"):
            print(u"  상관(차이%% 사이) %-14s r = %s"
                  % (s["metric"], s["pearson_r"]))
        else:
            print(u"  %-18s 기울기 %s · **절편(홉 보정 후 평균) %s%%** "
                  u"(t %s, p %s, p_holm %s)"
                  % (s["metric"], s["slope_vs_hop"], s["mean_pct"],
                     s["paired_t"], RS.fmt_p(float(s["paired_p"])),
                     RS.fmt_p(float(s["paired_p_holm"]))))
    print(u"  → 손실 편향이 분모(홉)에서 온 것이면 r(홉,손실) 이 음으로 커야 한다.")
    print()
    print(u"  도달률은 못 잰다 — 원본에 `july_reach` 만 있고 `nov_reach` 가 없다.")
    print(u"저장: " + OUT)


if __name__ == "__main__":
    main()
