"""표 14(미지 토폴로지 일반화) — 지연 6칸을 채우고 채울 수 있는 범위를 확정한다.

두 가지 일을 한다.

1. 출처 고정. 원고 표 14에 이미 인쇄된 18칸(3비율 x {OSPF, 제안} x
   {Loss, Hop, Risky})을 `review_results/topology_generalization_runs.csv`의
   토폴로지 평균과 전수 대조한다. 그 CSV에는 제안 모델이 둘 있다 —
   `frozen`(= ../SDN_DQN_new_env/risk_only_model_final.pth, 논문 본표 표 10/11이
   쓰는 동결본)과 `merged`(= review_response/ 사본, 친구 repo 유래). 둘은 재저장이
   아니라 별개로 학습된 가중치다(동일 원소 5.93%). 어느 쪽이 표 14를 만들었는지는
   인쇄된 18칸이 답한다. 표 11에서 쓴 방법과 같다 — 이미 인쇄된 칸이 출처를 고정하면
   빈칸은 추정이 아니라 같은 파일에서 읽어 오는 값이 된다.

2. 지연을 채우고, 그 칸에 붙일 수 있는 주장을 검정한다. 짝짓기 단위는 토폴로지다
   (비율별 n=10). 에피소드를 풀면 같은 토폴로지 안의 에피소드가 독립이 아니어서
   n이 부풀고 p가 거짓으로 작아진다. 그래서 토폴로지당 평균 하나를 관측으로 보고
   대응표본 t + Holm을 쓴다.

결론을 미리 적어 둔다. 지연 6칸은 로그에 있으므로 채울 수 있다. 그러나 "제안이
OSPF보다 빠르다"는 주장은 n=10에서 통과하지 못한다 — 세 비율 모두 관측 차이가
쌍대응 MDE보다 작고, 토폴로지별 차이가 양방향으로 100 ms 넘게 진동한다. 칸은 채우고
주장은 달지 않는 게 맞다.

주의. 이 표의 Loss/Risky 열은 홉당 평균 규약이다
(`FINDINGS_topology_generalization.md` 3절). 종단간 규약으로 다시 계산하면 우위는
12/12 유지되지만 크기가 25~27%p 깎인다. 지연 열만 채우고 나머지를 그대로 두면
표 안에서 규약이 섞인다.

재현:
    PYTHONIOENCODING=utf-8 python -X utf8 analyze_table14.py
"""

import collections
import csv
import math
import os

import review_stats as RS

BASE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(BASE, "review_results", "topology_generalization_runs.csv")
OUT = os.path.join(BASE, "review_results", "table14_delay.csv")

OSPF = "OSPF"
FROZEN = "Risk-DQN[frozen]"
MERGED = "Risk-DQN[merged]"

# 원고 표 14에 인쇄된 값. (평균, sd). 지연 열은 전부 【빈칸】이므로 여기 없다.
PUB = {
    ("0.2", OSPF): {"loss": (20.0, 5.9), "hop": (4.6, 0.5), "risky": (55.9, 17.6)},
    ("0.2", "P"): {"loss": (7.9, 4.4), "hop": (6.5, 0.9), "risky": (20.1, 13.0)},
    ("0.4", OSPF): {"loss": (20.6, 4.5), "hop": (4.1, 0.3), "risky": (57.6, 13.2)},
    ("0.4", "P"): {"loss": (7.9, 3.8), "hop": (6.9, 1.9), "risky": (20.1, 11.3)},
    ("0.6", OSPF): {"loss": (23.3, 3.0), "hop": (4.2, 0.3), "risky": (65.7, 8.8)},
    ("0.6", "P"): {"loss": (9.5, 5.7), "hop": (6.7, 1.2), "risky": (25.0, 16.6)},
}

# CSV 열 -> (원고 지표 이름, 100배 여부)
METRICS = [("loss", "loss", True), ("hop", "hop", False),
           ("risky_link_usage", "risky", False)]


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load():
    rows = list(csv.DictReader(open(RUNS, encoding="utf-8-sig")))
    index = {(r["method"], r["topology"]): r for r in rows}
    tops = collections.defaultdict(set)
    for r in rows:
        tops[r["rewire_ratio"]].add(r["topology"])
    return rows, index, {k: sorted(v) for k, v in tops.items()}


def cell(index, topologies, method, column, scale):
    """토폴로지 평균의 (평균, sd, n, 버린 개수). 짝짓기 단위가 토폴로지다."""
    vals, dropped = [], 0
    for top in topologies:
        row = index.get((method, top))
        if row is None:
            continue
        v = finite(row[column])
        if v is None:
            dropped += 1
            continue
        vals.append(v * (100.0 if scale else 1.0))
    if len(vals) < 2:
        return float("nan"), float("nan"), len(vals), dropped
    mean, sd = RS.mean_std(vals)
    return mean, sd, len(vals), dropped


def verdict(measured, published, prec=1):
    """표 11과 같은 3단 판정. 인쇄 자리수로 반올림해서 비교한다."""
    if not math.isfinite(measured):
        return "측정불가"
    tol = 0.5 * 10 ** (-prec)
    if abs(round(measured, prec) - published) < 1e-9:
        return "일치"
    if abs(measured - published) <= 2 * tol:
        return "반올림"
    return "** 불일치 **"


def paired(index, topologies, method_a, method_b, column):
    """method_a - method_b 의 토폴로지별 차이. 한쪽이 nan인 쌍은 버린다."""
    diffs, dropped = [], 0
    for top in topologies:
        a, b = index.get((method_a, top)), index.get((method_b, top))
        if not (a and b):
            continue
        x, y = finite(a[column]), finite(b[column])
        if x is None or y is None:
            dropped += 1
            continue
        diffs.append(x - y)
    return diffs, dropped


def one_sample(diffs, alpha=0.05):
    """(평균, sd, t, df, p, 쌍대응MDE). MDE = t_crit * sd/sqrt(n)."""
    n = len(diffs)
    if n < 2:
        return (float("nan"),) * 6
    mean, sd = RS.mean_std(diffs)
    se = sd / math.sqrt(n)
    if se == 0.0:
        return mean, sd, float("inf"), n - 1, 0.0, 0.0
    t = mean / se
    df = n - 1
    return mean, sd, t, df, RS.t_two_sided_p(t, df), RS.t_critical(df, alpha) * se


def required_pairs(diffs, alpha=0.05, power=0.80):
    """관측 효과를 80% 검정력으로 가릴 토폴로지 수. 검정력 항은 정규근사."""
    if len(diffs) < 2:
        return float("nan")
    mean, sd = RS.mean_std(diffs)
    if mean == 0.0 or sd == 0.0:
        return float("nan")
    z_power = 0.841621
    n = ((1.959964 + z_power) * sd / abs(mean)) ** 2
    for _ in range(60):
        n = ((RS.t_critical(max(1, int(round(n)) - 1), alpha) + z_power)
             * sd / abs(mean)) ** 2
    return int(math.ceil(n))


def section1(index, tops):
    """인쇄된 18칸으로 표 14의 제안 모델이 어느 체크포인트인지 고정한다."""
    print("=" * 78)
    print("1. 출처 대조 — 원고 표 14에 이미 인쇄된 18칸")
    print("=" * 78)
    print("%-5s %-9s %-7s %9s %9s %-11s" % (
        "비율", "행", "지표", "원고", "측정", "판정"))
    print("-" * 78)
    score = collections.Counter()
    for ratio in sorted(tops):
        for label, method in (("OSPF", OSPF), ("frozen", FROZEN),
                              ("merged", MERGED)):
            key = (ratio, OSPF if method == OSPF else "P")
            for column, name, scale in METRICS:
                mean, _, _, _ = cell(index, tops[ratio], method, column, scale)
                pub = PUB[key][name][0]
                tag = verdict(mean, pub)
                score[(label, tag)] += 1
                print("%-5s %-9s %-7s %9.1f %9.2f %-11s" % (
                    ratio, label, name, pub, mean, tag))
            if method == OSPF:
                print()
        print("-" * 78)
    print("판정 집계 (행 9칸씩):")
    for label in ("OSPF", "frozen", "merged"):
        hit = score[(label, "일치")] + score[(label, "반올림")]
        print("  %-7s 일치 %d / 반올림 %d / 불일치 %d  -> %d/9"
              % (label, score[(label, "일치")], score[(label, "반올림")],
                 score[(label, "** 불일치 **")], hit))
    print()
    print("OSPF 행은 모델과 무관하므로 9/9여야 정상이다. 제안 행에서 9/9를 내는")
    print("체크포인트가 표 14를 만든 것이다. frozen이 아니라면, 표 14의 '제안 기법'은")
    print("표 10/11의 '제안 기법'과 다른 모델이라는 뜻이다 (1.2절 상대경로 함정).")
    print()


def section2(index, tops):
    """지연 6칸."""
    print("=" * 78)
    print("2. 지연 6칸 — 같은 파일의 delay 열에서 읽는다")
    print("=" * 78)
    print("%-5s %-9s %7s %9s %9s %5s %s" % (
        "비율", "행", "n", "평균(ms)", "sd", "제외", "에피소드내 sd 평균"))
    print("-" * 78)
    filled = {}
    for ratio in sorted(tops):
        for label, method in (("OSPF", OSPF), ("frozen", FROZEN),
                              ("merged", MERGED)):
            mean, sd, n, dropped = cell(index, tops[ratio], method, "delay",
                                        False)
            inner = [finite(index[(method, t)]["delay_episode_std"])
                     for t in tops[ratio] if (method, t) in index]
            inner = [v for v in inner if v is not None]
            filled[(ratio, label)] = (mean, sd, n, dropped)
            print("%-5s %-9s %7d %9.2f %9.2f %5d %18.1f" % (
                ratio, label, n, mean, sd, dropped,
                sum(inner) / len(inner) if inner else float("nan")))
        print("-" * 78)
    print("sd 는 토폴로지 10개 사이의 표준편차다 (에피소드별 sd가 아니다).")
    print("표 14의 다른 열도 같은 규약이므로 ± 의 의미가 열마다 어긋나지 않는다.")
    print()
    return filled


def section3(index, tops):
    """채운 칸에 주장을 달 수 있는지 — 토폴로지 짝짓기 검정."""
    print("=" * 78)
    print("3. 그 칸에 '빠르다'를 쓸 수 있는가 — 대응표본 검정")
    print("=" * 78)
    results = {}
    for label, method in (("merged", MERGED), ("frozen", FROZEN)):
        print("[%s] 지연 차이 = 제안 - OSPF (음수가 제안에 유리)" % label)
        print("%-5s %3s %9s %9s %8s %8s %8s %8s %s" % (
            "비율", "n", "차이", "sd(차이)", "t", "p", "p_holm", "쌍MDE",
            "판정"))
        stash, ps = [], []
        for ratio in sorted(tops):
            diffs, dropped = paired(index, tops[ratio], method, OSPF, "delay")
            mean, sd, t, df, p, mde = one_sample(diffs)
            stash.append((ratio, diffs, mean, sd, t, p, mde, dropped))
            ps.append(p)
        for item, p_holm in zip(stash, RS.holm_bonferroni(ps)):
            ratio, diffs, mean, sd, t, p, mde, dropped = item
            call = "해상도 밖" if abs(mean) < mde else "해상도 안"
            if p_holm < 0.05:
                call = "유의"
            print("%-5s %3d %+9.2f %9.2f %8.3f %8s %8s %8.2f %s" % (
                ratio, len(diffs), mean, sd, t, RS.fmt_p(p),
                RS.fmt_p(p_holm), mde, call))
            results[(label, ratio)] = (mean, p, p_holm, mde,
                                       required_pairs(diffs), diffs)
        print("  토폴로지별 차이(ms):")
        for ratio, diffs, _, _, _, _, _, _ in stash:
            print("    %-5s %s" % (
                ratio, " ".join("%+.0f" % d for d in diffs)))
        print("  관측 효과를 80% 검정력으로 가리려면 필요한 토폴로지 수:")
        for ratio in sorted(tops):
            need = results[(label, ratio)][4]
            shown = ("-" if not math.isfinite(need)
                     else ">100,000" if need > 100000 else format(need, ","))
            print("    %-5s %s개 (현재 10개)" % (ratio, shown))
        print()
    print("세 비율 모두 |관측 차이| < 쌍MDE면, 이 실험은 지연 차이를 '있다'고도")
    print("'없다'고도 판정할 수 없다. 칸은 채우되 본문에 개선을 쓰면 과잉 주장이다.")
    print()
    return results


def section4(index, tops):
    """홉은 유의하게 길어졌는데 지연은 나빠지지 않았다 — 쓸 수 있는 문장."""
    print("=" * 78)
    print("4. 대신 쓸 수 있는 문장 — 홉은 길어졌는데 지연은 나빠지지 않았다")
    print("=" * 78)
    print("%-5s %-7s %10s %8s %8s %10s %s" % (
        "비율", "모델", "홉 차이", "p_holm", "판정", "지연 차이", "판정"))
    print("-" * 78)
    for label, method in (("merged", MERGED), ("frozen", FROZEN)):
        hop_ps, hop_items = [], []
        for ratio in sorted(tops):
            diffs, _ = paired(index, tops[ratio], method, OSPF, "hop")
            mean, sd, t, df, p, mde = one_sample(diffs)
            hop_items.append((ratio, mean, p, mde))
            hop_ps.append(p)
        for (ratio, mean, p, mde), p_holm in zip(hop_items,
                                                 RS.holm_bonferroni(hop_ps)):
            d_diffs, _ = paired(index, tops[ratio], method, OSPF, "delay")
            d_mean, _, _, _, _, d_mde = one_sample(d_diffs)
            print("%-5s %-7s %+10.3f %8s %8s %+10.2f %s" % (
                ratio, label, mean, RS.fmt_p(p_holm),
                "유의" if p_holm < 0.05 else "-", d_mean,
                "해상도 밖" if abs(d_mean) < d_mde else "해상도 안"))
    print()
    print("홉은 유의하게 길어졌고(+2~+3홉) 지연은 유의하게 나빠지지 않았다.")
    print("'위험 링크를 우회하는 더 긴 경로가 지연 측면에서 손해는 아니었다'까지가")
    print("측정이 지지하는 범위다. 개선 주장은 여기 들어가지 않는다.")
    print()


def section5(index, tops, filled, tests):
    """채운 표 14와 CSV 출력."""
    print("=" * 78)
    print("5. 채운 표 14 (제안 = merged, 원고가 인쇄한 18칸이 가리킨 체크포인트)")
    print("=" * 78)
    print("%-5s %-9s %14s %14s %12s %12s" % (
        "재배선", "방법", "지연(ms)", "손실(%)", "홉", "위험링크(%)"))
    print("-" * 78)
    out = []
    for ratio in sorted(tops):
        for label, method, key in (("OSPF", OSPF, OSPF),
                                   ("제안", MERGED, "P")):
            d_mean, d_sd, n, _ = filled[(ratio, "OSPF" if method == OSPF
                                         else "merged")]
            lo = cell(index, tops[ratio], method, "loss", True)
            hp = cell(index, tops[ratio], method, "hop", False)
            rk = cell(index, tops[ratio], method, "risky_link_usage", False)
            print("%-5s %-9s %7.1f ± %-4.1f %7.1f ± %-4.1f %5.1f ± %-4.1f"
                  " %5.1f ± %-4.1f" % (
                      "%d%%" % round(float(ratio) * 100), label,
                      d_mean, d_sd, lo[0], lo[1], hp[0], hp[1], rk[0], rk[1]))
            pub = PUB[(ratio, key)]
            out.append({
                "rewire_ratio": ratio,
                "row": label,
                "method": method,
                "n_topologies": n,
                "delay_ms": "%.4f" % d_mean,
                "delay_sd_across_topologies": "%.4f" % d_sd,
                "loss_pct": "%.4f" % lo[0],
                "loss_pct_sd": "%.4f" % lo[1],
                "hop": "%.4f" % hp[0],
                "hop_sd": "%.4f" % hp[1],
                "risky_link_pct": "%.4f" % rk[0],
                "risky_link_pct_sd": "%.4f" % rk[1],
                "pub_delay": "blank",
                "pub_loss_pct": pub["loss"][0],
                "pub_hop": pub["hop"][0],
                "pub_risky_link_pct": pub["risky"][0],
                "delay_claim": "not resolvable at n=10",
            })
        print("-" * 78)
    for ratio in sorted(tops):
        mean, p, p_holm, mde, need, _ = tests[("merged", ratio)]
        print("%s%%: 지연 차이 %+.1f ms, p_holm=%s, 쌍MDE %.1f ms -> %s" % (
            round(float(ratio) * 100), mean, RS.fmt_p(p_holm), mde,
            "유의" if p_holm < 0.05 else "주장 불가"))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        for row in out:
            writer.writerow(row)
    print()
    print("wrote %s (%d rows)" % (OUT, len(out)))


def main():
    _, index, tops = load()
    section1(index, tops)
    filled = section2(index, tops)
    tests = section3(index, tops)
    section4(index, tops)
    section5(index, tops, filled, tests)


if __name__ == "__main__":
    main()
