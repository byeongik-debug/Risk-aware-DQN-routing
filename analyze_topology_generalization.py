"""R3-4 일반화 결과 집계 — 토폴로지를 표본 단위로 본 대응표본 검정.

입력: review_results/topology_generalization_runs.csv (reeval_topology_generalization.py 산출)

설계. 미지 토폴로지 30개(재배선 20/40/60% x 10개)에서 OSPF와 Risk-DQN을 같은 토폴로지·
같은 시드로 평가했으므로 **토폴로지가 짝짓기 단위**다. 에피소드 전체를 풀어 검정하면
같은 토폴로지 안의 에피소드가 독립이 아니어서 n이 부풀고 p가 거짓으로 작아진다.
그래서 토폴로지당 평균을 하나의 관측으로 보고 대응표본 t를 쓴다 (비율별 n=10).

보고 3종:

1. `report_generalization` — 재배선 비율별로 Risk-DQN(동결) vs OSPF. 성능이 재배선
   정도에 따라 어떻게 무너지는지가 R3-4의 답이다.
2. `report_model_divergence` — 동결 모델 vs 병합 사본. 두 `.pth`는 구조가 같아 둘 다
   조용히 로드되지만 별개로 학습된 것이다(동일 원소 5.93%). 상대경로로 잘못된 모델을
   집었을 때 결론이 실제로 바뀌는지를 수치로 남긴다.
3. `report_power` — 유의하지 않은 비교에 대해 MDE를 함께 보고한다. "차이 없음"과
   "측정 불가"를 구별하지 않으면 과소주장이 된다.

재현: python -I -X utf8 analyze_topology_generalization.py
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from review_stats import (  # noqa: E402
    fmt_p,
    holm_bonferroni,
    mean_std,
    min_detectable_effect,
    t_two_sided_p,
)

IN_CSV = os.path.join(BASE_DIR, "review_results", "topology_generalization_runs.csv")
METRICS = ("success_rate", "delay", "loss", "risk", "hop", "risky_link_usage")
OSPF = "OSPF"
FROZEN = "Risk-DQN[frozen]"
MERGED = "Risk-DQN[merged]"


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def index_rows(rows):
    """{(method, topology): row} 와 비율별 토폴로지 목록."""
    index = {}
    by_ratio = defaultdict(list)
    for row in rows:
        index[(row["method"], row["topology"])] = row
        if row["method"] == OSPF:
            by_ratio[row["rewire_ratio"]].append(row["topology"])
    for ratio in by_ratio:
        by_ratio[ratio].sort()
    return index, by_ratio


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def paired_diffs(index, topologies, metric, method_a, method_b):
    """같은 토폴로지끼리 짝지은 method_a - method_b.

    성공률 0%인 토폴로지는 경로 지표가 nan이므로(평균할 경로가 없다) 그 쌍을 버린다.
    버린 개수를 함께 돌려줘서 표에 n을 적을 수 있게 한다 — nan을 조용히 평균에
    섞으면 비율 전체가 nan이 되고, 조용히 빼면 실패를 숨기게 된다.
    """
    diffs, dropped = [], 0
    for topology in topologies:
        a = index.get((method_a, topology))
        b = index.get((method_b, topology))
        if not (a and b):
            continue
        x, y = finite(a[metric]), finite(b[metric])
        if x is None or y is None:
            dropped += 1
            continue
        diffs.append(x - y)
    return diffs, dropped


def one_sample_t(diffs):
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan")
    mean, sd = mean_std(diffs)
    if sd == 0.0:
        # 모든 토폴로지에서 차이가 정확히 0이면 검정이 정의되지 않는다.
        return mean, 0.0, (1.0 if mean == 0.0 else 0.0)
    t = mean / (sd / math.sqrt(len(diffs)))
    return mean, sd, t_two_sided_p(t, len(diffs) - 1)


def cell_values(index, topologies, method, metric):
    """유한한 값만. nan(성공 0%인 토폴로지)은 제외하고 개수로 보고한다."""
    values = []
    for topology in topologies:
        row = index.get((method, topology))
        if not row:
            continue
        number = finite(row[metric])
        if number is not None:
            values.append(number)
    return values


def report_levels(index, by_ratio):
    print("=" * 96)
    print("1. 재배선 비율별 절대 성능 (토폴로지 10개 평균 +- 표준편차, 토폴로지가 표본 단위)")
    print("=" * 96)
    for metric in METRICS:
        print("\n  [{}]".format(metric))
        print("  {:<10}{:>22}{:>22}{:>22}".format(
            "재배선", "OSPF", "Risk-DQN(동결)", "Risk-DQN(병합)"))
        for ratio in sorted(by_ratio):
            cells = []
            for method in (OSPF, FROZEN, MERGED):
                values = cell_values(index, by_ratio[ratio], method, metric)
                if values:
                    mean, sd = mean_std(values)
                    suffix = "" if len(values) == len(by_ratio[ratio])                         else " n={}".format(len(values))
                    cells.append("{:>13.3f} +-{:6.3f}{}".format(mean, sd, suffix))
                else:
                    cells.append("{:>22}".format("-"))
            print("  {:<10}{}{}{}".format(ratio, *cells))


def report_comparison(index, by_ratio, method_a, method_b, title, note):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)
    print("  {}".format(note))
    print()
    print("  {:<10}{:<18}{:>12}{:>12}{:>12}{:>11}{:>11}{:>9}{:>5}{:>6}".format(
        "재배선", "지표", "차이", "A평균", "B평균", "p", "p_holm", "MDE", "n", "제외"))

    labelled = []
    for ratio in sorted(by_ratio):
        topologies = by_ratio[ratio]
        for metric in METRICS:
            diffs, dropped = paired_diffs(index, topologies, metric, method_a, method_b)
            if len(diffs) < 2:
                continue
            diff, _, p = one_sample_t(diffs)
            xs = cell_values(index, topologies, method_a, metric)
            ys = cell_values(index, topologies, method_b, metric)
            mde = min_detectable_effect(xs, ys) if len(xs) > 1 and len(ys) > 1 else float("nan")
            labelled.append((ratio, metric, diff, mean_std(xs)[0], mean_std(ys)[0], p, mde,
                             len(diffs), dropped))

    adjusted = holm_bonferroni([item[5] for item in labelled])
    significant = 0
    for (ratio, metric, diff, mean_a, mean_b, p, mde, n, dropped), p_holm in zip(
            labelled, adjusted):
        mark = ""
        if p_holm < 0.05:
            mark = " *"
            significant += 1
        print("  {:<10}{:<18}{:>+12.4f}{:>12.4f}{:>12.4f}{:>11}{:>11}{:>9.4f}"
              "{:>5}{:>6}{}".format(
                  ratio, metric, diff, mean_a, mean_b, fmt_p(p), fmt_p(p_holm), mde,
                  n, dropped, mark))
    print()
    print("  -> Holm 보정 후 유의한 비교: {}/{}건".format(significant, len(labelled)))
    return labelled, adjusted


def report_divergence_verdict(index, by_ratio, labelled, adjusted):
    """병합 함정이 결론을 바꿨는지 판정한다.

    주의. 상대 크기만으로 판정하면 안 된다. n=10이라 대부분의 지표가 검정력 부족이고
    (MDE가 관측 차이보다 크다), 유의하지 않은 차이를 "다르다"고 읽으면 그게 과잉 주장이다.
    그래서 근거를 두 층으로 나눈다:

    - 통계적 근거: Holm 보정 후 유의한 차이가 있는가.
    - 질적 근거: 한쪽은 경로를 찾고 다른 쪽은 못 찾는 토폴로지가 있는가. 성공/실패는
      평균 차이가 아니라 가용성 문제라서, 유의성과 별개로 보고 가치가 있다.
    """
    print()
    significant = [(item, p_holm) for item, p_holm in zip(labelled, adjusted) if p_holm < 0.05]

    worst = None
    for (ratio, metric, diff, _, mean_b, _, mde, _, _), p_holm in zip(labelled, adjusted):
        if mean_b == 0.0:
            continue
        relative = abs(diff) / abs(mean_b) * 100.0
        if worst is None or relative > worst[0]:
            worst = (relative, ratio, metric, diff, p_holm, mde)
    if worst is not None:
        relative, ratio, metric, diff, p_holm, mde = worst
        print("  최대 상대 차이: {} / {} 에서 {:+.4f} (병합본 대비 {:.2f}%, p_holm={}, MDE={:.4f})".format(
            ratio, metric, diff, relative, fmt_p(p_holm), mde))
        print("    -> 크기는 크지만 유의하지 않다. n=10에서 MDE가 관측 차이보다 크면")
        print("       '두 모델이 다르다'의 근거로 쓸 수 없다 — 측정 불가 구간이다.")

    # 질적 근거: 한쪽만 목적지에 도달하는 토폴로지.
    broken = []
    for ratio in sorted(by_ratio):
        for topology in by_ratio[ratio]:
            a = index.get((FROZEN, topology))
            b = index.get((MERGED, topology))
            if not (a and b):
                continue
            sa, sb = finite(a["success_rate"]), finite(b["success_rate"])
            if sa is None or sb is None or abs(sa - sb) < 1e-9:
                continue
            broken.append((ratio, topology, sa, sb))

    print()
    if broken:
        print("  두 모델의 성공률이 갈린 토폴로지 {}개:".format(len(broken)))
        for ratio, topology, sa, sb in broken:
            print("    {:<6} {:<12} 동결 {:6.2f}%  병합 {:6.2f}%".format(ratio, topology, sa, sb))
        print()
        print("  -> 판정: 평균 지표로는 두 모델을 구별할 수 없다({}/{}건 유의).".format(
            len(significant), len(labelled)))
        print("     그러나 위 토폴로지에서는 한 모델만 목적지에 도달한다. 이건 평균의")
        print("     문제가 아니라 가용성의 문제이므로, **어느 모델로 돌렸는지 모르는")
        print("     R3-4 수치는 재현 불가**로 봐야 한다. 폐기할 근거까지는 아니고,")
        print("     모델 해시를 명시해 다시 돌리면 된다.")
    elif significant:
        print("  -> 판정: Holm 보정 후 {}건이 유의하다. 잘못된 모델을 집으면 R3-4 수치가".format(
            len(significant)))
        print("     달라진다. 모델을 명시하지 않은 실행은 다시 돌려야 한다.")
    else:
        print("  -> 판정: 유의한 차이도, 성공/실패가 갈리는 토폴로지도 없다. 상대경로")
        print("     함정이 R3-4 결론을 바꾸지는 않았다. 그래도 md5는 CSV에 남긴다.")

    # 경로 지표가 성공 조건부라는 경고.
    conditioned = [(ratio, metric, n, dropped)
                   for (ratio, metric, _, _, _, _, _, n, dropped) in labelled if dropped]
    if conditioned:
        print()
        print("  [주의] 아래 비교는 양쪽이 모두 성공한 토폴로지만 쓴다(생존 조건부).")
        print("         실패한 토폴로지가 빠지므로 경로 품질이 실제보다 좋게 보인다.")
        seen = set()
        for ratio, metric, n, dropped in conditioned:
            if ratio in seen:
                continue
            seen.add(ratio)
            print("         재배선 {}: n={} (토폴로지 {}개 제외)".format(ratio, n, dropped))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=IN_CSV)
    args = parser.parse_args()

    rows = load(args.csv)
    index, by_ratio = index_rows(rows)
    print("rows={} topologies={} ratios={}\n".format(
        len(rows), len(by_ratio[sorted(by_ratio)[0]]) * len(by_ratio), sorted(by_ratio)))

    report_levels(index, by_ratio)
    report_comparison(
        index, by_ratio, FROZEN, OSPF,
        "2. Risk-DQN(동결 모델) vs OSPF — 미지 토폴로지 일반화 (A=DQN, B=OSPF)",
        "차이는 DQN - OSPF다. delay/loss/risk는 음수가 좋고 success_rate는 양수가 좋다.")
    labelled, adjusted = report_comparison(
        index, by_ratio, FROZEN, MERGED,
        "3. 동결 모델 vs 병합 사본 — 상대경로 함정이 결론을 바꿨는가 (A=동결, B=병합)",
        "구조가 같아 둘 다 로드되지만 별개로 학습된 모델이다(동일 원소 5.93%).")
    report_divergence_verdict(index, by_ratio, labelled, adjusted)


if __name__ == "__main__":
    main()
