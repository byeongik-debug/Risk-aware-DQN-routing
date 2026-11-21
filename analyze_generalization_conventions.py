"""R3-4 손실·risk를 종단간 규약으로 다시 검정한다.

입력: review_results/generalization_conventions.csv
      (reeval_generalization_conventions.py --model both 산출, 90행)

왜. `network_env_generalization.py:398-399`가 손실·risk를 홉당 평균으로 낸다.
R3-4에서 Risk-DQN은 OSPF보다 홉이 +69% 길므로(7.73 vs 4.56) 그 평균이 긴 경로를
유리하게 만든다. `analyze_hop_normalization.py`가 커밋된 CSV에서 홉을 되돌려
근사치를 냈지만, mean(x/h)*mean(h) != mean(x)라서 근사였다. 이 스크립트는
에피소드 단위로 다시 계산한 값을 쓰므로 **정확하다**. 근사와 어긋나면 이쪽이 맞다.

규약 3종:

    loss_published   mean(p_i)            현행 보고값
    loss_path_sum    sum(p_i)             delay가 이미 보고되는 형태
    loss_compounded  1 - prod(1 - p_i)    경로 어딘가에서 버려질 확률
    risk_path_sum    sum(risk_i)          risk의 경로 합

설계는 analyze_topology_generalization.py와 같다 — 토폴로지가 짝짓기 단위(비율별
n=10), 대응표본 t + Holm. 성공률 0%인 토폴로지는 경로 지표가 nan이라 빠지고,
빠진 개수를 보고한다(생존 조건부).

재현: python -I -X utf8 analyze_generalization_conventions.py
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analyze_topology_generalization import (  # noqa: E402
    FROZEN, MERGED, OSPF, finite, index_rows, load, one_sample_t,
)
from review_stats import (  # noqa: E402
    fmt_p, holm_bonferroni, mean_std, min_detectable_effect,
)

IN_CSV = os.path.join(BASE_DIR, "review_results", "generalization_conventions.csv")
CONVENTIONS = ("loss_published", "loss_path_sum", "loss_compounded", "risk_path_sum")

# analyze_hop_normalization.py가 낸 근사 개선%. 정확한 값과 맞는지 대조한다.
APPROX_GAIN = {
    ("0.2", "loss_path_sum"): 54.2, ("0.4", "loss_path_sum"): 37.4,
    ("0.6", "loss_path_sum"): 50.8,
    ("0.2", "risk_path_sum"): 34.6, ("0.4", "risk_path_sum"): 21.9,
    ("0.6", "risk_path_sum"): 36.0,
}


def paired(index, topologies, metric, method_a, method_b):
    """(차이 목록, a값, b값, 제외 개수)."""
    diffs, xs, ys, dropped = [], [], [], 0
    for topology in topologies:
        a, b = index.get((method_a, topology)), index.get((method_b, topology))
        if not (a and b):
            continue
        x, y = finite(a[metric]), finite(b[metric])
        if x is None or y is None:
            dropped += 1
            continue
        diffs.append(x - y)
        xs.append(x)
        ys.append(y)
    return diffs, xs, ys, dropped


def report(index, by_ratio, model, title):
    print()
    print("=" * 104)
    print(title)
    print("=" * 104)
    print("  개선%는 양수가 DQN 우위. 모든 지표에서 낮은 값이 좋다.\n")
    print("  {:<8}{:<18}{:>10}{:>10}{:>10}{:>11}{:>11}{:>9}{:>5}{:>6}".format(
        "재배선", "규약", "DQN", "OSPF", "개선%", "p", "p_holm", "MDE", "n", "제외"))

    labelled = []
    for ratio in sorted(by_ratio):
        for metric in CONVENTIONS:
            diffs, xs, ys, dropped = paired(
                index, by_ratio[ratio], metric, model, OSPF)
            if len(diffs) < 2:
                continue
            _, _, p = one_sample_t(diffs)
            mean_x, mean_y = mean_std(xs)[0], mean_std(ys)[0]
            gain = (mean_y - mean_x) / mean_y * 100.0 if mean_y else float("nan")
            mde = min_detectable_effect(xs, ys)
            labelled.append((ratio, metric, mean_x, mean_y, gain, p, mde,
                             len(diffs), dropped))

    adjusted = holm_bonferroni([item[5] for item in labelled])
    significant = 0
    for (ratio, metric, mean_x, mean_y, gain, p, mde, n, dropped), p_holm in zip(
            labelled, adjusted):
        mark = ""
        if p_holm < 0.05:
            mark = " *"
            significant += 1
        print("  {:<8}{:<18}{:>10.4f}{:>10.4f}{:>+10.1f}{:>11}{:>11}{:>9.4f}"
              "{:>5}{:>6}{}".format(
                  ratio, metric, mean_x, mean_y, gain, fmt_p(p), fmt_p(p_holm),
                  mde, n, dropped, mark))
    print()
    print("  -> Holm 보정 후 유의: {}/{}건".format(significant, len(labelled)))
    return labelled, adjusted


def report_shrinkage(labelled):
    """현행 보고값에서 종단간으로 갈 때 개선%가 얼마나 줄는가."""
    gains = {(ratio, metric): gain
             for ratio, metric, _, _, gain, _, _, _, _ in labelled}
    ratios = sorted({ratio for ratio, _ in gains})

    print()
    print("=" * 104)
    print("희석 분해 — 현행(홉당 평균)에서 종단간으로 갈 때 남는 개선%")
    print("=" * 104)
    print("  {:<8}{:>14}{:>14}{:>16}{:>14}{:>14}".format(
        "재배선", "published", "path_sum", "compounded",
        "sum 희석분", "comp 희석분"))
    for ratio in ratios:
        pub = gains.get((ratio, "loss_published"))
        tot = gains.get((ratio, "loss_path_sum"))
        comp = gains.get((ratio, "loss_compounded"))
        if pub is None or tot is None or comp is None:
            continue
        print("  {:<8}{:>+14.1f}{:>+14.1f}{:>+16.1f}{:>+14.1f}{:>+14.1f}".format(
            ratio, pub, tot, comp, pub - tot, pub - comp))

    print()
    print("  근사 대조 — analyze_hop_normalization.py의 홉 되돌리기가 맞았는가")
    print("  {:<8}{:<18}{:>12}{:>12}{:>10}".format(
        "재배선", "규약", "근사 개선%", "정확 개선%", "오차%p"))
    for (ratio, metric), approx in sorted(APPROX_GAIN.items()):
        exact = gains.get((ratio, metric))
        if exact is None:
            continue
        print("  {:<8}{:<18}{:>+12.1f}{:>+12.1f}{:>+10.1f}".format(
            ratio, metric, approx, exact, approx - exact))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=IN_CSV)
    args = parser.parse_args()

    rows = load(args.csv)
    index, by_ratio = index_rows(rows)
    print("rows={} ratios={}".format(len(rows), sorted(by_ratio)))

    labelled, _ = report(
        index, by_ratio, FROZEN,
        "Risk-DQN(동결 모델) vs OSPF — 종단간 규약 (토폴로지가 표본 단위, n=10)")
    report_shrinkage(labelled)
    report(index, by_ratio, MERGED,
           "Risk-DQN(병합 사본) vs OSPF — 종단간 규약 (참고)")


if __name__ == "__main__":
    main()
