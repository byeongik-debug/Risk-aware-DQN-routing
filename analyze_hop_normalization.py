"""R3-4 지표가 홉 정규화에 얼마나 의존하는지 분해한다.

왜. `run_topology_generalization_dynamic.py`와 `network_env_generalization.py`의
지표 정의가 서로 어긋난다 — 같은 반환 딕셔너리 안에서:

    "total_delay": total_delay,          # 경로 합
    "total_loss":  total_loss / hops,    # 홉당 평균
    "total_risk":  total_risk / hops,    # 홉당 평균
    risky_link_usage = risky_count / hop_count * 100    # 홉당 비율

R3-4에서 Risk-DQN은 OSPF보다 홉이 +3.1 길다(7.73 vs 4.56, +69%). 홉당 평균은 긴
경로를 유리하게 만들므로, 정규화된 세 지표의 우위는 부풀고 정규화되지 않은 delay는
우위가 없다. 실제로 그 패턴이 그대로 나온다.

이 스크립트는 커밋된 CSV만으로 홉을 되돌려 **경로 총량**을 추정한다:

    loss_sum  ~= loss * hop
    risk_sum  ~= risk * hop
    risky_cnt ~= risky_link_usage / 100 * hop

주의(근사의 한계). CSV의 각 값은 토폴로지 내 에피소드 평균이므로
mean(x/h) * mean(h) != mean(x) 이다. 에피소드마다 홉이 흔들리면 오차가 생긴다.
그래서 이건 **방향과 규모를 보는 근사**이고, 손실의 정확한 경로 합은
`reeval_generalization_conventions.py`가 에피소드 단위로 다시 계산한다.
그 둘이 어긋나면 정확한 쪽을 따른다.

재현: python -I -X utf8 analyze_hop_normalization.py
"""

import argparse
import math
import os
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from analyze_topology_generalization import (  # noqa: E402
    FROZEN, IN_CSV, OSPF, finite, index_rows, load, one_sample_t,
)
from review_stats import fmt_p, holm_bonferroni, mean_std  # noqa: E402

# (표시명, 정규화된 원래 지표, 되돌리는 배율)
DERIVED = (
    ("loss_sum",  "loss",             1.0),
    ("risk_sum",  "risk",             1.0),
    ("risky_cnt", "risky_link_usage", 0.01),
)


def derived_value(row, metric, scale):
    value, hop = finite(row[metric]), finite(row["hop"])
    if value is None or hop is None:
        return None
    return value * scale * hop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=IN_CSV)
    args = parser.parse_args()

    index, by_ratio = index_rows(load(args.csv))

    print("=" * 100)
    print("홉 정규화를 되돌린 경로 총량 — Risk-DQN(동결) vs OSPF")
    print("=" * 100)
    print("  정규화된 지표(홉당 평균)와 되돌린 총량을 나란히 둔다.")
    print("  총량에서 우위가 사라지면 그 우위는 경로가 길어서 생긴 희석이다.\n")
    print("  {:<8}{:<12}{:>11}{:>11}{:>10}{:>11}{:>11}{:>10}".format(
        "재배선", "지표", "DQN", "OSPF", "개선%", "p", "p_holm", "n"))

    labelled = []
    for ratio in sorted(by_ratio):
        for name, metric, scale in DERIVED:
            diffs, xs, ys = [], [], []
            for topology in by_ratio[ratio]:
                a, b = index.get((FROZEN, topology)), index.get((OSPF, topology))
                if not (a and b):
                    continue
                x, y = derived_value(a, metric, scale), derived_value(b, metric, scale)
                if x is None or y is None:
                    continue
                diffs.append(x - y)
                xs.append(x)
                ys.append(y)
            if len(diffs) < 2:
                continue
            _, _, p = one_sample_t(diffs)
            mean_x, mean_y = mean_std(xs)[0], mean_std(ys)[0]
            gain = (mean_y - mean_x) / mean_y * 100.0 if mean_y else float("nan")
            labelled.append((ratio, name, mean_x, mean_y, gain, p, len(diffs)))

    adjusted = holm_bonferroni([item[5] for item in labelled])
    for (ratio, name, mean_x, mean_y, gain, p, n), p_holm in zip(labelled, adjusted):
        mark = " *" if p_holm < 0.05 else ""
        print("  {:<8}{:<12}{:>11.4f}{:>11.4f}{:>+10.1f}{:>11}{:>11}{:>10}{}".format(
            ratio, name, mean_x, mean_y, gain, fmt_p(p), fmt_p(p_holm), n, mark))

    print("\n  * = Holm 보정 후 p<0.05. 개선%는 양수가 DQN 우위.")

    # 정규화된 쪽과 비교해 희석분을 적는다.
    print("\n" + "=" * 100)
    print("희석 분해 — 정규화 지표의 개선%가 총량에서 얼마나 남는가")
    print("=" * 100)
    print("  {:<8}{:<20}{:>14}{:>14}{:>14}".format(
        "재배선", "지표", "홉당평균 개선%", "총량 개선%", "희석분"))
    pairs = {"loss": "loss_sum", "risk": "risk_sum", "risky_link_usage": "risky_cnt"}
    derived_gain = {(ratio, name): gain for ratio, name, _, _, gain, _, _ in labelled}
    for ratio in sorted(by_ratio):
        for metric, name in pairs.items():
            xs, ys = [], []
            for topology in by_ratio[ratio]:
                a, b = index.get((FROZEN, topology)), index.get((OSPF, topology))
                if not (a and b):
                    continue
                x, y = finite(a[metric]), finite(b[metric])
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
            if not xs:
                continue
            mean_x, mean_y = mean_std(xs)[0], mean_std(ys)[0]
            norm_gain = (mean_y - mean_x) / mean_y * 100.0 if mean_y else float("nan")
            total_gain = derived_gain.get((ratio, name), float("nan"))
            if math.isnan(total_gain):
                continue
            print("  {:<8}{:<20}{:>+14.1f}{:>+14.1f}{:>+14.1f}".format(
                ratio, metric, norm_gain, total_gain, norm_gain - total_gain))


if __name__ == "__main__":
    main()
