"""재방문 배제 디코딩이 경로 품질을 망치지 않는지 확인한다.

왜. `diagnose_generalization_failure.py`가 재배선 60% 실패 100%가 2-주기 왕복임을
보였고, 미방문 이웃을 우선하게만 바꾸면 실패했던 네 토폴로지가 모두 100% 도달한다.
하지만 성공률만 올리고 손실·risk를 망치면 쓸 수 없다 — 미방문을 강제하면 risk 높은
링크로 밀릴 수 있다. 같은 토폴로지끼리 짝지어 비교한다.

입력:
    review_results/generalization_conventions.csv   원본 디코딩 (OSPF + 동결 + 병합)
    review_results/generalization_norevisit.csv     재방문 배제 (OSPF + 동결)

주의. 원본 디코딩에서 `r60_01`은 성공률 0%라 경로 지표가 nan이고 60% 집계에서
빠졌다(n=9). 재방문 배제에서는 n=10이다. 그래서 집계만 비교하면 **구성 변화를 품질
저하로 오독한다**. 이 스크립트는 두 가지를 다 낸다:

    (A) 각 디코딩의 자체 집계 — 논문에 쓸 수치
    (B) 양쪽 다 유한한 토폴로지만 쓴 짝지은 비교 — 품질 변화의 순수 측정

재현: python -I -X utf8 analyze_norevisit_decoding.py
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

ORIG_CSV = os.path.join(BASE_DIR, "review_results", "generalization_conventions.csv")
FIXED_CSV = os.path.join(BASE_DIR, "review_results", "generalization_norevisit.csv")
METRICS = ("success_rate", "hop", "delay", "loss_published", "loss_path_sum",
           "loss_compounded", "risk_path_sum")


def gain(index, topologies, metric, keep=None):
    """(DQN 평균, OSPF 평균, 개선%, n, 쓴 토폴로지 집합)."""
    xs, ys, used = [], [], set()
    for topology in topologies:
        if keep is not None and topology not in keep:
            continue
        a, b = index.get((FROZEN, topology)), index.get((OSPF, topology))
        if not (a and b):
            continue
        x, y = finite(a.get(metric)), finite(b.get(metric))
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        used.add(topology)
    if not xs:
        return None
    mean_x, mean_y = mean_std(xs)[0], mean_std(ys)[0]
    pct = (mean_y - mean_x) / mean_y * 100.0 if mean_y else float("nan")
    return mean_x, mean_y, pct, len(xs), used


def report_aggregate(label, index, by_ratio):
    print()
    print("=" * 96)
    print("(A) {} — 자체 집계 (DQN vs OSPF, 토폴로지가 표본 단위)".format(label))
    print("=" * 96)
    print("  {:<8}{:<18}{:>11}{:>11}{:>11}{:>6}".format(
        "재배선", "지표", "DQN", "OSPF", "개선%", "n"))
    for ratio in sorted(by_ratio):
        for metric in METRICS:
            got = gain(index, by_ratio[ratio], metric)
            if got is None:
                continue
            mean_x, mean_y, pct, n, _ = got
            # 성공률·홉은 "개선%"가 뜻이 다르므로 표시만 하고 부호를 뒤집지 않는다.
            print("  {:<8}{:<18}{:>11.4f}{:>11.4f}{:>+11.1f}{:>6}".format(
                ratio, metric, mean_x, mean_y, pct, n))
        print()


def report_paired(orig_index, fixed_index, by_ratio):
    """양쪽 다 유한한 토폴로지만 써서 디코딩 변경의 순효과를 잰다."""
    print()
    print("=" * 96)
    print("(B) 디코딩 변경의 순효과 — 같은 토폴로지끼리 짝지음")
    print("=" * 96)
    print("  양수 차이 = 재방문 배제 쪽 값이 더 크다. 손실·risk·홉은 작은 게 좋다.\n")
    print("  {:<8}{:<18}{:>11}{:>11}{:>10}{:>11}{:>11}{:>6}".format(
        "재배선", "지표", "원본", "배제", "차이", "p", "p_holm", "n"))

    labelled = []
    for ratio in sorted(by_ratio):
        for metric in METRICS:
            diffs, xs, ys = [], [], []
            for topology in by_ratio[ratio]:
                a = orig_index.get((FROZEN, topology))
                b = fixed_index.get((FROZEN, topology))
                if not (a and b):
                    continue
                x, y = finite(a.get(metric)), finite(b.get(metric))
                if x is None or y is None:
                    continue   # 원본에서 nan인 토폴로지는 짝이 없다
                xs.append(x)
                ys.append(y)
                diffs.append(y - x)
            if len(diffs) < 2:
                continue
            _, _, p = one_sample_t(diffs)
            labelled.append((ratio, metric, mean_std(xs)[0], mean_std(ys)[0],
                             sum(diffs) / len(diffs), p, len(diffs)))

    adjusted = holm_bonferroni([item[5] for item in labelled])
    last = None
    for (ratio, metric, mx, my, diff, p, n), p_holm in zip(labelled, adjusted):
        if last is not None and ratio != last:
            print()
        last = ratio
        print("  {:<8}{:<18}{:>11.4f}{:>11.4f}{:>+10.4f}{:>11}{:>11}{:>6}{}".format(
            ratio, metric, mx, my, diff, fmt_p(p), fmt_p(p_holm), n,
            " *" if p_holm < 0.05 else ""))

    print()
    print("  n이 10보다 작은 행은 원본 디코딩에서 성공률 0%로 경로 지표가 nan인")
    print("  토폴로지가 빠진 것이다 — 그 토폴로지는 품질 비교 자체가 불가능하다.")


def report_dropped(orig_index, fixed_index, by_ratio):
    """원본에서 빠졌던 토폴로지가 배제 디코딩에서 어떻게 되는가."""
    print()
    print("=" * 96)
    print("(C) 원본 디코딩이 실패했던 토폴로지")
    print("=" * 96)
    print("  {:<24}{:>11}{:>11}{:>11}{:>11}".format(
        "토폴로지", "원본 성공률", "배제 성공률", "배제 홉", "OSPF 홉"))
    for ratio in sorted(by_ratio):
        for topology in sorted(by_ratio[ratio]):
            a = orig_index.get((FROZEN, topology))
            b = fixed_index.get((FROZEN, topology))
            o = fixed_index.get((OSPF, topology))
            if not (a and b and o):
                continue
            before = finite(a["success_rate"])
            if before is None or before >= 100.0:
                continue
            print("  {:<24}{:>10.0f}%{:>10.0f}%{:>11.2f}{:>11.2f}".format(
                topology, before, finite(b["success_rate"]),
                finite(b["hop"]), finite(o["hop"])))


def report_delay(index, by_ratio, label):
    """delay는 원본에서 이미 경로 합이다 — 정규화 논쟁이 없는 유일한 지표.

    고친 디코딩은 경로를 늘리므로 delay가 나빠질 수 있다. 품질이 유지된다고
    말하려면 이걸 봐야 한다. MDE를 같이 적는다 — 유의하지 않은 걸 "차이 없음"으로
    읽으면 안 되고 "측정 불가"다.
    """
    print()
    print("=" * 96)
    print("(D) delay — {} (DQN vs OSPF, 고친 디코딩)".format(label))
    print("=" * 96)
    print("  {:<8}{:>11}{:>11}{:>11}{:>11}{:>11}{:>6}".format(
        "재배선", "DQN", "OSPF", "차이", "p", "MDE", "n"))
    rows = []
    for ratio in sorted(by_ratio):
        diffs, xs, ys = [], [], []
        for topology in by_ratio[ratio]:
            a, b = index.get((label, topology)), index.get((OSPF, topology))
            if not (a and b):
                continue
            x, y = finite(a.get("delay")), finite(b.get("delay"))
            if x is None or y is None:
                continue
            xs.append(x)
            ys.append(y)
            diffs.append(x - y)
        if len(diffs) < 2:
            continue
        _, _, p = one_sample_t(diffs)
        rows.append((ratio, mean_std(xs)[0], mean_std(ys)[0],
                     sum(diffs) / len(diffs), p,
                     min_detectable_effect(xs, ys), len(diffs)))
    adjusted = holm_bonferroni([r[4] for r in rows])
    for (ratio, mx, my, diff, p, mde, n), p_holm in zip(rows, adjusted):
        print("  {:<8}{:>11.2f}{:>11.2f}{:>+11.2f}{:>11}{:>11.2f}{:>6}{}".format(
            ratio, mx, my, diff, fmt_p(p), mde, n,
            " *" if p_holm < 0.05 else ""))
    print()
    print("  음수 차이 = DQN이 빠르다. 유의한 행이 없으면 §2.3의 \"측정 불가\" 결론이")
    print("  고친 디코딩에서도 유지된다는 뜻이다.")


def report_models(index, by_ratio):
    """고친 디코딩에서 동결 모델과 병합 사본이 구별되는가.

    왜 중요한가. 원본 디코딩에서 두 모델을 가른 유일한 증거는 **도달성**이었다 —
    동결은 네 토폴로지에서 실패하고 병합은 전부 100%였다. 그 실패가 디코딩
    artifact라면 그 증거가 사라지고, 두 모델은 다시 구별 불가가 된다. 그러면
    "어느 체크포인트를 집었는지가 R3-4 결론을 가른다"는 주장도 약해진다.
    """
    print()
    print("=" * 96)
    print("(E) 고친 디코딩에서 동결 모델 vs 병합 사본")
    print("=" * 96)
    print("  {:<8}{:<18}{:>11}{:>11}{:>10}{:>11}{:>11}{:>6}".format(
        "재배선", "지표", "동결", "병합", "차이", "p", "p_holm", "n"))
    labelled = []
    for ratio in sorted(by_ratio):
        for metric in METRICS:
            diffs, xs, ys = [], [], []
            for topology in by_ratio[ratio]:
                a, b = index.get((FROZEN, topology)), index.get((MERGED, topology))
                if not (a and b):
                    continue
                x, y = finite(a.get(metric)), finite(b.get(metric))
                if x is None or y is None:
                    continue
                xs.append(x)
                ys.append(y)
                diffs.append(y - x)
            if len(diffs) < 2:
                continue
            _, _, p = one_sample_t(diffs)
            labelled.append((ratio, metric, mean_std(xs)[0], mean_std(ys)[0],
                             sum(diffs) / len(diffs), p, len(diffs)))
    if not labelled:
        print("  병합 사본 행이 없다 — --model both 로 돌린 CSV가 필요하다.")
        return
    adjusted = holm_bonferroni([item[5] for item in labelled])
    significant = 0
    last = None
    for (ratio, metric, mx, my, diff, p, n), p_holm in zip(labelled, adjusted):
        if last is not None and ratio != last:
            print()
        last = ratio
        mark = ""
        if p_holm < 0.05:
            mark = " *"
            significant += 1
        print("  {:<8}{:<18}{:>11.4f}{:>11.4f}{:>+10.4f}{:>11}{:>11}{:>6}{}".format(
            ratio, metric, mx, my, diff, fmt_p(p), fmt_p(p_holm), n, mark))
    print()
    print("  -> Holm 보정 후 유의: {}/{}건".format(significant, len(labelled)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig", default=ORIG_CSV)
    parser.add_argument("--fixed", default=FIXED_CSV)
    args = parser.parse_args()

    orig_index, by_ratio = index_rows(load(args.orig))
    fixed_index, fixed_ratio = index_rows(load(args.fixed))
    print("원본 {} / 배제 {}".format(args.orig, args.fixed))
    print("재배선 비율 {} (배제 {})".format(sorted(by_ratio), sorted(fixed_ratio)))

    report_aggregate("원본 디코딩", orig_index, by_ratio)
    report_aggregate("재방문 배제", fixed_index, fixed_ratio)
    report_paired(orig_index, fixed_index, by_ratio)
    report_dropped(orig_index, fixed_index, by_ratio)
    report_delay(fixed_index, fixed_ratio, FROZEN)
    report_models(fixed_index, fixed_ratio)


if __name__ == "__main__":
    main()
