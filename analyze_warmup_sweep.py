"""reeval_warmup_sweep.py가 쓴 CSV를 집계해 윈도우 효과를 검정한다.

이 스윕을 만든 가설: "윈도우가 의미가 있다면 짧은 워밍업(시간 신호 살아 있음)에서는
정책을 구분하고, 긴 워밍업(포화로 신호 소멸)에서는 구분하지 못해야 한다."

reeval_warmup_sweep.py는 `seed=run_id`로 평가하므로 같은 run_id는 윈도우가 달라도
동일한 환경 궤적을 본다. 따라서 윈도우 비교는 **대응표본**이고 Welch(독립표본)보다
검정력이 훨씬 높다. 두 p값을 같이 출력해 결론이 검정 선택에 의존하지 않음을 보인다.

진단 4종:
  1. 워밍업별 판별비(stressed/safe jitter) — 신호가 언제 죽는가
  2. w=5 대비 윈도우별 유의차 — 신호가 살아 있을 때 성능이 갈리는가
  3. 동일 행동(delay/loss/hop 동일) 쌍에서 risk만 다른 사례 — 자(ruler) artifact 증명
  4. 윈도우 평균 delay를 홉 수에 회귀 — delay 차이가 경로 길이인가

재현: python -I analyze_warmup_sweep.py
원본 코드는 읽지도 수정하지도 않는다. 입력은 CSV 하나뿐이다.
"""

import argparse
import csv
import math
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from review_stats import fmt_p, holm_bonferroni, mean_std, t_two_sided_p, welch

IN_CSV = os.path.join(BASE_DIR, "review_results", "warmup_sweep_dynamic.csv")
BEHAVIOUR_KEYS = ("delay", "loss", "hop", "success_rate")
METRICS = ("delay", "loss", "risk", "hop")
REFERENCE_WINDOW = "5"


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def axes(rows):
    warmups = sorted({r["warmup"] for r in rows}, key=int)
    windows = sorted({r["window"] for r in rows}, key=int)
    runs = sorted({r["run_id"] for r in rows})
    return warmups, windows, runs


def cell_mean(rows, warmup, window, field):
    vals = [float(r[field]) for r in rows
            if r["warmup"] == warmup and r["window"] == window]
    return (mean_std(vals)[0] if vals else float("nan")), len(vals)


def paired_diffs(index, warmup, window, runs, metric):
    """같은 run_id(=같은 시드)끼리 짝지은 차이. 윈도우 간 환경이 동일하므로 유효하다."""
    diffs = []
    for run in runs:
        a = index.get((warmup, window, run))
        b = index.get((warmup, REFERENCE_WINDOW, run))
        if a and b:
            diffs.append(float(a[metric]) - float(b[metric]))
    return diffs


def one_sample_t(diffs):
    if len(diffs) < 2:
        return float("nan"), float("nan"), float("nan")
    mean, sd = mean_std(diffs)
    if sd == 0.0:
        # 모든 런에서 차이가 정확히 0이면 검정이 정의되지 않는다.
        return mean, 0.0, (1.0 if mean == 0.0 else 0.0)
    t = mean / (sd / math.sqrt(len(diffs)))
    return mean, sd, t_two_sided_p(t, len(diffs) - 1)


def report_signal_collapse(rows, warmups, windows):
    print("=" * 78)
    print("1. 판별비(stressed/safe jitter)가 워밍업과 함께 붕괴하는가")
    print("=" * 78)
    header = "  warmup" + "".join("{:>14}".format("w=" + w) for w in windows)
    print(header + "{:>12}{:>12}".format("delay(ms)", "loss"))
    for warmup in warmups:
        line = "  {:<6}".format(warmup)
        for window in windows:
            mean, n = cell_mean(rows, warmup, window, "discrimination_ratio")
            line += "{:>14}".format("{:.3f} [{}]".format(mean, n))
        pooled = [r for r in rows if r["warmup"] == warmup]
        line += "{:>12.2f}{:>12.4f}".format(
            mean_std([float(r["delay"]) for r in pooled])[0],
            mean_std([float(r["loss"]) for r in pooled])[0])
        print(line)
    print()
    print("  w=1은 deque(maxlen=1)의 std가 0이라 모든 워밍업에서 0.000 (구조적 결과).")
    print("  delay가 함께 오르면 포화 램프가 진행된 것이다.")


def report_window_tests(rows, index, warmups, windows, runs):
    print()
    print("=" * 78)
    print("2. 신호가 살아 있을 때 윈도우가 성능을 가르는가 (w=5 대비, Holm 보정)")
    print("=" * 78)
    significant = []
    for warmup in warmups:
        ratio, _ = cell_mean(rows, warmup, REFERENCE_WINDOW, "discrimination_ratio")
        print()
        print("  -- warmup={}  (w=5 판별비 {:.3f}x) --".format(warmup, ratio))
        for metric in METRICS:
            tests = []
            for window in windows:
                if window == REFERENCE_WINDOW:
                    continue
                diffs = paired_diffs(index, warmup, window, runs, metric)
                if len(diffs) < 2:
                    continue
                mean, _sd, p_paired = one_sample_t(diffs)
                a = [float(r[metric]) for r in rows
                     if r["warmup"] == warmup and r["window"] == window]
                b = [float(r[metric]) for r in rows
                     if r["warmup"] == warmup and r["window"] == REFERENCE_WINDOW]
                p_welch = welch(a, b)[3] if min(len(a), len(b)) >= 2 else float("nan")
                nonzero = sum(1 for d in diffs if abs(d) > 1e-12)
                tests.append((window, mean, p_paired, p_welch, len(diffs), nonzero))
            adjusted = holm_bonferroni([t[2] for t in tests])
            for test, p_holm in zip(tests, adjusted):
                window, mean, p_paired, p_welch, n, nonzero = test
                flag = " *" if p_holm < 0.05 else ""
                if p_holm < 0.05:
                    significant.append((warmup, metric, window, mean, p_holm))
                print("    {:<5} w={:<2}  diff={:+9.4f}  paired p={}  p_holm={}"
                      "  welch p={}  n={}  nonzero={}/{}{}".format(
                          metric, window, mean, fmt_p(p_paired), fmt_p(p_holm),
                          fmt_p(p_welch), n, nonzero, n, flag))
    print()
    print("  유의한 차이 {}건:".format(len(significant)))
    for warmup, metric, window, mean, p_holm in significant:
        print("    warmup={} {} w={} diff={:+.4f} p_holm={}".format(
            warmup, metric, window, mean, fmt_p(p_holm)))
    metrics_hit = {m for _, m, _, _, _ in significant}
    if metrics_hit and metrics_hit <= {"risk"}:
        print()
        print("  -> 유의차가 risk에만 나타난다. risk는 윈도우마다 자가 다르므로")
        print("     성능 비교 지표로 쓸 수 없다 (3번에서 증명).")


def report_same_route_different_risk(index, warmups, windows, runs):
    print()
    print("=" * 78)
    print("3. 같은 경로인데 risk 숫자만 다른가 (자 artifact의 직접 증거)")
    print("=" * 78)
    print("  동일 행동 = delay/loss/hop/success_rate가 문자열 단위로 전부 동일")
    print("  (같은 시드 -> 같은 환경이므로 이는 같은 경로 시퀀스를 뜻한다)")
    print()
    identical = differing = 0
    for warmup in warmups:
        for window in windows:
            if window == REFERENCE_WINDOW:
                continue
            for run in runs:
                a = index.get((warmup, window, run))
                b = index.get((warmup, REFERENCE_WINDOW, run))
                if not (a and b):
                    continue
                if any(a[k] != b[k] for k in BEHAVIOUR_KEYS):
                    continue
                identical += 1
                gap = float(a["risk"]) - float(b["risk"])
                if abs(gap) > 1e-12:
                    differing += 1
                    print("  warmup={:<3} w={:<2} run{}  delay={} hop={}  "
                          "risk {:.4f}(w5) vs {:.4f}  -> {:+.4f} ({:+.1f}%)".format(
                              warmup, window, run, a["delay"][:8], a["hop"][:4],
                              float(b["risk"]), float(a["risk"]), gap,
                              100.0 * gap / float(b["risk"])))
    print()
    print("  동일 행동 쌍 {}건 중 risk가 다른 것 {}건 ({:.0f}%)".format(
        identical, differing, 100.0 * differing / max(identical, 1)))
    if identical and differing == identical:
        print("  -> 라우팅이 완전히 동일한데 risk가 전부 다르다. 이 차이는 전적으로")
        print("     측정 자에서 나온 것이므로 risk로 윈도우를 비교할 수 없다.")


def report_delay_is_hops(rows, warmups, windows):
    print()
    print("=" * 78)
    print("4. 윈도우 간 delay 차이가 경로 길이인가 (윈도우 평균 delay ~ 홉 수)")
    print("=" * 78)
    print("  warmup   n   기울기(ms/홉)        r        r2")
    for warmup in warmups:
        xs, ys = [], []
        for window in windows:
            hop, n = cell_mean(rows, warmup, window, "hop")
            delay, _ = cell_mean(rows, warmup, window, "delay")
            if n:
                xs.append(hop)
                ys.append(delay)
        if len(xs) < 3:
            continue
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
        sxx = sum((a - mx) ** 2 for a in xs)
        syy = sum((b - my) ** 2 for b in ys)
        r = sxy / math.sqrt(sxx * syy) if sxx and syy else float("nan")
        print("  {:<7} {:<3} {:>14.2f} {:>8.4f} {:>9.4f}".format(
            warmup, n, sxy / sxx, r, r * r))
    print()
    print("  r2가 높으면 윈도우 사이의 delay 변동은 위험 회피가 아니라")
    print("  홉을 몇 개 쓰는지로 설명된다 (FINDINGS_loss_metric.md와 같은 축).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=IN_CSV)
    args = parser.parse_args()

    rows = load(args.csv)
    warmups, windows, runs = axes(rows)
    index = {(r["warmup"], r["window"], r["run_id"]): r for r in rows}
    print("입력: {}  ({}행, 워밍업 {} x window {} x 런 {})".format(
        args.csv, len(rows), len(warmups), len(windows), len(runs)))

    report_signal_collapse(rows, warmups, windows)
    report_window_tests(rows, index, warmups, windows, runs)
    report_same_route_different_risk(index, warmups, windows, runs)
    report_delay_is_hops(rows, warmups, windows)


if __name__ == "__main__":
    main()
