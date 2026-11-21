# -*- coding: utf-8 -*-
"""표 10 K 행(위험 윈도우) 품질 열을 끝낸다.

기존 review_results/window_sweep_significance.csv 는 두 군데가 비어 있었다.
  (1) w=10 행이 n=1 이라 t/df/p 가 nan 이다.
  (2) hop 열이 window_sweep_corrected.csv 에는 있는데 검정되지 않았다.

(1)은 "검정 불가"가 아니다. 새 관측 하나가 기존 표본의 분포에서 왔는지는
단일 관측 대 표본 t 로 검정한다:  t = (x - m) / (s * sqrt(1 + 1/n)), df = n-1.
n=1 에서 쓸 수 있는 유일한 정당한 검정이고, 이것이 비유의면 "런을 더 돌리면
차이가 나타날 것"이라는 기대의 근거가 없다는 뜻이다.

원본 폴더는 읽기만 한다. 출력은 review_results/ 로만 나간다.
"""
import csv
import glob
import math
import os

import review_stats as RS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, os.pardir, "SDN_DQN_new_env", "results"))
OUT = os.path.join(HERE, "review_results")

METRICS = ["delay", "loss", "risk", "hop"]
BASE_W = 5                      # 표 10 Full 행의 윈도우 (RISK_WINDOW 기본값)
HOURS_PER_RUN = 7.64            # 측정: 60 에피소드 206.2초 -> 8000 에피소드 458.3분
ZA, ZB = 1.959964, 0.841621     # alpha=.05 양측, power=.80


def load_runs():
    """{(scenario, window): {run_id: {metric: value}}}"""
    out = {}
    for path in glob.glob(os.path.join(SRC, "window*_run*_result.csv")):
        base = os.path.basename(path)
        head = base.split("_")
        window = int(head[0].replace("window", ""))
        scenario = head[1]
        run_id = int(head[2].replace("run", ""))
        with open(path, newline="", encoding="utf-8-sig") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("delay")]
        if not rows:
            continue
        rec = dict((m, float(rows[-1][m])) for m in METRICS)
        out.setdefault((scenario, window), {})[run_id] = rec
    return out


def check_reproduction(runs):
    """재계산한 평균이 window_sweep_corrected.csv 와 일치하는지 확인한다."""
    ref = os.path.join(OUT, "window_sweep_corrected.csv")
    if not os.path.exists(ref):
        return None
    worst, checked = 0.0, 0
    with open(ref, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            key = (row["scenario"], int(row["window"]))
            if key not in runs:
                raise AssertionError("기준 CSV의 %s 를 런별 파일로 못 만든다" % (key,))
            vals = runs[key]
            if int(row["n_runs"]) != len(vals):
                raise AssertionError("%s 런 수 불일치: 기준 %s vs 실제 %d"
                                     % (key, row["n_runs"], len(vals)))
            for m in METRICS:
                mean, _ = RS.mean_std([v[m] for v in vals.values()])
                worst = max(worst, abs(mean - float(row[m])))
                checked += 1
    assert worst < 1e-6, "재현 실패: 최대 오차 %.3e" % worst
    return worst, checked


def single_obs_t(x, sample):
    """새 관측 하나가 sample 의 분포에서 왔는가. 반환 (t, df, p, se)."""
    mean, sd = RS.mean_std(sample)
    n = len(sample)
    if n < 2 or sd == 0.0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    se = sd * math.sqrt(1.0 + 1.0 / n)
    t = (x - mean) / se
    return t, n - 1, RS.t_two_sided_p(t, n - 1), se


def pooled_sd(runs, scenario, metric):
    """윈도우 평균을 뺀 뒤의 런간 표준편차 (풀드)."""
    ss, df = 0.0, 0
    for (scen, _w), vals in runs.items():
        if scen != scenario or len(vals) < 2:
            continue
        series = [v[metric] for v in vals.values()]
        mean = sum(series) / len(series)
        ss += sum((v - mean) ** 2 for v in series)
        df += len(series) - 1
    return (math.sqrt(ss / df), df) if df else (float("nan"), 0)


def main():
    runs = load_runs()
    rep = check_reproduction(runs)
    if rep:
        print("기준 CSV 재현: %d칸, 최대오차 %.3e" % (rep[1], rep[0]))

    scenario = "dynamic"
    base = runs[(scenario, BASE_W)]
    base_n = len(base)
    windows = sorted(w for (s, w) in runs if s == scenario)

    test_rows = []
    raw_p = []
    for metric in METRICS:
        base_series = [v[metric] for v in base.values()]
        bmean, bsd = RS.mean_std(base_series)
        for w in windows:
            if w == BASE_W:
                continue
            series = [v[metric] for v in runs[(scenario, w)].values()]
            mean, sd = RS.mean_std(series)
            if len(series) >= 2:
                diff, t, df, p = RS.welch(series, base_series)
                mde = RS.min_detectable_effect(series, base_series)
                need = RS.required_n(series, base_series)
                kind = "welch"
            else:
                t, df, p, se = single_obs_t(series[0], base_series)
                diff = series[0] - bmean
                mde = RS.t_critical(df) * se
                psd, _ = pooled_sd(runs, scenario, metric)
                need = (math.ceil(2.0 * (ZA + ZB) ** 2 * psd * psd / (diff * diff))
                        if diff else float("nan"))
                kind = "single_obs_t"
            raw_p.append(p)
            test_rows.append({
                "scenario": scenario, "metric": metric, "window": w,
                "n": len(series), "mean": mean, "std": sd,
                "baseline_window": BASE_W, "baseline_mean": bmean,
                "baseline_std": bsd, "baseline_n": base_n,
                "diff_vs_w5": diff, "test": kind, "t": t, "df": df, "p": p,
                "mde": mde, "resolvable": "yes" if abs(diff) >= mde else "no",
                "required_n_per_arm": need,
            })

    adj = RS.holm_bonferroni(raw_p)
    for row, a in zip(test_rows, adj):
        row["p_holm"] = a
        row["significant_holm05"] = "yes" if a == a and a < 0.05 else "no"

    path = os.path.join(OUT, "window_sweep_tests_final.csv")
    cols = ["scenario", "metric", "window", "n", "mean", "std", "baseline_window",
            "baseline_mean", "baseline_std", "baseline_n", "diff_vs_w5", "test",
            "t", "df", "p", "p_holm", "significant_holm05", "mde", "resolvable",
            "required_n_per_arm"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for row in test_rows:
            wr.writerow(row)
    print("저장: %s (%d행)" % (path, len(test_rows)))

    # w=10 을 n=5 로 올리면 무엇이 가려지는가, 그리고 값이 얼마인가
    power_rows = []
    for metric in METRICS:
        psd, pdf = pooled_sd(runs, scenario, metric)
        x = list(runs[(scenario, 10)].values())[0][metric]
        for w in [w for w in windows if w != 10]:
            series = [v[metric] for v in runs[(scenario, w)].values()]
            m = sum(series) / len(series)
            d = abs(x - m)
            n2 = len(series)
            mde5 = (ZA + ZB) * psd * math.sqrt(1.0 / 5 + 1.0 / n2)
            need = math.ceil(2.0 * (ZA + ZB) ** 2 * psd * psd / (d * d)) if d else float("nan")
            hours_need = need * 2 * HOURS_PER_RUN if need == need else float("nan")
            power_rows.append({
                "metric": metric, "pooled_run_sd": psd, "pooled_df": pdf,
                "w10_value": x, "vs_window": w, "vs_n": n2, "vs_mean": m,
                "abs_diff": d, "diff_over_sd": d / psd,
                "mde_if_w10_had_n5": mde5,
                "resolved_at_n5": "yes" if d >= mde5 else "no",
                "required_n_per_arm": need,
                "extra_hours_to_reach_n5": 4 * HOURS_PER_RUN,
                "hours_for_required_n": hours_need,
            })
    path = os.path.join(OUT, "window10_power.csv")
    cols = list(power_rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for row in power_rows:
            wr.writerow(row)
    print("저장: %s (%d행)" % (path, len(power_rows)))

    print()
    print("=== w=10 (n=1) 대 w=5 (n=5), 단일 관측 대 표본 t ===")
    for row in test_rows:
        if row["window"] == 10:
            print("  %-6s w10=%11.5f  w5=%11.5f  차이 %+10.5f  t=%+6.3f df=%d  p=%s  Holm=%s"
                  % (row["metric"], row["mean"], row["baseline_mean"], row["diff_vs_w5"],
                     row["t"], int(row["df"]), RS.fmt_p(row["p"]), RS.fmt_p(row["p_holm"])))
    sig = [r for r in test_rows if r["significant_holm05"] == "yes"]
    print()
    print("Holm 보정 후 유의한 칸: %d / %d" % (len(sig), len(test_rows)))
    for row in sig:
        print("   %s w=%d  차이 %+.5f  p_holm=%.4f"
              % (row["metric"], row["window"], row["diff_vs_w5"], row["p_holm"]))


if __name__ == "__main__":
    main()
