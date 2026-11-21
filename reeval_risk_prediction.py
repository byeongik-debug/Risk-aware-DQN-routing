"""R3-2를 성립하는 형태로 다시 묻는다 — 위험을 **예측**할 수 있는 환경인가.

발표된 R3-2 실험(`results/risk_future_detection_accuracy_summary.csv`)은 다섯 지표가
소수 16자리까지 동일하다. `analyze_risk_identification.py`가 원인을 둘로 갈라놨다.

  (1) 프로토콜 결함 — 정답 집합이 항상 50링크 중 15~16개인데 예측이 top k=10이라
      모든 지표의 top10이 100% 정답 집합 안에 들어간다. 정밀도가 1.0에 고정되고
      tp=k, fp=0이 되어 혼동행렬이 **무엇을 골랐는지와 무관**해진다.
  (2) 환경 결함 — dynamic에서 혼잡 플래그가 reset 후 한 번도 바뀌지 않는다.

그래서 "`unstable`에서 다시 돌리면 된다"로는 부족하다. (2)만 고치고 (1)을 남기면
또 동률이 난다. 이 스크립트는 둘을 같이 고친다: **k 없는 순위 지표(AUC)**를 쓰고,
혼잡 전이가 실제로 일어나는 `unstable`을 포함해 네 시나리오를 전부 잰다.

그리고 코드를 읽다가 더 상위의 사실이 나왔다. `update_dynamic_qos`의 루프 안에서

    traffic_ratio = random.uniform(...)            # 매 스텝 새로 뽑는다
    q["traffic"]      = clip(traffic_ratio * capacity, ...)   # 누적이 아니라 대입
    q["utilization"]  = clip(q["traffic"] / capacity, ...)
    q["is_congested"] = q["utilization"] >= 0.8

이므로 **이용률은 스텝 간 독립**이고, 혼잡은 링크의 (심은 집합 소속, 타입)만으로
결정되는 독립 베르누이다. 그러면 과거로부터 미래 혼잡을 맞추는 일은 원리적으로
"그 링크가 어느 범주인가"를 맞추는 일로 환원된다. 반면 `delay`·`loss`는 누산되므로
`delay_t`는 과거 이용률의 **합**에 비례한다 — 즉 환경이 윈도우 평균을 공짜로 계산해
상태에 넣어주고 있다는 뜻이 된다. 사실이면 윈도우 스윕이 전부 무효였던 단일 원인이다.

재기 전에 예측을 박아둔다. 틀리면 틀린 대로 적는다.

  P1 util의 lag-1 자기상관이 네 시나리오 전부 |r| < 0.05다 (스텝 간 독립).
  P2 corr(delay_t, 누적평균 util) > corr(delay_t, util_t)이고 격차가 t와 함께 커진다
     (누산기가 이미 윈도우 평균이다 → 명시적 윈도우는 중복이다).
  P3 제안 risk의 AUC가 순간 util의 AUC를 어느 시나리오에서도 0.02 넘게 못 이긴다.
  P4 모든 예측자가 범주 오라클 천장 이하이고, unstable에서는 그 천장 자체가 1.0보다
     훨씬 낮다 (어떤 방법도 넘을 수 없는 내재 한계가 있다).
  P5 **그런데 unstable에서는 윈도우 평균이 순간값을 이긴다.** 심은 backup 링크
     U(0.4125,0.825)와 비심 링크 U(0.15,0.45)가 [0.4125,0.45]에서 겹치므로 한 표본으로는
     범주가 안 갈리고 평균이 유리해야 한다. 이게 맞으면 논문은 "라우팅"이 아니라
     **"식별"** 쪽에서 시간 항 주장을 세울 수 있다.
  P6 AUC(Δ=1) ≈ AUC(Δ=10)이다. 진짜 시간 과정이면 가까운 미래가 더 잘 맞아야 하므로,
     같다는 것은 i.i.d.의 지문이다.

원본(`SDN_DQN_new_env/`, `Risk-aware-DQN-routing/`)은 건드리지 않는다. 모델도 쓰지 않는다.
출력은 `review_results/risk_prediction.csv`.
"""

import csv
import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_risk_oracle_leak as ROL  # noqa: E402
from network_env import RISK_WINDOW, NetworkEnv  # noqa: E402
from topology import EDGES  # noqa: E402

SCENARIOS = ("normal", "unstable", "congestion", "dynamic")
SEEDS = 20
T_MAX = 40
T0 = 10            # 논문의 평가 워밍업 지점
DELTAS = (1, 5, 10)
WINDOWS = (3, 5, 10)
CORR_TIMES = (5, 10, 20, 40)

OUT_DIR = os.path.join(BASE_DIR, "review_results")
OUT_CSV = os.path.join(OUT_DIR, "risk_prediction.csv")


# --------------------------------------------------------------------------- 통계
def pearson(xs, ys):
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x.size < 3 or x.std() == 0.0 or y.std() == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def auc(pos, neg):
    """P(양성 > 음성) + 0.5 x 동률. 완전 분리면 1.0, 무정보면 0.5."""
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


# --------------------------------------------------------------------------- 수집
def trace(scenario, seed):
    """한 에피소드를 T_MAX 스텝 굴리며 링크별 시계열을 모은다. 경로는 밟지 않는다."""
    np.random.seed(seed)
    random.seed(seed)
    env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RISK_WINDOW)
    env.reset()
    danger, _support = ROL.scenario_sets(env)
    edges = [env.normalize_edge(u, v) for u, v in EDGES]

    rec = {k: np.zeros((T_MAX + 1, len(edges))) for k in
           ("util", "delay", "loss", "risk", "cong")}
    for t in range(T_MAX + 1):
        if t > 0:
            env.update_dynamic_qos()
        for i, e in enumerate(edges):
            q = env.qos[e]
            rec["util"][t, i] = q["utilization"]
            rec["delay"][t, i] = q["delay"]
            rec["loss"][t, i] = q["loss"]
            rec["risk"][t, i] = q["risk"]
            rec["cong"][t, i] = 1.0 if q["is_congested"] else 0.0
    # 범주 = (심은 집합 소속, 링크 타입). 미래 혼잡의 유일한 지속 정보원이다.
    rec["category"] = [("danger" if e in danger else "plain") + "/" + env.qos[e]["type"]
                       for e in edges]
    return rec


def collect():
    return {s: [trace(s, seed) for seed in range(SEEDS)] for s in SCENARIOS}


# ------------------------------------------------------------------- P1 자기상관
def autocorr_rows(traces):
    rows = []
    for scenario in SCENARIOS:
        for lag in (1, 5):
            per_link = []
            for rec in traces[scenario]:
                u = rec["util"]
                for i in range(u.shape[1]):
                    series = u[1:, i]          # t=0은 init_qos라 분포가 다르다
                    r = pearson(series[:-lag], series[lag:])
                    if not np.isnan(r):
                        per_link.append(r)
            arr = np.asarray(per_link)
            rows.append({
                "block": "autocorr", "scenario": scenario, "key": "lag%d" % lag,
                "mean": float(arr.mean()), "max_abs": float(np.abs(arr).max()),
                "n": int(arr.size),
            })
    return rows


# ------------------------------------------------- P2 누산기가 윈도우 평균인가
def accumulator_rows(traces):
    rows = []
    for scenario in SCENARIOS:
        for t in CORR_TIMES:
            inst, runmean = [], []
            for rec in traces[scenario]:
                u, d = rec["util"], rec["delay"]
                inst.append(pearson(d[t], u[t]))
                runmean.append(pearson(d[t], u[1:t + 1].mean(axis=0)))
            rows.append({
                "block": "accumulator", "scenario": scenario, "key": "t%d" % t,
                "r_instant": float(np.nanmean(inst)),
                "r_runmean": float(np.nanmean(runmean)),
                "gap": float(np.nanmean(runmean) - np.nanmean(inst)),
            })
    return rows


# ------------------------------------------- P3~P6 미래 혼잡 예측 (k 없는 AUC)
def scores_at(rec, t0):
    u = rec["util"]
    out = {
        "util_now": u[t0].copy(),
        "delay_now": rec["delay"][t0].copy(),
        "loss_now": rec["loss"][t0].copy(),
        "risk_now": rec["risk"][t0].copy(),
        "cong_flag_now": rec["cong"][t0].copy(),
    }
    for k in WINDOWS:
        lo = max(1, t0 - k + 1)
        out["util_win%d" % k] = u[lo:t0 + 1].mean(axis=0)
    # 낡은 창: 최근 5스텝(t0-4..t0)과 겹치지 않는 그 앞의 5스텝.
    # 윈도우 이득이 시간 구조에서 오면 최근 창이 이겨야 하고,
    # 정적 범주를 추정할 표본이 늘어서 오면 둘이 같아야 한다.
    lo = max(1, t0 - 9)
    out["util_stale5"] = u[lo:max(lo + 1, t0 - 4)].mean(axis=0)
    return out


def category_oracle(traces, scenario):
    """범주별 혼잡 빈도. util이 스텝 간 독립이면 이것이 베이즈 최적이다."""
    hit, tot = {}, {}
    for rec in traces[scenario]:
        for i, cat in enumerate(rec["category"]):
            hit[cat] = hit.get(cat, 0.0) + rec["cong"][1:, i].sum()
            tot[cat] = tot.get(cat, 0.0) + (T_MAX)
    return {c: hit[c] / tot[c] for c in hit}


def prediction_rows(traces):
    rows = []
    for scenario in SCENARIOS:
        oracle = category_oracle(traces, scenario)
        for delta in DELTAS:
            bank = {}
            label, oracle_score, open_mask = [], [], []
            for rec in traces[scenario]:
                sc = scores_at(rec, T0)
                for name, vals in sc.items():
                    bank.setdefault(name, []).extend(vals.tolist())
                label.extend(rec["cong"][T0 + delta].tolist())
                oracle_score.extend([oracle[c] for c in rec["category"]])
                open_mask.extend((rec["cong"][T0] == 0.0).tolist())
            bank["category_oracle"] = oracle_score
            label = np.asarray(label)
            open_mask = np.asarray(open_mask, dtype=bool)

            for scope, keep in (("all", np.ones_like(open_mask)),
                                ("open", open_mask)):
                y = label[keep]
                pos_n, neg_n = int((y == 1).sum()), int((y == 0).sum())
                for name, vals in bank.items():
                    v = np.asarray(vals, dtype=float)[keep]
                    rows.append({
                        "block": "prediction", "scenario": scenario,
                        "delta": delta, "scope": scope, "score": name,
                        "auc": auc(v[y == 1], v[y == 0]),
                        "n_pos": pos_n, "n_neg": neg_n,
                    })
    return rows


# --------------------------------------------------------------------------- 출력
def show_autocorr(rows):
    print("=" * 78)
    print("P1. 이용률이 스텝 간 독립인가 — 링크별 자기상관")
    print("=" * 78)
    print("  {:<12}{:>8}{:>12}{:>12}".format("시나리오", "lag", "평균 r", "최대 |r|"))
    for r in rows:
        print("  {:<12}{:>8}{:>12.4f}{:>12.4f}".format(
            r["scenario"], r["key"], r["mean"], r["max_abs"]))
    worst = max(abs(r["mean"]) for r in rows)
    print()
    print("  판정: {} (평균 |r| 최대 {:.4f})".format(
        "지지 — 스텝 간 독립이다" if worst < 0.05 else "반증 — 시간 구조가 있다", worst))
    return worst


def show_accumulator(rows):
    print()
    print("=" * 78)
    print("P2. delay 누산기가 이미 윈도우 평균인가")
    print("=" * 78)
    print("  링크 간 상관: delay_t 대 순간 util / 누적평균 util")
    print("  {:<12}{:>6}{:>12}{:>12}{:>10}".format(
        "시나리오", "t", "순간", "누적평균", "격차"))
    for r in rows:
        print("  {:<12}{:>6}{:>12.4f}{:>12.4f}{:>+10.4f}".format(
            r["scenario"], r["key"], r["r_instant"], r["r_runmean"], r["gap"]))
    sub = [r for r in rows if r["scenario"] != "normal"]
    ok = all(r["gap"] > 0 for r in sub)
    print()
    print("  판정: {}".format(
        "지지 — 누적평균이 더 설명한다. 명시적 윈도우는 중복이다" if ok
        else "반증 — 순간값이 더 설명하는 칸이 있다"))
    return ok


def show_prediction(rows):
    print()
    print("=" * 78)
    print("P3~P6. 미래 혼잡 예측 — k 없는 AUC (0.5 = 무정보)")
    print("=" * 78)
    names = ["util_now", "util_win3", "util_win5", "util_win10",
             "delay_now", "loss_now", "risk_now", "cong_flag_now",
             "category_oracle"]
    for scope in ("all", "open"):
        print()
        print("  [{}]  {}".format(
            scope, "전체 링크" if scope == "all"
            else "t=10에 아직 혼잡하지 않은 링크만 — 진짜 예측 과제"))
        head = "  {:<12}{:>6}".format("시나리오", "Δ")
        for n in names:
            head += "{:>11}".format(n[:10])
        print(head)
        for scenario in SCENARIOS:
            for delta in DELTAS:
                pick = {r["score"]: r for r in rows
                        if r["scenario"] == scenario and r["delta"] == delta
                        and r["scope"] == scope}
                if not pick:
                    continue
                any_row = next(iter(pick.values()))
                line = "  {:<12}{:>6}".format(scenario, delta)
                for n in names:
                    line += "{:>11}".format("{:.3f}".format(pick[n]["auc"])
                                             if n in pick and not np.isnan(pick[n]["auc"])
                                             else "n/a")
                print(line + "   (+{}/-{})".format(any_row["n_pos"], any_row["n_neg"]))

    def grab(scenario, delta, scope, name):
        for r in rows:
            if (r["scenario"] == scenario and r["delta"] == delta
                    and r["scope"] == scope and r["score"] == name):
                return r["auc"]
        return float("nan")

    print()
    print("  P3 — 제안 risk가 순간 util을 이기는가 (scope=open, Δ=10)")
    worst = -9.0
    for scenario in SCENARIOS:
        a = grab(scenario, 10, "open", "risk_now")
        b = grab(scenario, 10, "open", "util_now")
        if np.isnan(a) or np.isnan(b):
            print("    {:<12} n/a".format(scenario))
            continue
        worst = max(worst, a - b)
        print("    {:<12} risk {:.3f}  util {:.3f}  차 {:+.3f}".format(
            scenario, a, b, a - b))
    print("    판정: {} (최대 차 {:+.3f})".format(
        "지지 — risk가 순간 관측을 못 넘는다" if worst <= 0.02
        else "반증 — risk가 순간 관측보다 낫다", worst))

    print()
    print("  P5 — unstable에서 윈도우 평균이 순간값을 이기는가 (scope=open)")
    for delta in DELTAS:
        base = grab("unstable", delta, "open", "util_now")
        best_k, best_v = None, -9.0
        for k in WINDOWS:
            v = grab("unstable", delta, "open", "util_win%d" % k)
            if not np.isnan(v) and v > best_v:
                best_k, best_v = k, v
        print("    Δ={:<3} 순간 {:.3f}  최고 윈도우 k={} {:.3f}  차 {:+.3f}".format(
            delta, base, best_k, best_v, best_v - base))
    gain = max(grab("unstable", d, "open", "util_win%d" % k) - grab("unstable", d, "open", "util_now")
               for d in DELTAS for k in WINDOWS)
    print("    판정: {} (최대 이득 {:+.3f})".format(
        "지지 — 식별에서는 이력이 보탬이 된다" if gain > 0.02
        else "반증 — 이력이 보태는 게 없다", gain))

    print()
    print("  P7 — 윈도우 이득이 시간 구조인가 표본 수인가 (scope=open)")
    print("       최근 5스텝 평균 대 겹치지 않는 낡은 5스텝 평균")
    gaps = []
    for scenario in SCENARIOS:
        for delta in DELTAS:
            fresh = grab(scenario, delta, "open", "util_win5")
            stale = grab(scenario, delta, "open", "util_stale5")
            if np.isnan(fresh) or np.isnan(stale):
                continue
            gaps.append(fresh - stale)
            print("    {:<12} Δ={:<3} 최근 {:.3f}  낡은 {:.3f}  차 {:+.3f}".format(
                scenario, delta, fresh, stale, fresh - stale))
    if gaps:
        mx = max(abs(g) for g in gaps)
        print("    판정: {} (최대 |차| {:.3f})".format(
            "정적 범주 추정이다 — 최근성이 무관하다" if mx <= 0.02
            else "시간 구조가 있다 — 최근 관측이 더 유용하다", mx))

    print()
    print("  P6 — Δ가 커져도 AUC가 같은가 (i.i.d.의 지문, scope=open, util_now)")
    for scenario in SCENARIOS:
        vals = [grab(scenario, d, "open", "util_now") for d in DELTAS]
        txt = "  ".join("Δ{}={:.3f}".format(d, v) for d, v in zip(DELTAS, vals))
        spread = np.nanmax(vals) - np.nanmin(vals)
        print("    {:<12}{}   폭 {:.3f}".format(scenario, txt, spread))
    return worst


def main():
    print("수집 — {}시나리오 x {}시드 x {}스텝 (모델 불필요)".format(
        len(SCENARIOS), SEEDS, T_MAX))
    traces = collect()

    rows_a = autocorr_rows(traces)
    rows_b = accumulator_rows(traces)
    rows_c = prediction_rows(traces)

    show_autocorr(rows_a)
    show_accumulator(rows_b)
    show_prediction(rows_c)

    print()
    print("  범주별 혼잡 빈도 (오라클의 내용)")
    for scenario in SCENARIOS:
        oracle = category_oracle(traces, scenario)
        parts = ["{} {:.3f}".format(c, p) for c, p in sorted(oracle.items())]
        print("    {:<12}{}".format(scenario, "  ".join(parts)))

    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    keys = sorted({k for r in rows_a + rows_b + rows_c for k in r})
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_a + rows_b + rows_c:
            w.writerow(r)
    print()
    print("저장: {} ({}행)".format(OUT_CSV, len(rows_a) + len(rows_b) + len(rows_c)))


if __name__ == "__main__":
    main()
