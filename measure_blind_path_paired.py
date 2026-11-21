# -*- coding: utf-8 -*-
"""`measure_blind_path_optimality.py` 의 순위가 흔들려서 다시 짰다.

첫 측정은 학습 혼합 기대값 표를 몬테카를로로 쌓고 후보 246개를 **절대 기대보상**
으로 줄 세웠다. 표 네 개(에피소드 100/400 x 시드 기점 둘)를 쌓아 보니 관측 경로
`0-4-8-14-15-21-22-24` 의 순위가 **3위 / 5위 / 6위 / 3위**로 흔들렸고 7홉 안에서도
1위와 2위를 오갔다. 순위 자체가 추정량이라 그렇다. 그래서 두 가지를 고친다.

  (1) **쌍체로 바꾼다.** 후보들은 **같은 에피소드**에서 점수가 난다(환경 궤적이
      행동과 무관하다는 것은 앞 스크립트 V2 에서 0.0 으로 확인했다). 그러면
      경로끼리의 차는 짝지어진 양이고, 공통 난수가 분산을 상쇄한다. 절대값의
      표본오차가 아니라 **차의 표본오차**로 판정한다.
  (2) **할인을 넣는다.** `train_risk_blind.py:60` 의 `GAMMA = 0.99` 다. 에이전트가
      최대화하는 것은 할인 보상이고, 할인은 긴 경로의 도착 보너스를 깎으므로
      8홉 우위가 뒤집힐 수 있다. gamma 1.00 과 0.99 를 모두 센다.

보상이 간선 가산이므로 에피소드 하나에서 후보 전부의 보상을 **근사 없이** 계산할
수 있다(앞 스크립트와 같은 논리, 여기서는 평균을 내기 전에 에피소드별로 남긴다):

    R_gamma(P) = sum_k gamma^k * r_k(e_k, k)  +  gamma^(L-1) * (45 - 6*mean_risk_L)

V3 로 앞 스크립트와 맞물리는지 확인한다: gamma=1.00 에서 에피소드별 보상의 평균이
기대값 표의 값과 같아야 한다(같은 양을 다른 순서로 더한 것이므로 부동소수 오차만
남아야 한다).

원본은 읽기만 한다. 산출은 `review_results/` 로.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_blind_path_paired.py
"""
from __future__ import print_function

import argparse
import csv
import io
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import review_stats as RS                                   # noqa: E402
from measure_blind_path_optimality import (                  # noqa: E402
    EVAL_SEED, EVAL_WARMUP, OBSERVED, RESULT_DIR, RISK_WINDOW,
    TRAIN_SCENARIOS, TRAIN_WEIGHT, Expect, build_adj, enumerate_simple_paths,
    hop_dist_to_dest, norm, snapshots, step_cost)

GAMMAS = (1.00, 0.99)
PAIR_CSV = os.path.join(RESULT_DIR, "blind_path_paired.csv")
SUM_CSV = os.path.join(RESULT_DIR, "blind_path_paired_summary.csv")


def episode_return(path, snaps, scenario, dist, gamma):
    """에피소드 하나에서 경로 하나의 보상 (정확값, 근사 없음)."""
    L = len(path) - 1
    tot = 0.0
    for k in range(L):
        u, v = path[k], path[k + 1]
        risk, _d, _l, cong = snaps[k][norm(u, v)]
        prog = dist.get(v, 99) < dist.get(u, 99)
        tot += (gamma ** k) * step_cost(scenario, risk, cong, prog)
    fin = snaps[L - 1]
    mr = sum(fin[norm(path[k], path[k + 1])][0] for k in range(L)) / L
    tot += (gamma ** (L - 1)) * (45.0 - 6.0 * mr)
    return tot


def one_sample_p(diffs):
    """차 평균이 0 인가 (양측). n 이 크므로 정규 근사, z 와 함께 돌려준다."""
    n = len(diffs)
    m = sum(diffs) / n
    if n < 2:
        return m, 0.0, float("nan"), 1.0
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0.0:
        return m, 0.0, float("inf") if m else 0.0, 0.0 if m else 1.0
    z = m / se
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return m, se, z, p


def collect(cands, dist, max_hops, episodes_spec):
    """에피소드별 x 후보별 보상 행렬 (감가율마다)."""
    out = dict((g, dict((p, []) for p in cands)) for g in GAMMAS)
    for scenario, seed, warmup in episodes_spec:
        snaps = snapshots(scenario, seed, warmup, max_hops, RISK_WINDOW)
        for g in GAMMAS:
            col = out[g]
            for p in cands:
                col[p].append(episode_return(p, snaps, scenario, dist, g))
    return out


def report(tag, mat, cands, fh_rows, label):
    print("\n" + "-" * 74)
    print("[%s] %s" % (tag, label))
    for g in GAMMAS:
        col = mat[g]
        n = len(col[OBSERVED])
        obs = col[OBSERVED]
        means = dict((p, sum(v) / len(v)) for p, v in col.items())
        order = sorted(cands, key=lambda p: -means[p])
        rank = order.index(OBSERVED) + 1
        same_len = [p for p in order if len(p) == len(OBSERVED)]
        rank_len = same_len.index(OBSERVED) + 1

        stats = []
        for p in cands:
            if p == OBSERVED:
                continue
            d = [a - b for a, b in zip(obs, col[p])]
            m, se, z, pv = one_sample_p(d)
            stats.append((p, m, se, z, pv))
        pvals = [s[4] for s in stats]
        adj = RS.holm_bonferroni(pvals)   # 조정 p 만 돌려준다
        rej = [(x == x and x < 0.05) for x in adj]
        better = [(s, aj) for s, rj, aj in zip(stats, rej, adj)
                  if rj and s[1] < 0]
        worse = [(s, aj) for s, rj, aj in zip(stats, rej, adj)
                 if rj and s[1] > 0]
        tie = len(stats) - len(better) - len(worse)

        print("\n  gamma=%.2f · 에피소드 %d개 · 후보 %d개"
              % (g, n, len(cands)))
        print("    관측 경로 평균보상 %+9.4f -> 절대 순위 %d위 / %d · "
              "같은 길이 %d위 / %d"
              % (means[OBSERVED], rank, len(cands), rank_len, len(same_len)))
        print("    쌍체 + Holm(족 %d): 관측보다 **유의하게 높은** 경로 %d개 · "
              "유의하게 낮은 경로 %d개 · 못 가름 %d개"
              % (len(stats), len(better), len(worse), tie))
        better.sort(key=lambda x: x[0][1])
        for (p, m, se, z, pv), aj in better[:5]:
            print("      %-30s %d홉 차 %+8.4f ± %.4f  p_holm %s"
                  % ("-".join(str(x) for x in p), len(p) - 1, m, se,
                     ("%.3g" % aj) if aj < 1e-4 else RS.fmt_p(aj)))
        if not better:
            print("      (없다 — 관측 경로가 이 족에서 최적과 구별되지 않는다)")
        for p, m, se, z, pv in stats:
            fh_rows.append(dict(
                scope=tag, gamma="%.2f" % g,
                path="-".join(str(x) for x in p), hop=len(p) - 1,
                n=n, diff_obs_minus_rival="%.6f" % m, se="%.6f" % se,
                z="%.4f" % z, p_raw="%.6g" % pv))
        yield (tag, g, n, means[OBSERVED], rank, rank_len, len(better),
               len(worse), tie)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-episodes", type=int, default=400)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--seed-base", type=int, default=90000)
    ap.add_argument("--extra-hops", type=int, default=2)
    a = ap.parse_args()

    adj = build_adj()
    dist = hop_dist_to_dest(adj)
    d0 = dist[0]
    max_hops = d0 + a.extra_hops
    cands = enumerate_simple_paths(adj, max_hops)
    assert OBSERVED in cands
    print("후보 %d개 (홉<=%d) · 관측 경로 %d홉 · 최단 %d홉 · gamma %s"
          % (len(cands), max_hops, len(OBSERVED) - 1, d0,
             " ".join("%.2f" % g for g in GAMMAS)), flush=True)

    train_spec = []
    for si, sc in enumerate(TRAIN_SCENARIOS):
        for i in range(a.train_episodes):
            train_spec.append((sc, a.seed_base + si * 1000 + i, 0))
    eval_spec = [("dynamic", EVAL_SEED + i, EVAL_WARMUP)
                 for i in range(a.eval_episodes)]

    print("학습 혼합 %d에피소드 수집 중..." % len(train_spec), flush=True)
    m_train = collect(cands, dist, max_hops, train_spec)
    print("평가 %d에피소드 수집 중..." % len(eval_spec), flush=True)
    m_eval = collect(cands, dist, max_hops, eval_spec)

    # ---- V3: gamma=1.00 에서 기대값 표와 맞물리는가
    print("\n" + "=" * 74)
    print("V3 - gamma=1.00 에피소드 평균 == 기대값 표 (같은 양, 다른 더하기 순서)")
    ex = Expect(dist, max_hops)
    for sc, seed, warmup in train_spec:
        ex.add_episode(sc, snapshots(sc, seed, warmup, max_hops, RISK_WINDOW),
                       weight=TRAIN_WEIGHT)
    worst = 0.0
    for p in cands:
        got = sum(m_train[1.00][p]) / len(m_train[1.00][p])
        worst = max(worst, abs(got - ex.expected_return(p)))
    print("  후보 %d개 최대 절대차 %.3e -> %s"
          % (len(cands), worst, "일치" if worst < 1e-9 else "**어긋남**"))
    if worst >= 1e-9:
        print("  두 계산이 어긋난다. 중단한다.")
        return 1

    rows = []
    summ = []
    summ += list(report("train", m_train, cands, rows,
                        "학습 혼합(네 시나리오 균등) — 에이전트가 최적화한 분포"))
    summ += list(report("eval", m_eval, cands, rows,
                        "평가(dynamic 단일 + 워밍업 10) — 표 10 이 재는 분포"))

    with io.open(PAIR_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "scope", "gamma", "path", "hop", "n", "diff_obs_minus_rival",
            "se", "z", "p_raw"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with io.open(SUM_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scope", "gamma", "n_episodes", "observed_mean_return",
                    "observed_rank", "observed_rank_same_len",
                    "n_sig_better", "n_sig_worse", "n_tie"])
        for t in summ:
            w.writerow([t[0], "%.2f" % t[1], t[2], "%.6f" % t[3], t[4],
                        t[5], t[6], t[7], t[8]])
    print("\n저장: %s (%d행) · %s (%d행)"
          % (PAIR_CSV, len(rows), SUM_CSV, len(summ)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
