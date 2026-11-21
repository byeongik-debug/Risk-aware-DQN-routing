# -*- coding: utf-8 -*-
"""28.2 의 마지막 괄호 한 줄을 재려고 만들었다.

    (왜 그 경로인지는 재지 않았다.)

상태를 가린 재학습 10런이 전부 `0-4-8-14-15-21-22-24`(7홉)로 수렴하고, 추론
시점 가림(`edgemean`)도 같은 자리로 떨어진다. FINDINGS 는 거기서 "그 경로는
가중치의 성질이 아니라 환경과 보상의 성질이다" 까지 적고 멈췄다.

**코드를 읽으니 질문이 더 좁아진다.** `network_env.get_state()` 는

    [ 간선 50칸 위험도 ] + curr_onehot(25) + dest_onehot(25) + visited_onehot(25)

이고, Blind 는 앞 50칸을 상수로 덮는다. 그러면 남는 입력은 **현재 노드 · 목적지
(상수) · 방문 집합**뿐이다 — 즉 환경 정보가 **하나도** 남지 않는다. 그래서 Blind
정책은 "위험을 못 보는 정책" 이 아니라 **열린 루프(open-loop) 고정 경로**이고,
최적 Blind 정책은 **기대보상이 가장 큰 고정 경로**다. 질문은 이렇게 선다:

    `0-4-8-14-15-21-22-24` 는 평균 환경에서 기대보상 최적 경로인가?

**정확히 계산할 수 있다.** 두 가지가 맞아떨어진다.
  (1) `update_dynamic_qos()` 는 `timestep` 만 올리고 간선마다 난수를 뽑는다 —
      **행동을 보지 않는다.** 그래서 환경 궤적은 (시드, 갱신 횟수)만의 함수이고,
      후보 경로가 달라도 k번째 홉이 보는 링크 값은 같다.
  (2) 보상이 간선 가산이다. 홉별 항은 그 홉의 스냅샷에서, 종단 항
      `-total_risk*6.0` 은 `calculate_path_metrics` 가 hop_count 로 나눈
      **홉당 평균**이라 길이 L 을 고정하면 역시 가산이다.
  따라서 E[R(P)] = Σ_k E[홉비용(간선, k)] + 45 − (6/L)·Σ_k E[최종위험(간선, L)]
  로, 기대값 표 하나를 쌓으면 후보 전부를 **근사 없이** 비교할 수 있다.

규약(읽어서 맞춘 것):
  학습  `train_risk_blind.py` — 네 시나리오 **균등 혼합**(각 0.25), 워밍업 없음
        (`env.reset()` 직후 step), 보상은 시나리오별 계수.
  평가  `eval_ablation_risky.evaluate` — **dynamic** 단일, EVAL_SEED 20261114+i,
        `reset()` 뒤 `update_dynamic_qos()` 10회, epsilon 0.

검증 둘을 통과해야 숫자를 쓴다:
  V1 관측 경로를 평가 규약으로 다시 채점해 커밋된 `risk_blind_runs.csv` 의
     delay 471.089876 · loss 0.158481 · risk 0.373444 · hop 7 을 재현한다.
     (재현되면 내 스냅샷이 하네스와 같은 환경을 본다는 뜻이다.)
  V2 환경 궤적이 **행동과 무관**한지 실측한다 — 같은 시드로 서로 다른 경로를
     걸은 뒤 같은 스텝의 링크 값이 비트 동일한지 본다. (1)의 가정이다.

원본은 읽기만 한다. 산출은 `review_results/` 로.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_blind_path_optimality.py
"""
from __future__ import print_function

import argparse
import collections
import csv
import io
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "SDN_DQN_new_env"))

import network_env                                        # noqa: E402
import risk_calculator                                     # noqa: E402
from topology import DESTINATION, EDGES, NUM_NODES, SOURCE  # noqa: E402

# eval_ablation_risky.evaluate 는 Blind 평가에서 zeroed=[] 로 들어온다
# (eval_risk_blind.eval_blind -> E.evaluate(model_path, [], episodes)).
risk_calculator.ZEROED_FACTORS = []

RISK_WINDOW = 5   # train_risk_blind.py:70 과 같은 값

RESULT_DIR = os.path.join(HERE, "review_results")
CAND_CSV = os.path.join(RESULT_DIR, "blind_path_candidates.csv")
RIVAL_CSV = os.path.join(RESULT_DIR, "blind_path_rivals.csv")
EDGE_CSV = os.path.join(RESULT_DIR, "blind_path_edge_costs.csv")

OBSERVED = (0, 4, 8, 14, 15, 21, 22, 24)
EVAL_SEED = 20261114            # eval_ablation_risky.EVAL_SEED
EVAL_WARMUP = 10                # eval_ablation_risky.evaluate
TRAIN_SCENARIOS = ("normal", "congestion", "unstable", "dynamic")
TRAIN_WEIGHT = 0.25             # train_risk_blind.SCENARIO_PROBS

# 커밋된 risk_blind_runs.csv 의 Blind 값 (모든 런이 동일)
V1_TARGET = dict(delay=471.089876, loss=0.158481, risk=0.373444, hop=7.0)
V1_TOL = 1e-5


# ------------------------------------------------------------------ 위상

def build_adj():
    adj = collections.defaultdict(set)
    for u, v in EDGES:
        adj[u].add(v)
        adj[v].add(u)
    return dict((k, sorted(v)) for k, v in adj.items())


def hop_dist_to_dest(adj):
    """DESTINATION 까지의 홉 거리 (network_env 의 진행 보상이 쓰는 그 거리)."""
    dist = {DESTINATION: 0}
    frontier = [DESTINATION]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


def enumerate_simple_paths(adj, max_hops):
    """SOURCE -> DESTINATION 단순 경로 전수 (홉 <= max_hops)."""
    out = []
    path = [SOURCE]
    seen = set([SOURCE])

    def rec(u):
        if u == DESTINATION:
            out.append(tuple(path))
            return
        if len(path) - 1 >= max_hops:
            return
        for v in adj.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            path.append(v)
            rec(v)
            path.pop()
            seen.discard(v)

    rec(SOURCE)
    return out


# ------------------------------------------------------------- 환경 궤적

def norm(u, v):
    return (u, v) if u <= v else (v, u)


def snapshots(scenario, seed, warmup, n_updates, risk_window):
    """reset + warmup 뒤, 갱신 1..n_updates 시점의 간선 값을 그대로 기록한다.

    `step()` 이 `update_dynamic_qos()` 를 **먼저** 부르고 값을 읽으므로,
    k번째 홉이 보는 값은 warmup + k 번째 갱신 직후의 값이다.
    """
    random.seed(seed)
    np.random.seed(seed)
    env = network_env.NetworkEnv(use_risk=True, scenario=scenario,
                                 risk_window=risk_window)
    env.reset()
    for _ in range(warmup):
        env.update_dynamic_qos()
    snaps = []
    for _ in range(n_updates):
        env.update_dynamic_qos()
        snaps.append(dict((e, (float(q["risk"]), float(q["delay"]),
                               float(q["loss"]),
                               bool(q.get("is_congested", False))))
                          for e, q in env.qos.items()))
    return snaps


def step_cost(scenario, risk, congested, progress):
    """network_env.step 의 홉 단위 보상 (부호 그대로, 재방문 없음 가정)."""
    r = 0.0
    if scenario == "normal":
        r -= risk * 0.7
        r += 2.0 if progress else -2.0
        r -= 0.10
    elif scenario == "congestion":
        r -= risk * 10.0
        if congested:
            r -= 4.0
        r += 0.7 if progress else -0.7
        r -= 0.06
    elif scenario == "unstable":
        r -= risk * 10.0
        if risk >= 0.22:
            r -= 3.0
        r += 0.7 if progress else -0.7
        r -= 0.06
    elif scenario == "dynamic":
        r -= risk * 22.0
        if congested:
            r -= 5.0
        if risk >= 0.25:
            r -= 5.0
        r += 0.35 if progress else -0.35
        r -= 0.08
    else:
        raise ValueError(scenario)
    return r


class Expect(object):
    """E[홉비용(간선, k)] 과 E[최종위험(간선, L)] 을 쌓는다."""

    def __init__(self, dist, max_hops):
        self.dist = dist
        self.max_hops = max_hops
        self.cost = collections.defaultdict(float)   # (k, e) -> sum
        self.riskfin = collections.defaultdict(float)  # (L, e) -> sum
        self.n = 0

    def add_episode(self, scenario, snaps, weight=1.0):
        self.n += weight
        for k in range(min(self.max_hops, len(snaps))):
            snap = snaps[k]
            for (u, v) in EDGES:
                e = norm(u, v)
                risk, _d, _l, cong = snap[e]
                # 진행 여부는 방향에 따라 다르므로 두 방향을 따로 쌓는다.
                for a, b in ((u, v), (v, u)):
                    prog = self.dist.get(b, 99) < self.dist.get(a, 99)
                    self.cost[(k, (a, b))] += weight * step_cost(
                        scenario, risk, cong, prog)
        for L in range(1, min(self.max_hops, len(snaps)) + 1):
            snap = snaps[L - 1]
            for (u, v) in EDGES:
                e = norm(u, v)
                self.riskfin[(L, e)] += weight * snap[e][0]

    def decompose(self, path):
        """E[R] 을 홉비용 합 / 도착 보너스 / 종단 위험항으로 가른다."""
        L = len(path) - 1
        steps = sum(self.cost[(k, (path[k], path[k + 1]))] / self.n
                    for k in range(L))
        rf = sum(self.riskfin[(L, norm(path[k], path[k + 1]))] / self.n
                 for k in range(L))
        meanrisk = rf / L
        return dict(hop=L, steps=steps, arrive=45.0, meanrisk=meanrisk,
                    terminal=-6.0 * meanrisk,
                    total=steps + 45.0 - 6.0 * meanrisk)

    def expected_return(self, path):
        L = len(path) - 1
        tot = 0.0
        for k in range(L):
            a, b = path[k], path[k + 1]
            tot += self.cost[(k, (a, b))] / self.n
        rf = 0.0
        for k in range(L):
            rf += self.riskfin[(L, norm(path[k], path[k + 1]))] / self.n
        return tot + 45.0 - 6.0 * (rf / L)


# ----------------------------------------------------------------- 검증

def v1_replay(path, episodes, risk_window):
    """관측 경로를 평가 규약으로 다시 채점한다 (walk_final = 종단 스냅샷)."""
    L = len(path) - 1
    acc = dict(delay=[], loss=[], risk=[])
    for i in range(episodes):
        snaps = snapshots("dynamic", EVAL_SEED + i, EVAL_WARMUP, L,
                          risk_window)
        fin = snaps[L - 1]
        d = s_l = s_r = 0.0
        for k in range(L):
            e = norm(path[k], path[k + 1])
            risk, delay, loss, _c = fin[e]
            d += delay
            s_l += loss
            s_r += risk
        acc["delay"].append(d)
        acc["loss"].append(s_l / L)
        acc["risk"].append(s_r / L)
    return dict((k, sum(v) / len(v)) for k, v in acc.items())


def v2_action_independence(risk_window):
    """같은 시드로 다른 경로를 걸어도 같은 스텝의 링크 값이 같은가."""
    outs = []
    for actions in ([4, 8, 14], [1, 2, 3]):
        random.seed(777)
        np.random.seed(777)
        env = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                     risk_window=risk_window)
        env.reset()
        rec = []
        node = SOURCE
        adj = build_adj()
        for a in actions:
            if a not in adj.get(node, ()):
                rec.append(None)
                break
            env.step(a)
            node = a
            rec.append(dict((e, float(q["risk"])) for e, q in env.qos.items()))
        outs.append(rec)
    if any(r is None or any(x is None for x in r) for r in outs):
        return None
    worst = 0.0
    n = min(len(outs[0]), len(outs[1]))
    for t in range(n):
        for e in outs[0][t]:
            worst = max(worst, abs(outs[0][t][e] - outs[1][t][e]))
    return worst, n


# ------------------------------------------------------------------ 본체

def rival_paths(scenario, seed, warmup, risk_window):
    """워밍업 직후 상태에서 비용별 최단경로 (OSPF 비교군이 쓰는 그 정의)."""
    random.seed(seed)
    np.random.seed(seed)
    env = network_env.NetworkEnv(use_risk=True, scenario=scenario,
                                 risk_window=risk_window)
    env.reset()
    for _ in range(warmup):
        env.update_dynamic_qos()
    out = {}
    for ct in ("hop", "bandwidth", "delay", "risk"):
        try:
            out[ct] = tuple(env.ospf_path(cost_type=ct))
        except Exception as exc:                        # noqa: BLE001
            out[ct] = ("err", str(exc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--train-episodes", type=int, default=100,
                    help="시나리오마다 이만큼 (균등 혼합이라 가중 평균과 같다)")
    ap.add_argument("--seed-base", type=int, default=90000,
                    help="학습 혼합 표의 시드 기점 (추정 안정성 확인용)")
    ap.add_argument("--extra-hops", type=int, default=2,
                    help="최단 홉 + 이만큼까지 후보로 센다")
    a = ap.parse_args()

    rw = RISK_WINDOW
    print("RISK_WINDOW=%d · SOURCE=%d · DESTINATION=%d · 노드 %d · 간선 %d"
          % (rw, SOURCE, DESTINATION, NUM_NODES, len(EDGES)), flush=True)

    adj = build_adj()
    dist = hop_dist_to_dest(adj)
    d0 = dist[SOURCE]
    print("\n[위상] SOURCE->DESTINATION 홉 거리 = %d" % d0)
    print("  관측 경로 %s : %d홉 → %s"
          % ("-".join(str(x) for x in OBSERVED), len(OBSERVED) - 1,
             "최단" if len(OBSERVED) - 1 == d0 else "**최단 아님**"))

    max_hops = d0 + a.extra_hops
    cands = enumerate_simple_paths(adj, max_hops)
    by_len = collections.Counter(len(p) - 1 for p in cands)
    print("  단순 경로 전수(홉<=%d): %d개 — 길이별 %s"
          % (max_hops, len(cands),
             " · ".join("%d홉 %d개" % (k, by_len[k]) for k in sorted(by_len))))
    assert OBSERVED in cands, "관측 경로가 후보에 없다"
    shortest = [p for p in cands if len(p) - 1 == d0]

    # ---------------- V2 (가정 먼저)
    print("\n" + "=" * 74)
    print("V2 - 환경 궤적이 행동과 무관한가 (같은 시드, 다른 경로)")
    v2 = v2_action_independence(rw)
    if v2 is None:
        print("  측정 실패 (간선 없음) -> 아래 계산을 신뢰할 수 없다")
        return 1
    worst, nsteps = v2
    print("  스텝 %d개 비교 · 링크 위험도 최대 절대차 %.3e -> %s"
          % (nsteps, worst,
             "행동과 무관하다" if worst == 0.0 else "**행동에 의존한다**"))
    if worst != 0.0:
        print("  가정이 깨졌다. 기대값 표 방식을 쓸 수 없다.")
        return 1

    # ---------------- V1
    print("\n" + "=" * 74)
    print("V1 - 관측 경로를 평가 규약으로 재채점 (dynamic, EVAL_SEED+i, warmup 10)")
    got = v1_replay(OBSERVED, a.episodes, rw)
    ok = True
    for k in ("delay", "loss", "risk"):
        diff = abs(got[k] - V1_TARGET[k])
        flag = "일치" if diff <= V1_TOL else "**어긋남**"
        ok = ok and diff <= V1_TOL
        print("  %-6s 재채점 %.6f · 커밋 %.6f · 차 %.2e  %s"
              % (k, got[k], V1_TARGET[k], diff, flag))
    if not ok:
        print("  -> 하네스와 같은 환경을 보고 있지 않다. 중단한다.")
        return 1

    # ---------------- 기대값 표
    print("\n" + "=" * 74)
    print("[기대값] 홉별 간선 비용을 쌓는다 (평가 %d에피소드 · 학습 혼합 %dx4)"
          % (a.episodes, a.train_episodes), flush=True)

    ex_eval = Expect(dist, max_hops)
    for i in range(a.episodes):
        ex_eval.add_episode("dynamic",
                            snapshots("dynamic", EVAL_SEED + i, EVAL_WARMUP,
                                      max_hops, rw))
    print("  평가(dynamic) 표 완료", flush=True)

    ex_train = Expect(dist, max_hops)
    for si, sc in enumerate(TRAIN_SCENARIOS):
        for i in range(a.train_episodes):
            ex_train.add_episode(sc, snapshots(sc, a.seed_base + si * 1000 + i,
                                               0, max_hops, rw),
                                 weight=TRAIN_WEIGHT)
        print("  학습 혼합 %-11s 완료" % sc, flush=True)

    # ---------------- 순위
    rows = []
    for p in cands:
        rows.append(dict(path="-".join(str(x) for x in p),
                         hop=len(p) - 1,
                         is_observed=1 if p == OBSERVED else 0,
                         r_eval=ex_eval.expected_return(p),
                         r_train=ex_train.expected_return(p)))
    for key in ("r_eval", "r_train"):
        rows.sort(key=lambda r: -r[key])
        for n, r in enumerate(rows, 1):
            r["rank_" + key] = n

    obs = [r for r in rows if r["is_observed"]][0]
    print("\n" + "=" * 74)
    print("[순위] 관측 경로의 기대보상 순위 (후보 %d개 중)" % len(rows))
    for key, lab in (("r_train", "학습 혼합(네 시나리오 균등)"),
                     ("r_eval", "평가(dynamic 단일)")):
        rows.sort(key=lambda r: -r[key])
        best = rows[0]
        print("\n  %s — 관측 경로 %d위 / %d, 기대보상 %+.4f (최적 %+.4f, 차 %.4f)"
              % (lab, obs["rank_" + key], len(rows), obs[key], best[key],
                 best[key] - obs[key]))
        print("    상위 5:")
        for r in rows[:5]:
            print("      %2d위 %-30s %2d홉 %+9.4f%s"
                  % (r["rank_" + key], r["path"], r["hop"], r[key],
                     "   <= 관측" if r["is_observed"] else ""))
        same_len = [r for r in rows if r["hop"] == len(OBSERVED) - 1]
        same_len.sort(key=lambda r: -r[key])
        rk = [n for n, r in enumerate(same_len, 1) if r["is_observed"]][0]
        print("    같은 길이(%d홉 %d개) 안에서는 %d위"
              % (len(OBSERVED) - 1, len(same_len), rk))

    # ---------------- 분해
    print("\n" + "=" * 74)
    print("[분해] 학습 혼합에서 관측 경로와 상위 경로의 기대보상 내역")
    rows.sort(key=lambda r: -r["r_train"])
    show = [r["path"] for r in rows[:3]]
    if obs["path"] not in show:
        show.append(obs["path"])
    print("  %-30s %3s %9s %9s %9s %9s"
          % ("경로", "홉", "홉비용합", "평균위험", "종단항", "합계"))
    deco = {}
    for ps in show:
        pt = tuple(int(x) for x in ps.split("-"))
        dd = ex_train.decompose(pt)
        deco[ps] = dd
        print("  %-30s %3d %+9.4f %9.6f %+9.4f %+9.4f%s"
              % (ps, dd["hop"], dd["steps"], dd["meanrisk"], dd["terminal"],
                 dd["total"], "   <= 관측" if ps == obs["path"] else ""))
    o = deco[obs["path"]]
    t = deco[rows[0]["path"]]
    DEC_CSV = os.path.join(RESULT_DIR, "blind_path_decompose.csv")
    with io.open(DEC_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["path", "hop", "is_observed", "steps_sum", "arrive",
                    "mean_risk", "terminal", "total"])
        for ps in show:
            dd = deco[ps]
            w.writerow([ps, dd["hop"], 1 if ps == obs["path"] else 0,
                        "%.6f" % dd["steps"], "%.1f" % dd["arrive"],
                        "%.6f" % dd["meanrisk"], "%.6f" % dd["terminal"],
                        "%.6f" % dd["total"]])
        gap = t["total"] - o["total"]
        w.writerow(["GAP_best_minus_observed", t["hop"] - o["hop"], "",
                    "%.6f" % (t["steps"] - o["steps"]), "0.0", "",
                    "%.6f" % (t["terminal"] - o["terminal"]),
                    "%.6f" % gap])
        w.writerow(["GAP_terminal_share_pct", "", "", "", "", "", "",
                    "%.6f" % (100.0 * (t["terminal"] - o["terminal"]) / gap)])
    print("  분해 저장: %s" % DEC_CSV)
    if t["hop"] != o["hop"]:
        print("  1위(%d홉) - 관측(%d홉): 홉비용합 %+.4f · 종단항 %+.4f"
              " -> 길이 차가 종단항을 %s"
              % (t["hop"], o["hop"], t["steps"] - o["steps"],
                 t["terminal"] - o["terminal"],
                 "완화한다" if t["terminal"] > o["terminal"] else "악화한다"))

    # ---------------- 경쟁 설명
    print("\n" + "=" * 74)
    print("[경쟁 설명] 관측 경로가 다른 규칙의 산물인가 (평가 %d에피소드)"
          % a.episodes)
    hit = collections.Counter()
    tot = 0
    rivrows = []
    for i in range(a.episodes):
        rp = rival_paths("dynamic", EVAL_SEED + i, EVAL_WARMUP, rw)
        tot += 1
        for ct, p in rp.items():
            if p == OBSERVED:
                hit[ct] += 1
        rivrows.append(dict(episode=i, **dict(
            (ct, "-".join(str(x) for x in p)) for ct, p in rp.items())))
    for ct in ("hop", "bandwidth", "delay", "risk"):
        print("  %-10s 최단경로가 관측 경로와 같은 비율: %5.1f%% (%d/%d)"
              % (ct, 100.0 * hit[ct] / tot, hit[ct], tot))
    print("  최단 홉 경로 %d개 중 관측 경로 포함: %s"
          % (len(shortest), "예" if OBSERVED in shortest else "아니오"))

    # ---------------- 저장
    with io.open(CAND_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "hop", "is_observed",
                                           "r_train", "r_eval",
                                           "rank_r_train", "rank_r_eval"])
        w.writeheader()
        rows.sort(key=lambda r: -r["r_train"])
        for r in rows:
            r["r_train"] = "%.6f" % r["r_train"]
            r["r_eval"] = "%.6f" % r["r_eval"]
            w.writerow(r)
    with io.open(RIVAL_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["episode", "hop", "bandwidth",
                                           "delay", "risk"])
        w.writeheader()
        for r in rivrows:
            w.writerow(r)
    SUM_CSV = os.path.join(RESULT_DIR, "blind_path_optimality_summary.csv")
    srows = [
        ("hop_distance_src_dst", "%d" % d0),
        ("observed_hop", "%d" % (len(OBSERVED) - 1)),
        ("n_shortest_paths", "%d" % len(shortest)),
        ("n_candidates", "%d" % len(rows)),
        ("observed_rank_train", "%d" % obs["rank_r_train"]),
        ("observed_rank_eval", "%d" % obs["rank_r_eval"]),
        ("observed_return_train", "%.6f" % o["total"]),
        ("observed_meanrisk_train", "%.6f" % o["meanrisk"]),
        ("best_return_train", "%.6f" % t["total"]),
        ("v1_delay", "%.6f" % got["delay"]),
        ("v1_loss", "%.6f" % got["loss"]),
        ("v1_risk", "%.6f" % got["risk"]),
        ("v2_max_abs_diff", "%.6e" % worst),
    ]
    with io.open(SUM_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "value"])
        for k, v in srows:
            w.writerow([k, v])
    print("요약 저장: %s (%d행)" % (SUM_CSV, len(srows)))
    print("\n저장: %s (%d행) · %s (%d행)"
          % (CAND_CSV, len(rows), RIVAL_CSV, len(rivrows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
