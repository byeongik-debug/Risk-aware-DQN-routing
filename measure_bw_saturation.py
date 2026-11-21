# -*- coding: utf-8 -*-
"""`ospf_bw` 가 고손실 링크에 머무는 **이유** 후보 하나를 검정한다.

`network_env.py:update_dynamic_qos` 의 대역폭 노이즈는 어느 분기에서도 평균이
양수고, 시나리오가 건드리는 링크에서는 **하한이 0** 이다:

    평시          bw_noise = uniform(-5.0, 15.0)   평균 +5.0
    congestion    uniform(0.0, 30.0)               평균 +15.0   (하한 0)
    unstable      uniform(0.0, 25.0)               평균 +12.5   (하한 0)
    그리고 np.clip(..., 50.0, 1000.0)

가설: WARMUP(=50) 갱신이면 대역폭이 상한 1000 에 **포화**하고, 그러면
`ospf_path(cost_type="bandwidth")` 의 비용 `1000/bandwidth` 가 전 간선 1.0 으로
동점이 되어 **홉 수 최소화로 퇴화**한다. 그게 사실이면 `ospf_bw` 의 '고손실 링크
체류' 는 링크를 고른 결과가 아니라 **비용함수가 사라진 결과**다.

재는 것 (시나리오마다, 시드 5개):
  - 갱신 0/10/25/50 회 뒤 대역폭 분포와 **상한 1000 도달 간선 비율**
  - 포화 뒤 `1000/bw` 비용의 서로 다른 값 수 · 최대/최소비
  - `ospf_path("bandwidth")` 가 **홉 최소 경로인가** (BFS 와 대조)
  - 그 경로가 시나리오 지정 나쁜 링크(CONGESTION_LINKS / UNSTABLE_LINKS)를
    몇 개 밟는가 — 홉최소 경로가 원래 그런가(시나리오 무관 BFS 와 비교)

원본은 읽기만 한다. 산출은 `review_results/`.
"""
from __future__ import print_function

import collections
import csv
import io
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import measure_convention_gap_mechanism as M                 # noqa: E402
import network_env as NE                                     # noqa: E402

RESULT_DIR = os.path.join(HERE, "review_results")
OUT_CSV = os.path.join(RESULT_DIR, "bw_saturation.csv")

CHECKPOINTS = (0, 10, 25, 50)
BW_CEIL = 1000.0


def bfs_hop_path(env):
    """시나리오·비용과 무관한 홉 최소 경로 (사전식 tie-break)."""
    adj = collections.defaultdict(list)
    for u, v in NE.EDGES:
        adj[u].append(v)
        adj[v].append(u)
    for k in adj:
        adj[k].sort()
    src, dst = 0, M.DESTINATION
    prev = {src: None}
    frontier = [src]
    while frontier and dst not in prev:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in prev:
                    prev[v] = u
                    nxt.append(v)
        frontier = nxt
    if dst not in prev:
        return None
    out, cur = [], dst
    while cur is not None:
        out.append(cur)
        cur = prev[cur]
    return list(reversed(out))


def main():
    rows = []
    bad_sets = {
        "congestion": set(tuple(sorted(x)) for x in NE.CONGESTION_LINKS),
        "unstable": set(tuple(sorted(x)) for x in NE.UNSTABLE_LINKS),
    }
    print("대역폭 포화 검정 · 시나리오 %s · 시드 %s · WARMUP=%d"
          % (",".join(M.SCENARIOS), list(M.SEEDS), M.WARMUP), flush=True)

    for sc in M.SCENARIOS:
        for sd in M.SEEDS:
            np.random.seed(sd)
            random.seed(sd)
            env = M.NetworkEnv(use_risk=False, scenario=sc)
            env.reset()
            done = 0
            for cp in CHECKPOINTS:
                while done < cp:
                    env.update_dynamic_qos()
                    done += 1
                bws = [env.qos[tuple(sorted(e))]["bandwidth"]
                       for e in set(tuple(sorted(x)) for x in NE.EDGES)]
                at_ceil = sum(1 for b in bws if b >= BW_CEIL - 1e-9)
                costs = [1000.0 / max(b, 1e-6) for b in bws]
                uniq = len(set(round(c, 12) for c in costs))
                path = env.ospf_path(cost_type="bandwidth")
                hop_path = bfs_hop_path(env)
                nhop = (len(path) - 1) if path else -1
                nhop_min = (len(hop_path) - 1) if hop_path else -1
                bad = bad_sets.get(sc, set())
                def badcount(p):
                    if not p:
                        return -1
                    return sum(1 for a, b in zip(p[:-1], p[1:])
                               if tuple(sorted((a, b))) in bad)
                rows.append(dict(
                    scenario=sc, seed=sd, updates=cp,
                    n_edges=len(bws), at_ceiling=at_ceil,
                    ceil_share=at_ceil / float(len(bws)),
                    bw_min=min(bws), bw_mean=sum(bws) / len(bws),
                    cost_uniq=uniq,
                    cost_ratio=max(costs) / min(costs),
                    ospf_bw_hops=nhop, hop_min=nhop_min,
                    is_hop_min=int(nhop == nhop_min),
                    same_as_bfs=int(path == hop_path),
                    bad_on_ospf_bw=badcount(path),
                    bad_on_hopmin=badcount(hop_path),
                    n_bad_links=len(bad)))
        print("  %-11s 완료" % sc, flush=True)

    print("\n" + "=" * 100)
    print("  %-11s %8s %8s %9s %9s %8s %9s %7s %7s %8s %8s"
          % ("시나리오", "갱신", "상한몫", "bw최소", "bw평균", "비용종",
             "비용비", "bw홉", "최소홉", "홉최소?", "나쁜링크"))
    for sc in M.SCENARIOS:
        for cp in CHECKPOINTS:
            sub = [r for r in rows if r["scenario"] == sc
                   and r["updates"] == cp]
            n = float(len(sub))
            f = lambda k: sum(r[k] for r in sub) / n                  # noqa
            print("  %-11s %8d %8.3f %9.1f %9.1f %8.1f %9.4f %7.1f %7.1f "
                  "%8.2f %8.1f"
                  % (sc, cp, f("ceil_share"), f("bw_min"), f("bw_mean"),
                     f("cost_uniq"), f("cost_ratio"), f("ospf_bw_hops"),
                     f("hop_min"), f("is_hop_min"), f("bad_on_ospf_bw")))
    print("\n  * 나쁜링크 = 시나리오 지정 링크(congestion/unstable)를 밟은 수."
          " 홉최소 경로의 같은 수는 아래.")
    for sc in ("congestion", "unstable"):
        sub = [r for r in rows if r["scenario"] == sc
               and r["updates"] == M.WARMUP]
        if sub:
            n = float(len(sub))
            print("    %-11s 지정 %d개 중  ospf_bw %.1f  홉최소 %.1f"
                  % (sc, sub[0]["n_bad_links"],
                     sum(r["bad_on_ospf_bw"] for r in sub) / n,
                     sum(r["bad_on_hopmin"] for r in sub) / n))

    # ---- 왜 홉이 안 늘어나는가: 홉수별 최소 대역폭비용의 실제 여유
    # 비용비만으로는 못 말한다. 7홉이 6홉을 이기려면 max/min > 7/6 = 1.167 면
    # 충분조건이 열리고 관측 비용비는 1.51~2.43 이라 **열려 있다**. 그래서
    # 열린 채로도 안 일어났는지를 직접 센다.
    from measure_blind_path_optimality import enumerate_simple_paths, build_adj
    adj = build_adj()
    cands = enumerate_simple_paths(adj, 8)
    byhop = collections.defaultdict(list)
    for p in cands:
        byhop[len(p) - 1].append(p)
    print("\n" + "-" * 100)
    print("[홉 여유] 후보 %d개 (홉<=8) · 홉수별 %s"
          % (len(cands), dict((k, len(v)) for k, v in sorted(byhop.items()))))
    print("  %-11s %6s %10s %10s %10s %9s %8s"
          % ("시나리오", "갱신", "최소6홉", "최소7홉", "최소8홉", "7홉여유%",
             "비용비"))
    mrows = []
    for sc in M.SCENARIOS:
        for cp in (0, M.WARMUP, 50):
            best = collections.defaultdict(list)
            ratios = []
            for sd in M.SEEDS:
                np.random.seed(sd)
                random.seed(sd)
                env = M.NetworkEnv(use_risk=False, scenario=sc)
                env.reset()
                for _ in range(cp):
                    env.update_dynamic_qos()
                cost = dict((tuple(sorted(e)),
                             1000.0 / max(env.qos[tuple(sorted(e))]
                                          ["bandwidth"], 1e-6))
                            for e in set(tuple(sorted(x)) for x in NE.EDGES))
                ratios.append(max(cost.values()) / min(cost.values()))
                for h, ps in byhop.items():
                    best[h].append(min(
                        sum(cost[tuple(sorted((a, b)))]
                            for a, b in zip(p[:-1], p[1:])) for p in ps))
            m = dict((h, sum(v) / len(v)) for h, v in best.items())
            marg = ((m.get(7, float("nan")) - m.get(6, float("nan")))
                    / m.get(6, float("nan")) * 100.0)
            print("  %-11s %6d %10.4f %10.4f %10.4f %9.2f %8.4f"
                  % (sc, cp, m.get(6, float("nan")), m.get(7, float("nan")),
                     m.get(8, float("nan")), marg,
                     sum(ratios) / len(ratios)))
            mrows.append((sc, cp, m.get(6), m.get(7), m.get(8), marg,
                          sum(ratios) / len(ratios)))
    with io.open(os.path.join(RESULT_DIR, "bw_hop_margin.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scenario", "updates", "min_cost_6hop", "min_cost_7hop",
                    "min_cost_8hop", "margin_7_over_6_pct", "cost_ratio"])
        for r in mrows:
            w.writerow([r[0], r[1]] + ["%.6f" % x for x in r[2:]])
    print("  * 7홉여유%% > 0 이면 6홉이 이긴다. 충분조건 문턱은 비용비 7/6=1.1667"
          " 이고 관측 비용비는 그보다 크다 - **열려 있는데 안 일어났다**.")

    fields = list(rows[0].keys())
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, ("%.6f" % r[k]) if isinstance(r[k], float)
                             else r[k]) for k in fields))
    print("\n저장: %s (%d행)" % (OUT_CSV, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
