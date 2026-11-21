# -*- coding: utf-8 -*-
"""`FINDINGS_environment_dynamics.md:364` 의 남은 칸을 닫는다.

    남겨 두는 것: ospf_bw가 손실 수준이 높은 링크에 머무는 이유(같은 링크를
    반복해서 고르는지, 경로 간 중복도)는 이 측정에 없다. 천장 비율과 통과 시
    손실 수준은 쟀지만 **링크 재사용은 재지 않았다.**

`convention_gap_links.csv` 는 링크 단위로 집계돼 있지만 **간선 정체가 빠져
있다**(`episode_record` 가 `hop_index`·손실만 남기고 `e` 를 버린다). 그래서
같은 보행을 다시 돌려 **간선 수열**을 기록한다. 규약은 그 탐침을 그대로
import 해서 쓴다 — `WARMUP`, `walk_logging`, 플래너 목록, 시드/에피소드 수까지
`measure_convention_gap_mechanism` / `reeval_scoring_convention` 의 것이다.

재는 것 (정책마다, 시드 5개 x 200에피소드 풀):
  - **반복해서 고르는가**: 쓴 간선 수 · 통과 점유율 HHI · 상위 1/3/5 간선 몫.
  - **경로 간 중복도**: 서로 다른 경로 수 · 최빈 경로 몫 · 에피소드 쌍
    간선집합 **자카드** 평균(쌍 20,000개 표집, 고정 시드).
  - **그 반복이 고손실 링크에 쏠렸는가**: 간선의 통과 횟수와 통과 시 평균
    손실의 **스피어만 rho** · 평균 손실 >= 0.30(천장 0.35 근처) 간선에 간
    통과 몫.

원본은 읽기만 한다. 산출은 `review_results/`.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_link_reuse.py
"""
from __future__ import print_function

import argparse
import collections
import csv
import io
import itertools
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import measure_convention_gap_mechanism as M                 # noqa: E402
import review_stats as RS                                    # noqa: E402

RESULT_DIR = os.path.join(HERE, "review_results")
OUT_CSV = os.path.join(RESULT_DIR, "link_reuse.csv")
TEST_CSV = os.path.join(RESULT_DIR, "link_reuse_tests.csv")
EDGE_CSV = os.path.join(RESULT_DIR, "link_reuse_edges.csv")

JACCARD_PAIRS = 20000
JACCARD_SEED = 12345
HI_LOSS = 0.30          # 천장 0.35 (network_env.py:229 clip) 근처


def walk_edges(env, next_node_of):
    """`M.walk_logging` 과 같은 함수를 쓰고 간선만 꺼낸다."""
    _samples, log = M.walk_logging(env, next_node_of)
    return [(e, float(loss)) for e, _idx, loss, _cong in log]


def run_planner_edges(scenario, seed, episodes, planner):
    """`M.run_planner` 과 같은 순서 · 같은 난수 소비. 간선 수열을 돌려준다."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = M.NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(M.WARMUP):
            env.update_dynamic_qos()
        planned = planner(env)
        if not planned or planned[-1] != M.DESTINATION:
            continue
        M.QBH.path_metrics(env, planned)      # 난수 소비 없음 (원 탐침과 동일)
        queue = list(planned[1:])

        def next_node_of(_env, queue=queue):
            return queue.pop(0) if queue else None

        seq = walk_edges(env, next_node_of)
        if env.current_node != M.DESTINATION:
            continue
        out.append((tuple(env.visited), seq))
    return out


def run_risk_dqn_edges(scenario, seed, episodes, model):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    out = []
    for _ in range(episodes):
        env = M.NetworkEnv(use_risk=True, scenario=scenario,
                           risk_window=M.RLC.RISK_WINDOW)
        env.reset()
        for _ in range(M.WARMUP):
            env.update_dynamic_qos()

        def next_node_of(e):
            valid = e.get_valid_actions()
            if not valid:
                return None
            state = e.get_state()
            with torch.no_grad():
                q = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            return int(torch.argmax(masked).item())

        seq = walk_edges(env, next_node_of)
        if env.current_node != M.DESTINATION:
            continue
        out.append((tuple(env.visited), seq))
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    return num / ((dx * dy) ** 0.5) if dx > 0 and dy > 0 else float("nan")


def cell_stats(episodes):
    """에피소드 목록 하나(한 시드)에서 재사용·중복 지표를 뽑는다."""
    cnt = collections.Counter()
    lsum = collections.defaultdict(float)
    esets, paths = [], collections.Counter()
    for path, seq in episodes:
        paths[path] += 1
        s = set()
        for e, loss in seq:
            cnt[e] += 1
            lsum[e] += loss
            s.add(e)
        esets.append(s)
    n_tr = float(sum(cnt.values())) or 1.0
    share = sorted((c / n_tr for c in cnt.values()), reverse=True)
    hhi = sum(s * s for s in share)
    rng = random.Random(JACCARD_SEED)
    jac = []
    if len(esets) >= 2:
        for _ in range(JACCARD_PAIRS):
            i = rng.randrange(len(esets))
            j = rng.randrange(len(esets))
            if i == j:
                continue
            a, b = esets[i], esets[j]
            u = len(a | b)
            jac.append(len(a & b) / float(u) if u else 0.0)
    mloss = dict((e, lsum[e] / cnt[e]) for e in cnt)
    hi_tr = sum(c for e, c in cnt.items() if mloss[e] >= HI_LOSS) / n_tr
    ks = sorted(cnt)
    rho = spearman([cnt[e] for e in ks], [mloss[e] for e in ks])
    mhop = n_tr / float(len(episodes)) if episodes else 0.0
    # 길이 보정: 쓴 간선 수는 경로가 길면 자동으로 커진다. 1.0 = 고정 경로.
    rep = len(cnt) / mhop if mhop > 0 else float("nan")
    # HHI 는 '간선을 적게 쓰면' 저절로 커진다 -> 균등 대비로 정규화. 1.0 = 균등.
    hhi_n = hhi * len(cnt) if cnt else float("nan")
    # 재사용이 고손실 쪽에 쏠렸는가: 통과수 가중 평균손실 - 간선 단순 평균손실.
    loss_u = sum(mloss.values()) / len(mloss) if mloss else float("nan")
    loss_w = sum(cnt[e] * mloss[e] for e in cnt) / n_tr if cnt else float("nan")
    return dict(
        n_episodes=len(episodes), n_traversals=int(n_tr),
        n_edges_used=len(cnt), hhi=hhi, hhi_norm=hhi_n,
        edges_per_hop=rep,
        top1_share=share[0] if share else 0.0,
        top3_share=sum(share[:3]), top5_share=sum(share[:5]),
        n_distinct_paths=len(paths),
        top_path_share=(paths.most_common(1)[0][1] / float(len(episodes)))
        if episodes else 0.0,
        mean_jaccard=sum(jac) / len(jac) if jac else float("nan"),
        hi_loss_traverse_share=hi_tr, rho_count_loss=rho,
        loss_unweighted=loss_u, loss_weighted=loss_w,
        loss_reuse_delta=loss_w - loss_u,
        mean_hop=mhop), cnt, mloss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--scenarios", default=None)
    a = ap.parse_args()

    scen = (tuple(a.scenarios.split(",")) if a.scenarios else M.SCENARIOS)
    seeds = list(M.SEEDS)[:a.seeds] if a.seeds else list(M.SEEDS)
    neps = a.episodes or M.EPISODES
    print("시나리오 %s · 시드 %s · 에피소드 %d (규약은 원 탐침과 같다)"
          % (",".join(scen), seeds, neps), flush=True)

    rows = []
    edge_rows = []
    percell = collections.defaultdict(dict)    # (sc, pol) -> seed -> stats
    for sc in scen:
        for name, planner in M.RSC.PLANNERS:
            for sd in seeds:
                st, cnt, mloss = cell_stats(
                    run_planner_edges(sc, sd, neps, planner))
                st.update(scenario=sc, policy=name, seed=sd)
                rows.append(st)
                percell[(sc, name)][sd] = st
                for e in sorted(cnt):
                    edge_rows.append(dict(
                        scenario=sc, policy=name, seed=sd,
                        edge="%d-%d" % e, count=cnt[e],
                        mean_loss="%.6f" % mloss[e]))
            print("  %-11s %-12s 완료" % (sc, name), flush=True)
    model = M.RLC.load_risk_model()
    for sc in scen:
        for sd in seeds:
            st, cnt, mloss = cell_stats(
                run_risk_dqn_edges(sc, sd, neps, model))
            st.update(scenario=sc, policy="risk_dqn", seed=sd)
            rows.append(st)
            percell[(sc, "risk_dqn")][sd] = st
            for e in sorted(cnt):
                edge_rows.append(dict(
                    scenario=sc, policy="risk_dqn", seed=sd,
                    edge="%d-%d" % e, count=cnt[e],
                    mean_loss="%.6f" % mloss[e]))
        print("  %-11s %-12s 완료" % (sc, "risk_dqn"), flush=True)

    print("\n" + "=" * 104)
    print("[링크 재사용] 시드 %d개 평균 (풀 = 시드 x %d에피소드)"
          % (len(seeds), neps))
    print("  %-11s %-10s %6s %6s %7s %7s %6s %7s %8s %8s %8s %8s"
          % ("시나리오", "정책", "간선", "홉", "간선/홉", "HHIn",
             "경로수", "최빈", "자카드", "고손실몫", "손실델타", "rho"))
    for sc in scen:
        for pol in M.POLICIES:
            d = percell[(sc, pol)]
            f = lambda k: sum(d[s][k] for s in seeds) / float(len(seeds))  # noqa
            print("  %-11s %-10s %6.1f %6.3f %7.3f %7.3f %6.1f %7.3f %8.4f "
                  "%8.4f %8.4f %8.3f"
                  % (sc, pol, f("n_edges_used"), f("mean_hop"),
                     f("edges_per_hop"), f("hhi_norm"),
                     f("n_distinct_paths"), f("top_path_share"),
                     f("mean_jaccard"), f("hi_loss_traverse_share"),
                     f("loss_reuse_delta"), f("rho_count_loss")))

    # ---- ospf_bw 대 나머지 셋: 재사용 지표 셋만 (족을 좁게)
    focus = [s for s in ("unstable", "congestion") if s in scen]
    mets = ("edges_per_hop", "mean_jaccard", "hi_loss_traverse_share",
            "loss_reuse_delta")
    stats = []
    for sc in focus:
        for met in mets:
            base = [percell[(sc, "ospf_bw")][s][met] for s in seeds]
            for pol in ("ospf_delay", "qos", "risk_dqn"):
                other = [percell[(sc, pol)][s][met] for s in seeds]
                t, dof, pv, _ = RS.welch(base, other)
                stats.append((sc, met, pol, RS.mean_std(base)[0],
                              RS.mean_std(other)[0], t, pv))
    if stats:
        adj = RS.holm_bonferroni([s[6] for s in stats])
        print("\n" + "-" * 92)
        print("[검정] ospf_bw 대 나머지 · Welch + Holm (족 %d)" % len(stats))
        for s, aj in zip(stats, adj):
            mark = "유의" if (aj == aj and aj < 0.05) else "    "
            print("  %-11s %-22s 대 %-10s  %8.4f 대 %8.4f  p_holm %-9s %s"
                  % (s[0], s[1], s[2], s[3], s[4],
                     ("%.3g" % aj) if aj == aj and aj < 1e-4 else RS.fmt_p(aj),
                     mark))
        with io.open(TEST_CSV, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["scenario", "metric", "ospf_bw_vs", "ospf_bw_mean",
                        "other_mean", "t", "p_raw", "p_holm"])
            for s, aj in zip(stats, adj):
                w.writerow([s[0], s[1], s[2], "%.6f" % s[3], "%.6f" % s[4],
                            "%.4f" % s[5], "%.6g" % s[6],
                            ("%.6g" % aj) if aj == aj else "nan"])

    fields = ["scenario", "policy", "seed", "n_episodes", "n_traversals",
              "n_edges_used", "mean_hop", "edges_per_hop", "hhi", "hhi_norm",
              "top1_share", "top3_share", "top5_share", "n_distinct_paths",
              "top_path_share", "mean_jaccard", "hi_loss_traverse_share",
              "loss_unweighted", "loss_weighted", "loss_reuse_delta",
              "rho_count_loss"]
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, ("%.6f" % r[k]) if isinstance(r[k], float)
                             else r[k]) for k in fields))
    with io.open(EDGE_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scenario", "policy", "seed",
                                           "edge", "count", "mean_loss"])
        w.writeheader()
        for r in edge_rows:
            w.writerow(r)
    print("\n저장: %s (%d행) · %s (%d행)"
          % (OUT_CSV, len(rows), EDGE_CSV, len(edge_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
