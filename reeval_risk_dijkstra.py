"""학습이 위험 공식 위에 무엇을 더하는가 — 위험 가중 Dijkstra와 직접 맞춰 본다.

리뷰어 R1/R3-1은 위험 공식의 가중치를 정당화하라고 했다. `analyze_risk_formula.py`는
"가중치를 바꿔도 결과가 안 바뀐다(측정 불가)"고 답했다. 그런데 더 앞선 질문이 있다 —
**그 공식을 그냥 Dijkstra에 넣으면 되는 것 아닌가?** 그러면 학습이 필요한 이유를
논문이 따로 대야 한다.

이 질문이 떠오른 경로: `reeval_ospf_replan.py`에서 발표된 OSPF 비용(1000/bandwidth)이
혼잡 정보를 구조적으로 안 담는다는 것을 확인하면서 `network_env.py:462-479`를 보니
`cost_type` 중 `"risk"`가 **구현돼 있는데 어느 실험에서도 쓰이지 않았다**(전 저장소를
grep하면 모든 호출이 `"bandwidth"`다). 그리고 congestion 시드 0에서 그 경로가
`[0, 4, 8, 14, 15, 21, 22, 24]`로, 전에 측정한 Risk-DQN의 congestion 경로와 같았다.
한 시드 우연일 수 있으니 센다.

세 변형:

    risk_dijkstra_static    워밍업 후 q["risk"] Dijkstra 한 번            (학습 없음)
    risk_dijkstra_replan    워밍업 후 매 홉 재계획 + 방문 노드 제외        (학습 없음)
    risk_dqn                학습된 risk_model_final.pth                   (제안 방법)

핵심은 **짝지음**이다. 에피소드마다 환경 인스턴스를 **하나만** 만들고, 워밍업 후 같은
상태에서 (1) 위험 Dijkstra 경로를 먼저 계산한 다음 (2) 같은 환경에서 DQN을 굴린다.
둘이 같은 상태에서 결정하므로 경로를 글자 단위로 비교할 수 있다. 환경을 따로 만들면
난수 소비가 달라져 비교가 무의미해진다 — 그래서 한 환경이어야 한다.

`q["risk"]`는 `use_risk`와 무관하게 `update_dynamic_qos`에서 항상 계산된다
(`network_env.py:236-248`, 분기 밖). `use_risk=False`와 `True`에서 50개 링크 risk가
동일함은 측정으로 확인했다. 그래서 `use_risk=True` 환경 하나로 둘 다 할 수 있다.

주의 — 이 비교에 유리한 쪽으로 기울 수 있는 지점:
  * `q["risk"]`는 `delay_history`/`loss_history`를 쓰므로 워밍업 10스텝이 쌓인 뒤에야
    의미가 있다. 둘 다 워밍업 후에 보므로 조건은 같다.
  * 위험 Dijkstra는 risk만 보고 홉 수를 안 본다. 홉이 길어지면 종단간 손실이 커지므로
    손실 규약에서 불리할 수 있다 — 그래서 홉도 같이 적는다.
  * Risk-DQN은 매 홉 재결정한다. 그래서 `risk_dijkstra_replan`을 같이 둬서 재계획
    여부와 학습 여부를 분리한다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_risk_dijkstra.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_risk_dijkstra.py --from-csv
"""

import argparse
import csv
import heapq
import os
import random
import sys
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
import reeval_qos_baseline_handicap as QBH  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import (  # noqa: E402
    fmt_p, holm_bonferroni, min_detectable_effect, welch)
from topology import DESTINATION  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "risk_dijkstra.csv")
PATH_CSV = os.path.join(BASE_DIR, "review_results", "risk_dijkstra_paths.csv")

# 재계획 기준선이 방문 노드를 어떻게 다루는가. 기본값 visited가 현행
# 측정에 쓴 규칙이므로 기존 CSV가 그대로 재현된다. 문제: 이 규칙은
# **기준선에만** 걸려 있고 제안(risk_dqn, 215-223행)은 이웃 전체에서
# argmax를 고른다 — 원본 평가와 같은 규칙이다. 즉 두 쪽이 다른 규칙으로
# 평가됐다. reeval_ospf_replan.py는 같은 선택을 loopfree 플래그로 양쪽
# 다 보고하는데 여기엔 그 절제가 없어서, 재계획 열이 정적 열보다 나쁜
# 것이 (ㄱ) 재계획 자체 탓인지 (ㄴ) 이 배제 탓인지 갈리지 않는다.
REPLAN_BLOCKED = "visited"
REF_CSV = os.path.join(BASE_DIR, "review_results", "qos_baseline_handicap.csv")

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
EPISODES = RLC.EVAL_EPISODES
WARMUP = RLC.WARMUP

METRICS = QBH.METRICS
path_metrics = QBH.path_metrics
seed_mean = QBH.seed_mean


# ------------------------------------------------------------------- 플래너

def risk_plan(env, blocked):
    """현재 노드에서 출발하는 위험 가중 최단경로.

    비용은 `network_env.py:474-475`의 `cost = q["risk"]`와 같다. `env.ospf_path`는
    SOURCE 고정이라 중간 노드에서 못 쓰므로 직접 Dijkstra를 돈다. blocked가 비고
    현재 노드가 SOURCE면 `env.ospf_path("risk")`와 같아야 하고 --verify가 확인한다.
    """
    source = env.current_node
    dist = {source: 0.0}
    prev = {source: None}
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float("inf")):
            continue
        for nb in RLC.ADJ[node]:
            if nb == source or nb in blocked:
                continue
            e = env.normalize_edge(node, nb)
            q = env.qos.get(e)
            if q is None:
                continue
            new_cost = cost + float(q["risk"])
            if new_cost < dist.get(nb, float("inf")):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    if DESTINATION not in prev:
        return []
    path, curr = [], DESTINATION
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return path


def verify_planner():
    same = diff = 0
    for scenario in SCENARIOS:
        for seed in SEEDS[:2]:
            np.random.seed(seed)
            random.seed(seed)
            for _ in range(5):
                env = NetworkEnv(use_risk=True, scenario=scenario,
                                 risk_window=RLC.RISK_WINDOW)
                env.reset()
                for _ in range(WARMUP):
                    env.update_dynamic_qos()
                a = env.ospf_path(cost_type="risk")
                b = risk_plan(env, set())
                if a == b:
                    same += 1
                    continue
                diff += 1
                ca = sum(env.qos[env.normalize_edge(u, v)]["risk"]
                         for u, v in zip(a[:-1], a[1:]))
                cb = sum(env.qos[env.normalize_edge(u, v)]["risk"]
                         for u, v in zip(b[:-1], b[1:]))
                assert abs(ca - cb) < 1e-9, (
                    "위험 플래너 재구현이 원본보다 나쁘다: {} {:.6f} vs {} {:.6f}"
                    .format(a, ca, b, cb))
    print("[검증] risk_plan = env.ospf_path('risk'): 일치 {}건, 동률 {}건".format(same, diff))


# ------------------------------------------------------------------- 러너

def run_paired(scenario, seed, episodes, model):
    """에피소드마다 환경 하나로 세 정책의 경로를 모두 뽑는다 — 완전 짝지음.

    순서가 중요하다. 위험 Dijkstra 경로는 env를 step하지 않으므로 먼저 둘 다 계산하고,
    상태를 바꾸는 롤아웃(재계획·DQN)은 각각 자기 환경 사본에서 해야 한다. 사본을 만들
    수단이 환경에 없으므로, 같은 시드·같은 호출 순서로 환경을 다시 만들어 재현한다.
    재현성은 아래 assert가 확인한다 — 재생성한 환경의 risk 벡터가 같아야 한다.
    """
    import torch

    device = next(model.parameters()).device
    out = {k: [] for k in ("risk_dijkstra_static", "risk_dijkstra_replan", "risk_dqn")}
    reached = Counter()
    paths = {k: [] for k in out}

    for ep in range(episodes):
        # (1) 정적 위험 Dijkstra — 환경을 건드리지 않는다.
        np.random.seed(seed * 100003 + ep)
        random.seed(seed * 100003 + ep)
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RLC.RISK_WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        risk_snapshot = [env.qos[e]["risk"] for e in sorted(env.qos)]
        p_static = risk_plan(env, set())
        if p_static and p_static[-1] == DESTINATION:
            out["risk_dijkstra_static"].append(path_metrics(env, p_static))
            paths["risk_dijkstra_static"].append(tuple(p_static))
            reached["risk_dijkstra_static"] += 1
        else:
            paths["risk_dijkstra_static"].append(None)

        # (2) 매 홉 재계획 — 같은 상태에서 시작하도록 환경을 같은 시드로 다시 만든다.
        np.random.seed(seed * 100003 + ep)
        random.seed(seed * 100003 + ep)
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RLC.RISK_WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        assert [env.qos[e]["risk"] for e in sorted(env.qos)] == risk_snapshot, \
            "환경 재생성이 같은 상태를 주지 않는다 — 짝지음이 깨졌다"
        done = False
        while not done:
            planned = risk_plan(env, replan_blocked(env))
            if len(planned) < 2:
                break
            _, _, done, _ = env.step(planned[1])
            if len(env.visited) >= MAX_STEPS:
                break
        if env.current_node == DESTINATION:
            out["risk_dijkstra_replan"].append(path_metrics(env, env.visited))
            paths["risk_dijkstra_replan"].append(tuple(env.visited))
            reached["risk_dijkstra_replan"] += 1
        else:
            paths["risk_dijkstra_replan"].append(None)

        # (3) 학습된 Risk-DQN — 같은 시작 상태.
        np.random.seed(seed * 100003 + ep)
        random.seed(seed * 100003 + ep)
        env = NetworkEnv(use_risk=True, scenario=scenario, risk_window=RLC.RISK_WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked = torch.full_like(q_values, -1e9)
            for a in valid:
                masked[a] = q_values[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node == DESTINATION:
            out["risk_dqn"].append(path_metrics(env, env.visited))
            paths["risk_dqn"].append(tuple(env.visited))
            reached["risk_dqn"] += 1
        else:
            paths["risk_dqn"].append(None)

    means = {k: seed_mean(v) for k, v in out.items()}
    success = {k: reached[k] / episodes * 100 for k in out}
    return means, success, paths


# --------------------------------------------------------------- 측정/입출력

def measure():
    model = RLC.load_risk_model()
    rows, path_rows = [], []
    for scenario in SCENARIOS:
        for seed in SEEDS:
            means, success, paths = run_paired(scenario, seed, EPISODES, model)
            for name in means:
                row = {"scenario": scenario, "variant": name, "seed": seed,
                       "success_rate": success[name]}
                row.update(means[name])
                rows.append(row)

            n = len(paths["risk_dqn"])
            both = [(a, b) for a, b in zip(paths["risk_dijkstra_static"],
                                           paths["risk_dqn"])
                    if a is not None and b is not None]
            exact = sum(1 for a, b in both if a == b)
            both_rp = [(a, b) for a, b in zip(paths["risk_dijkstra_replan"],
                                              paths["risk_dqn"])
                       if a is not None and b is not None]
            exact_rp = sum(1 for a, b in both_rp if a == b)
            path_rows.append({
                "scenario": scenario, "seed": seed, "episodes": n,
                "static_vs_dqn_pairs": len(both),
                "static_vs_dqn_exact": exact,
                "static_vs_dqn_pct": exact / len(both) * 100 if both else float("nan"),
                "replan_vs_dqn_pairs": len(both_rp),
                "replan_vs_dqn_exact": exact_rp,
                "replan_vs_dqn_pct": exact_rp / len(both_rp) * 100 if both_rp else float("nan"),
                "distinct_static": len({p for p in paths["risk_dijkstra_static"] if p}),
                "distinct_replan": len({p for p in paths["risk_dijkstra_replan"] if p}),
                "distinct_dqn": len({p for p in paths["risk_dqn"] if p}),
            })
            print("  {:<11} seed {}  경로 일치 정적 {:5.1f}%  재계획 {:5.1f}%  "
                  "고유경로 {}/{}/{}  성공 {:.0f}/{:.0f}/{:.0f}%".format(
                      scenario, seed, path_rows[-1]["static_vs_dqn_pct"],
                      path_rows[-1]["replan_vs_dqn_pct"],
                      path_rows[-1]["distinct_static"], path_rows[-1]["distinct_replan"],
                      path_rows[-1]["distinct_dqn"],
                      success["risk_dijkstra_static"], success["risk_dijkstra_replan"],
                      success["risk_dqn"]), flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["scenario", "variant", "seed", "success_rate"] + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    with open(PATH_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(path_rows[0].keys()))
        w.writeheader()
        w.writerows(path_rows)
    print("\n저장: {} ({}행), {} ({}행)".format(
        OUT_CSV, len(rows), PATH_CSV, len(path_rows)))
    return rows, path_rows


def load_csv(path, int_cols=("seed",)):
    if not os.path.exists(path):
        raise SystemExit("CSV가 없다: " + path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("scenario", "variant"):
                continue
            r[k] = int(v) if k in int_cols else float(v)
    return rows


def by_variant(rows, scenario):
    out = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        d = out.setdefault(r["variant"],
                           {k: [] for k in ("success_rate",) + tuple(METRICS)})
        for k in d:
            d[k].append(r[k])
    return out


# ------------------------------------------------------------------- 보고

def report(rows, path_rows, ref_rows):
    print("\n" + "=" * 78)
    print("A0. 표본 차이 점검 — 짝지음 때문에 난수 소비가 달라진다")
    print("=" * 78)
    print("  에피소드마다 시드를 다시 박아야 세 정책이 같은 환경을 보게 된다. 그래서")
    print("  이 실행의 에피소드 집합은 qos_baseline_handicap.csv와 다르다. 같은 정책")
    print("  (risk_dqn)의 평균이 얼마나 벌어지는지를 적어두고, 아래 사다리 표에서")
    print("  OSPF·QoS 열은 그 오차 범위를 감안해 읽어야 한다.")
    worst = 0.0
    for scenario in SCENARIOS:
        mine = by_variant(rows, scenario)["risk_dqn"]
        theirs = by_variant(ref_rows, scenario)["risk_dqn"]
        line = "  {:<11}".format(scenario)
        for k in ("hop", "delay", "compounded"):
            a, b = float(np.mean(mine[k])), float(np.mean(theirs[k]))
            dev = abs(a - b) / abs(b) * 100 if b else 0.0
            worst = max(worst, dev)
            line += "  {} {:.4f} vs {:.4f} ({:+.2f}%)".format(k, a, b, dev)
        print(line)
    print("  -> 같은 정책의 표본 간 최대 편차 {:.2f}%. 짝지은 비교(A·C절)는 이 오차와"
          .format(worst))
    print("     무관하다 — 세 정책이 에피소드마다 같은 환경을 봤기 때문이다.")

    print("\n" + "=" * 78)
    print("A. 경로가 같은가 — 위험 Dijkstra 대 학습된 Risk-DQN (에피소드 단위 짝지음)")
    print("=" * 78)
    print("  {:<12}{:>14}{:>14}{:>26}".format(
        "시나리오", "정적 일치%", "재계획 일치%", "고유경로 정적/재계획/DQN"))
    for scenario in SCENARIOS:
        g = [r for r in path_rows if r["scenario"] == scenario]
        print("  {:<12}{:>14.1f}{:>14.1f}{:>26}".format(
            scenario,
            float(np.mean([r["static_vs_dqn_pct"] for r in g])),
            float(np.mean([r["replan_vs_dqn_pct"] for r in g])),
            "{:.1f} / {:.1f} / {:.1f}".format(
                float(np.mean([r["distinct_static"] for r in g])),
                float(np.mean([r["distinct_replan"] for r in g])),
                float(np.mean([r["distinct_dqn"] for r in g])))))

    print("\n" + "=" * 78)
    print("B. 품질 — 학습이 위험 공식 위에 무엇을 더하는가")
    print("=" * 78)
    for metric in ("compounded", "delay", "hop", "cong_live", "published"):
        print("\n  [{}]".format(metric))
        print("    {:<12}{:>13}{:>13}{:>12}{:>12}{:>11}{:>11}".format(
            "시나리오", "risk-Dijk", "risk-Dijk재", "Risk-DQN",
            "공정QoS", "OSPF", "DQN이득%"))
        for scenario in SCENARIOS:
            g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
            st = float(np.mean(g["risk_dijkstra_static"][metric]))
            rp = float(np.mean(g["risk_dijkstra_replan"][metric]))
            dq = float(np.mean(g["risk_dqn"][metric]))
            qo = float(np.mean(gr["qos_post_warmup"][metric]))
            os_ = float(np.mean(gr["ospf"][metric]))
            best = min(st, rp)
            gain = (best - dq) / best * 100 if best else float("nan")
            print("    {:<12}{:>13.4f}{:>13.4f}{:>12.4f}{:>12.4f}{:>11.4f}{:>10.1f}%"
                  .format(scenario, st, rp, dq, qo, os_, gain))

    print("\n  도달률(%)")
    print("    {:<12}{:>13}{:>13}{:>12}".format(
        "시나리오", "risk-Dijk", "risk-Dijk재", "Risk-DQN"))
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        print("    {:<12}{:>13.1f}{:>13.1f}{:>12.1f}".format(
            scenario,
            float(np.mean(g["risk_dijkstra_static"]["success_rate"])),
            float(np.mean(g["risk_dijkstra_replan"]["success_rate"])),
            float(np.mean(g["risk_dqn"]["success_rate"]))))

    print("\n" + "=" * 78)
    print("C. Risk-DQN 대 위험 Dijkstra — Welch + Holm, 최소검출효과 병기")
    print("=" * 78)
    tests = []
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        dq = g["risk_dqn"]
        for name in ("risk_dijkstra_static", "risk_dijkstra_replan"):
            for metric in METRICS:
                base = g[name][metric]
                diff, _, _, p = welch(base, dq[metric])
                mde = min_detectable_effect(base, dq[metric])
                bm = float(np.mean(base))
                tests.append({"scenario": scenario, "variant": name, "metric": metric,
                              "rel": diff / bm * 100 if bm else float("nan"),
                              "mde_rel": mde / bm * 100 if bm else float("nan"),
                              "p": p})
    adjusted = holm_bonferroni([x["p"] for x in tests])
    for x, p_adj in zip(tests, adjusted):
        x["p_holm"] = p_adj
        x["sig"] = bool(p_adj == p_adj and p_adj < 0.05)
    untestable = sum(1 for x in tests if x["p"] != x["p"])

    for scenario in SCENARIOS:
        print("\n  {}".format(scenario))
        print("    {:<24}{:<16}{:>11}{:>9}{:>11}".format(
            "기준선", "지표", "DQN이득", "MDE", "p_holm"))
        for x in tests:
            if x["scenario"] != scenario or x["metric"] not in (
                    "compounded", "delay", "hop", "cong_live"):
                continue
            print("    {:<24}{:<16}{:>10.1f}%{:>8.1f}%{:>11}{}".format(
                x["variant"], x["metric"], x["rel"], x["mde_rel"],
                fmt_p(x["p_holm"]), "  *" if x["sig"] else ""))

    print("\n  유의 {}/{}건, 검정 불가(분산 0) {}건".format(
        sum(1 for x in tests if x["sig"]), len(tests), untestable))
    print("  시드당 {}에피소드 평균이라 유의성은 판단에 못 쓴다 — 크기로 읽는다."
          .format(EPISODES))

    print("\n" + "=" * 78)
    print("정리 — 종단간 손실(compounded) 기준 사다리")
    print("=" * 78)
    for scenario in SCENARIOS:
        g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
        vals = [
            ("OSPF(발표)", float(np.mean(gr["ospf"]["compounded"]))),
            ("QoS(발표)", float(np.mean(gr["qos_t0_stale"]["compounded"]))),
            ("QoS(공정)", float(np.mean(gr["qos_post_warmup"]["compounded"]))),
            ("위험Dijkstra", float(np.mean(g["risk_dijkstra_static"]["compounded"]))),
            ("위험Dijkstra재", float(np.mean(g["risk_dijkstra_replan"]["compounded"]))),
            ("Risk-DQN", float(np.mean(g["risk_dqn"]["compounded"]))),
        ]
        print("\n  {}".format(scenario))
        for name, v in sorted(vals, key=lambda x: x[1]):
            mark = "  <- 제안" if name == "Risk-DQN" else ""
            print("    {:<16}{:.4f}{}".format(name, v, mark))


def replan_blocked(env):
    """재계획 기준선이 피할 노드 집합.

    visited: 현행 — 현재 노드만 빼고 지나온 노드를 전부 뺀다.
    prev:    직전 노드만 뺀다(2-주기만 막는다).
    none:    아무것도 빼지 않는다 — 제안(risk_dqn)과 같은 규칙이다.
    """
    if REPLAN_BLOCKED == "visited":
        return set(env.visited) - {env.current_node}
    if REPLAN_BLOCKED == "prev":
        v = env.visited
        return {v[-2]} if len(v) >= 2 else set()
    return set()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replan-blocked",
                        choices=("visited", "prev", "none"),
                        default="visited",
                        help="재계획 기준선의 방문 노드 규칙. visited가 현행.")
    args = parser.parse_args()

    if args.replan_blocked != "visited":
        # 원래 CSV를 덮지 않는다 — 규칙별로 파일을 가른다.
        global REPLAN_BLOCKED, OUT_CSV, PATH_CSV
        REPLAN_BLOCKED = args.replan_blocked
        sfx = "_%s.csv" % args.replan_blocked
        OUT_CSV = OUT_CSV.replace(".csv", sfx)
        PATH_CSV = PATH_CSV.replace(".csv", sfx)
        print("재계획 배제 규칙 = %s · 출력 = %s"
              % (REPLAN_BLOCKED, os.path.basename(OUT_CSV)))

    if args.verify:
        verify_planner()
        return

    ref_rows = load_csv(REF_CSV)
    if args.from_csv:
        rows = load_csv(OUT_CSV)
        path_rows = load_csv(PATH_CSV, int_cols=("seed", "episodes"))
    else:
        verify_planner()
        print("\n측정 시작 — 3정책 x {}시나리오 x {}시드 x {}에피소드 (짝지음)".format(
            len(SCENARIOS), len(SEEDS), EPISODES))
        rows, path_rows = measure()

    report(rows, path_rows, ref_rows)


if __name__ == "__main__":
    main()
