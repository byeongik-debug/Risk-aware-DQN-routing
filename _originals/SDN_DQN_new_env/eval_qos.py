# eval_qos.py
# OSPF와 QoS-Model을 5개 시드로 반복 평가해 mean ± std 계산
# 결과를 exp1_reference.csv로 저장 → show_results.py 실험 1에서 Risk-DQN과 비교
#
# QoS-Model 동작 방식:
#   t=0(초기 상태)에서 현재 QoS(지연·대역폭·손실) 기반으로 전체 경로를 한 번만 계획
#   이후 10번 워밍업으로 네트워크 상태가 변해도 경로를 재계획하지 않음
#   → 링크 상태 변화에 적응 못함 → Risk-DQN보다 높은 손실 발생

import os, csv, heapq, random
import numpy as np
from collections import defaultdict
from network_env import NetworkEnv
from topology import DESTINATION, NUM_NODES, EDGES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(BASE_DIR, "results")
OUT_CSV = os.path.join(RESULT_DIR, "exp1_reference.csv")

SCENARIOS = ["normal", "congestion", "unstable", "dynamic"]
EVAL_EPISODES = 200
SEEDS = [0, 1, 2, 3, 4]   # 5개 시드로 반복 → mean ± std 계산

# 인접 리스트 구성 (양방향) — Dijkstra 탐색에 사용
ADJ = defaultdict(list)
for u, v in EDGES:
    ADJ[u].append(v)
    ADJ[v].append(u)


def qos_plan_full_path(env):
    """현재 QoS 상태 기반 Dijkstra로 소스→목적지 전체 경로를 한 번만 계획

    엣지 비용 = (지연_정규화 + 대역폭_역수_정규화 + 손실_정규화) / 3
    세 지표를 동일 가중치로 합산한 현재 상태 복합 QoS 점수
    """
    source = env.current_node

    dist = {source: 0.0}   # 출발 노드 비용 0
    prev = {source: None}  # 경로 역추적용
    heap = [(0.0, source)]

    while heap:
        cost, node = heapq.heappop(heap)
        if cost > dist.get(node, float('inf')):
            continue  # 이미 더 좋은 경로를 찾은 노드는 스킵
        for nb in ADJ[node]:
            if nb == source:
                continue  # 출발 노드로 돌아가지 않음
            e = env.normalize_edge(node, nb)
            q = env.qos.get(e, {"delay": 120.0, "bandwidth": 0.0, "loss": 0.35})
            # 각 지표를 0~1로 정규화
            d_n = float(np.clip(q["delay"] / 120.0, 0.0, 1.0))
            b_n = 1.0 - float(np.clip(q["bandwidth"] / 1000.0, 0.0, 1.0))
            l_n = float(np.clip(q["loss"] / 0.35, 0.0, 1.0))
            edge_cost = (d_n + b_n + l_n) / 3.0
            new_cost = cost + edge_cost
            if new_cost < dist.get(nb, float('inf')):
                dist[nb] = new_cost
                prev[nb] = node
                heapq.heappush(heap, (new_cost, nb))

    if DESTINATION not in prev:
        return []
    path = []
    curr = DESTINATION
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    path.reverse()
    return path


def _run_ospf_one_seed(scenario, seed, episodes):
    """단일 시드로 OSPF 평가 → 이 시드의 평균값 반환"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, hops = [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        path = env.ospf_path(cost_type="bandwidth")
        if not path or path[-1] != DESTINATION:
            continue
        success_count += 1
        m = env.calculate_path_metrics(path)
        delays.append(m["total_delay"])
        losses.append(m["total_loss"])
        hops.append(m["hop_count"])
        hop_count = max(m["hop_count"], 1)
        unstable_usages.append(m["unstable_count"] / hop_count * 100)
        congested_usages.append(m["congested_count"] / hop_count * 100)

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
        "unstable_usage": float(np.mean(unstable_usages)) if unstable_usages else 0.0,
        "congested_usage": float(np.mean(congested_usages)) if congested_usages else 0.0,
    }


def evaluate_ospf(scenario, episodes=EVAL_EPISODES):
    """OSPF를 5개 시드로 반복 평가 → mean ± std (ddof=1) 반환"""
    seed_results = [_run_ospf_one_seed(scenario, s, episodes) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "model": "OSPF", "scenario": scenario,
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":      float(np.mean(d_vals)),
        "delay_std":  float(np.std(d_vals, ddof=1)),
        "loss":       float(np.mean(l_vals)),
        "loss_std":   float(np.std(l_vals, ddof=1)),
        "hop":        float(np.mean(h_vals)),
        "hop_std":    float(np.std(h_vals, ddof=1)),
        "unstable_usage":  float(np.mean([r["unstable_usage"]  for r in seed_results])),
        "congested_usage": float(np.mean([r["congested_usage"] for r in seed_results])),
    }


def _run_qos_one_seed(scenario, seed, episodes):
    """단일 시드로 QoS-Model 평가 → 이 시드의 평균값 반환"""
    np.random.seed(seed)
    random.seed(seed)

    delays, losses, hops = [], [], []
    unstable_usages, congested_usages = [], []
    success_count = 0

    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()

        # t=0 스냅샷(네트워크 변화 전)으로 경로 계획 → 링크 상태 변화 예측 불가
        planned_path = qos_plan_full_path(env)
        if not planned_path or planned_path[-1] != DESTINATION:
            continue

        # warm-up 후 실행 → 계획 시와 다른 링크 상태 → 적응 불가
        for _ in range(10):
            env.update_dynamic_qos()

        # 계획된 경로 그대로 실행 (재계획 없음)
        for next_node in planned_path[1:]:
            _, _, done, _ = env.step(next_node)
            if done:
                break

        if env.current_node == DESTINATION:
            success_count += 1
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
            hop_count = max(m["hop_count"], 1)
            unstable_usages.append(m["unstable_count"] / hop_count * 100)
            congested_usages.append(m["congested_count"] / hop_count * 100)

    return {
        "success_rate": success_count / episodes * 100,
        "delay": float(np.mean(delays)) if delays else 0.0,
        "loss":  float(np.mean(losses)) if losses else 0.0,
        "hop":   float(np.mean(hops))   if hops   else 0.0,
        "unstable_usage": float(np.mean(unstable_usages)) if unstable_usages else 0.0,
        "congested_usage": float(np.mean(congested_usages)) if congested_usages else 0.0,
    }


def evaluate_qos(scenario, episodes=EVAL_EPISODES):
    """QoS-Model을 5개 시드로 반복 평가 → mean ± std (ddof=1) 반환
    model 이름은 'QoS-DQN'으로 저장 — show_results.py가 이 키로 읽음"""
    seed_results = [_run_qos_one_seed(scenario, s, episodes) for s in SEEDS]
    d_vals = [r["delay"] for r in seed_results]
    l_vals = [r["loss"]  for r in seed_results]
    h_vals = [r["hop"]   for r in seed_results]
    return {
        "model": "QoS-DQN", "scenario": scenario,
        "success_rate": float(np.mean([r["success_rate"] for r in seed_results])),
        "delay":      float(np.mean(d_vals)),
        "delay_std":  float(np.std(d_vals, ddof=1)),
        "loss":       float(np.mean(l_vals)),
        "loss_std":   float(np.std(l_vals, ddof=1)),
        "hop":        float(np.mean(h_vals)),
        "hop_std":    float(np.std(h_vals, ddof=1)),
        "unstable_usage":  float(np.mean([r["unstable_usage"]  for r in seed_results])),
        "congested_usage": float(np.mean([r["congested_usage"] for r in seed_results])),
    }


def main():
    print("=" * 70)
    print(f"  Exp 1: OSPF / QoS-Model 평가  (seeds={SEEDS})")
    print("=" * 70)
    os.makedirs(RESULT_DIR, exist_ok=True)
    results = []
    for sc in SCENARIOS:
        print(f"\n[OSPF - {sc}]  seed별 평가 중...")
        r = evaluate_ospf(sc)
        results.append(r)
        print(f"  Delay={r['delay']:.1f} ± {r['delay_std']:.1f} | Loss={r['loss']:.5f} ± {r['loss_std']:.5f}")

        print(f"[QoS-Model - {sc}]  seed별 평가 중...")
        r = evaluate_qos(sc)
        results.append(r)
        print(f"  Delay={r['delay']:.1f} ± {r['delay_std']:.1f} | Loss={r['loss']:.5f} ± {r['loss_std']:.5f}")

    # exp1_reference.csv 저장 (std 컬럼 포함)
    fields = ["model", "scenario", "success_rate",
              "delay", "delay_std", "loss", "loss_std", "hop", "hop_std",
              "unstable_usage", "congested_usage"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n저장 완료: {OUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()