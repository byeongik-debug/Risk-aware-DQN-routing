"""발표된 QoS 비교군의 핸디캡이 Risk-DQN 우위에 얼마를 보태는지 뺀다.

왜 필요한가. `FINDINGS_loss_metric.md` 4절에 두 가지를 적어뒀다.

  (1) 메인 표의 "QoS-DQN" 행에는 신경망이 없다 — `eval_qos.py:31`의 Dijkstra 한 번이다.
      `"QoS-DQN"`은 `show_results.py`가 찾는 딕셔너리 키일 뿐이다 (`eval_qos.py:196`).
  (2) 그 비교군은 **워밍업 전** t=0 상태로 경로를 계획하고, 워밍업 10스텝이 지난 뒤
      낡은 계획을 재계획 없이 실행한다 (`eval_qos.py:139-155`). OSPF와 Risk-DQN은
      둘 다 워밍업 **후**에 계획/행동한다. 즉 QoS만 자기가 겪지 않는 상태로 계획한다.

적어만 뒀고 **크기를 재지 않았다.** 리뷰어가 "같은 정보를 주면 어떻게 되냐"고 물으면
답할 수치가 없다. 이 스크립트가 그 수치를 만든다. 같은 환경·같은 시드·같은 200
에피소드·같은 워밍업으로 QoS 비교군의 다섯 변형을 돌린다:

    qos_t0_stale          워밍업 전 계획, 재계획 없음        (발표된 설계, 재현 확인용)
    qos_post_warmup       워밍업 후 계획, 재계획 없음        (다른 방법과 같은 정보)
    qos_replan            워밍업 후, 매 홉 재계획            (원본과 같은 규칙: 마스킹 없음)
    qos_replan_loopfree   워밍업 후, 매 홉 재계획 + 방문 제외 (가장 센 공정 기준선)
    qos_dqn               학습된 qos_model_final.pth         (평가된 적이 한 번도 없다)

평가 루프는 `reeval_loss_conventions.py`에서 import해서 쓴다 — 루프가 글자 하나까지
같다는 것이 이 비교의 전제다. `qos_t0_stale`이 커밋된 `exp1_reference.csv`를 재현하는지
assert로 확인하고 시작한다. 재현이 깨지면 나머지 비교도 믿을 수 없다.

`qos_dqn`의 상태 구성과 신경망은 `train_qos_dqn.py`에서 그대로 import한다 (그 파일은
`if __name__` 가드가 있어 import해도 학습이 돌지 않는다. 모듈 수준에서 시드를 42로
고정하지만 러너마다 다시 시드를 박으므로 영향이 없다).

지표는 홉·지연 + 손실 3규약에, **살아 있는 혼잡 플래그**(`q["is_congested"]`)를 센
것을 더했다. 원본 `calculate_path_metrics`의 `congested_count`는 시나리오별로 심어둔
정적 링크 집합 소속만 세므로 normal에서 항상 0이고, 홉으로 나눠 보고된다
(`eval_qos.py:84`) — 홉 정규화가 우위를 부풀린다는 1절 발견이 여기에도 걸린다.
그래서 원시 개수와 홉당 비율을 같이 적는다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_qos_baseline_handicap.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_qos_baseline_handicap.py --from-csv
"""

import argparse
import csv
import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import (  # noqa: E402
    fmt_p, holm_bonferroni, mean_std, min_detectable_effect, welch)
from topology import DESTINATION  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "qos_baseline_handicap.csv")
QOS_MODEL = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "qos_model_final.pth")

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
EPISODES = RLC.EVAL_EPISODES
WARMUP = RLC.WARMUP

# 비교할 지표. published/path_sum/compounded는 손실 3규약(`reeval_loss_conventions.py` 머리말).
METRICS = ("hop", "delay", "published", "path_sum", "compounded",
           "cong_live", "cong_live_per_hop")

# 작을수록 좋은 지표들 — 전부 그렇다(홉도 경로 길이라 짧은 게 낫다).
LOWER_IS_BETTER = set(METRICS)


def path_metrics(env, path):
    """RLC.loss_conventions에 살아 있는 혼잡 링크 수를 더한다."""
    out = dict(RLC.loss_conventions(env, path))
    live = 0
    for u, v in zip(path[:-1], path[1:]):
        e = env.normalize_edge(u, v)
        q = env.qos.get(e)
        if q is not None and q.get("is_congested", False):
            live += 1
    hop = max(len(path) - 1, 1)
    out["cong_live"] = float(live)
    out["cong_live_per_hop"] = live / hop * 100.0
    return out


def seed_mean(per_episode):
    if not per_episode:
        return {k: float("nan") for k in METRICS}
    return {k: float(np.mean([e[k] for e in per_episode])) for k in METRICS}


# ---------------------------------------------------------------- QoS 변형들

def qos_plan_excluding(env, blocked):
    """QoS 복합 비용 Dijkstra, 단 blocked에 든 노드는 경유하지 않는다.

    `RLC.qos_plan_full_path`와 비용 함수가 같다 — 다른 건 후보 노드 집합뿐이다.
    blocked가 비면 결과가 같아야 하고, 그걸 --verify로 확인한다.
    """
    import heapq

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
            q = env.qos.get(e, {"delay": 120.0, "bandwidth": 0.0, "loss": 0.35})
            d_n = float(np.clip(q["delay"] / 120.0, 0.0, 1.0))
            b_n = 1.0 - float(np.clip(q["bandwidth"] / 1000.0, 0.0, 1.0))
            l_n = float(np.clip(q["loss"] / 0.35, 0.0, 1.0))
            new_cost = cost + (d_n + b_n + l_n) / 3.0
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


def run_qos_t0_stale(scenario, seed, episodes, model=None):
    """발표된 설계. RLC.run_qos_seed를 그대로 쓰되 지표만 더 받는다."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        planned = RLC.qos_plan_full_path(env)   # 워밍업 전 상태로 계획
        if not planned or planned[-1] != DESTINATION:
            continue
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        for next_node in planned[1:]:
            _, _, done, _ = env.step(next_node)
            if done:
                break
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


def run_qos_post_warmup(scenario, seed, episodes, model=None):
    """같은 플래너, 계획 시점만 워밍업 뒤로 옮긴다 — OSPF·Risk-DQN과 같은 정보."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        planned = RLC.qos_plan_full_path(env)
        if not planned or planned[-1] != DESTINATION:
            continue
        for next_node in planned[1:]:
            _, _, done, _ = env.step(next_node)
            if done:
                break
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


def _run_qos_replan(scenario, seed, episodes, loopfree):
    """매 홉 재계획. loopfree면 이미 방문한 노드를 경유 후보에서 뺀다.

    loopfree=False는 원본 평가와 같은 규칙이다 — 방문 노드를 거르지 않으므로
    A->B->A 진동이 가능하고 MAX_STEPS에서 끊긴다. 그 실패율을 숨기지 않고 센다.
    """
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        done = False
        while not done:
            blocked = set(env.visited) - {env.current_node} if loopfree else set()
            planned = qos_plan_excluding(env, blocked)
            if len(planned) < 2:
                break
            _, _, done, _ = env.step(planned[1])
            if len(env.visited) >= MAX_STEPS:
                break
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


def run_qos_replan(scenario, seed, episodes, model=None):
    return _run_qos_replan(scenario, seed, episodes, loopfree=False)


def run_qos_replan_loopfree(scenario, seed, episodes, model=None):
    return _run_qos_replan(scenario, seed, episodes, loopfree=True)


def run_qos_dqn(scenario, seed, episodes, model=None):
    """학습된 QoS-DQN. 행동 선택은 RLC.run_dqn_seed와 같은 마스킹 greedy다."""
    import torch
    from train_qos_dqn import compute_qos_state

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            state = compute_qos_state(env)
            with torch.no_grad():
                q_values = model(torch.FloatTensor(state).unsqueeze(0).to(device))[0]
            masked = torch.full_like(q_values, -1e9)
            for a in valid:
                masked[a] = q_values[a]
            _, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


def run_ospf(scenario, seed, episodes, model=None):
    """워밍업 후 대역폭 Dijkstra (`eval_qos.py:74`)."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        path = env.ospf_path(cost_type="bandwidth")
        if not path or path[-1] != DESTINATION:
            continue
        out.append(path_metrics(env, path))
    return seed_mean(out), len(out) / episodes * 100


def run_risk_dqn(scenario, seed, episodes, model=None):
    """제안 방법. RLC.run_dqn_seed와 같은 루프 (use_risk=True, risk_window=5)."""
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario,
                         risk_window=RLC.RISK_WINDOW)
        state = env.reset()
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
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


# 순서가 출력 순서다. model_kind는 어떤 체크포인트를 넘길지 고른다.
VARIANTS = (
    ("ospf", run_ospf, None),
    ("qos_t0_stale", run_qos_t0_stale, None),
    ("qos_post_warmup", run_qos_post_warmup, None),
    ("qos_replan", run_qos_replan, None),
    ("qos_replan_loopfree", run_qos_replan_loopfree, None),
    ("qos_dqn", run_qos_dqn, "qos"),
    ("risk_dqn", run_risk_dqn, "risk"),
)


def load_qos_model():
    import torch
    from train_qos_dqn import QNetwork, QOS_STATE_SIZE

    if not os.path.exists(QOS_MODEL):
        raise SystemExit("QoS-DQN checkpoint not found: " + QOS_MODEL)
    device = torch.device("cpu")
    probe = NetworkEnv(use_risk=False, scenario="normal")
    model = QNetwork(QOS_STATE_SIZE, probe.action_size).to(device)
    model.load_state_dict(torch.load(QOS_MODEL, map_location=device,
                                     weights_only=False))
    model.eval()
    return model


def verify_planner():
    """blocked가 비면 qos_plan_excluding이 원본 플래너와 같은 경로를 내야 한다."""
    np.random.seed(0)
    random.seed(0)
    for scenario in SCENARIOS:
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        a = RLC.qos_plan_full_path(env)
        b = qos_plan_excluding(env, set())
        assert a == b, "플래너 재구현이 원본과 다르다: {} vs {}".format(a, b)
    print("  플래너 동치 확인: 4 시나리오 전부 동일 경로")


def published_reference():
    """커밋된 exp1_reference.csv의 QoS 행을 (시나리오 -> 지표) 로 읽는다."""
    path = os.path.join(BASE_DIR, "..", "SDN_DQN_new_env", "results",
                        "exp1_reference.csv")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("model") == "QoS-DQN":
                out[row["scenario"]] = {
                    "delay": float(row["delay"]),
                    "loss": float(row["loss"]),
                    "hop": float(row["hop"]),
                }
    return out


# ---------------------------------------------------------------- 실행

def run_all():
    rows = []
    models = {}
    if any(kind for _, _, kind in VARIANTS):
        models["risk"] = RLC.load_risk_model()
        models["qos"] = load_qos_model()

    for name, runner, kind in VARIANTS:
        model = models.get(kind)
        for scenario in SCENARIOS:
            for seed in SEEDS:
                stats, success = runner(scenario, seed, EPISODES, model)
                row = {"scenario": scenario, "variant": name, "seed": seed,
                       "success_rate": success}
                row.update(stats)
                rows.append(row)
            got = [r for r in rows if r["variant"] == name
                   and r["scenario"] == scenario]
            print("  {:<20} {:<11} 성공 {:5.1f}%  홉 {:5.2f}  지연 {:7.1f}  "
                  "compounded {:.4f}".format(
                      name, scenario,
                      float(np.mean([r["success_rate"] for r in got])),
                      float(np.nanmean([r["hop"] for r in got])),
                      float(np.nanmean([r["delay"] for r in got])),
                      float(np.nanmean([r["compounded"] for r in got]))))

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["scenario", "variant", "seed", "success_rate"] + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n저장: {} ({}행)".format(OUT_CSV, len(rows)))
    return rows


def load_rows():
    if not os.path.exists(OUT_CSV):
        raise SystemExit("먼저 --from-csv 없이 실행해 CSV를 만들어야 한다: " + OUT_CSV)
    rows = []
    with open(OUT_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out = {"scenario": row["scenario"], "variant": row["variant"],
                   "seed": int(row["seed"]),
                   "success_rate": float(row["success_rate"])}
            for k in METRICS:
                out[k] = float(row[k])
            rows.append(out)
    return rows


def pick(rows, scenario, variant, metric):
    return [r[metric] for r in rows
            if r["scenario"] == scenario and r["variant"] == variant
            and not np.isnan(r[metric])]


# ---------------------------------------------------------------- 보고

def report_reproduction(rows):
    """발표된 설계를 재현했는지부터 확인한다. 깨지면 나머지 비교가 무의미하다."""
    ref = published_reference()
    print("\n" + "=" * 78)
    print("A. 발표된 QoS 행 재현 확인 (qos_t0_stale vs exp1_reference.csv)")
    print("=" * 78)
    if not ref:
        print("  exp1_reference.csv 없음 — 재현 확인 생략")
        return
    print("  {:<11} {:>10} {:>10} {:>9} {:>10} {:>10} {:>8}".format(
        "시나리오", "지연(발표)", "지연(재현)", "차이", "손실(발표)", "손실(재현)", "차이"))
    worst = 0.0
    for scenario in SCENARIOS:
        if scenario not in ref:
            continue
        d_mine = float(np.mean(pick(rows, scenario, "qos_t0_stale", "delay")))
        l_mine = float(np.mean(pick(rows, scenario, "qos_t0_stale", "published")))
        d_ref = ref[scenario]["delay"]
        l_ref = ref[scenario]["loss"]
        d_pct = (d_mine - d_ref) / d_ref * 100 if d_ref else 0.0
        l_pct = (l_mine - l_ref) / l_ref * 100 if l_ref else 0.0
        worst = max(worst, abs(d_pct), abs(l_pct))
        print("  {:<11} {:>10.2f} {:>10.2f} {:>8.2f}% {:>10.5f} {:>10.5f} "
              "{:>7.2f}%".format(scenario, d_ref, d_mine, d_pct,
                                 l_ref, l_mine, l_pct))
    print("\n  최대 편차 {:.2f}% — 0에 가까우면 루프가 발표된 것과 같다.".format(worst))


def report_handicap(rows):
    """계획 시점만 바꿨을 때 QoS 비교군 자체가 얼마나 좋아지는지."""
    print("\n" + "=" * 78)
    print("B. 핸디캡의 크기 — 같은 플래너, 계획 시점만 워밍업 뒤로")
    print("=" * 78)
    print("  {:<11} {:>9} {:>9} {:>9}   {:>9} {:>9} {:>9}".format(
        "시나리오", "지연(t0)", "지연(후)", "개선", "복합(t0)", "복합(후)", "개선"))
    for scenario in SCENARIOS:
        a_d = float(np.mean(pick(rows, scenario, "qos_t0_stale", "delay")))
        b_d = float(np.mean(pick(rows, scenario, "qos_post_warmup", "delay")))
        a_c = float(np.mean(pick(rows, scenario, "qos_t0_stale", "compounded")))
        b_c = float(np.mean(pick(rows, scenario, "qos_post_warmup", "compounded")))
        print("  {:<11} {:>9.1f} {:>9.1f} {:>8.1f}%   {:>9.4f} {:>9.4f} "
              "{:>8.1f}%".format(scenario, a_d, b_d,
                                 (a_d - b_d) / a_d * 100 if a_d else 0.0,
                                 a_c, b_c,
                                 (a_c - b_c) / a_c * 100 if a_c else 0.0))
    print("\n  개선이 크면 발표된 비교군은 '적응 못 하는' 게 아니라 '볼 수 없게 만든' 것이다.")


def report_success(rows):
    print("\n" + "=" * 78)
    print("C. 변형별 목적지 도달률 (재계획이 진동을 일으키는지)")
    print("=" * 78)
    header = "  {:<20}".format("변형") + "".join(
        "{:>12}".format(s) for s in SCENARIOS)
    print(header)
    for name, _, _ in VARIANTS:
        line = "  {:<20}".format(name)
        for scenario in SCENARIOS:
            vals = [r["success_rate"] for r in rows
                    if r["scenario"] == scenario and r["variant"] == name]
            line += "{:>11.1f}%".format(float(np.mean(vals)) if vals else float("nan"))
        print(line)


def report_vs_risk(rows):
    """Risk-DQN 대 각 QoS 변형. 시드가 단위(n=5), Welch + Holm + MDE."""
    print("\n" + "=" * 78)
    print("D. Risk-DQN 대 각 QoS 변형 (시드 단위 n=5, Welch, Holm 보정)")
    print("=" * 78)
    print("  MDE는 이 분산·n에서 유의해질 수 있는 최소 차이다. 관측 < MDE면")
    print("  '차이 없음'이 아니라 '이 검정력으로는 못 가른다'로 읽어야 한다.\n")

    for scenario in SCENARIOS:
        risk = {m: pick(rows, scenario, "risk_dqn", m) for m in METRICS}
        tests = []
        for name, _, _ in VARIANTS:
            if name in ("risk_dqn", "ospf"):
                continue
            for metric in METRICS:
                base = pick(rows, scenario, name, metric)
                if len(base) < 2 or len(risk[metric]) < 2:
                    continue
                diff, _, _, p = welch(base, risk[metric])
                mde = min_detectable_effect(base, risk[metric])
                b_m, _ = mean_std(base)
                r_m, _ = mean_std(risk[metric])
                tests.append({"variant": name, "metric": metric, "p": p,
                              "base": b_m, "risk": r_m, "mde": mde,
                              "diff": diff})
        if not tests:
            continue
        # 결정론적 정책은 시드 간 분산이 0이 되어 표준오차가 0 -> p가 nan이다.
        # holm_bonferroni가 nan을 건너뛰므로 보정 자체는 안전하고, 검정 불가 건수를 따로 센다.
        adjusted = holm_bonferroni([x["p"] for x in tests])
        for x, p_adj in zip(tests, adjusted):
            x["p_holm"] = p_adj
            x["sig"] = bool(p_adj == p_adj and p_adj < 0.05)

        untestable = sum(1 for x in tests if x["p"] != x["p"])
        print("[{}]  유의 {}/{} (Holm){}".format(
            scenario, sum(1 for x in tests if x["sig"]), len(tests),
            "  / 분산 0으로 검정 불가 {}건".format(untestable) if untestable else ""))
        print("  {:<20} {:<18} {:>9} {:>9} {:>9} {:>9} {:>10} {:>3}".format(
            "변형", "지표", "변형값", "Risk-DQN", "차이", "MDE", "p_holm", "유의"))
        for x in tests:
            better = "+" if (x["diff"] > 0) == (x["metric"] in LOWER_IS_BETTER) else "-"
            print("  {:<20} {:<18} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>10} "
                  "{:>3}".format(
                      x["variant"], x["metric"], x["base"], x["risk"], x["diff"],
                      x["mde"], fmt_p(x["p_holm"]),
                      ("*" + better) if x["sig"] else ""))
        print()


def report_shrinkage(rows):
    """발표된 비교군 대비 우위가 공정한 비교군에서 얼마나 줄어드는지 — 핵심 표."""
    print("=" * 78)
    print("E. 우위 축소 — 발표된 비교군 대비 vs 가장 센 공정 비교군 대비")
    print("=" * 78)
    print("  상대 우위(%) = (QoS변형 - Risk-DQN) / QoS변형 × 100, 클수록 Risk-DQN 유리\n")
    show = ("delay", "published", "compounded", "hop", "cong_live")
    strong = "qos_replan_loopfree"
    print("  {:<11} {:<13} {:>10} {:>10} {:>10} {:>10} {:>9}".format(
        "시나리오", "지표", "t0(발표)", "워밍업후", "재계획", "loopfree", "축소pp"))
    for scenario in SCENARIOS:
        for metric in show:
            risk = float(np.mean(pick(rows, scenario, "risk_dqn", metric)))
            vals = {}
            for name in ("qos_t0_stale", "qos_post_warmup", "qos_replan", strong):
                base = pick(rows, scenario, name, metric)
                b = float(np.mean(base)) if base else float("nan")
                vals[name] = (b - risk) / b * 100 if b else float("nan")
            print("  {:<11} {:<13} {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}% "
                  "{:>8.1f}".format(
                      scenario, metric, vals["qos_t0_stale"],
                      vals["qos_post_warmup"], vals["qos_replan"], vals[strong],
                      vals["qos_t0_stale"] - vals[strong]))
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true",
                        help="재실행 없이 저장된 CSV로 분석만 한다")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    if args.from_csv:
        rows = load_rows()
    else:
        if not args.skip_verify:
            verify_planner()
        print("\n시드 {}  에피소드 {}  워밍업 {}\n".format(SEEDS, EPISODES, WARMUP))
        rows = run_all()

    report_reproduction(rows)
    report_handicap(rows)
    report_success(rows)
    report_vs_risk(rows)
    report_shrinkage(rows)


if __name__ == "__main__":
    main()
