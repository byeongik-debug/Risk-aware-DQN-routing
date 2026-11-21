"""살아남은 주장 하나를 의심한다 — OSPF 대비 우위도 재계획 비대칭을 쓰고 있는가.

`reeval_qos_baseline_handicap.py`가 보인 것: QoS 비교군 대비 우위의 95%는 그 비교군이
**자기가 평가받는 상태를 보지 못해서** 생긴 것이었다. 계획 시점만 옮기니 사라졌다.
그 실험에서 유일하게 멀쩡하게 남은 것이 **OSPF 대비 36~56%**다.

그런데 비대칭이 완전히 닫힌 게 아니다 (`FINDINGS_loss_metric.md` 5절 표):

    OSPF      워밍업 후 계획  O     매 홉 재결정  X   (`eval_qos.py:74`)
    Risk-DQN  워밍업 후 행동  O     매 홉 재결정  O   (`eval_baseline.py:56-75`)

OSPF는 계획 시점은 공정하다. 하지만 경로를 **한 번 깔고 끝**이고, Risk-DQN은 홉마다
다시 고른다. QoS 쪽에서는 재계획을 더 줘도 값이 거의 안 변했다(0.3968 -> 0.3962).
그렇다고 OSPF도 같을 거라고 **가정하면 안 된다** — OSPF의 비용은 대역폭뿐이고, 대역폭은
이용률에 따라 매 스텝 바뀐다. 재계획이 QoS보다 더 먹힐 여지가 있다. 재서 확인한다.

돌리는 변형:

    ospf_static               워밍업 후 대역폭 Dijkstra 한 번        (발표된 설계, 재현 확인용)
    ospf_replan               매 홉 재계획, 방문 노드 안 거름        (원본과 같은 규칙)
    ospf_replan_loopfree      매 홉 재계획 + 방문 노드 제외          (공정 비교의 상한)
    ospf_delay_static         워밍업 후 지연 Dijkstra 한 번          (비용 함수만 바꿈)
    ospf_delay_replan_loopfree  지연 비용 + 매 홉 재계획             (최단경로 계열의 최강)

뒤의 둘은 질문이 다르다. 앞의 셋은 "적응을 허용하면 어떻게 되냐"를 재고, 이것은
"비용 함수 선택이 얼마를 설명하냐"를 잰다. `cost_type="bandwidth"`는 Cisco OSPF의
기본값(기준대역폭/대역폭)이라 발표된 선택 자체는 방어 가능하다 — 다만 그 비용이
손실률을 아예 안 보므로, 손실 지표에서 불리한 게 당연하다는 반론이 가능하다.
그 반론의 크기를 미리 알아두자는 것이다.

`ospf_delay_static`이 있어야 비용 함수 효과가 재계획 효과와 분리된다. `ospf_static`과
재계획 조건이 똑같고 비용만 다르므로, 둘의 차이는 전부 비용 함수에서 온다. 재계획이
대역폭 비용에서 무의미했으니 지연 비용에서도 그럴 거라고 **가정하지 않기 위해** 넣는다.

Risk-DQN과 공정 QoS는 **다시 돌리지 않는다.** `review_results/qos_baseline_handicap.csv`에
같은 환경·같은 시드·같은 200에피소드로 이미 있고, 그 실행은 커밋된 `exp1_reference.csv`를
편차 0.00%로 재현했다. 여기서는 `ospf_static`이 그 CSV의 `ospf` 행을 재현하는지를
assert로 확인해서 두 실행이 같은 루프라는 것만 보인다. 재현이 깨지면 비교를 중단한다.

플래너는 직접 Dijkstra를 쓴다 — `env.ospf_path`가 항상 SOURCE에서 출발해서 중간 노드에서
재계획을 할 수 없기 때문이다. 재구현이 원본과 같은지는 `--verify`가 확인한다(출발 노드에서
blocked가 비면 같은 경로 또는 같은 비용이어야 한다).

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_ospf_replan.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_ospf_replan.py --from-csv
"""

import argparse
import csv
import heapq
import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
import reeval_qos_baseline_handicap as QBH  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import (  # noqa: E402
    fmt_p, holm_bonferroni, mean_std, min_detectable_effect, welch)
from topology import DESTINATION, SOURCE  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "ospf_replan.csv")
REF_CSV = os.path.join(BASE_DIR, "review_results", "qos_baseline_handicap.csv")

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
EPISODES = RLC.EVAL_EPISODES
WARMUP = RLC.WARMUP

# 지표 정의와 path_metrics를 QBH에서 그대로 가져온다 — 두 CSV를 나란히 읽으려면
# 지표가 같은 함수에서 나와야 한다.
METRICS = QBH.METRICS
path_metrics = QBH.path_metrics
seed_mean = QBH.seed_mean

# 재현 허용 편차. 같은 루프·같은 시드면 0.00%여야 한다.
REPRO_TOL_PCT = 0.05


# ------------------------------------------------------------------- 플래너

def ospf_cost(q, cost_type):
    """`network_env.py:462-479`의 비용 함수와 같다."""
    if cost_type == "bandwidth":
        return 1000.0 / max(q["bandwidth"], 1e-6)
    if cost_type == "delay":
        return q["delay"]
    if cost_type == "hop":
        return 1.0
    raise ValueError("모르는 cost_type: " + str(cost_type))


def ospf_plan_excluding(env, blocked, cost_type):
    """현재 노드에서 출발하는 최단경로. blocked에 든 노드는 경유하지 않는다.

    `env.ospf_path`는 SOURCE 고정이라 중간 노드에서 못 쓴다. 비용 함수는 같다.
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
            new_cost = cost + ospf_cost(q, cost_type)
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


def path_cost(env, path, cost_type):
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        total += ospf_cost(env.qos[env.normalize_edge(u, v)], cost_type)
    return total


def verify_planner():
    """출발 노드에서 blocked가 비면 재구현이 `env.ospf_path`와 같아야 한다.

    경로가 다를 수 있는 유일한 정당한 이유는 동률이다. 그래서 경로 일치와 비용 일치를
    따로 센다. 비용이 다르면 재구현이 틀린 것이므로 assert로 멈춘다.
    """
    same_path = diff_path = 0
    for cost_type in ("bandwidth", "delay"):
        for scenario in SCENARIOS:
            for seed in SEEDS[:2]:
                np.random.seed(seed)
                random.seed(seed)
                for _ in range(5):
                    env = NetworkEnv(use_risk=False, scenario=scenario)
                    env.reset()
                    for _ in range(WARMUP):
                        env.update_dynamic_qos()
                    assert env.current_node == SOURCE
                    a = env.ospf_path(cost_type=cost_type)
                    b = ospf_plan_excluding(env, set(), cost_type)
                    if a == b:
                        same_path += 1
                        continue
                    diff_path += 1
                    ca, cb = path_cost(env, a, cost_type), path_cost(env, b, cost_type)
                    assert abs(ca - cb) < 1e-9, (
                        "플래너 재구현이 원본보다 나쁜 경로를 낸다: "
                        "{} cost={:.6f} vs {} cost={:.6f} ({}, {}, seed {})".format(
                            a, ca, b, cb, cost_type, scenario, seed))
    print("[검증] 플래너 재구현 = 원본: 경로 일치 {}건, 동률로 경로만 다름 {}건".format(
        same_path, diff_path))


# ------------------------------------------------------------------- 러너들

def _run_ospf_static(scenario, seed, episodes, cost_type):
    """워밍업 후 한 번 계획하고 끝. 발표된 OSPF 루프(`eval_qos.py:74`)와 같다."""
    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        path = env.ospf_path(cost_type=cost_type)
        if not path or path[-1] != DESTINATION:
            continue
        out.append(path_metrics(env, path))
    return seed_mean(out), len(out) / episodes * 100


def run_ospf_static(scenario, seed, episodes):
    """발표된 설계. `QBH.run_ospf`와 같은 루프여야 하고, 재현 assert가 그걸 확인한다."""
    return _run_ospf_static(scenario, seed, episodes, "bandwidth")


def run_ospf_delay_static(scenario, seed, episodes):
    """비용만 지연으로 바꿨다. ospf_static과의 차이는 전부 비용 함수에서 온다."""
    return _run_ospf_static(scenario, seed, episodes, "delay")


def _run_ospf_replan(scenario, seed, episodes, loopfree, cost_type):
    """매 홉 재계획. 환경을 실제로 step해서 상태가 흐르게 둔다.

    loopfree=False는 원본 평가와 같은 규칙이다(방문 노드를 안 거른다). 진동하면
    MAX_STEPS에서 끊기고, 그 실패를 도달률로 드러낸다.
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
            planned = ospf_plan_excluding(env, blocked, cost_type)
            if len(planned) < 2:
                break
            _, _, done, _ = env.step(planned[1])
            if len(env.visited) >= MAX_STEPS:
                break
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
    return seed_mean(out), len(out) / episodes * 100


def run_ospf_replan(scenario, seed, episodes):
    return _run_ospf_replan(scenario, seed, episodes, False, "bandwidth")


def run_ospf_replan_loopfree(scenario, seed, episodes):
    return _run_ospf_replan(scenario, seed, episodes, True, "bandwidth")


def run_ospf_delay_replan_loopfree(scenario, seed, episodes):
    return _run_ospf_replan(scenario, seed, episodes, True, "delay")


VARIANTS = (
    ("ospf_static", run_ospf_static),
    ("ospf_replan", run_ospf_replan),
    ("ospf_replan_loopfree", run_ospf_replan_loopfree),
    ("ospf_delay_static", run_ospf_delay_static),
    ("ospf_delay_replan_loopfree", run_ospf_delay_replan_loopfree),
)

# 비교 상대. 다시 돌리지 않고 검증된 CSV에서 읽는다.
FROM_REF = ("ospf", "qos_post_warmup", "risk_dqn")


# --------------------------------------------------------------- 입출력/집계

def measure():
    rows = []
    for scenario in SCENARIOS:
        for name, fn in VARIANTS:
            for seed in SEEDS:
                means, success = fn(scenario, seed, EPISODES)
                row = {"scenario": scenario, "variant": name, "seed": seed,
                       "success_rate": success}
                row.update(means)
                rows.append(row)
                print("  {:<10} {:<28} seed {}  성공 {:5.1f}%  "
                      "compounded {:.4f}  홉 {:.2f}".format(
                          scenario, name, seed, success,
                          means["compounded"], means["hop"]), flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["scenario", "variant", "seed", "success_rate"] + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("\n저장: {} ({}행)".format(OUT_CSV, len(rows)))
    return rows


def load_csv(path):
    if not os.path.exists(path):
        raise SystemExit("CSV가 없다: " + path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["seed"] = int(r["seed"])
        for k in ("success_rate",) + tuple(METRICS):
            r[k] = float(r[k])
    return rows


def by_variant(rows, scenario):
    """시나리오 하나에서 변형 -> 지표 -> 시드별 값 리스트."""
    out = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        d = out.setdefault(r["variant"], {k: [] for k in ("success_rate",) + tuple(METRICS)})
        for k in d:
            d[k].append(r[k])
    return out


# ------------------------------------------------------------------- 보고

def report_metric_information(episodes=40, seeds=5):
    """비용 함수가 왜 전부를 설명하는지 — 링크 단위 상관을 직접 센다.

    메커니즘을 단정하지 않기 위한 측정이다. 두 가지를 본다.

      r(bandwidth, loss)  발표된 OSPF 비용 1000/bandwidth가 손실을 어느 방향으로 보는가.
                          양수면 비용이 작은(=선호되는) 고대역폭 링크가 손실도 크다는
                          뜻이고, 그 기준선은 혼잡을 못 피하는 게 아니라 혼잡으로 간다.
      r(delay, loss)      지연이 손실의 대리변수로 얼마나 충분한가. 1에 가까우면
                          지연만 보는 최단경로가 손실도 같이 줄이게 된다.

    `network_env.py:227-229`에서 이용률은 delay·loss만 올리고 bandwidth에는 bw_noise만
    더한다 — 구조적으로 그렇게 되어 있다. 그 구조가 수치로 얼마인지를 적는다.
    """
    print("\n" + "=" * 78)
    print("F. 왜 비용 함수가 전부를 설명하는가 — 링크 단위 상관 (워밍업 후 상태)")
    print("=" * 78)
    print("  {:<12}{:>8}{:>14}{:>14}{:>16}{:>14}".format(
        "시나리오", "n", "r(bw,util)", "r(bw,loss)", "r(delay,loss)", "r(delay,util)"))
    for scenario in SCENARIOS:
        bw, ut, ls, dl = [], [], [], []
        for seed in range(seeds):
            np.random.seed(seed)
            random.seed(seed)
            for _ in range(episodes):
                env = NetworkEnv(use_risk=False, scenario=scenario)
                env.reset()
                for _ in range(WARMUP):
                    env.update_dynamic_qos()
                for q in env.qos.values():
                    bw.append(q["bandwidth"])
                    ut.append(q["utilization"])
                    ls.append(q["loss"])
                    dl.append(q["delay"])

        def r(a, b):
            if np.std(a) < 1e-12 or np.std(b) < 1e-12:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        print("  {:<12}{:>8}{:>14.3f}{:>14.3f}{:>16.3f}{:>14.3f}".format(
            scenario, len(bw), r(bw, ut), r(bw, ls), r(dl, ls), r(dl, ut)))
    print("\n  읽는 법: r(bw,loss)가 양수면 발표된 OSPF 비용(1000/bandwidth)이 손실 큰")
    print("  링크를 선호한다는 뜻이다. r(delay,loss)가 1에 가까우면 지연만 보는")
    print("  최단경로가 위험 모델 없이도 손실을 피하게 된다.")


def report(rows, ref_rows):
    print("\n" + "=" * 78)
    print("A. 재현 확인 — ospf_static이 qos_baseline_handicap.csv의 ospf와 같은가")
    print("=" * 78)
    worst = 0.0
    for scenario in SCENARIOS:
        mine = by_variant(rows, scenario)["ospf_static"]
        theirs = by_variant(ref_rows, scenario)["ospf"]
        line = "  {:<11}".format(scenario)
        for k in ("hop", "delay", "compounded"):
            a, b = float(np.mean(mine[k])), float(np.mean(theirs[k]))
            dev = abs(a - b) / abs(b) * 100 if b else 0.0
            worst = max(worst, dev)
            line += "  {} {:.4f} vs {:.4f} ({:+.2f}%)".format(k, a, b, dev)
        print(line)
    assert worst <= REPRO_TOL_PCT, (
        "재현 실패: 최대 편차 {:.3f}% > {:.3f}%. 두 실행이 같은 루프가 아니므로 "
        "이하 비교를 믿을 수 없다.".format(worst, REPRO_TOL_PCT))
    print("  -> 최대 편차 {:.3f}% — 같은 루프다. 비교 진행.".format(worst))

    print("\n" + "=" * 78)
    print("B. 재계획을 허용하면 OSPF가 얼마나 좋아지는가 (compounded = 1-prod(1-p))")
    print("=" * 78)
    names = [n for n, _ in VARIANTS]
    print("  {:<11}".format("시나리오")
          + "".join("{:>14}".format(n.replace("ospf_", "")[:13]) for n in names)
          + "{:>12}{:>12}".format("공정QoS", "Risk-DQN"))
    for scenario in SCENARIOS:
        g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
        line = "  {:<11}".format(scenario)
        for n in names:
            line += "{:>14.4f}".format(float(np.mean(g[n]["compounded"])))
        line += "{:>12.4f}{:>12.4f}".format(
            float(np.mean(gr["qos_post_warmup"]["compounded"])),
            float(np.mean(gr["risk_dqn"]["compounded"])))
        print(line)

    print("\n  도달률(%)")
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        print("  {:<11}".format(scenario)
              + "".join("{:>14.1f}".format(float(np.mean(g[n]["success_rate"])))
                        for n in names))

    print("\n" + "=" * 78)
    print("C. 보고된 OSPF 대비 우위 중 재계획 비대칭이 설명하는 몫")
    print("=" * 78)
    print("  몫 = (격차_발표 - 격차_공정) / 격차_발표.  격차 = 기준선 - RiskDQN")
    for metric in ("compounded", "path_sum", "delay", "hop", "cong_live"):
        print("\n  [{}]".format(metric))
        print("    {:<11}{:>12}{:>12}{:>12}{:>10}{:>10}".format(
            "시나리오", "OSPF발표", "OSPF공정", "Risk-DQN", "발표대비", "공정대비"))
        for scenario in SCENARIOS:
            g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
            pub = float(np.mean(g["ospf_static"][metric]))
            fair = float(np.mean(g["ospf_replan_loopfree"][metric]))
            risk = float(np.mean(gr["risk_dqn"][metric]))
            imp_pub = (pub - risk) / pub * 100 if pub else float("nan")
            imp_fair = (fair - risk) / fair * 100 if fair else float("nan")
            share = ((pub - risk) - (fair - risk)) / (pub - risk) * 100 \
                if abs(pub - risk) > 1e-12 else float("nan")
            print("    {:<11}{:>12.4f}{:>12.4f}{:>12.4f}{:>9.1f}%{:>9.1f}%   "
                  "(핸디캡 몫 {:.1f}%)".format(
                      scenario, pub, fair, risk, imp_pub, imp_fair, share))

    print("\n" + "=" * 78)
    print("D. Risk-DQN 대 각 OSPF 변형 — Welch + Holm, 최소검출효과 병기")
    print("=" * 78)
    tests = []
    for scenario in SCENARIOS:
        g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
        risk = gr["risk_dqn"]
        for n in names:
            for metric in METRICS:
                base = g[n][metric]
                diff, _, _, p = welch(base, risk[metric])
                mde = min_detectable_effect(base, risk[metric])
                bm = float(np.mean(base))
                tests.append({
                    "scenario": scenario, "variant": n, "metric": metric,
                    "base": bm, "risk": float(np.mean(risk[metric])),
                    "diff": diff, "rel": diff / bm * 100 if bm else float("nan"),
                    "mde": mde, "mde_rel": mde / bm * 100 if bm else float("nan"),
                    "p": p})

    # 결정론적 정책은 시드 간 분산이 0이 되어 표준오차가 0 -> p가 nan이다.
    # holm_bonferroni가 nan을 건너뛰므로 보정 자체는 안전하고, 검정 불가 건수를 따로 센다.
    adjusted = holm_bonferroni([x["p"] for x in tests])
    for x, p_adj in zip(tests, adjusted):
        x["p_holm"] = p_adj
        x["sig"] = bool(p_adj == p_adj and p_adj < 0.05)
    untestable = sum(1 for x in tests if x["p"] != x["p"])

    for scenario in SCENARIOS:
        print("\n  {}".format(scenario))
        print("    {:<28}{:<20}{:>10}{:>9}{:>10}".format(
            "기준선", "지표", "상대우위", "MDE", "p_holm"))
        for x in tests:
            if x["scenario"] != scenario or x["metric"] not in (
                    "compounded", "delay", "hop", "cong_live", "published"):
                continue
            print("    {:<28}{:<20}{:>9.1f}%{:>8.1f}%{:>10}{}".format(
                x["variant"], x["metric"], x["rel"], x["mde_rel"],
                fmt_p(x["p_holm"]), "  *" if x["sig"] else ""))

    sig = sum(1 for x in tests if x["sig"])
    print("\n  유의 {}/{}건 (Holm 보정 후), 검정 불가(분산 0) {}건".format(
        sig, len(tests), untestable))
    print("  주의: 시드당 {}에피소드를 평균하므로 시드 간 분산이 거의 0이다. "
          "유의성은 이 해상도에서 판단에 쓸 수 없고 크기로만 읽어야 한다.".format(EPISODES))

    print("\n" + "=" * 78)
    print("E. 두 축을 분리한다 — 비용 함수(대역폭/지연) x 재계획(없음/매홉)")
    print("=" * 78)
    print("  ospf_static과 ospf_delay_static은 재계획 조건이 같으므로 둘의 차이는")
    print("  전부 비용 함수에서 온다. 같은 열 안의 차이가 재계획 효과다.")
    for metric in ("compounded", "delay", "hop", "cong_live"):
        print("\n  [{}]".format(metric))
        print("    {:<11}{:>12}{:>12}{:>12}{:>12}{:>12}".format(
            "시나리오", "대역폭1회", "대역폭매홉", "지연1회", "지연매홉", "Risk-DQN"))
        for scenario in SCENARIOS:
            g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
            vals = [float(np.mean(g[n][metric])) for n in
                    ("ospf_static", "ospf_replan_loopfree",
                     "ospf_delay_static", "ospf_delay_replan_loopfree")]
            risk = float(np.mean(gr["risk_dqn"][metric]))
            print("    {:<11}".format(scenario)
                  + "".join("{:>12.4f}".format(v) for v in vals)
                  + "{:>12.4f}".format(risk))
            best = min(vals)
            print("      {:<9}비용 함수 효과 {:+.1f}%  재계획 효과 {:+.1f}%(대역폭) "
                  "{:+.1f}%(지연)  |  최강 최단경로 대비 Risk-DQN {:+.1f}%".format(
                      "", (vals[0] - vals[2]) / vals[0] * 100 if vals[0] else float("nan"),
                      (vals[0] - vals[1]) / vals[0] * 100 if vals[0] else float("nan"),
                      (vals[2] - vals[3]) / vals[2] * 100 if vals[2] else float("nan"),
                      (best - risk) / best * 100 if best else float("nan")))

    report_metric_information()

    print("\n" + "=" * 78)
    print("정리")
    print("=" * 78)
    worst_share = []
    for scenario in SCENARIOS:
        g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
        pub = float(np.mean(g["ospf_static"]["compounded"]))
        fair = float(np.mean(g["ospf_replan_loopfree"]["compounded"]))
        risk = float(np.mean(gr["risk_dqn"]["compounded"]))
        if abs(pub - risk) > 1e-12:
            worst_share.append((scenario, ((pub - risk) - (fair - risk)) / (pub - risk) * 100,
                                (pub - risk) / pub * 100, (fair - risk) / fair * 100))
    for scenario, share, imp_pub, imp_fair in worst_share:
        print("  {:<11} 발표 {:5.1f}% -> 공정 {:5.1f}% 개선 (재계획 비대칭 몫 {:5.1f}%)".format(
            scenario, imp_pub, imp_fair, share))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true",
                        help="측정을 건너뛰고 저장된 CSV로 보고만 다시 만든다")
    parser.add_argument("--verify", action="store_true",
                        help="플래너 재구현이 원본과 같은지만 확인하고 끝낸다")
    args = parser.parse_args()

    if args.verify:
        verify_planner()
        return

    ref_rows = load_csv(REF_CSV)
    for name in FROM_REF:
        assert any(r["variant"] == name for r in ref_rows), \
            "참조 CSV에 {} 행이 없다: {}".format(name, REF_CSV)

    if args.from_csv:
        rows = load_csv(OUT_CSV)
    else:
        verify_planner()
        print("\n측정 시작 — {}변형 x {}시나리오 x {}시드 x {}에피소드".format(
            len(VARIANTS), len(SCENARIOS), len(SEEDS), EPISODES))
        rows = measure()

    report(rows, ref_rows)


if __name__ == "__main__":
    main()
