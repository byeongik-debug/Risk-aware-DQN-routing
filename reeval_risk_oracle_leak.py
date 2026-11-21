"""위험 점수에 섞인 정답 정보가 라우팅에 얼마를 보태는가.

`risk_calculator.calculate_link_risk`의 다섯 요소 중 네 개는 관측 가능한 텔레메트리다
(이용률, 혼잡 플래그, 지연 표준편차, 손실 표준편차). 다섯 번째 `link_type`만 다르다 —
`network_env.py:193-214`가 거기에 **시나리오가 심어둔 집합의 소속을 넣는다**:

    e in CONGESTION_LINKS       -> risk_type = "congested"            -> 0.8
    e in UNSTABLE_LINKS         -> risk_type = "unstable"             -> 0.9
    e in dynamic_danger_set     -> risk_type = "unstable_congested"   -> 1.0
    그 외                        -> 토폴로지 타입(backbone/backup/normal) -> 0.05/0.3/0.15

그리고 공식이 끝난 뒤 환경이 한 번 더 곱한다 (`network_env.py:244-248`):

    congestion·unstable에서 backup 링크      -> risk x 0.60
    dynamic에서 dynamic_support_set 링크     -> risk x 0.55

둘 다 실제 컨트롤러가 알 수 없는 정보다. 심어둔 집합의 소속은 실험 설계자만 안다.

크기는 이미 재뒀다 (`FINDINGS` 참조). 나쁜 링크와 나머지의 risk 차이에서 라벨 항이
차지하는 몫은 congestion 8.6%, unstable 15.0%고, 지배 신호는 관측 가능한 이용률이다.
**하지만 분리도가 작다는 것과 라우팅이 안 바뀐다는 것은 다른 얘기다.** backup 링크를
40% 깎아주는 보정은 하필 "우회로를 타야 하는" 시나리오에서만 걸리므로, 분리 통계에는
거의 안 보이면서 경로 선택은 바꿀 수 있다. 그래서 경로를 직접 돌려 본다.

2x2로 끈다:

    risk_published    심은 라벨 + 시나리오 보정   (= 환경의 q["risk"], 발표된 그대로)
    risk_nomult       심은 라벨 + 보정 없음       (보정만 분리)
    risk_obs_mult     관측 타입 + 시나리오 보정   (라벨만 분리)
    risk_observable   관측 타입 + 보정 없음       (실제 컨트롤러가 계산할 수 있는 것)

앞의 둘에 매 홉 재계획판을 더한다 — QoS·OSPF에서 재계획이 무의미했지만 위험 비용에서도
그럴 거라고 가정하지 않는다.

공식 복제가 정확한지 먼저 확인한다. congestion·unstable의 비backup 링크에서 환경의
`q["risk"]`와 소수점까지 같아야 하고, backup은 정확히 0.60배여야 한다. dynamic은 위상
집합을 `env.get_dynamic_phase_sets()`로 받아 같은 방식으로 맞춘다. 불일치가 남으면
숫자를 그대로 보고하고 넘어간다 — 숨기면 이 비교의 근거가 사라진다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_risk_oracle_leak.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_risk_oracle_leak.py --verify
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
from network_env import (  # noqa: E402
    CONGESTION_LINKS, MAX_STEPS, UNSTABLE_LINKS, NetworkEnv)
from review_stats import fmt_p, holm_bonferroni, min_detectable_effect, welch  # noqa: E402
from risk_calculator import DEFAULT_WEIGHTS  # noqa: E402
from topology import DESTINATION  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "risk_oracle_leak.csv")
REF_CSV = os.path.join(BASE_DIR, "review_results", "qos_baseline_handicap.csv")

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
EPISODES = RLC.EVAL_EPISODES
WARMUP = RLC.WARMUP

METRICS = QBH.METRICS
path_metrics = QBH.path_metrics
seed_mean = QBH.seed_mean

# `risk_calculator.py:62-74`의 link_type -> 고유 위험도 표.
TYPE_RISK = {"unstable_congested": 1.0, "unstable": 0.9, "congested": 0.8,
             "backup": 0.3, "backbone": 0.05}
OTHER_TYPE_RISK = 0.15


def type_risk(name):
    return TYPE_RISK.get(name, OTHER_TYPE_RISK)


def scenario_sets(env):
    """환경이 이 스텝에 쓴 집합들을 되돌려 받는다 (`network_env.py:170-181`)."""
    danger = support = set()
    if env.scenario == "congestion":
        danger = {env.normalize_edge(*x) for x in CONGESTION_LINKS}
    elif env.scenario == "unstable":
        danger = {env.normalize_edge(*x) for x in UNSTABLE_LINKS}
    elif env.scenario == "dynamic":
        danger, support = env.get_dynamic_phase_sets()
        danger = {env.normalize_edge(*x) for x in danger}
        support = {env.normalize_edge(*x) for x in support}
    return danger, support


PLANTED_LABEL = {"congestion": "congested", "unstable": "unstable",
                 "dynamic": "unstable_congested"}


def risk_of(env, e, q, danger, support, use_planted_label, use_multiplier):
    """`calculate_link_risk` + 환경의 사후 보정을 복제한다. 두 스위치로 끈다."""
    w = DEFAULT_WEIGHTS
    if use_planted_label and e in danger:
        label = PLANTED_LABEL[env.scenario]
    else:
        label = q["type"]
    risk = (w["util"] * min(max(q["utilization"], 0.0), 1.0)
            + w["cong"] * (1.0 if q["is_congested"] else 0.0)
            + w["delay"] * min(float(np.std(env.delay_history[e])) / 20.0, 1.0)
            + w["loss"] * min(float(np.std(env.loss_history[e])) / 0.03, 1.0)
            + w["type"] * type_risk(label))
    risk = float(np.clip(risk, 0.0, 1.0))

    if use_multiplier:
        if env.scenario in ("congestion", "unstable") and q["type"] == "backup":
            risk = float(np.clip(risk * 0.60, 0.0, 1.0))
        if env.scenario == "dynamic" and e in support:
            risk = float(np.clip(risk * 0.55, 0.0, 1.0))
    return risk


VARIANT_SWITCHES = {
    "risk_published": (True, True),
    "risk_nomult": (True, False),
    "risk_obs_mult": (False, True),
    "risk_observable": (False, False),
}


def verify_formula(episodes=20, seeds=3):
    """복제가 환경의 q["risk"]와 같은지 — risk_published 설정에서 확인한다."""
    print("[검증] 공식 복제 대 환경의 q['risk']")
    ok_all = True
    for scenario in SCENARIOS:
        n = bad = 0
        worst = 0.0
        for seed in range(seeds):
            np.random.seed(seed)
            random.seed(seed)
            for _ in range(episodes):
                env = NetworkEnv(use_risk=True, scenario=scenario,
                                 risk_window=RLC.RISK_WINDOW)
                env.reset()
                for _ in range(WARMUP):
                    env.update_dynamic_qos()
                danger, support = scenario_sets(env)
                for e, q in env.qos.items():
                    mine = risk_of(env, e, q, danger, support, True, True)
                    n += 1
                    d = abs(mine - q["risk"])
                    worst = max(worst, d)
                    if d > 1e-9:
                        bad += 1
        flag = "" if bad == 0 else "   <- 불일치"
        if bad:
            ok_all = False
        print("  {:<12} {}/{}건 불일치, 최대 편차 {:.3e}{}".format(
            scenario, bad, n, worst, flag))
    if not ok_all:
        print("  주의: 불일치가 있는 시나리오에서는 아래 2x2 비교를 쓰지 말 것.")
    else:
        print("  -> 전 시나리오 복제 정확. 라벨·보정을 끈 변형이 유효하다.")
    return ok_all


# ------------------------------------------------------------------- 플래너

def plan(env, blocked, use_planted_label, use_multiplier):
    danger, support = scenario_sets(env)
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
            new_cost = cost + risk_of(env, e, q, danger, support,
                                      use_planted_label, use_multiplier)
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


def run_variant(scenario, seed, episodes, switches, replan):
    label, mult = switches
    np.random.seed(seed)
    random.seed(seed)
    out = []
    distinct = set()
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario,
                         risk_window=RLC.RISK_WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        if not replan:
            p = plan(env, set(), label, mult)
            if p and p[-1] == DESTINATION:
                out.append(path_metrics(env, p))
                distinct.add(tuple(p))
            continue
        done = False
        while not done:
            p = plan(env, set(env.visited) - {env.current_node}, label, mult)
            if len(p) < 2:
                break
            _, _, done, _ = env.step(p[1])
            if len(env.visited) >= MAX_STEPS:
                break
        if env.current_node == DESTINATION:
            out.append(path_metrics(env, env.visited))
            distinct.add(tuple(env.visited))
    return seed_mean(out), len(out) / episodes * 100, len(distinct)


VARIANTS = (
    ("risk_published", "risk_published", False),
    ("risk_nomult", "risk_nomult", False),
    ("risk_obs_mult", "risk_obs_mult", False),
    ("risk_observable", "risk_observable", False),
    ("risk_published_replan", "risk_published", True),
    ("risk_observable_replan", "risk_observable", True),
)


def measure():
    rows = []
    for scenario in SCENARIOS:
        for name, key, replan in VARIANTS:
            for seed in SEEDS:
                means, success, distinct = run_variant(
                    scenario, seed, EPISODES, VARIANT_SWITCHES[key], replan)
                row = {"scenario": scenario, "variant": name, "seed": seed,
                       "success_rate": success, "distinct_paths": distinct}
                row.update(means)
                rows.append(row)
            print("  {:<11} {:<24} 완료".format(scenario, name), flush=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["scenario", "variant", "seed", "success_rate", "distinct_paths"] \
        + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n저장: {} ({}행)".format(OUT_CSV, len(rows)))
    return rows


def load_csv(path):
    if not os.path.exists(path):
        raise SystemExit("CSV가 없다: " + path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in ("scenario", "variant"):
                continue
            r[k] = float(v)
    return rows


def by_variant(rows, scenario):
    out = {}
    for r in rows:
        if r["scenario"] != scenario:
            continue
        keys = [k for k in r if k not in ("scenario", "variant")]
        d = out.setdefault(r["variant"], {k: [] for k in keys})
        for k in d:
            d[k].append(r[k])
    return out


def report(rows, ref_rows):
    names = [n for n, _, _ in VARIANTS]

    print("\n" + "=" * 78)
    print("A. 발표된 위험 비용과 관측만으로 만든 위험 비용 — 경로 품질")
    print("=" * 78)
    for metric in ("compounded", "delay", "hop", "cong_live"):
        print("\n  [{}]".format(metric))
        print("    {:<11}".format("시나리오")
              + "".join("{:>15}".format(n.replace("risk_", "")[:14]) for n in names)
              + "{:>12}".format("Risk-DQN"))
        for scenario in SCENARIOS:
            g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
            print("    {:<11}".format(scenario)
                  + "".join("{:>15.4f}".format(float(np.mean(g[n][metric])))
                            for n in names)
                  + "{:>12.4f}".format(float(np.mean(gr["risk_dqn"][metric]))))

    print("\n  도달률(%) / 고유 경로 수")
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        print("    {:<11}".format(scenario)
              + "".join("{:>15}".format("{:.0f} / {:.1f}".format(
                  float(np.mean(g[n]["success_rate"])),
                  float(np.mean(g[n]["distinct_paths"])))) for n in names))

    print("\n" + "=" * 78)
    print("B. 두 누출이 각각 얼마를 보태는가 (compounded 기준)")
    print("=" * 78)
    print("  보정 효과 = published - nomult   (심은 라벨은 둘 다 켠 상태)")
    print("  라벨 효과 = published - obs_mult (보정은 둘 다 켠 상태)")
    print("  합계      = published - observable")
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        pub = float(np.mean(g["risk_published"]["compounded"]))
        nm = float(np.mean(g["risk_nomult"]["compounded"]))
        om = float(np.mean(g["risk_obs_mult"]["compounded"]))
        ob = float(np.mean(g["risk_observable"]["compounded"]))
        print("  {:<11} 발표 {:.4f}  보정만끔 {:.4f}({:+.1f}%)  "
              "라벨만끔 {:.4f}({:+.1f}%)  둘다끔 {:.4f}({:+.1f}%)".format(
                  scenario, pub, nm, (nm - pub) / pub * 100,
                  om, (om - pub) / pub * 100, ob, (ob - pub) / pub * 100))

    print("\n" + "=" * 78)
    print("C. 재계획이 위험 비용에서는 다른가")
    print("=" * 78)
    for scenario in SCENARIOS:
        g = by_variant(rows, scenario)
        for base, rp in (("risk_published", "risk_published_replan"),
                         ("risk_observable", "risk_observable_replan")):
            a = float(np.mean(g[base]["compounded"]))
            b = float(np.mean(g[rp]["compounded"]))
            print("  {:<11} {:<18} {:.4f} -> 매홉 {:.4f} ({:+.1f}%)  홉 {:.2f} -> {:.2f}"
                  .format(scenario, base.replace("risk_", ""), a, b,
                          (b - a) / a * 100 if a else float("nan"),
                          float(np.mean(g[base]["hop"])),
                          float(np.mean(g[rp]["hop"]))))

    print("\n" + "=" * 78)
    print("D. Risk-DQN 대 관측만으로 만든 위험 Dijkstra — Welch + Holm")
    print("=" * 78)
    tests = []
    for scenario in SCENARIOS:
        g, gr = by_variant(rows, scenario), by_variant(ref_rows, scenario)
        dq = gr["risk_dqn"]
        for name in ("risk_published", "risk_observable", "risk_observable_replan"):
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
    for scenario in SCENARIOS:
        print("\n  {}".format(scenario))
        for x in tests:
            if x["scenario"] != scenario or x["metric"] not in (
                    "compounded", "delay", "hop", "cong_live"):
                continue
            print("    {:<24}{:<14}DQN이득 {:>7.1f}%  MDE {:>6.1f}%  {}{}".format(
                x["variant"], x["metric"], x["rel"], x["mde_rel"],
                fmt_p(x["p_holm"]), "  *" if x["sig"] else ""))
    print("\n  유의 {}/{}건. 시드당 {}에피소드 평균이라 크기로 읽는다.".format(
        sum(1 for x in tests if x["sig"]), len(tests), EPISODES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    if args.verify:
        verify_formula()
        return

    ref_rows = load_csv(REF_CSV)
    if args.from_csv:
        rows = load_csv(OUT_CSV)
    else:
        if not verify_formula():
            print("\n복제가 정확하지 않은 시나리오가 있다 — 그래도 측정은 진행하고,")
            print("보고에서 해당 시나리오를 인용할 때 이 경고를 함께 써야 한다.")
        print("\n측정 시작 — {}변형 x {}시나리오 x {}시드 x {}에피소드".format(
            len(VARIANTS), len(SCENARIOS), len(SEEDS), EPISODES))
        rows = measure()

    report(rows, ref_rows)


if __name__ == "__main__":
    main()
