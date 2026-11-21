"""메인 표는 세 가지 다른 채점 규약을 섞어 쓴다.

`network_env.py:300` — `step()`은 매번 `update_dynamic_qos()`를 호출해 50개 링크의
지연·손실·이용률 잡음을 **다시 굴린다**. 그래서 "언제의 링크 상태로 채점하는가"가
방법마다 다르면 비교가 성립하지 않는다. 발표된 평가 코드를 보면 다르다:

    OSPF      `eval_qos.py:86-92`   워밍업 -> 계획 -> **step 없이** 그 스냅샷에서 채점
    QoS       `eval_qos.py:150-160` 계획 -> 계획대로 step -> **최종** 스냅샷에서 채점
    Risk-DQN  (같은 규약)            매 홉 선택하며 step -> **최종** 스냅샷에서 채점

OSPF만 **자기가 최적화한 바로 그 스냅샷에서 채점된다**. 잡음이 있는 양을 argmin으로
고르고 같은 표본에서 점수를 매기면 그 잡음만큼 낙관적으로 나온다(선택 편의).

편의의 크기는 "고르는 양"과 "채점하는 양"이 얼마나 같은지에 달렸다. 발표된 OSPF는
`1000/bandwidth`로 고르는데 채점은 손실·지연이고, 둘의 상관은 약하다
(r(bw,loss) = -0.03/+0.38/+0.24/+0.34) -> 편의가 작을 것이다. 반면 지연 비용 플래너는
지연으로 고르고 지연으로 채점되며 손실과도 r(delay,loss) = 0.85~0.99로 묶여 있다
-> 편의가 클 것이다. 실제로 `reeval_ospf_replan.py`에서 unstable compounded가
0.2405(1회 계획, step 없음) 대 0.3404(매 홉 재계획, step 있음)로 41.5% 벌어졌다.
나는 그걸 "재계획이 해롭다"고 읽었는데, 재계획이 아니라 **채점 규약**이 원인일 수 있다.
그래서 재계획을 주지 않고 규약만 바꾼 변형을 넣는다.

세 규약을 교차시킨다:

    plan_snapshot  계획만 하고 step하지 않는다. 계획에 쓴 상태에서 채점.  (발표된 OSPF)
    walk_final     계획대로 step한 뒤 **최종** 상태에서 채점.             (발표된 QoS/Risk-DQN)
    walk_realized  각 링크를 **지나간 그 순간의** 상태로 채점.

`walk_realized`가 필요한 이유: `walk_final`도 엄밀히는 틀렸다. 6~8번 잡음이 다시 굴려진
뒤의 상태로 첫 홉을 채점하기 때문이다. 어느 방법도 유리해지지 않는 공통 오차지만,
잡음이 큰 시나리오에서 절대값을 왜곡한다. 실현값은 `step()`이 상태를 갱신한 **직후**의
그 링크 값으로 잰다 — 전이에 실제로 쓰인 상태다.

검증 두 개를 먼저 통과해야 아래 숫자를 쓴다:
  - ospf_bw의 `plan_snapshot`이 커밋된 `exp1_reference.csv` 계보(`qos_baseline_handicap.csv`의
    `ospf` 행)를 재현해야 한다. 재현되면 내가 발표된 규약을 정확히 복제했다는 뜻이다.
  - qos의 `walk_final`이 같은 CSV의 `qos_post_warmup`을 재현해야 한다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_scoring_convention.py
    KMP_DUPLICATE_LIB_OK=TRUE python -I -X utf8 reeval_scoring_convention.py --from-csv
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
import reeval_qos_baseline_handicap as QBH  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import fmt_p, holm_bonferroni, min_detectable_effect, welch  # noqa: E402
from topology import DESTINATION  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results", "scoring_convention.csv")
REF_CSV = os.path.join(BASE_DIR, "review_results", "qos_baseline_handicap.csv")

SCENARIOS = RLC.SCENARIOS
SEEDS = RLC.SEEDS
EPISODES = RLC.EVAL_EPISODES
WARMUP = RLC.WARMUP
METRICS = QBH.METRICS
path_metrics = QBH.path_metrics
seed_mean = QBH.seed_mean

# 재현 허용 오차(%). 발표된 규약을 복제한 변형은 이 안에 들어와야 한다.
REPRO_TOL_PCT = 0.05


# --------------------------------------------------------------- 규약 계산

def realized_metrics(samples):
    """지나간 순간의 링크 값으로 `RLC.loss_conventions` + `cong_live`를 계산한다.

    samples는 (delay, loss, is_congested) 튜플들 — 홉을 밟은 순서대로.
    """
    hop = max(len(samples), 1)
    delay = sum(d for d, _, _ in samples)
    rates = [l for _, l, _ in samples]
    survival = 1.0
    for rate in rates:
        survival *= (1.0 - rate)
    live = sum(1 for _, _, c in samples if c)
    return {
        "hop": len(samples),
        "delay": delay,
        "published": sum(rates) / hop,
        "path_sum": sum(rates),
        "compounded": 1.0 - survival,
        "cong_live": float(live),
        "cong_live_per_hop": live / hop * 100.0,
    }


def walk_collecting(env, next_node_of):
    """계획대로(또는 정책대로) 전진하며 실현값을 모은다.

    간선은 step **전에** 정하고(current_node가 바뀌므로), 값은 step **후에** 읽는다 —
    `step()`이 `update_dynamic_qos()`를 먼저 호출하므로 그게 전이에 쓰인 상태다.
    """
    samples = []
    done = False
    while not done:
        nxt = next_node_of(env)
        if nxt is None:
            break
        e = env.normalize_edge(env.current_node, nxt)
        _, _, done, _ = env.step(nxt)
        q = env.qos.get(e)
        if q is not None:
            samples.append((q["delay"], q["loss"],
                            bool(q.get("is_congested", False))))
        if len(env.visited) >= MAX_STEPS:
            break
    return samples


# --------------------------------------------------------------- 플래너

def plan_ospf_bw(env):
    return env.ospf_path(cost_type="bandwidth")


def plan_ospf_delay(env):
    return env.ospf_path(cost_type="delay")


def plan_qos(env):
    return RLC.qos_plan_full_path(env)


PLANNERS = (
    ("ospf_bw", plan_ospf_bw),
    ("ospf_delay", plan_ospf_delay),
    ("qos", plan_qos),
)


def run_planner(scenario, seed, episodes, planner):
    """한 플래너를 세 규약으로 동시에 잰다 — 같은 에피소드, 같은 계획.

    규약마다 따로 돌리면 난수 소비가 갈려 표본이 달라진다. 계획은 한 번만 하고
    plan_snapshot을 먼저 기록한 뒤 전진시키므로, 세 숫자가 같은 에피소드에서 나온다.
    """
    np.random.seed(seed)
    random.seed(seed)
    snap, final, real = [], [], []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        planned = planner(env)
        if not planned or planned[-1] != DESTINATION:
            continue
        # 발표된 OSPF 규약: 여기서 멈추고 채점한다.
        snap.append(path_metrics(env, planned))
        queue = list(planned[1:])

        def next_node_of(_env, queue=queue):
            return queue.pop(0) if queue else None

        samples = walk_collecting(env, next_node_of)
        if env.current_node != DESTINATION:
            continue
        final.append(path_metrics(env, env.visited))
        real.append(realized_metrics(samples))
    n = float(episodes)
    return {"plan_snapshot": (seed_mean(snap), len(snap) / n * 100),
            "walk_final": (seed_mean(final), len(final) / n * 100),
            "walk_realized": (seed_mean(real), len(real) / n * 100)}


def run_risk_dqn(scenario, seed, episodes, model):
    """제안 방법. QBH.run_risk_dqn과 같은 루프에 실현값 수집만 더했다."""
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    final, real = [], []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=scenario,
                         risk_window=RLC.RISK_WINDOW)
        env.reset()
        for _ in range(WARMUP):
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

        samples = walk_collecting(env, next_node_of)
        if env.current_node != DESTINATION:
            continue
        final.append(path_metrics(env, env.visited))
        real.append(realized_metrics(samples))
    n = float(episodes)
    return {"walk_final": (seed_mean(final), len(final) / n * 100),
            "walk_realized": (seed_mean(real), len(real) / n * 100)}


# --------------------------------------------------------------- 측정

def measure():
    rows = []
    for scenario in SCENARIOS:
        for name, planner in PLANNERS:
            for seed in SEEDS:
                got = run_planner(scenario, seed, EPISODES, planner)
                for conv, (means, success) in got.items():
                    row = {"scenario": scenario, "policy": name,
                           "convention": conv, "seed": seed,
                           "success_rate": success}
                    row.update(means)
                    rows.append(row)
            print("  {:<11} {:<12} 완료".format(scenario, name), flush=True)

    model = RLC.load_risk_model()
    for scenario in SCENARIOS:
        for seed in SEEDS:
            got = run_risk_dqn(scenario, seed, EPISODES, model)
            for conv, (means, success) in got.items():
                row = {"scenario": scenario, "policy": "risk_dqn",
                       "convention": conv, "seed": seed,
                       "success_rate": success}
                row.update(means)
                rows.append(row)
        print("  {:<11} {:<12} 완료".format(scenario, "risk_dqn"), flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["scenario", "policy", "convention", "seed", "success_rate"] \
        + list(METRICS)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print("\n저장: {} ({}행)".format(OUT_CSV, len(rows)))
    return rows


def load_csv(path, keys):
    if not os.path.exists(path):
        raise SystemExit("CSV가 없다: " + path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in keys:
                continue
            r[k] = float(v)
    return rows


def pick(rows, scenario, policy, convention, metric):
    return [r[metric] for r in rows
            if r["scenario"] == scenario and r["policy"] == policy
            and r["convention"] == convention]


def ref_pick(ref_rows, scenario, variant, metric):
    return [r[metric] for r in ref_rows
            if r["scenario"] == scenario and r["variant"] == variant]


# --------------------------------------------------------------- 보고

def report(rows, ref_rows):
    print("\n" + "=" * 78)
    print("A. 재현 검증 — 발표된 두 규약을 정확히 복제했는가")
    print("=" * 78)
    print("  ospf_bw/plan_snapshot  vs 커밋된 ospf 행")
    print("  qos/walk_final         vs 커밋된 qos_post_warmup 행")
    worst = 0.0
    for scenario in SCENARIOS:
        for policy, conv, variant in (("ospf_bw", "plan_snapshot", "ospf"),
                                      ("qos", "walk_final", "qos_post_warmup")):
            line = "    {:<11} {:<24}".format(scenario, policy + "/" + conv)
            for metric in ("delay", "published", "compounded"):
                mine = float(np.mean(pick(rows, scenario, policy, conv, metric)))
                theirs = float(np.mean(ref_pick(ref_rows, scenario, variant, metric)))
                dev = (mine - theirs) / theirs * 100 if theirs else 0.0
                worst = max(worst, abs(dev))
                line += "  {} {:+.2f}%".format(metric[:5], dev)
            print(line)
    verdict = "통과" if worst <= REPRO_TOL_PCT else "실패"
    print("\n  최대 편차 {:.3f}% -> {} (허용 {:.2f}%)".format(
        worst, verdict, REPRO_TOL_PCT))
    if worst > REPRO_TOL_PCT:
        print("  복제가 어긋났다 — 아래 숫자를 인용하기 전에 원인을 찾아야 한다.")

    print("\n" + "=" * 78)
    print("B. 채점 규약만 바꾸면 얼마가 움직이는가 (step 없음 -> 계획대로 전진)")
    print("=" * 78)
    print("  고르는 양과 채점하는 양이 같을수록 편의가 커야 한다는 예측을 검증한다.")
    for metric in ("compounded", "delay", "published", "cong_live"):
        print("\n  [{}]".format(metric))
        print("    {:<11}".format("시나리오")
              + "".join("{:>26}".format(n) for n, _ in PLANNERS))
        for scenario in SCENARIOS:
            line = "    {:<11}".format(scenario)
            for name, _ in PLANNERS:
                a = float(np.mean(pick(rows, scenario, name, "plan_snapshot", metric)))
                b = float(np.mean(pick(rows, scenario, name, "walk_final", metric)))
                rel = (b - a) / a * 100 if a else float("nan")
                line += "{:>26}".format("{:.4f}->{:.4f} {:+.1f}%".format(a, b, rel))
            print(line)

    print("\n" + "=" * 78)
    print("C. 같은 규약에서 다시 세운다 — Risk-DQN 대 최강 비학습 플래너")
    print("=" * 78)
    for conv in ("walk_final", "walk_realized"):
        print("\n  [{}]".format(conv))
        for metric in ("compounded", "delay", "hop", "cong_live"):
            print("    {:<9}".format(metric)
                  + "{:<11}".format("시나리오")
                  + "".join("{:>13}".format(n) for n, _ in PLANNERS)
                  + "{:>13}{:>16}".format("Risk-DQN", "최강대비"))
            for scenario in SCENARIOS:
                vals = {n: float(np.mean(pick(rows, scenario, n, conv, metric)))
                        for n, _ in PLANNERS}
                dqn = float(np.mean(pick(rows, scenario, "risk_dqn", conv, metric)))
                best = min(vals.values())
                gain = (best - dqn) / best * 100 if best else float("nan")
                print("    {:<9}{:<11}".format("", scenario)
                      + "".join("{:>13.4f}".format(vals[n]) for n, _ in PLANNERS)
                      + "{:>13.4f}{:>15.1f}%".format(dqn, gain))
            print()

    print("=" * 78)
    print("D. walk_final과 walk_realized는 다른 결론을 주는가")
    print("=" * 78)
    print("  같은 주행, 채점 시점만 다르다. 순위가 바뀌면 메인 표의 절대값을 못 믿는다.")
    for metric in ("compounded", "delay"):
        print("\n  [{}]".format(metric))
        names = [n for n, _ in PLANNERS] + ["risk_dqn"]
        print("    {:<11}".format("시나리오")
              + "".join("{:>22}".format(n) for n in names))
        for scenario in SCENARIOS:
            line = "    {:<11}".format(scenario)
            for n in names:
                a = float(np.mean(pick(rows, scenario, n, "walk_final", metric)))
                b = float(np.mean(pick(rows, scenario, n, "walk_realized", metric)))
                rel = (b - a) / a * 100 if a else float("nan")
                line += "{:>22}".format("{:.4f}/{:.4f} {:+.1f}%".format(a, b, rel))
            print(line)

    print("\n" + "=" * 78)
    print("E. Welch + Holm — 같은 규약에서 Risk-DQN 대 각 플래너")
    print("=" * 78)
    tests = []
    for scenario in SCENARIOS:
        for conv in ("walk_final", "walk_realized"):
            dqn = {m: pick(rows, scenario, "risk_dqn", conv, m) for m in METRICS}
            for name, _ in PLANNERS:
                for metric in METRICS:
                    base = pick(rows, scenario, name, conv, metric)
                    diff, _, _, p = welch(base, dqn[metric])
                    mde = min_detectable_effect(base, dqn[metric])
                    bm = float(np.mean(base))
                    tests.append({"scenario": scenario, "conv": conv,
                                  "policy": name, "metric": metric,
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
            if x["scenario"] != scenario or x["conv"] != "walk_realized":
                continue
            if x["metric"] not in ("compounded", "delay", "hop", "cong_live"):
                continue
            print("    {:<12}{:<12}DQN이득 {:>7.1f}%  MDE {:>6.1f}%  {}{}".format(
                x["policy"], x["metric"], x["rel"], x["mde_rel"],
                fmt_p(x["p_holm"]), "  *" if x["sig"] else ""))
    print("\n  (walk_realized만 표시. 유의 {}/{}건, 시드당 {}에피소드 평균이라"
          " 크기로 읽는다.)".format(
              sum(1 for x in tests if x["sig"]), len(tests), EPISODES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true")
    args = parser.parse_args()

    ref_rows = load_csv(REF_CSV, ("scenario", "variant"))
    if args.from_csv:
        rows = load_csv(OUT_CSV, ("scenario", "policy", "convention"))
    else:
        print("측정 시작 — ({}플래너 x 3규약 + Risk-DQN x 2규약)"
              " x {}시나리오 x {}시드 x {}에피소드".format(
                  len(PLANNERS), len(SCENARIOS), len(SEEDS), EPISODES))
        rows = measure()
    report(rows, ref_rows)


if __name__ == "__main__":
    main()
