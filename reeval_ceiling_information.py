"""누산기는 정보를 더하고 클립은 정보를 지운다 — 어느 쪽이 지배하는가.

`network_env.py:228-229`가 누산기라는 것(FINDINGS_environment_dynamics 1절)에서
**내 결론에 불리한 가설**이 하나 나온다. 매 스텝 더해지는 항이 `utilization * 2.0`이므로
천장에 닿기 전까지 delay는 이용률의 **적분**이다. 적분은 평균 0 잡음을 지우므로
`delay`는 시간이 갈수록 이용률의 **더 좋은** 대리변수가 된다. 그러면 문제는 "누산"이
아니라 "클립"이고, 고칠 곳도 갱신식이 아니라 상한이다.

그리고 더 불리한 가능성이 있다. 천장은 심어둔 집합 **안의 순위**를 지우지만
심은 링크 전부가 120에 박히고 나머지는 안 박히면 **심은/안 심은 분리는 완벽해진다.**
라우팅이 필요한 건 "나쁜 집합을 피하라"뿐이므로, 그렇다면 환경은 어려워진 게 아니라
**거저먹기로 쉬워진** 것이다. 그쪽이 사실이면 "천장이 위험 추정을 망친다"는 서술은
과장이고, 올바른 서술은 "천장이 위험 추정을 **불필요하게** 만든다"가 된다.

검증 가능한 예측을 먼저 적는다. 틀리면 틀린 대로 적는다.

  P1  rho(delay, util)이 t=0에서 올라가 정점을 지나고 천장이 차면서 **내려간다.**
  P2  AUC(심은 링크 대 나머지 | delay)는 1.0으로 올라가 **그대로 머문다.**
      (깨지면 — 즉 내려가면 — "쉬워졌다"가 반증되고 천장이 식별을 지우는 게 맞다.)
  P3  심은 집합 **안의** rho(delay, util)은 0으로 붕괴한다.
  P4  normal은 심은 링크가 없고 포화도 없으므로 rho가 단조 상승하고 내려오지 않는다.
  P5  unstable은 포화가 늦으므로(예측 25스텝) 정점이 늦다.

P2와 P3이 동시에 성립하면 결론은 이렇게 바뀐다: **위험 모델이 쓸모없는 이유가
신호가 지워져서가 아니라 과제가 이진 판별로 퇴화해서다.** 그건 더 강한 비판이다 —
"제안 방법이 정확한 위험 추정을 한다"는 주장을 뒷받침할 데이터가 아예 없다는 뜻이니까.
"""

import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
import reeval_risk_oracle_leak as ROL  # noqa: E402
from network_env import NetworkEnv  # noqa: E402

SCENARIOS = ("normal", "unstable", "congestion", "dynamic")
TIMES = (0, 2, 5, 10, 14, 18, 25, 40)
SEEDS = 20
DELAY_CEIL = 120.0
LOSS_CEIL = 0.35


def ranks(values):
    """동률은 평균 순위를 준다."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def spearman(xs, ys):
    if len(xs) < 3:
        return float("nan")
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return float("nan")
    return num / (dx * dy) ** 0.5


def auc(pos, neg):
    """P(양성 > 음성) + 0.5 x 동률. 완전 분리면 1.0, 무정보면 0.5."""
    if not pos or not neg:
        return float("nan")
    total = 0.0
    for p in pos:
        for n in neg:
            total += 1.0 if p > n else (0.5 if p == n else 0.0)
    return total / (len(pos) * len(neg))


def measure():
    acc = {s: {t: [] for t in TIMES} for s in SCENARIOS}
    for scenario in SCENARIOS:
        for seed in range(SEEDS):
            np.random.seed(seed)
            random.seed(seed)
            env = NetworkEnv(use_risk=True, scenario=scenario,
                             risk_window=RLC.RISK_WINDOW)
            env.reset()
            danger, _support = ROL.scenario_sets(env)
            for t in range(max(TIMES) + 1):
                if t > 0:
                    env.update_dynamic_qos()
                if t not in TIMES:
                    continue
                edges = list(env.qos.keys())
                dly = [env.qos[e]["delay"] for e in edges]
                los = [env.qos[e]["loss"] for e in edges]
                utl = [env.qos[e]["utilization"] for e in edges]
                pos_i = [i for i, e in enumerate(edges) if e in danger]
                neg_i = [i for i, e in enumerate(edges) if e not in danger]
                pd = [dly[i] for i in pos_i]
                nd = [dly[i] for i in neg_i]
                pl = [los[i] for i in pos_i]
                nl = [los[i] for i in neg_i]
                acc[scenario][t].append({
                    "rho_all": spearman(dly, utl),
                    "rho_planted": spearman(pd, [utl[i] for i in pos_i]),
                    "auc_delay": auc(pd, nd),
                    "auc_loss": auc(pl, nl),
                    "ceil_d_all": sum(1 for v in dly if v >= DELAY_CEIL - 1e-9) / len(dly),
                    "ceil_d_pl": (sum(1 for v in pd if v >= DELAY_CEIL - 1e-9) / len(pd)
                                  if pd else float("nan")),
                    "ceil_l_pl": (sum(1 for v in pl if v >= LOSS_CEIL - 1e-9) / len(pl)
                                  if pl else float("nan")),
                    "distinct_pl": (len({round(v, 6) for v in pd}) / len(pd)
                                    if pd else float("nan")),
                    "n_planted": float(len(pd)),
                    "std_delay_pl": (float(np.mean([np.std(env.delay_history[edges[i]])
                                                    for i in pos_i])) if pos_i
                                     else float("nan")),
                })
    return acc


def measure_init():
    """`init_qos()` 직후 — 심은 정보가 아직 하나도 안 들어간 상태.

    `network_env.py:109-134`의 초기값은 토폴로지 타입(backbone/backup/normal)만 보고
    정해지고 traffic은 전 링크 `uniform(0.10, 0.35)`다. 시나리오가 심은 집합은 여기
    등장하지 않으므로 AUC가 0.5 근처여야 한다. 그리고 `reset()`은 `init_qos()` 뒤에
    `update_dynamic_qos()`를 **한 번 호출한다**(`:260-261`). 즉 위 표의 t=0은 이미
    1스텝 누산된 상태다. 둘을 비교하면 "한 번의 갱신이 얼마를 만드는가"가 나온다.
    """
    out = {}
    for scenario in SCENARIOS:
        rows = []
        for seed in range(SEEDS):
            np.random.seed(seed)
            random.seed(seed)
            env = NetworkEnv(use_risk=True, scenario=scenario,
                             risk_window=RLC.RISK_WINDOW)
            env.reset()
            danger, _ = ROL.scenario_sets(env)
            env.init_qos()  # 누산분을 버리고 초기 분포만 다시 뽑는다
            edges = list(env.qos.keys())
            pos = [env.qos[e]["delay"] for e in edges if e in danger]
            neg = [env.qos[e]["delay"] for e in edges if e not in danger]
            posl = [env.qos[e]["loss"] for e in edges if e in danger]
            negl = [env.qos[e]["loss"] for e in edges if e not in danger]
            rows.append({"auc_delay": auc(pos, neg), "auc_loss": auc(posl, negl)})
        out[scenario] = rows
    return out


def mean_of(rows, key):
    vals = [r[key] for r in rows if not np.isnan(r[key])]
    return float(np.mean(vals)) if vals else float("nan")


def fmt(v, w=9, p=3):
    return (" " * w if np.isnan(v) else "{:>{w}.{p}f}".format(v, w=w, p=p))


def main():
    acc = measure()
    print("누산 대 클립 — 50링크 x {}시드, 심은 집합 대 나머지".format(SEEDS))
    print("rho = Spearman(delay, utilization), AUC = P(심은 링크 delay > 나머지)")

    for key, label, pred in (
        ("rho_all", "P1  rho(delay, util) 전체 링크", "올랐다 내려와야 한다"),
        ("auc_delay", "P2  AUC(심은 대 나머지 | delay)", "1.0으로 올라가 머물러야 한다"),
        ("rho_planted", "P3  rho(delay, util) 심은 집합 안", "0으로 붕괴해야 한다"),
        ("auc_loss", "    AUC(심은 대 나머지 | loss)", ""),
        ("ceil_d_pl", "    delay 천장 비율 (심은 집합)", ""),
        ("ceil_l_pl", "    loss 천장 비율 (심은 집합)", ""),
        ("distinct_pl", "    심은 집합의 서로 다른 delay 비율", "1.0 = 전부 구분"),
        ("std_delay_pl", "    std(delay_history) 심은 집합", ""),
    ):
        print()
        print("  {}{}".format(label, "   [{}]".format(pred) if pred else ""))
        print("    {:<12}{}".format("시나리오", "".join(
            "{:>9}".format("t=" + str(t)) for t in TIMES)))
        for s in SCENARIOS:
            row = "".join(fmt(mean_of(acc[s][t], key)) for t in TIMES)
            print("    {:<12}{}".format(s, row))

    init = measure_init()
    print()
    print("  한 번의 갱신이 만드는 분리 — init_qos() 직후 대 t=0(=갱신 1회)")
    print("    {:<12}{:>12}{:>12}{:>12}{:>12}".format(
        "시나리오", "AUC_delay", "init", "AUC_loss", "init"))
    for s in SCENARIOS:
        if np.isnan(mean_of(init[s], "auc_delay")):
            print("    {:<12}{:>12}".format(s, "심은 집합 없음"))
            continue
        print("    {:<12}{}{}{}{}".format(
            s, fmt(mean_of(acc[s][0], "auc_delay"), 12),
            fmt(mean_of(init[s], "auc_delay"), 12),
            fmt(mean_of(acc[s][0], "auc_loss"), 12),
            fmt(mean_of(init[s], "auc_loss"), 12)))

    print()
    print("=" * 78)
    print("판정")
    print("=" * 78)
    for s in SCENARIOS:
        series = [mean_of(acc[s][t], "rho_all") for t in TIMES]
        clean = [(t, v) for t, v in zip(TIMES, series) if not np.isnan(v)]
        if not clean:
            continue
        peak_t, peak_v = max(clean, key=lambda kv: kv[1])
        last_t, last_v = clean[-1]
        first_v = clean[0][1]
        a = [mean_of(acc[s][t], "auc_delay") for t in TIMES]
        a_clean = [v for v in a if not np.isnan(v)]
        rp = [mean_of(acc[s][t], "rho_planted") for t in TIMES]
        rp_clean = [(t, v) for t, v in zip(TIMES, rp) if not np.isnan(v)]
        dist = [mean_of(acc[s][t], "distinct_pl") for t in TIMES]
        dist_clean = [(t, v) for t, v in zip(TIMES, dist) if not np.isnan(v)]
        print()
        print("  {}".format(s))
        # P1: 정점을 지나 내려와야 맞는 예측이었다. 정점이 마지막 시점이면 반증.
        p1 = "반증 — 정점이 마지막 시점이고 하락이 없다" if peak_t == last_t else \
             "지지 — 정점 후 {:+.3f} 하락".format(last_v - peak_v)
        print("    P1  rho {:.3f}(t={}) -> {:.3f}(t={})   {}"
              .format(first_v, clean[0][0], last_v, last_t, p1))
        if a_clean:
            # 단조 상승이면 유지다. t=0의 낮은 값은 "하락"이 아니다.
            drop = min((b - a) for a, b in zip(a_clean, a_clean[1:])) if len(a_clean) > 1 else 0.0
            print("    P2  AUC {:.3f}(t={}) -> {:.3f}(t={})   최대하락 {:+.3f}   {}"
                  .format(a_clean[0], clean[0][0], a_clean[-1], last_t, min(drop, 0.0),
                          "지지 — 하락 없음" if drop >= -0.005 else "반증 — 중간에 떨어진다"))
        else:
            print("    P2  심은 집합 없음 — 해당 없음")
        if rp_clean:
            worst = min(rp_clean, key=lambda kv: abs(kv[1]))
            print("    P3  집합 내 rho {:+.3f}(t={}) -> {:+.3f}(t={}), 최소절대값 {:+.3f}(t={})"
                  .format(rp_clean[0][1], rp_clean[0][0], rp_clean[-1][1],
                          rp_clean[-1][0], worst[1], worst[0]))
            if dist_clean:
                print("        서로 다른 delay 비율 {:.3f} -> {:.3f}  (심은 링크 {:.0f}개 중 {:.1f}개 값)"
                      .format(dist_clean[0][1], dist_clean[-1][1],
                              mean_of(acc[s][TIMES[-1]], "n_planted"),
                              dist_clean[-1][1] * mean_of(acc[s][TIMES[-1]], "n_planted")))
        else:
            print("    P3  심은 집합 없음 — 해당 없음")


if __name__ == "__main__":
    main()
