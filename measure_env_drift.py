# -*- coding: utf-8 -*-
"""환경 누산기가 경로 길이를 지표로 새게 하는지 **정책 없이** 측정한다.

왜 필요한가. `SDN_DQN_new_env/network_env.py`:
  - `step()` **첫 줄(300)** 이 `self.update_dynamic_qos()` 다.
  - `update_dynamic_qos()` 의 228-229 는 **재추출이 아니라 누산**이다:
    `q["delay"] = clip(q["delay"] + delay_noise + util*2.0, 3, 120)`,
    `q["loss"]  = clip(q["loss"]  + loss_noise + max(0,util-.8)*.03, 0, .35)`.
    `delay_noise` 는 평균 0 이지만 `util*2.0` 은 **항상 양수**다.
  - `calculate_path_metrics` 는 **도착 시점** qos 로 계산된다.
따라서 경로가 한 홉 길면 (a) 링크 하나를 더 더하고 (b) **모든** 링크가
누산 한 틱을 더 받는다. 23.9.6.10 의 기울기(+31.17 ms/홉, r² 0.957)는
둘을 합친 값이므로 분해해야 한다. 여기서는 모델을 안 쓰고 환경만 굴려
(b)를 직접 재고, 이어서 조건 간 에피소드 시작 환경이 실제로 갈라지는지
지문으로 확인한다.

출력: review_results/env_drift.csv
"""
import os
import sys
import csv
import random

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import occlude_state_blocks as OC  # noqa: E402
sys.path.append(OC.ORIG)
sys.path.append(HERE)  # OC 가 import 시 HERE 를 빼내므로 다시 넣는다
import measure_rank_vs_value as RV  # noqa: E402
from network_env import NetworkEnv  # noqa: E402
from topology import DESTINATION  # noqa: E402

OUT = os.path.join(OC.OUT_DIR, "env_drift.csv")
TRIALS = int(os.environ.get("ED_TRIALS", "200"))
KMAX = int(os.environ.get("ED_KMAX", "14"))
FP_EP = int(os.environ.get("ED_FP_EP", "60"))
LINE = "=" * 86


def fingerprint(env):
    """에피소드 시작 환경의 지문. 전 링크 지연 합."""
    return sum(float(q["delay"]) for q in env.qos.values())


def drift_curve():
    """같은 궤적을 k 틱 더 진행시키며 **고정 경로** 지표를 잰다.

    긴 경로가 겪는 것과 **동일한** 상황이다(같은 에피소드 안에서
    업데이트가 더 쌓인다). 정책은 전혀 안 쓴다.
    """
    random.seed(OC.EVAL_SEED)
    np.random.seed(OC.EVAL_SEED)
    ref = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                     risk_window=OC.WINDOW)
    ref.reset()
    path = ref.ospf_path(cost_type="hop")  # 홉비용은 위상만 → 고정
    rows = {k: [] for k in range(KMAX + 1)}
    for t in range(TRIALS):
        random.seed(OC.EVAL_SEED + 1000 * t)
        np.random.seed(OC.EVAL_SEED + 1000 * t)
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        for k in range(KMAX + 1):
            m = env.calculate_path_metrics(path)
            util = np.mean([q["utilization"] for q in env.qos.values()])
            rsk = np.mean([q["risk"] for q in env.qos.values()])
            rows[k].append((float(m["total_delay"]),
                            float(m["total_loss"]),
                            float(util), float(rsk)))
            env.update_dynamic_qos()
    return path, rows


def ols(xs, ys):
    n = float(len(xs))
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else float("nan")
    a = my - b * mx
    syy = sum((y - my) ** 2 for y in ys)
    pred = [a + b * x for x in xs]
    ssr = sum((y - p) ** 2 for y, p in zip(ys, pred))
    r2 = 1.0 - ssr / syy if syy else float("nan")
    return b, a, r2


def episode_fingerprints(cond, model_path, device, pool):
    """조건 하나로 FP_EP 에피소드를 굴리며 시작 환경 지문을 기록."""
    net = OC.load_net(model_path, device)
    random.seed(OC.EVAL_SEED)
    np.random.seed(OC.EVAL_SEED)
    torch.manual_seed(OC.EVAL_SEED)
    drng = random.Random(OC.EVAL_SEED + 7)
    prng = np.random.RandomState(OC.EVAL_SEED + 13)
    out = []
    for _ep in range(FP_EP):
        donor = pool[drng.randrange(len(pool))]
        perm = prng.permutation(50)
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        fp = fingerprint(env)
        done = False
        nstep = 0
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            q = OC.qvals(net, RV.make_fed(state, cond, donor, perm), device)
            state, _r, done, _i = env.step(
                OC.masked_argmax_from_q(q, list(valid)))
            nstep += 1
        out.append((fp, nstep, 1 if env.current_node == DESTINATION else 0))
    return out


def main():
    device = torch.device("cpu")
    print(LINE)
    print("[0] 환경 누산기 — `step`:300 이 매 스텝 `update_dynamic_qos` 호출")
    print("-" * 86)
    print("  시나리오 %s · WARMUP %d · 시행 %d · k 최대 %d"
          % (OC.SCENARIO, OC.WARMUP, TRIALS, KMAX))

    path, rows = drift_curve()
    hops = float(len(path) - 1)
    print("  고정 경로(위상 최단, 정책 무관): %s (%d홉)"
          % (path, len(path) - 1))

    print()
    print("[1] 고정 경로의 지표가 누산 틱 k 에 따라 어떻게 변하나")
    print("-" * 86)
    print("%4s %12s %12s %12s %12s"
          % ("k", "지연합", "손실(홉당)", "평균이용률", "평균위험"))
    ks, dls, lss = [], [], []
    for k in range(KMAX + 1):
        v = rows[k]
        d = sum(r[0] for r in v) / len(v)
        ls = sum(r[1] for r in v) / len(v)
        u = sum(r[2] for r in v) / len(v)
        rk = sum(r[3] for r in v) / len(v)
        print("%4d %12.3f %12.6f %12.4f %12.4f" % (k, d, ls, u, rk))
        ks.append(float(k))
        dls.append(d)
        lss.append(ls)

    print()
    print("  증분(틱 하나당 실제 변화) — 선형인지 본다:")
    inc = "    " + " ".join("%+.2f" % (dls[i + 1] - dls[i])
                            for i in range(len(dls) - 1))
    print(inc)

    bd, _ad, r2d = ols(ks, dls)
    bl, _al, r2l = ols(ks, lss)
    # **전 구간 OLS 는 쓰면 안 된다.** 링크당 지연이 clip 상한(120)으로
    # 포화하므로 k=0~4 가 과도구간이고, 전 구간 기울기는 한계기울기를
    # 크게 과대평가한다. 실제 에피소드는 reset(1틱)+WARMUP(10틱)+h스텝
    # = 11+h 틱에서 측정되므로 관련 구간은 k=h≈7~9, 즉 **포화 이후**다.
    K0 = 6
    bd2, _a2, r2d2 = ols(ks[K0:], dls[K0:])
    bl2, _b2, r2l2 = ols(ks[K0:], lss[K0:])
    print()
    print("  누산 기울기(고정 경로, %.0f홉):" % hops)
    print("    전 구간 k=0~%d : 지연 %+.4f ms/틱 (r² %.3f) "
          "— **과도구간 포함, 쓰지 말 것**" % (KMAX, bd, r2d))
    print("    포화 후 k=%d~%d : 지연 **%+.4f ms/틱** (r² %.3f) "
          "= 링크당 %+.4f" % (K0, KMAX, bd2, r2d2, bd2 / hops))
    print("    포화 후 k=%d~%d : 손실 %+.8f /틱 (r² %.3f)"
          % (K0, KMAX, bl2, r2l2))

    print()
    print("[2] 23.9.6.10 의 기울기 분해 — +31.17 ms/홉 중 누산 몫")
    print("-" * 86)
    print("  홉이 하나 늘면 (a) 링크 하나를 더 더하고 (b) 전 링크가 틱 하나")
    print("  더 받는다. (b) 몫 = 링크당 틱기울기 × 평균 경로 홉수.")
    LOSS_B_C = -0.00860   # 23.9.6.10 (C) 무해 조건만
    DELAY_B_C = 31.17
    hbar = 7.6            # 무해 조건 평균 홉
    per_link_d = bd2 / hops
    per_link_l = bl2 / hops
    print("    평균 홉 %.1f 기준(포화 후 기울기 사용):" % hbar)
    print("      지연 누산 몫 %+.3f ms/홉 → (C) %.2f 의 **%.1f%%**"
          % (per_link_d * hbar, DELAY_B_C,
             100.0 * per_link_d * hbar / DELAY_B_C))
    print("      손실 누산 몫 %+.8f /홉 → (C) %.5f 의 **%.2f%%**"
          % (per_link_l * hbar, LOSS_B_C,
             100.0 * per_link_l * hbar / LOSS_B_C))
    print("    (과도구간 포함 기울기를 쓰면 지연 몫이 %.1f%% 로 "
          "뻥튀김 — 약 %.1f배 과대)"
          % (100.0 * (bd / hops) * hbar / DELAY_B_C,
             (bd / hops) / (per_link_d) if per_link_d else float("nan")))
    print("    이용률은 누산이 아니라 **매 틱 재추출**(`traffic = ratio *")
    print("    capacity`)이라 정상이다 — 위 표에서 0.58 고정. 손실도 ")
    print("    `max(0,util-0.8)*0.03` 이 util 0.58 에서 **0** 이라 ")
    print("    평균 0 랜덤워크이고, 표에서 k 에 따라 안 움직인다.")

    print()
    print("[3] 조건 간 **에피소드 시작 환경**이 같은가 (지문)")
    print("-" * 86)
    models = []
    for arm, d, pat in OC.ARMS:
        for r in OC.RUNS:
            q = os.path.join(d, pat % r)
            if os.path.exists(q):
                models.append(("%s_run%02d" % (arm, r), q))
    refnet = OC.load_net(models[0][1], device)
    pool = [s for s, _ in OC.build_battery("greedy", device, refnet)]
    mname, mpath = models[0]
    CONDS = ("intact", "risk_rankpow05n", "risk_rankpow05")
    fps = {}
    for c in CONDS:
        fps[c] = episode_fingerprints(c, mpath, device, pool)
        print("  %-18s 모델 %s · 도달 %d/%d · 평균스텝 %.3f"
              % (c, mname, sum(r[2] for r in fps[c]), FP_EP,
                 sum(r[1] for r in fps[c]) / float(FP_EP)))
    base = fps["intact"]
    print()
    print("%4s %14s %14s %14s %7s %9s"
          % ("ep", "지문 intact", "지문 05n", "지문 pow05", "스텝", "일치"))
    first_diff = {c: None for c in CONDS[1:]}
    ndiff = {c: 0 for c in CONDS[1:]}
    for ep in range(FP_EP):
        same = all(abs(fps[c][ep][0] - base[ep][0]) < 1e-9
                   for c in CONDS[1:])
        for c in CONDS[1:]:
            if abs(fps[c][ep][0] - base[ep][0]) >= 1e-9:
                ndiff[c] += 1
                if first_diff[c] is None:
                    first_diff[c] = ep
        if ep < 6:
            print("%4d %14.4f %14.4f %14.4f %7d %9s"
                  % (ep, base[ep][0], fps["risk_rankpow05n"][ep][0],
                     fps["risk_rankpow05"][ep][0], base[ep][1],
                     "같음" if same else "**다름**"))
    print()
    for c in CONDS[1:]:
        print("  %-18s 지문 다른 에피소드 %d/%d · 첫 불일치 ep %s"
              % (c, ndiff[c], FP_EP, first_diff[c]))

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value"])
        w.writerow(["path", "nodes", "-".join(str(x) for x in path)])
        for k in range(KMAX + 1):
            v = rows[k]
            w.writerow(["drift", "k%d_delay" % k,
                        "%.6f" % (sum(r[0] for r in v) / len(v))])
            w.writerow(["drift", "k%d_loss" % k,
                        "%.8f" % (sum(r[1] for r in v) / len(v))])
            w.writerow(["drift", "k%d_util" % k,
                        "%.6f" % (sum(r[2] for r in v) / len(v))])
            w.writerow(["drift", "k%d_risk" % k,
                        "%.6f" % (sum(r[3] for r in v) / len(v))])
        w.writerow(["slope", "delay_per_tick_allk_UNUSABLE", "%.6f" % bd])
        w.writerow(["slope", "delay_per_tick_allk_r2", "%.4f" % r2d])
        w.writerow(["slope", "delay_per_tick_sat", "%.6f" % bd2])
        w.writerow(["slope", "delay_per_tick_sat_r2", "%.4f" % r2d2])
        w.writerow(["slope", "delay_per_link_per_tick_sat",
                    "%.6f" % (bd2 / hops)])
        w.writerow(["slope", "loss_per_tick_allk", "%.8f" % bl])
        w.writerow(["slope", "loss_per_tick_sat", "%.8f" % bl2])
        w.writerow(["slope", "loss_per_tick_sat_r2", "%.4f" % r2l2])
        w.writerow(["slope", "sat_k_from", K0])
        w.writerow(["slope", "fixed_path_hops", "%.1f" % hops])
        w.writerow(["slope", "trials", TRIALS])
        for c in CONDS:
            w.writerow(["fp", "%s_mean_steps" % c,
                        "%.4f" % (sum(r[1] for r in fps[c]) / float(FP_EP))])
        for c in CONDS[1:]:
            w.writerow(["fp", "%s_ndiff" % c, ndiff[c]])
            w.writerow(["fp", "%s_first_diff" % c, first_diff[c]])
        w.writerow(["fp", "episodes", FP_EP])
    print()
    print("저장: %s" % OUT)
    print(LINE)


if __name__ == "__main__":
    main()
