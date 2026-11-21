# -*- coding: utf-8 -*-
"""표 10 의 '못 채운 행' 을 내 하네스로 실제 측정한다.

왜 필요한가
-----------
`build_table10_fill.py` 가 `Equal weights`, `K=3`, `K=10` 행에
"이 설정으로 학습한 모델이 없다" 고 적었다. **틀렸다.**
`SDN_DQN_new_env/models/` 에 `weight_equal`(5런), `weight_temporal`(5),
`window3`(10), `window5`(5), `window7`(8), `window10`(**1**)이 다 있다.

그럼 원본 `results/*_result.csv` 를 그대로 쓰면 되지 않나? **안 된다.**
그 CSV 들은 학습 스크립트 자신의 평가기에서 나왔고, 표 10 Full 지연
348.7 이 다른 하네스(동결 모델 1개 × 평가 시드)에서 차용된 것과 **같은
종류의 혼합**이 된다. 그래서 `eval_ablation_risky.py` 와 **완전히 같은
하네스**(EVAL_SEED 20261114, 200 에피소드/모델, 같은 지표 추출, 같은
phase 하한)로 다시 롤아웃한다 -> `table10_risky.csv` 의 열과 직접 비교
가능해진다.

막아야 하는 함정 둘 (롤아웃 전에 assert 로 막는다)
----------------------------------------------------
1. `risk_calculator.CUSTOM_WEIGHTS` 몽키패치가 죽으면 가중치 팔이
   **조용히 기본 가중치로** 평가된다 -> 위험값이 실제로 달라지는지 탐침.
2. `risk_window` 가 안 먹으면 모든 K 가 5 로 평가된다(이 저장소의
   default-arg binding 함정) -> `env.loss_history[e].maxlen == K` 확인.
"""
from __future__ import print_function

import argparse
import csv
import glob
import io
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# eval_ablation_risky 를 임포트하면 그 모듈의 sys.path 수술 + "원본이
# 아닌 사본이 임포트됐다" assert + 절제 훅 확인 + PHASE_MIN 계산이
# 그대로 재사용된다. main() 은 __main__ 가드 안이라 실행되지 않는다.
import eval_ablation_risky as E               # noqa: E402

risk_calculator = E.risk_calculator
network_env = E.network_env
T = E.T
torch = E.torch

ORIG = E.ORIG
EVAL_SEED = E.EVAL_SEED
FIELDS = E.FIELDS
PHASE_MIN = E.PHASE_MIN
RESULT_DIR = E.RESULT_DIR

WEIGHTS = {
    "equal": {"util": 0.20, "cong": 0.20, "delay": 0.20,
              "loss": 0.20, "type": 0.20},
    "temporal": {"util": 0.25, "cong": 0.20, "delay": 0.25,
                 "loss": 0.20, "type": 0.10},
}

# (라벨, 모델 접두사, risk_window, 가중치 키, 원고 표 10/12 의 행 이름)
ARMS = [
    ("W-Equal", "weight_equal", T.RISK_WINDOW, "equal",
     "Equal weights (0.2)"),
    ("W-Temporal", "weight_temporal", T.RISK_WINDOW, "temporal",
     "(표 12 전용)"),
    ("K=3", "window3", 3, None, "K = 3 (K/2)"),
    ("K=5", "window5", 5, None, "(K 스윕 자체 기준선)"),
    ("K=7", "window7", 7, None, "(원고에 없음 — 스윕 보강)"),
    ("K=10", "window10", 10, None, "K = 10 (2K)"),
]

PROBE = dict(utilization=0.7, is_congested=True,
             delay_history=[10.0, 40.0, 25.0, 60.0, 15.0],
             loss_history=[0.01, 0.06, 0.02, 0.09, 0.03],
             link_type="unstable")


def probe_weights(wkey):
    """가중치 몽키패치가 실제로 위험값을 움직이는지 확인한다."""
    risk_calculator.CUSTOM_WEIGHTS = None
    base = risk_calculator.calculate_link_risk(**PROBE)
    risk_calculator.CUSTOM_WEIGHTS = (None if wkey is None
                                      else dict(WEIGHTS[wkey]))
    got = risk_calculator.calculate_link_risk(**PROBE)
    if wkey is None:
        assert abs(got - base) < 1e-12, "가중치 미변경인데 위험값이 움직였다"
    else:
        assert abs(got - base) > 1e-9, (
            "CUSTOM_WEIGHTS 가 위험값을 바꾸지 않았다 - 몽키패치 사망 (%s)"
            % wkey)
    return base, got


def probe_window(k):
    """risk_window 가 실제로 이력 길이를 바꾸는지 확인한다."""
    env = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                 risk_window=k)
    e0 = list(env.loss_history.keys())[0]
    got_l = env.loss_history[e0].maxlen
    got_d = env.delay_history[e0].maxlen
    assert got_l == k and got_d == k, (
        "risk_window=%d 가 안 먹었다 (loss %s / delay %s)"
        % (k, got_l, got_d))
    assert env.state_size == 125, "상태 차원이 %d 다" % env.state_size
    return got_l


def evaluate_arm(model_path, risk_window, wkey, episodes):
    """E.evaluate 와 같은 절차. 다른 점은 risk_window/가중치를 받는 것뿐.

    E.evaluate 는 risk_window=T.RISK_WINDOW 로 못 박혀 있어서 K 팔에
    쓸 수 없다. 절제는 없으므로 ZEROED_FACTORS 는 항상 [] 다.
    """
    risk_calculator.ZEROED_FACTORS = []
    risk_calculator.CUSTOM_WEIGHTS = (None if wkey is None
                                      else dict(WEIGHTS[wkey]))

    base = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                  risk_window=risk_window)
    agent = T.DQNAgent(base.state_size, base.action_size)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device,
                   weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    out = dict((k, []) for k in FIELDS)
    reached = 0
    for i in range(episodes):
        random.seed(EVAL_SEED + i)
        np.random.seed(EVAL_SEED + i)
        torch.manual_seed(EVAL_SEED + i)

        env = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                     risk_window=risk_window)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            st = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.policy_net(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node != network_env.DESTINATION:
            continue
        reached += 1
        m = env.calculate_path_metrics(env.visited)
        hop = max(int(m["hop_count"]), 1)
        risky = max(int(m["unstable_count"]), int(m["congested_count"]))
        out["delay"].append(float(m["total_delay"]))
        out["loss"].append(float(m["total_loss"]))
        out["risk"].append(float(m["total_risk"]))
        out["hop"].append(float(m["hop_count"]))
        out["risky_cnt"].append(float(risky))
        out["risky_pct"].append(risky / hop * 100.0)
        out["loss_total"].append(float(m["total_loss"]) * hop)
        out["risk_total"].append(float(m["total_risk"]) * hop)
        floor = PHASE_MIN[env.dynamic_phase]
        out["risky_min"].append(float(floor))
        out["risky_excess"].append(float(risky - floor))
        out["at_floor"].append(1.0 if risky == floor else 0.0)

    risk_calculator.CUSTOM_WEIGHTS = None
    res = dict((k, float(np.mean(v)) if v else float("nan"))
               for k, v in out.items())
    res["success_rate"] = 100.0 * reached / episodes
    return res


def find_models(prefix):
    pat = os.path.join(ORIG, "models", "%s_run*.pth" % prefix)
    got = []
    for p in sorted(glob.glob(pat)):
        name = os.path.basename(p)
        mo = re.search(r"_run(\d+)\.pth$", name)
        if mo and name.startswith(prefix + "_run"):
            got.append((int(mo.group(1)), p))
    return sorted(got)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=T.EVAL_EPISODES)
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    print("=" * 92)
    print("표 10/12 의 '못 채운 행' 재측정 - eval_ablation_risky 와 같은 하네스")
    print("=" * 92)
    print("EVAL_SEED=%d, 에피소드/모델=%d, 지표 추출 동일"
          % (EVAL_SEED, args.episodes))
    print("")
    print("[사전 탐침] 몽키패치 둘이 살아 있는지 먼저 확인한다")
    print("-" * 92)
    for wk in (None, "equal", "temporal"):
        b, g = probe_weights(wk)
        print("  가중치 %-9s 위험값 %.6f -> %.6f  %s"
              % (str(wk), b, g,
                 "변함없음(정상)" if wk is None else "바뀜(정상)"))
    risk_calculator.CUSTOM_WEIGHTS = None
    for k in (1, 3, 5, 7, 10):
        print("  risk_window=%-2d 이력 maxlen=%d (정상)" % (k, probe_window(k)))
    print("")

    rows = []
    runs = []
    for label, prefix, rw, wkey, msrow in ARMS:
        if args.only and args.only not in label:
            continue
        mods = find_models(prefix)
        if not mods:
            print("!! %s: 모델 없음 (%s_run*.pth)" % (label, prefix))
            continue
        probe_window(rw)
        probe_weights(wkey)
        risk_calculator.CUSTOM_WEIGHTS = None
        print("[%s] 모델 %d개, risk_window=%d, 가중치=%s"
              % (label, len(mods), rw,
                 wkey or "기본(0.35/0.30/0.15/0.10/0.10)"))
        per = []
        for rid, p in mods:
            r = evaluate_arm(p, rw, wkey, args.episodes)
            per.append(r)
            row = {"arm": label, "run": rid,
                   "model": os.path.basename(p),
                   "risk_window": rw, "weights": wkey or "default"}
            row.update(dict((k, "%.6f" % r[k]) for k in FIELDS))
            row["success_rate"] = "%.4f" % r["success_rate"]
            runs.append(row)
            print("   run%02d  delay %8.3f  loss %.5f  hop %5.3f"
                  "  risky%% %6.3f  도달 %.1f%%"
                  % (rid, r["delay"], r["loss"], r["hop"],
                     r["risky_pct"], r["success_rate"]))
        agg = {"arm": label, "manuscript_row": msrow, "n": len(per),
               "risk_window": rw, "weights": wkey or "default"}
        for k in FIELDS + ("success_rate",):
            v = [x[k] for x in per]
            agg[k] = "%.6f" % float(np.mean(v))
            agg[k + "_std"] = ("%.6f" % float(np.std(v, ddof=1))
                               if len(v) > 1 else "")
        rows.append(agg)
        sd = ("" if len(per) < 2
              else " +- %.3f" % float(agg["delay_std"]))
        print("   -> 평균 delay %.3f%s  loss %.5f  hop %.3f  risky%% %.3f"
              % (float(agg["delay"]), sd, float(agg["loss"]),
                 float(agg["hop"]), float(agg["risky_pct"])))
        if len(per) < 2:
            print("   ** n=1 - 표준편차 없음. 표에 n 을 반드시 병기할 것. **")
        print("")

    if rows:
        p1 = os.path.join(RESULT_DIR, "table10_extra.csv")
        cols = (["arm", "manuscript_row", "n", "risk_window", "weights"]
                + [c for k in FIELDS + ("success_rate",)
                   for c in (k, k + "_std")])
        with io.open(p1, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("wrote %s" % p1)
    if runs:
        p2 = os.path.join(RESULT_DIR, "table10_extra_runs.csv")
        cols2 = (["arm", "run", "model", "risk_window", "weights"]
                 + list(FIELDS) + ["success_rate"])
        with io.open(p2, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols2)
            w.writeheader()
            for r in runs:
                w.writerow(r)
        print("wrote %s  (모델별 - Welch 검정용)" % p2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
