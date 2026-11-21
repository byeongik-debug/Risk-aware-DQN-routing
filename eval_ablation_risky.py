# -*- coding: utf-8 -*-
"""Table 10's Risky% column, from the already-trained ablation models.

Why this needs no retraining
----------------------------
train_factor_ablation.py's evaluation already computes everything we need --
calculate_path_metrics() returns unstable_count and congested_count -- but the
trainer keeps only delay/loss/hop/success_rate and throws the counts away. So
Risky% can be recovered by replaying the same evaluation with the saved
weights. No script in the main pipeline has ever produced a Risky% number for
Table 10 or Table 11; the only existing definition lives in the R3-4 runner,
and this script copies it exactly
(run_topology_generalization_dynamic.py:142-167):

    hop_count   = max(int(metrics["hop_count"]), 1)
    risky_count = max(int(metrics["unstable_count"]),
                      int(metrics["congested_count"]))
    risky_link_usage = risky_count / hop_count * 100.0

In the dynamic scenario the two counters track the same set, so the max() is a
no-op there; it is kept for identity with the published definition.

What the metrics mean (network_env.py:448-456)
----------------------------------------------
    "total_delay": total_delay,              <- path SUM
    "total_loss":  total_loss / hop_count,   <- per-hop MEAN
    "total_risk":  total_risk / hop_count,   <- per-hop MEAN

Delay is a sum and Loss/Risk are means, so hop count moves them in opposite
directions by construction. Risky% is also per-hop normalized, so extra hops
dilute it unless those hops are themselves dangerous. This script therefore
records both the normalized value and the raw count, and reports Loss and Risk
on both bases (per-hop as published, and times hop as the path total).

Three things this does that the original evaluation did not
-----------------------------------------------------------
1. It restores risk_calculator.ZEROED_FACTORS to the model's own ablation. The
   state vector contains the risk score, so evaluating a -Util model under the
   full risk function would feed it states it never saw in training. The
   trainer got this right implicitly because training and evaluation shared
   one process.
2. It seeds PER EPISODE, not once before the loop. Seeding once is not enough:
   the number of random draws a rollout consumes depends on how many hops the
   policy walks, so after episode 1 the environments would desynchronize
   between models -- precisely between the models whose path lengths differ
   most. With a per-episode seed every model faces the identical 200 initial
   conditions.
3. It cross-checks the recomputed delay/loss against the committed per-run
   CSVs. Those came from an unseeded draw, so they will not match exactly; the
   check is that they agree within run-to-run spread.

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE python -X utf8 eval_ablation_risky.py
        [--episodes 50]
"""
import argparse
import csv
import glob
import os
import random
import sys

import networkx as nx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))

# 자기 디렉터리를 치우고 원본을 맨 앞에 — review_response/risk_calculator.py 는
# 절제 훅이 없는 구버전이라, 그놈이 임포트되면 절제가 조용히 무효가 된다.
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import risk_calculator                       # noqa: E402
import network_env                           # noqa: E402
import train_factor_ablation as T            # noqa: E402
import torch                                 # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

for _m in (risk_calculator, network_env, T):
    _got = os.path.abspath(_m.__file__)
    assert _got.startswith(ORIG), "원본이 아닌 사본이 임포트됐다: %s" % _got
assert hasattr(risk_calculator, "ZEROED_FACTORS"), "절제 훅이 없다"

sys.path.insert(0, HERE)
import review_stats as RS                    # noqa: E402

EVAL_SEED = 20261114
RESULT_DIR = os.path.join(HERE, "review_results")
TEMPORAL = ["delay_inst", "loss_inst"]

# (factor_key, label, 모델 디렉터리)
SETTINGS = [
    ("full", "Full", os.path.join(HERE, "models_ablation")),
    ("util", "-Util", os.path.join(ORIG, "models")),
    ("cong", "-Cong", os.path.join(ORIG, "models")),
    ("delay_inst", "-dDelay", os.path.join(ORIG, "models")),
    ("loss_inst", "-dLoss", os.path.join(ORIG, "models")),
    ("link_type", "-Type", os.path.join(ORIG, "models")),
    ("temporal", "-VD,VL", os.path.join(HERE, "models_ablation")),
]
FIELDS = ("delay", "loss", "risk", "hop", "risky_cnt", "risky_pct",
          "loss_total", "risk_total", "risky_min", "risky_excess",
          "at_floor")


def phase_minima():
    """phase마다 '어떤 경로로도 더 적게 밟을 수 없는' 위험 링크 수를 풀어서 구한다.

    위험 간선에 1, 안전 간선에 0 가중을 주고 최단경로를 구하면 그 비용이
    최소 적중 수다. 하드코딩이 아니라 토폴로지와 DYNAMIC_PHASES에서 유도한다.
    """
    def norm(u, v):
        return tuple(sorted((u, v)))

    out = {}
    for ph, info in network_env.DYNAMIC_PHASES.items():
        danger = set(norm(*e) for e in info["danger"])
        h = nx.Graph()
        for u, v in network_env.EDGES:
            h.add_edge(u, v, w=1.0 if norm(u, v) in danger else 0.0)
        best = nx.shortest_path(h, network_env.SOURCE,
                                network_env.DESTINATION, weight="w")
        out[ph] = sum(1 for a, b in zip(best[:-1], best[1:])
                      if norm(a, b) in danger)
    return out


PHASE_MIN = phase_minima()


def probe_hook(zeroed):
    """절제가 실제로 위험도를 움직이는지(full이면 안 움직이는지) 확인."""
    probe = dict(utilization=0.7, is_congested=True,
                 delay_history=[10.0, 40.0, 25.0, 60.0, 15.0],
                 loss_history=[0.01, 0.06, 0.02, 0.09, 0.03],
                 link_type="unstable")
    risk_calculator.ZEROED_FACTORS = []
    base = risk_calculator.calculate_link_risk(**probe)
    risk_calculator.ZEROED_FACTORS = list(zeroed)
    got = risk_calculator.calculate_link_risk(**probe)
    if zeroed:
        assert abs(got - base) > 1e-9, "절제가 위험도를 바꾸지 않았다 — 훅 미적용"
    else:
        assert abs(got - base) < 1e-12, "절제가 없는데 위험도가 움직였다"
    return base, got


def evaluate(model_path, zeroed, episodes):
    """train_factor_ablation.py 의 평가를 그대로 재생하고 위험 링크 수를 보존."""
    risk_calculator.ZEROED_FACTORS = list(zeroed)
    base = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                  risk_window=T.RISK_WINDOW)
    agent = T.DQNAgent(base.state_size, base.action_size)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device,
                   weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    out = dict((k, []) for k in FIELDS)
    reached = 0
    for i in range(episodes):
        # 에피소드마다 시드 — 롤아웃 길이가 난수 소비량을 바꾸므로
        # 루프 앞 한 번으로는 모델 간 짝짓기가 깨진다.
        random.seed(EVAL_SEED + i)
        np.random.seed(EVAL_SEED + i)
        torch.manual_seed(EVAL_SEED + i)

        env = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                     risk_window=T.RISK_WINDOW)
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

    risk_calculator.ZEROED_FACTORS = []
    res = dict((k, float(np.mean(v)) if v else float("nan"))
               for k, v in out.items())
    res["success_rate"] = 100.0 * reached / episodes
    return res


def published(key):
    """커밋된 per-run CSV 평균 — 교차 확인용."""
    paths = []
    for d in (os.path.join(ORIG, "results"), RESULT_DIR):
        paths += glob.glob(os.path.join(
            d, "factor_%s_dynamic_run*_result.csv" % key))
    vals = {"delay": [], "loss": [], "hop": []}
    for p in sorted(set(paths)):
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                for k in vals:
                    vals[k].append(float(r[k]))
    if not vals["delay"]:
        return None
    return dict((k, float(np.mean(v))) for k, v in vals.items()), \
        len(vals["delay"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=T.EVAL_EPISODES)
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()

    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    rows = []
    for key, label, mdir in SETTINGS:
        zeroed = T.FACTOR_MAP.get(key, TEMPORAL if key == "temporal" else None)
        if zeroed is None:
            print("[skip] %s: FACTOR_MAP에 없다" % key)
            continue
        paths = [os.path.join(mdir, "factor_%s_run%02d.pth" % (key, r))
                 for r in range(1, args.runs + 1)]
        have = [p for p in paths if os.path.exists(p)]
        if not have:
            print("[skip] %-7s 모델 없음 (%s)" % (label, mdir))
            continue
        b, g = probe_hook(zeroed)
        print("[eval] %-7s zeroed=%-26s 모델 %d개  탐침 %.4f -> %.4f"
              % (label, ",".join(zeroed) or "(없음)", len(have), b, g))
        per = []
        for p in have:
            r = evaluate(p, zeroed, args.episodes)
            per.append(r)
            print("    %-26s hop=%.2f  risky=%.3f(최소 %.3f 초과 %+.3f "
                  "최소달성 %.1f%%)  delay=%.1f  loss=%.5f  succ=%.1f%%"
                  % (os.path.basename(p), r["hop"], r["risky_cnt"],
                     r["risky_min"], r["risky_excess"],
                     100.0 * r["at_floor"], r["delay"], r["loss"],
                     r["success_rate"]))
        agg = {"factor": label, "key": key, "n": len(per)}
        for k in FIELDS + ("success_rate",):
            m, s = RS.mean_std([x[k] for x in per])
            agg[k], agg[k + "_std"] = m, s
        pub = published(key)
        if pub:
            pv, pn = pub
            agg["pub_delay"], agg["pub_loss"] = pv["delay"], pv["loss"]
            agg["pub_hop"], agg["pub_n"] = pv["hop"], pn
        rows.append(agg)

    if not rows:
        print("\n평가할 모델이 없다.")
        return 1

    print("\n표 10 Risky%% 열  (에피소드별 시드 %d+i, %d에피소드, 모델 n=5)"
          % (EVAL_SEED, args.episodes))
    hdr = ("%-7s | %6s | %13s | %13s | %9s | %13s"
           % ("factor", "홉", "Risky%(홉당)", "위험링크수", "Delay(합)",
              "Loss(홉당)"))
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-7s | %6.3f | %6.2f±%-6.2f | %6.3f±%-6.3f | %9.2f | "
              "%7.5f±%-7.5f"
              % (r["factor"], r["hop"], r["risky_pct"], r["risky_pct_std"],
                 r["risky_cnt"], r["risky_cnt_std"], r["delay"],
                 r["loss"], r["loss_std"]))

    print("\n분자가 포화됐는가 — 이론적 최소 적중 대비")
    print("  phase별 최소 적중(토폴로지에서 유도): %s" % PHASE_MIN)
    hdr3 = ("%-7s | %11s | %11s | %11s | %12s"
            % ("factor", "실제 적중", "최소 적중", "초과", "최소달성률"))
    print(hdr3)
    print("-" * len(hdr3))
    for r in rows:
        print("%-7s | %11.4f | %11.4f | %+11.4f | %11.2f%%"
              % (r["factor"], r["risky_cnt"], r["risky_min"],
                 r["risky_excess"], 100.0 * r["at_floor"]))

    print("\n희석을 되돌린 값 — Risky%는 홉당 비율이므로 홉이 늘면 묽어진다")
    hdr2 = ("%-7s | %6s | %12s | %12s | %13s | %12s"
            % ("factor", "홉", "Risky%(홉당)", "위험링크수", "Loss×홉",
               "Risk×홉"))
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        print("%-7s | %6.3f | %12.2f | %12.3f | %13.6f | %12.6f"
              % (r["factor"], r["hop"], r["risky_pct"], r["risky_cnt"],
                 r["loss_total"], r["risk_total"]))
    print("  주의: Risk x 홉 열은 설정 간 비교가 불가능하다. total_risk 는 그")
    print("  설정의 '절제된' 위험 함수로 계산되므로 요인마다 척도가 다르다")
    print("  (위 탐침: 0.5688 ~ 0.8688, 최대 35% 차이). 설정 간 비교가 되는")
    print("  위험 지표는 위험링크수 하나뿐이다 — 그건 토폴로지의 위험 집합으로")
    print("  정의되고 위험 함수를 쓰지 않는다. 그리고 그 값은 전부 같다.")

    print("\n재평가 대 커밋값 (에피소드 추출이 달라 정확히 같을 수는 없다)")
    print("%-7s | %19s | %21s | %13s" % ("factor", "delay 재/커밋",
                                         "loss 재/커밋", "hop 재/커밋"))
    for r in rows:
        if "pub_delay" not in r:
            print("%-7s | %19s | %21s | %13s"
                  % (r["factor"], "(커밋값 없음)", "-", "-"))
            continue
        print("%-7s | %8.1f / %8.1f | %9.5f / %9.5f | %5.2f / %5.2f"
              % (r["factor"], r["delay"], r["pub_delay"],
                 r["loss"], r["pub_loss"], r["hop"], r["pub_hop"]))

    out = os.path.join(RESULT_DIR, "table10_risky.csv")
    cols = ["factor", "key", "n"]
    for k in FIELDS + ("success_rate",):
        cols += [k, k + "_std"]
    cols += ["pub_delay", "pub_loss", "pub_hop", "pub_n"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
