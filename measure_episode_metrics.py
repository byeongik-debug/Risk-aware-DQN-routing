# -*- coding: utf-8 -*-
"""2×2 네 칸(+`scale054`)을 **에피소드 단위**로 기록한다.

왜 필요한가. 23.9.6.10 의 '합 버전' 괄호는
`mean(loss_i) × mean(hop_i)` 였다 — 에피소드별 값이 CSV 에 없어서
`mean(loss_i × hop_i)` 를 못 냈고, 그 차이(Jensen 격차)를 **한계로
적어 두기만** 했다. `calculate_path_metrics` 는 손실을
`total_loss / hop_count` 로 돌려주므로(`network_env.py:453`),
에피소드별로 받아 `× hop_count` 하면 종단간 합이 **근사 없이** 나온다.

그래서 이 스크립트는 집계값이 아니라 **에피소드 행**을 남긴다. 앞으로
"에피소드 수준에서 보면?" 류 질문에 롤아웃을 다시 돌릴 필요가 없다.

함정 기록: `calculate_path_metrics` 는 경로에 **고리가 있으면 같은
간선을 두 번 더한다**(`diagnose_generalization_failure.py:168`).
23.9.6.12 에서 `pow05` 만 재방문 2.8% 이므로 `has_rev` 열을 같이
남겨 분리할 수 있게 한다.

[2026-11-14 추가] 기준 경로 열. 23.9.6.4 는 "코사인이 피해 **유무**는
맞추고 **양**은 못 맞춘다" 로 끝났는데, 양을 말할 **분모**가 없었다.
도착 시점 그 환경에서 `ospf_path(cost_type=...)` 네 가지를 뽑으면
"그 환경에서 달성 가능한 범위" 가 생긴다. Dijkstra 는 RNG 를 안 쓰므로
롤아웃은 비트 단위로 그대로여야 하고, 겹치는 9열이 기존 CSV 와
**정확히 일치하는지**로 그걸 검증한다(일치하지 않으면 기준 열은 못 쓴다).

출력: review_results/$EM_OUT (기본 episode_metrics.csv)
환경변수: EM_EPISODES(기본 200) · EM_CONDS(기본 2×2 + scale054)
          EM_OUT(출력 파일명 — 기존 10,000행 CSV 를 덮지 않게)
"""
import csv
import io
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402

sys.path.append(OC.ORIG)
# OC 는 import 시 sys.path 에서 HERE 를 **빼낸다**(원본 섀도잉 방지,
# occlude_state_blocks.py:36). 그래서 다시 붙여야 review_response 안의
# 모듈을 import 할 수 있다.
sys.path.append(HERE)

import measure_rank_vs_value as RV  # noqa: E402
from network_env import NetworkEnv  # noqa: E402
from topology import DESTINATION  # noqa: E402

OUT_DIR = OC.OUT_DIR
EPISODES = int(os.environ.get("EM_EPISODES", "200"))
DEFAULT = ("intact", "risk_scale184", "risk_rankpow05n",
           "risk_rankpow05", "risk_scale054")
CONDS = tuple(x.strip() for x in
              os.environ.get("EM_CONDS", ",".join(DEFAULT)).split(",")
              if x.strip())
OUTNAME = os.environ.get("EM_OUT", "episode_metrics.csv")
# 도착 **그 시점 그 환경**에서 달성 가능한 기준 경로들. 23.9.6.4 가
# 남긴 "양을 설명 못 한다" 에 **분모**를 주기 위한 것이다.
# `ospf_path` 는 Dijkstra 라 RNG 를 **안 쓰므로** 롤아웃은 비트 단위로
# 그대로여야 한다 — 겹치는 열이 기존 CSV 와 일치하는지로 검증한다.
# `bandwidth` 는 논문의 모든 평가 스크립트가 쓰는 비용이고(메모리:
# eval_qos.py:88 등), `delay` 는 그 환경의 지연 최적이다.
REFS = ("delay", "hop", "bandwidth", "risk")
REFCOLS = tuple("ref_%s_%s" % (ct, sfx) for ct in REFS
                for sfx in ("delay", "lossph", "losssum", "hop"))
BASECOLS = ("condition", "model", "episode", "delay", "loss_perhop",
            "loss_sum", "hop", "hop_count", "has_rev")


def run_one(net, device, cond, donor_pool, episodes, seed):
    """조건 하나. 에피소드별 행 리스트를 돌려준다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    drng = random.Random(seed + 1)
    prng = np.random.RandomState(seed + 2)

    rows = []
    for ep in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        donor = donor_pool[drng.randrange(len(donor_pool))]
        perm = prng.permutation(RV.HI - RV.LO)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            fed = RV.make_fed(state, cond, donor, perm)
            q = OC.qvals(net, fed, device)
            action = OC.masked_argmax_from_q(q, list(valid))
            state, _, done, _ = env.step(action)
        if env.current_node != DESTINATION:
            continue
        vis = list(env.visited)
        hop = max(len(vis) - 1, 1)
        uniq = len(set(vis))
        m = env.calculate_path_metrics(vis)
        lp = float(m["total_loss"])
        row = {
            "episode": ep,
            "delay": float(m["total_delay"]),
            "loss_perhop": lp,
            # `total_loss / hop_count` 를 되돌린 **정확한** 종단간 합.
            "loss_sum": lp * hop,
            "hop": float(len(vis) - 1),
            "hop_count": float(hop),
            "has_rev": 1 if len(vis) != uniq else 0,
        }
        # 기준 경로는 **같은 env 인스턴스·도착 그 시점**에서 뽑는다.
        # `ospf_path` 는 SOURCE→DESTINATION 고정이라 현재 노드와 무관
        # 하고, `nx.shortest_path` 는 결정론적이라 RNG 를 안 쓴다.
        # 기준 경로 손실도 `/hop_count` 로 나오므로 홉당·합 둘 다 남긴다.
        for ct in REFS:
            rp = env.ospf_path(cost_type=ct)
            rm = env.calculate_path_metrics(rp)
            rh = max(len(rp) - 1, 1)
            row["ref_%s_delay" % ct] = float(rm["total_delay"])
            row["ref_%s_lossph" % ct] = float(rm["total_loss"])
            row["ref_%s_losssum" % ct] = float(rm["total_loss"]) * rh
            row["ref_%s_hop" % ct] = float(len(rp) - 1)
        rows.append(row)
    return rows


def main():
    device = torch.device("cpu")
    # 23.9.6.12 와 **같은 엄격 비교군 10개**를 써야 홉이 대조된다.
    models = []
    for arm, d, pat in OC.ARMS:
        for r in OC.RUNS:
            q = os.path.join(d, pat % r)
            if os.path.exists(q):
                models.append(("%s_run%02d" % (arm, r), q))
    if len(models) != 10:
        raise SystemExit("엄격 비교군 10개가 아니다: %d" % len(models))
    ref = OC.load_net(models[0][1], device)
    pool = [s for s, _ in OC.build_battery("greedy", device, ref)]
    path = os.path.join(OUT_DIR, OUTNAME)
    bar = "=" * 86
    print(bar)
    print("에피소드 단위 측정 — 조건 %d개 × 모델 %d개 × %d 에피소드"
          % (len(CONDS), len(models), EPISODES))
    print("  평가시드 %d · 시나리오 %s · window %d · warmup %d"
          % (OC.EVAL_SEED, OC.SCENARIO, OC.WINDOW, OC.WARMUP))
    print(bar)
    n = 0
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(list(BASECOLS) + list(REFCOLS))
        for cond in CONDS:
            print("")
            print("[%s]" % cond)
            for name, ckpt in models:
                net = OC.load_net(ckpt, device)
                rows = run_one(net, device, cond, pool, EPISODES,
                               OC.EVAL_SEED)
                for r in rows:
                    # 앞 9열은 **형식까지 그대로** 둔다. 기존
                    # episode_metrics.csv 와 바이트 단위로 맞는지가
                    # "Dijkstra 가 롤아웃을 안 건드린다" 의 검증이다.
                    out = [cond, name, r["episode"],
                           "%.6f" % r["delay"],
                           "%.8f" % r["loss_perhop"],
                           "%.8f" % r["loss_sum"],
                           "%.1f" % r["hop"], "%.1f" % r["hop_count"],
                           r["has_rev"]]
                    for c in REFCOLS:
                        if c.endswith("_hop"):
                            out.append("%.1f" % r[c])
                        elif "loss" in c:
                            out.append("%.8f" % r[c])
                        else:
                            out.append("%.6f" % r[c])
                    w.writerow(out)
                n += len(rows)
                print("  %-20s 도달 %3d/%3d · 홉 %.3f · 손실합 %.5f"
                      % (name, len(rows), EPISODES,
                         sum(r["hop"] for r in rows) / max(len(rows), 1),
                         sum(r["loss_sum"] for r in rows)
                         / max(len(rows), 1)))
                fh.flush()
    print("")
    print("행 %d개 저장: %s" % (n, path))
    print(bar)


if __name__ == "__main__":
    main()
