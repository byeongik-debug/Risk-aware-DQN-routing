# -*- coding: utf-8 -*-
"""23.9.6.9 — 홉수는 '우회' 와 '재방문(고리)' 을 섞고 있다. 분리해서 센다.

왜 필요한가
-----------
`SDN_DQN_new_env/network_env.py` 의 `get_valid_actions` 는 현재 노드의
**이웃 전부**를 돌려준다(264-266행). 방문한 노드를 빼지 않으므로
**재방문이 허용**되고, `calculate_path_metrics(self.visited)` 는 그
되돌아간 엣지까지 더한다. 그래서 `hop_count` 한 숫자에
  (ㄱ) 더 긴 '우회' 경로를 고른 것과
  (ㄴ) 같은 노드를 다시 밟는 '고리' 가
섞여 있다. 23.9 의 "짧고 나쁜 경로" 서술과 표 10 계열 홉 수치가 전부
이 숫자를 쓴다. 섞인 채로 해석하면 안 되므로 **갈라서 센다**.

무엇을 세는가 (에피소드마다)
--------------------------
- `hop_count`            : 지금 쓰는 숫자 (= len(visited)-1)
- 고리 뺀 길이            : len(set(visited))-1  (서로 다른 노드 수 - 1)
- 재방문 스텝 수          : len(visited) - len(set(visited))
- 재방문 있는 에피소드 비율
- 위상 최단경로 길이과 같은 에피소드 비율

한계를 먼저 적는다
-----------------
- 고리를 뺀 길이는 '방문 집합' 기준이라, 고리를 돌고 나온 뒤 실제로
  더 긴 경로를 쓴 경우를 **과소평가**한다(집합이라 순서를 버린다).
  그래서 '재방문 스텝 수' 를 같이 낸다.
- MAX_STEPS=35 이고 관측 홉은 7~9 라 절단은 없다. 성공률 100%는
  rank_vs_value.csv 전 조건에서 이미 확인했다.
- 조건은 기본 `intact` 하나다. `REV_CONDS` 로 늘릴 수 있지만 롤아웃
  비용이 조건 수에 비례한다.

출력: review_results/revisit_rate.csv
"""
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import occlude_state_blocks as OC                   # noqa: E402

sys.path.append(OC.ORIG)
# OC 는 import 시 sys.path 에서 HERE 를 **빼낸다**(원본 모듈 섀도잉
# 방지, occlude_state_blocks.py:36). 그래서 여기서 다시 붙여야
# review_response 안의 모듈을 import 할 수 있다.
sys.path.append(HERE)
import numpy as np                                  # noqa: E402
import torch                                        # noqa: E402
import networkx as nx                               # noqa: E402
import measure_rank_vs_value as RV                  # noqa: E402
from network_env import NetworkEnv                  # noqa: E402
from network_env import DESTINATION                 # noqa: E402

EPISODES = int(os.environ.get("REV_EPISODES", "100"))
CONDS = tuple(x.strip() for x in
              os.environ.get("REV_CONDS", "intact").split(",") if x.strip())


def run_one(net, device, cond, donor_pool, episodes, seed):
    """조건 하나에 대해 에피소드별 경로 통계를 모은다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    drng = random.Random(seed + 1)
    prng = np.random.RandomState(seed + 2)

    rec = {"hop": [], "loopfree": [], "revisit": [], "has_rev": [],
           "is_short": [], "reached": 0, "sp": []}
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        donor = donor_pool[drng.randrange(len(donor_pool))]
        perm = prng.permutation(RV.HI - RV.LO)
        src = env.visited[0] if env.visited else env.current_node
        try:
            sp = nx.shortest_path_length(env.graph, src, DESTINATION)
        except nx.NetworkXNoPath:
            sp = -1
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
        rec["reached"] += 1
        vis = list(env.visited)
        hop = len(vis) - 1
        uniq = len(set(vis))
        rec["hop"].append(float(hop))
        rec["loopfree"].append(float(uniq - 1))
        rec["revisit"].append(float(len(vis) - uniq))
        rec["has_rev"].append(1.0 if len(vis) != uniq else 0.0)
        rec["sp"].append(float(sp))
        rec["is_short"].append(1.0 if hop == sp else 0.0)
    return rec


def mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()

    models = []
    for arm, d, pat in OC.ARMS:
        for r in OC.RUNS:
            p = os.path.join(d, pat % r)
            if os.path.exists(p):
                models.append(("%s_run%02d" % (arm, r), p))
    if len(models) != 10:
        raise SystemExit("엄격 비교군 10개가 아니다: %d" % len(models))
    nets = [(nm, OC.load_net(p, device)) for nm, p in models]
    battery = OC.build_battery("greedy", device, nets[0][1])
    donor_pool = [s for s, _ in battery]
    print("모델 %d · 에피소드 %d · 조건 %s · 시드 %d"
          % (len(nets), EPISODES, ",".join(CONDS), RV.EVAL_SEED))
    print("")

    rows = []
    bar = "=" * 86
    for cond in CONDS:
        print("%-18s %7s %9s %9s %9s %9s %8s"
              % ("모델 (%s)" % cond, "홉", "고리뺀길이", "재방문스텝",
                 "재방문%", "최단경로%", "도달"))
        for nm, net in nets:
            r = run_one(net, device, cond, donor_pool, EPISODES,
                        RV.EVAL_SEED)
            row = {"condition": cond, "model": nm,
                   "hop": mean(r["hop"]), "loopfree": mean(r["loopfree"]),
                   "revisit": mean(r["revisit"]),
                   "has_rev_pct": 100.0 * mean(r["has_rev"]),
                   "is_short_pct": 100.0 * mean(r["is_short"]),
                   "sp_len": mean(r["sp"]), "reached": r["reached"],
                   "episodes": EPISODES}
            rows.append(row)
            print("%-18s %7.3f %9.3f %9.3f %8.1f%% %8.1f%% %5d/%d"
                  % (nm, row["hop"], row["loopfree"], row["revisit"],
                     row["has_rev_pct"], row["is_short_pct"],
                     row["reached"], EPISODES))
        sub = [r for r in rows if r["condition"] == cond]
        print("%-18s %7.3f %9.3f %9.3f %8.1f%% %8.1f%%"
              % ("  평균", mean([r["hop"] for r in sub]),
                 mean([r["loopfree"] for r in sub]),
                 mean([r["revisit"] for r in sub]),
                 mean([r["has_rev_pct"] for r in sub]),
                 mean([r["is_short_pct"] for r in sub])))
        print("")

    print(bar)
    print("읽는 법")
    print("-" * 86)
    print("  위상 최단경로 길이 평균 = %.3f 홉 (에피소드마다 다시 계산)"
          % mean([r["sp_len"] for r in rows]))
    print("  · '재방문%' 가 0 이면 hop_count 는 순수한 경로 길이다 —")
    print("    23.9 의 '짧고 나쁜 경로' 서술을 그대로 써도 된다.")
    print("  · 0 이 아니면 홉 차이의 일부는 **고리**다. 그때는 '고리 뺀")
    print("    길이' 쪽으로 다시 말해야 하고, 홉 비교는 두 숫자를 같이")
    print("    인용해야 한다.")
    print("  · '최단경로%' 는 품질이 아니라 **탐색 폭** 지표다. 100%면")
    print("    위험 정보를 안 쓰고 최단경로만 따라간 것이다")
    print("    (path_diversity 의 risk_zero 가 그랬다).")
    print(bar)

    path = os.path.join(OC.OUT_DIR, "revisit_rate.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "condition", "model", "hop", "loopfree", "revisit",
            "has_rev_pct", "is_short_pct", "sp_len", "reached",
            "episodes"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
