# -*- coding: utf-8 -*-
"""위험 블록을 가리면 정책이 **상태의존성 자체를 잃는가** — 경로 다양성.

동기: 탐침에서 `risk_zero` 조건의 평균 홉이 **정확히 6.000** 으로 붕괴했다
(원본 7.6 대). 6 은 0→24 의 위상적 최단 홉수다. 이것은 "위험 50칸을 가리면
정책이 고정 최단경로 하나만 내보낸다" 와 들어맞지만, **아직 재지 않았으므로
단정할 수 없다** (occlude_state_blocks.py 는 평균 지표만 낸다).

그래서 조건별로 **노드열 자체를 기록**해 서로 다른 경로의 개수·최빈
경로 점유율·엔트로피를 낸다.

**1차 탐침에서 내가 두 번 틀렸고, 둘 다 여기 고쳐 넣었다:**

(가) 가설 자체가 동어반복이었다. 위험 블록을 **상수로 고정**하면 0스텝 상태가
     매 에피소드 똑같아진다(dest 는 19절에서 상수, curr·visited 는 경로가 결정).
     그러면 롤아웃이 결정적이므로 **경로가 1개인 것은 정책의 성질이 아니라 산수**다.
     실제로 `risk_zero` 와 `risk_mean` 이 **둘 다** 10/10 붕괴했다 — 전자는 6홉
     최단경로(점유 100%), 후자는 8.5홉 비최단경로로, 서로 다른 경로이지만 둘 다
     "1개" 라는 사실에서는 아무 정보가 없다.
     → 동어반복을 뺀 질문은 **"고정하는 값을 바꾸면 경로가 바뀌는가"** 다.
     그래서 `risk_frozen` 을 넣었다: 에피소드마다 **다른 공여**의 위험 블록을 뽑아
     끝까지 고정한다. 경로수가 공여수에 가까우면 정책은 위험 **내용**을 읽고,
     1이면 **크기 오프셋**에만 반응하는 것이다.
     다만 이 붕괴가 말해 주는 구조는 남는다 — **에피소드 간 변동을 주는 관측
     블록은 위험 50칸 하나뿐이다.**

(나) 무작동 통제를 잘못 골랐다. `dest_zero` 가 intact 와 같을 것이라고 적었지만
     5/10 모델에서 달랐다. 19절의 "상수" 는 **정보가 0** 이라는 뜻이고, 상수 1.0
     입력은 학습된 **치우침**으로 작동한다 — 지우면 결정이 뒤집힐 수 있다
     (배치 행동변화율 0.50% 와 같은 크기). 코드 검증용 **진짜 무작동 통제는
     `dest_mean`** 이다: 블록이 상수이므로 배치 평균이 곧 그 값이고, 따라서
     intact 와 **비트 단위로 같아야** 한다. 다르면 가림 경로 자체가 틀린 것이다.

남는 통제:
 (1) `intact` 가 경로 1개면 결론은 "가림의 효과" 가 아니라 "환경이 애초에
     변동을 안 준다" 다. intact 의 경로 수가 분모다.
 (2) 최빈 경로가 **진짜 BFS 최단경로인지** 위상에서 직접 확인한다
     ("6홉이니 최단경로일 것" 이라고 넘기지 않는다).
 (3) 경로가 갈리는 것과 경로가 **좋은** 것은 다르다. 품질 판단은
     `state_occlusion_rollout.csv` 의 risk_mean 지연으로 따로 한다.

비교군은 occlude_state_blocks.py 와 동일한 125차원 엄격 10개 모델이고,
가림 함수·배치 묶음·RNG 규약을 그 모듈에서 **그대로 임포트해 쓴다**
(로직이 갈라지지 않게 하기 위함).

출력: review_results/path_diversity.csv
"""
import csv
import math
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# occlude_state_blocks 가 임포트 시 sys.path 에서 HERE 를 지우므로 먼저 받는다.
import occlude_state_blocks as OC                # noqa: E402

import numpy as np                               # noqa: E402
import torch                                     # noqa: E402
from network_env import NetworkEnv               # noqa: E402
from topology import EDGES, SOURCE, DESTINATION  # noqa: E402

# OC.ROLLOUT_CONDS 에 두 조건을 더한다.
#  dest_mean  : **진짜 무작동 통제**. dest 블록은 19절에서 상수이므로 배치 평균이
#               곧 그 값이고, 따라서 intact 와 비트 단위로 같아야 한다. 다르면
#               측정 코드가 틀렸다. (dest_zero 는 무작동이 아니다 - 상수 1.0 은
#               정보는 0이지만 학습된 **치우침**으로 작동하므로 지우면 흔들린다.)
#  risk_frozen: 공여 상태의 위험 블록을 **에피소드 시작에 한 번 뽑아 끝까지 고정**.
#               위험을 고정하면 경로가 1개로 붕괴하는 것은 동어반복이므로(아래),
#               "정책이 위험의 **내용**에 반응하는가" 는 공여를 바꿔 가며 재야 한다.
#               에피소드마다 다른 공여 → 경로가 갈리면 내용을 읽는 것이고,
#               전부 같은 경로면 내용이 아니라 크기에만 반응하는 것이다.
CONDS = OC.ROLLOUT_CONDS + (("dest_mean", "dest", "mean"),
                            ("risk_frozen", "risk", "shuffle_frozen"))

EPISODES = int(os.environ.get("PD_EPISODES", "100"))
EVAL_SEED = int(os.environ.get("PD_SEED", "20261114"))
OUT = os.path.join(OC.OUT_DIR, "path_diversity.csv")


def adjacency():
    adj = {}
    for a, b in EDGES:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def shortest_paths(src, dst):
    """홉 기준 최단경로 전체 집합과 그 홉수."""
    adj = adjacency()
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    if dst not in dist:
        return set(), None

    def up(node):
        if node == src:
            return [(src,)]
        out = []
        for v in adj.get(node, ()):
            if dist.get(v, 10 ** 9) == dist[node] - 1:
                for p in up(v):
                    out.append(p + (node,))
        return out

    return set(up(dst)), dist[dst]


def rollout_paths(net, device, blk, mode, donor_pool, blockmean,
                  episodes=EPISODES, seed=EVAL_SEED):
    """occlude_state_blocks.run_eval_occluded 와 같은 열·같은 RNG 규약.

    다른 점은 지표 대신 **노드열**을 모으는 것뿐이다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    drng = random.Random(seed + 1)

    lo = hi = None
    if blk is not None:
        for nm, a, b in OC.BLOCKS:
            if nm == blk:
                lo, hi = a, b

    frozen = (mode == "shuffle_frozen")
    eff_mode = "shuffle" if frozen else mode

    paths, hops, fails, donor_ids = [], [], 0, []
    for _ in range(episodes):
        ep_donor = None
        if lo is not None and frozen:
            j = drng.randrange(len(donor_pool))
            donor_ids.append(j)
            ep_donor = donor_pool[j]
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            if lo is None:
                fed = state
            else:
                if frozen:
                    donor = ep_donor
                else:
                    donor = donor_pool[drng.randrange(len(donor_pool))]
                fed = OC.occlude(state, lo, hi, eff_mode, donor, blockmean)
            action = OC.masked_argmax_from_q(OC.qvals(net, fed, device),
                                             list(valid))
            state, _, done, _ = env.step(action)
        if env.current_node == DESTINATION:
            paths.append(tuple(env.visited))
            hops.append(env.calculate_path_metrics(env.visited)["hop_count"])
        else:
            fails += 1
    return paths, hops, fails, donor_ids


def entropy_bits(counter, total):
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        p = c / total
        if p > 0:
            h -= p * math.log(p, 2)
    return h


def summarize(paths, hops, fails, sp_set):
    cnt = Counter(paths)
    n = len(paths)
    if n == 0:
        return {"n_success": 0, "n_fail": fails, "n_paths": 0,
                "top1_share_pct": float("nan"), "entropy_bits": float("nan"),
                "mean_hop": float("nan"), "modal_path": "",
                "modal_is_shortest": "", "shortest_share_pct": float("nan")}
    modal, mc = cnt.most_common(1)[0]
    sp_hits = sum(c for p, c in cnt.items() if p in sp_set)
    return {"n_success": n, "n_fail": fails, "n_paths": len(cnt),
            "top1_share_pct": mc / n * 100.0,
            "entropy_bits": entropy_bits(cnt, n),
            "mean_hop": float(np.mean(hops)),
            "modal_path": "-".join(str(x) for x in modal),
            "modal_is_shortest": "Y" if modal in sp_set else "N",
            "shortest_share_pct": sp_hits / n * 100.0}


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()

    sp_set, sp_hop = shortest_paths(SOURCE, DESTINATION)
    print("위상 최단경로: %d홉, 서로 다른 최단경로 %d개 (%d→%d)"
          % (sp_hop, len(sp_set), SOURCE, DESTINATION))

    models = []
    for arm, d, pat in OC.ARMS:
        for r in OC.RUNS:
            p = os.path.join(d, pat % r)
            if os.path.exists(p):
                models.append(("%s_run%02d" % (arm, r), arm, p))
    if len(models) != 10:
        raise SystemExit("엄격 비교군 10개가 아니다: %d" % len(models))
    nets = [(nm, arm, OC.load_net(p, device)) for nm, arm, p in models]

    ref = dict((nm, n) for nm, _, n in nets)["window5_run01"]
    battery = OC.build_battery("greedy", device, ref)
    blockmean = np.mean(np.stack([s for s, _ in battery]), axis=0)
    donor_pool = [s for s, _ in battery]
    print("배치 상태 묶음 %d개 · 에피소드 %d · 평가시드 %d"
          % (len(battery), EPISODES, EVAL_SEED))

    rows = []
    for cond, blk, mode in CONDS:
        for nm, arm, net in nets:
            paths, hops, fails, dids = rollout_paths(net, device, blk, mode,
                                                     donor_pool, blockmean)
            r = {"cond": cond, "model": nm, "arm": arm,
                 "n_donors": len(set(dids))}
            r.update(summarize(paths, hops, fails, sp_set))
            rows.append(r)
        done = [x for x in rows if x["cond"] == cond]
        print("  %-16s 경로수 %4.1f · 최빈점유 %5.1f%% · 홉 %.3f"
              % (cond,
                 float(np.mean([x["n_paths"] for x in done])),
                 float(np.nanmean([x["top1_share_pct"] for x in done])),
                 float(np.nanmean([x["mean_hop"] for x in done]))))

    fields = ["cond", "model", "arm", "n_donors", "n_success",
              "n_fail", "n_paths",
              "top1_share_pct", "entropy_bits", "mean_hop",
              "shortest_share_pct", "modal_is_shortest", "modal_path"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    report(rows, sp_set, sp_hop)
    print("저장: %s" % OUT)


def report(rows, sp_set, sp_hop):
    print("")
    print("=" * 78)
    print("[1] 조건별 경로 다양성 (모델 10개 평균, 에피소드 %d)" % EPISODES)
    print("%-16s %8s %8s %9s %8s %9s %7s"
          % ("조건", "경로수", "최빈%", "엔트로피", "홉", "최단경로%", "실패"))
    base = {}
    for cond, _, _ in CONDS:
        sub = [x for x in rows if x["cond"] == cond]
        npaths = float(np.mean([x["n_paths"] for x in sub]))
        base[cond] = npaths
        print("%-16s %8.1f %8.1f %9.3f %8.3f %9.1f %7.1f"
              % (cond, npaths,
                 float(np.nanmean([x["top1_share_pct"] for x in sub])),
                 float(np.nanmean([x["entropy_bits"] for x in sub])),
                 float(np.nanmean([x["mean_hop"] for x in sub])),
                 float(np.nanmean([x["shortest_share_pct"] for x in sub])),
                 float(np.mean([x["n_fail"] for x in sub]))))

    print("")
    print("[2] 통제 셋 — 먼저 이게 통과해야 [3] 을 읽을 수 있다")
    intact = {x["model"]: x for x in rows if x["cond"] == "intact"}
    keys = ("n_paths", "top1_share_pct", "entropy_bits", "mean_hop",
            "modal_path")

    bad = [m for m in intact if intact[m]["n_paths"] <= 1]
    print(" (1) intact 경로가 2개 이상인 모델: %d/%d  %s"
          % (len(intact) - len(bad), len(intact),
             "OK" if not bad else "실패 → 환경이 변동을 안 준다: %s" % bad))

    dm = {x["model"]: x for x in rows if x["cond"] == "dest_mean"}
    diff = [m for m in intact if any(intact[m][k] != dm[m][k] for k in keys)]
    print(" (2) **무작동 통제** dest_mean 지문이 intact 와 같은 모델: %d/%d  %s"
          % (len(intact) - len(diff), len(intact),
             "OK (가림 경로 자체는 무해하다)" if not diff
             else "실패 → 측정 코드가 틀렸다: %s" % diff))

    dz = {x["model"]: x for x in rows if x["cond"] == "dest_zero"}
    dzd = [m for m in intact if any(intact[m][k] != dz[m][k] for k in keys)]
    print(" (3) 참고: dest_zero 는 무작동이 **아니다** — 지문이 달라진 모델 %d/%d."
          % (len(dzd), len(intact)))
    print("     상수 1.0 은 정보는 0이지만 학습된 치우침으로 작동하므로 지우면")
    print("     결정 몇 개가 뒤집힌다(배치 행동변화율 0.60%와 같은 크기).")
    print(" (4) 위상 최단경로는 %d홉, 서로 다른 것 %d개"
          % (sp_hop, len(sp_set)))

    print("")
    print("[3] 경로 1개 붕괴는 **동어반복이다** — 먼저 이걸 빼고 읽어야 한다")
    print("  위험을 상수로 고정하면 0스텝 상태가 매 에피소드 똑같아진다")
    print("  (dest 는 상수, curr·visited 는 경로가 결정) → 경로가 1개인 것은")
    print("  정책의 성질이 아니라 **산수**다. risk_zero·risk_mean 의 1개는 증거가")
    print("  아니다. 다만 이것이 말해 주는 구조는 있다: **에피소드 간 변동을 주는")
    print("  관측 블록은 위험 50칸 하나뿐이다.**")
    print("")
    for cond in ("intact", "risk_zero", "risk_mean", "risk_frozen",
                 "risk_shuffle", "curr_zero", "visited_shuffle"):
        sub = [x for x in rows if x["cond"] == cond]
        if not sub:
            continue
        one = sum(1 for x in sub if x["n_paths"] == 1)
        print("  %-15s 경로1개 %2d/10 · 경로수중위 %4.1f · 홉 %6.3f "
              "· 최단경로 %5.1f%% · 공여수 %3.0f"
              % (cond, one,
                 float(np.median([x["n_paths"] for x in sub])),
                 float(np.nanmean([x["mean_hop"] for x in sub])),
                 float(np.nanmean([x["shortest_share_pct"] for x in sub])),
                 float(np.mean([x["n_donors"] for x in sub]))))

    print("")
    print("[4] 그래서 정책이 위험의 **내용**을 읽는가 — risk_frozen 이 답한다")
    rf = [x for x in rows if x["cond"] == "risk_frozen"]
    it = [x for x in rows if x["cond"] == "intact"]
    if rf:
        nd = float(np.mean([x["n_donors"] for x in rf]))
        npz = float(np.mean([x["n_paths"] for x in rf]))
        print("  에피소드마다 **다른 공여의 위험 블록을 끝까지 고정**했다.")
        print("  공여 %0.1f개 → 서로 다른 경로 %0.1f개 (모델 평균)." % (nd, npz))
        collapsed = sum(1 for x in rf if x["n_paths"] == 1)
        print("  경로 1개로 붕괴한 모델 %d/10." % collapsed)
        print("")
        print("  읽는 법 — 동어반복이 제거된 유일한 칸이다:")
        print("   · **비교 기준은 공여수가 아니라 intact 의 경로수다.**")
        print("     정책의 출력 레퍼토리 자체가 ~3개라서 어떤 입력을 먹여도")
        print("     3개를 넘을 수 없다. risk_frozen 경로수 ~= intact 경로수면")
        print("     '틀리지만 그럴듯한 위험이 실제 위험만큼 경로를 가른다'")
        print("     → 내용을 읽는다.")
        print("   · 경로수가 1이면 위험이 주는 것은 내용이 아니라 **크기 오프셋**")
        print("     뿐이다(블록이 제곱노름의 57.07%, 차원별 sd 중위 0.031).")
        print("   · 단 절대수가 작은 것(공여 %d개 -> 경로 %0.1f개)은 **따로"
              % (nd, npz))
        print("     봐야 할 발견**이고 intact 에도 똑같이 적용된다 — 위험")
        print("     상태를 아주 거친 경로 집합으로 뭉갠다(차원별 민감도")
        print("     6.3/50 과 같은 방향).")
        print("  단, 경로가 갈려도 **그게 더 나은 경로라는 뜻은 아니다** — 품질은")
        print("  state_occlusion_rollout.csv 의 risk_mean 지연으로 따로 판단한다.")
        print("  intact 경로수 %0.1f 과 비교할 것."
              % float(np.mean([x["n_paths"] for x in it])))
    print("=" * 78)


if __name__ == "__main__":
    main()
