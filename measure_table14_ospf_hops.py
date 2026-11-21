# -*- coding: utf-8 -*-
u"""표 14 OSPF 경로의 홉을 분류해 산출물로 남긴다 (기록 20.3 · 20.7절).

왜 다시 재는가. 기록(`FINDINGS_table10_f1.md` 20.7)은 OSPF 홉의 분류표를
적는다 — 지지 경로 겹침이 45.4 / 44.5 / 35.9%, 후보 3합집합이 99.6 / 98.9 /
99.8%. 또 20.3 의 주석(`DRAFT_4_9_table14.md` 주3)은 워밍업을 빼면 Risky 가
40.8 / 44.2 / 47.8% 로 내려가고 경로 변경률이 19.0 / 18.0 / 25.7% 라고 적는다.
그런데 이 값들을 쓴 스크립트가 없다. 그때 즉석으로 돌려 출력만 보고 문서에
옮겨 적었고 CSV 가 남지 않았다 — 감사(32절)에서 `DOC_ONLY` 경보로 뜬다.
내 문서에 적힌 것은 출처가 아니다.

둘을 한 롤아웃에서 잰다. `ospf_path()` 는 nx 최단경로라 **난수를 소비하지
않는다**. 그래서 한 에피소드 안에서
  reset -> (워밍업 전 경로) -> update_dynamic_qos x 10 -> (워밍업 후 경로)
를 하면 워밍업 후 경로는 `reeval_table14_final.run_episode()` 가 뽑는 경로와
비트 단위로 같다. 검증: 워밍업 후 집계가 커밋된 `table14_final_runs.csv` 의
OSPF 행과 일치해야 한다(에피소드 100 기준).

분류 정의 — `network_env_generalization._build_dynamic_phases()` 그대로:
  후보 경로 3개의 간선 집합 path_sets[0..2] 에서
    support = path_sets[phase]
    danger  = (나머지 둘의 합집합) - support
  Risky 홉 = danger 에 든 홉(`calculate_path_metrics` 의 unstable/congested_count).
  지지 홉  = support 에 든 홉.
  합집합 홉 = danger | support (= 후보 3경로 합집합).
`DYNAMIC_PHASE_PER_EPISODE = True` 이므로 phase 는 에피소드 내내 고정이고,
워밍업은 phase 를 바꾸지 않는다 — 경로가 바뀐다면 대역폭이 흘러서다.

집계 규약은 커밋된 런과 같다: 에피소드 평균을 토폴로지별로 내고, 토폴로지
10개를 다시 평균한다(에피소드 전체를 한 풀로 섞지 않는다).

재현: python measure_table14_ospf_hops.py
원본 폴더(`SDN_DQN_new_env/`, `Risk-aware-DQN-routing/`)는 건드리지 않는다.
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from network_env_generalization import (  # noqa: E402
    DYNAMIC_PHASE_PER_EPISODE, normalize_edge, path_to_edges)
from reeval_table14_final import Table14Env  # noqa: E402
import run_topology_generalization_dynamic as RTG  # noqa: E402

TOPO_DIR = os.path.join(HERE, "topologies_dynamic")
SCENARIO = "dynamic"
BASE_EVAL_SEED = RTG.BASE_EVAL_SEED
WARMUP_STEPS = RTG.WARMUP_STEPS


def classify(env, path):
    u"""홉을 위험/지지/합집합으로 가른다. 난수를 쓰지 않는다."""
    ph = env.dynamic_phases[env.dynamic_phase]
    danger, support = ph["danger"], ph["support"]
    union = danger | support
    hops = [normalize_edge(u, v) for (u, v) in zip(path[:-1], path[1:])]
    n = len(hops)
    n_risky = sum(1 for e in hops if e in danger)
    n_sup = sum(1 for e in hops if e in support)
    n_uni = sum(1 for e in hops if e in union)
    return dict(hop=n, risky=n_risky, support=n_sup, union=n_uni,
                all_in_union=1.0 if (n and n_uni == n) else 0.0)


def one_topology(topo, episodes):
    u"""한 토폴로지. 에피소드마다 워밍업 전/후 경로를 모두 뽑는다."""
    acc = {w: {k: [] for k in
               ("hop", "risky", "support", "union", "all_in_union",
                "risky_pct", "support_pct", "union_pct")}
           for w in (0, WARMUP_STEPS)}
    changed = []
    for ep in range(episodes):
        seed = BASE_EVAL_SEED + ep
        env = Table14Env(topo, use_risk=False, scenario=SCENARIO, seed=seed)
        env.reset()
        phase0 = env.dynamic_phase

        p0 = env.ospf_path(cost_type="bandwidth")
        for _ in range(WARMUP_STEPS):
            env.update_dynamic_qos()
        p1 = env.ospf_path(cost_type="bandwidth")
        assert env.dynamic_phase == phase0 or not DYNAMIC_PHASE_PER_EPISODE

        changed.append(0.0 if list(p0) == list(p1) else 1.0)
        for w, p in ((0, p0), (WARMUP_STEPS, p1)):
            if not p or p[-1] != env.destination:
                continue
            c = classify(env, p)
            h = max(c["hop"], 1)
            # 자체 검증: 워밍업 후 경로의 Risky 는 원본 지표 정의와 같아야 한다.
            if w == WARMUP_STEPS:
                m = env.calculate_path_metrics(p)
                assert int(m["risky_count"]) == c["risky"], (
                    topo, ep, m["risky_count"], c["risky"])
            for k in ("hop", "risky", "support", "union", "all_in_union"):
                acc[w][k].append(float(c[k]))
            acc[w]["risky_pct"].append(100.0 * c["risky"] / h)
            acc[w]["support_pct"].append(100.0 * c["support"] / h)
            acc[w]["union_pct"].append(100.0 * c["union"] / h)
    out = {}
    for w in acc:
        out[w] = {k: float(np.mean(v)) if v else float("nan")
                  for k, v in acc[w].items()}
    out["path_changed_pct"] = 100.0 * float(np.mean(changed))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, nargs="+", default=[30, 100],
                    help=u"에피소드 수. 30=기록 20.7절, 100=커밋된 표 14 규약")
    args = ap.parse_args()

    topos = sorted(os.listdir(TOPO_DIR))
    runs, summ = [], []
    for episodes in args.episodes:
        per_ratio = {}
        for name in topos:
            ratio = {"r20": 0.2, "r40": 0.4, "r60": 0.6}[name.split("_")[2]]
            res = one_topology(os.path.join(TOPO_DIR, name), episodes)
            per_ratio.setdefault(ratio, []).append(res)
            for w in (0, WARMUP_STEPS):
                runs.append(dict(
                    episodes=episodes, warmup=w, rewire_ratio=ratio,
                    topology=name.replace(".json", ""),
                    **{k: "%.6f" % v for k, v in res[w].items()}))
                runs[-1]["path_changed_pct"] = "%.6f" % res["path_changed_pct"]
            print(u"  %-28s ep=%3d  warm Risky %5.2f  지지 %5.2f  "
                  u"합집합 %6.2f | nowarm Risky %5.2f  변경 %5.2f"
                  % (name, episodes,
                     res[WARMUP_STEPS]["risky_pct"],
                     res[WARMUP_STEPS]["support_pct"],
                     res[WARMUP_STEPS]["union_pct"],
                     res[0]["risky_pct"], res["path_changed_pct"]))

        for ratio in sorted(per_ratio):
            rs = per_ratio[ratio]
            for w in (0, WARMUP_STEPS):
                row = dict(episodes=episodes, warmup=w, rewire_ratio=ratio,
                           topologies=len(rs))
                # 6자리로 적는다. 4자리면 아래 자체 검증의 최대차가 반올림
                # 때문에 3e-5 로 떠서 "불일치" 로 읽힌다.
                for k in rs[0][w]:
                    row[k] = "%.6f" % float(np.mean([r[w][k] for r in rs]))
                row["path_changed_pct"] = "%.6f" % float(
                    np.mean([r["path_changed_pct"] for r in rs]))
                summ.append(row)

    rp = os.path.join(RES, "table14_ospf_hops_runs.csv")
    with io.open(rp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(runs[0].keys()))
        w.writeheader()
        for r in runs:
            w.writerow(r)
    sp = os.path.join(RES, "table14_ospf_hops.csv")
    with io.open(sp, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        for r in summ:
            w.writerow(r)

    print()
    print(u"=== 요약 (토폴로지 10개 평균) ===")
    print(u"%-4s %-6s %-6s %8s %8s %8s %8s %9s %8s"
          % (u"ep", u"워밍업", u"재배선", u"위험홉", u"총홉", u"Risky%",
             u"지지%", u"합집합%", u"변경%"))
    for r in summ:
        print(u"%-4s %-6s %-6s %8s %8s %8s %8s %9s %8s"
              % (r["episodes"], r["warmup"], r["rewire_ratio"],
                 r["risky"], r["hop"], r["risky_pct"], r["support_pct"],
                 r["union_pct"], r["path_changed_pct"]))

    # 커밋된 표 14 OSPF 행과 대조 (에피소드 100 · 워밍업 10).
    ref = os.path.join(RES, "table14_final_runs.csv")
    if os.path.exists(ref) and 100 in args.episodes:
        rows = [r for r in csv.DictReader(io.open(ref, encoding="utf-8"))
                if r["method"] == "OSPF"]
        print()
        print(u"=== 자체 검증: 커밋된 table14_final_runs.csv OSPF 행과 대조 ===")
        worst = 0.0
        for ratio in (0.2, 0.4, 0.6):
            sub = [r for r in rows if abs(float(r["rewire_ratio"]) - ratio) < 1e-9]
            mine = [s for s in summ if s["episodes"] == 100
                    and s["warmup"] == WARMUP_STEPS
                    and abs(s["rewire_ratio"] - ratio) < 1e-9][0]
            for col, key in (("risky_link_usage", "risky_pct"),
                             ("hop", "hop"), ("risky_count", "risky")):
                a = float(np.mean([float(s[col]) for s in sub]))
                b = float(mine[key])
                worst = max(worst, abs(a - b))
                print(u"  r=%.1f %-18s 커밋 %8.4f  재계산 %8.4f  차 %.2e"
                      % (ratio, col, a, b, abs(a - b)))
        print(u"  최대 절대차 %.3e  %s"
              % (worst, u"(동일)" if worst < 1e-6 else u"(불일치!)"))

    print()
    print(u"저장: %s (%d행)\n      %s (%d행)" % (rp, len(runs), sp, len(summ)))


if __name__ == "__main__":
    sys.exit(main())
