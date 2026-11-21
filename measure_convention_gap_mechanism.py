# -*- coding: utf-8 -*-
"""`FINDINGS_environment_dynamics.md` D절이 단정을 보류한 문장 하나를 재려고 만들었다.

그 문장은 이렇게 적혀 있었다:

    채점 규약을 `walk_final` -> `walk_realized` 로 바꾸면 네 플래너가 같은 방향으로
    움직이는데, unstable 에서 `ospf_bw` 만 -5.1% 로 덜 움직인다(나머지 셋은 -26~-29%).
    **이유는 추정이다** - ospf_bw 는 혼잡 링크를 4배 많이 밟고, 혼잡 링크는 가장 빨리
    드리프트하지만 가장 빨리 천장에도 닿는다. 천장에 닿은 링크는 지날 때 재도 끝에 재도
    0.35 이므로 두 규약이 같은 값을 준다. **재지 않았으므로 단정하지 않는다.**

두 부분으로 나뉜다. (ㄱ) "천장에 닿은 링크는 두 규약이 같은 값을 준다" - 링크 단위
전제. (ㄴ) "그래서 ospf_bw 가 덜 움직인다" - 경로 단위 결론. 전제가 참이어도 결론은
거짓일 수 있다. 그래서 둘을 따로 잰다.

결론을 먼저 반증할 수 있는 길이 하나 있다. `compounded = 1 - S`(S = 생존율)이고 C 는
S 의 **선형**함수이므로 평균 수준에서 항등식이 선다:

    rel = (C_real - C_final)/C_final = -D x (S_final / C_final),
    D = (S_real - S_final)/S_final

오른쪽 둘째 항은 **기준 손실 수준만의 함수**다(압축계수). 즉 상대 이동폭이 작은 것은
(i) 물리적 드리프트 D 가 작아서일 수도 있고 (ii) 압축계수가 작아서일 수도 있다. 천장
가설은 (i)을 예측한다. 커밋된 `scoring_convention.csv` 로 이 항등식을 계산하면 unstable
에서 ospf_bw 의 D 는 **+26.45%** 로 넷 중 가장 크고(나머지 +12.5~14.9%) 압축계수는
0.193 대 1.89~2.10 이다. 즉 (ii)다 - 천장 가설의 예측과 **반대 방향**이다.

이 스크립트는 그 계산을 링크 단위로 내려가 확인하고, 세 경쟁가설을 같은 표에서 가른다:

  H_ceil  천장에 닿은 링크는 규약 간 차이가 0 이다 (D절이 적은 전제)
  H_den   ospf_bw 의 작은 상대 이동폭은 분모(압축계수)다 (위 항등식)
  H_len   ospf_bw 는 경로가 짧아(6.01 대 8.13홉) 지난 시점과 끝 사이 경과 스텝이 달라서다

링크마다 (1) 지나간 순간의 손실 (2) 같은 간선의 에피소드 **끝** 손실 (3) 지날 때 천장·
혼잡 여부 (4) 지난 시점과 끝 사이 경과 스텝을 남긴다. 원본은 읽기만 한다.

검증 둘을 통과해야 아래 숫자를 쓴다:
  V1 링크 단위로 다시 더한 C_real / C_final 이 하네스의 `realized_metrics` /
     `path_metrics` 가 준 값과 **정확히** 같아야 한다(내 기록이 채점과 같은 간선·같은
     값을 봤다는 뜻).
  V2 시드 평균이 커밋된 `scoring_convention.csv` 를 REPRO_TOL_PCT 안에서 재현해야 한다
     (같은 난수 흐름이라는 뜻 - 기록을 더해 난수 소비가 바뀌면 여기서 깨진다).

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_convention_gap_mechanism.py
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_convention_gap_mechanism.py --from-csv
"""

import argparse
import csv
import math
import os
import random
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "..", "SDN_DQN_new_env"))

import reeval_loss_conventions as RLC  # noqa: E402
import reeval_qos_baseline_handicap as QBH  # noqa: E402
import reeval_scoring_convention as RSC  # noqa: E402
from network_env import MAX_STEPS, NetworkEnv  # noqa: E402
from review_stats import fmt_p, holm_bonferroni, mean_std, welch  # noqa: E402
from topology import DESTINATION  # noqa: E402

OUT_CSV = os.path.join(BASE_DIR, "review_results",
                       "convention_gap_mechanism.csv")
LINK_CSV = os.path.join(BASE_DIR, "review_results", "convention_gap_links.csv")
REF_CSV = os.path.join(BASE_DIR, "review_results", "scoring_convention.csv")

SCENARIOS = RSC.SCENARIOS
SEEDS = RSC.SEEDS
EPISODES = RSC.EPISODES
WARMUP = RSC.WARMUP
REPRO_TOL_PCT = RSC.REPRO_TOL_PCT

# network_env.py:229 -- q["loss"] = np.clip(..., 0.0, 0.35)
LOSS_CEIL = 0.35
CEIL_EPS = 1e-12

POLICIES = ("ospf_bw", "ospf_delay", "qos", "risk_dqn")


def hazard(p):
    """-ln(1-p). 경로 생존율의 로그가 이 값들의 **합**이라 링크 단위 귀속이 선다."""
    return -math.log(max(1.0 - p, 1e-15))


# --------------------------------------------------------------- 보행 + 기록

def walk_logging(env, next_node_of):
    """`RSC.walk_collecting` 과 같은 순서로 걸으며 간선·시점까지 남긴다.

    원본과 같은 자리에서 같은 것을 읽어야 한다 - 간선은 step **전에** 정하고
    (current_node 가 바뀌므로) 값은 step **후에** 읽는다. 난수를 더 쓰지 않으므로
    V2 가 이걸 지킨다.
    """
    samples = []       # RSC.realized_metrics 가 먹는 모양 그대로
    log = []           # (edge, hop_index, loss_at_traverse, congested_at_traverse)
    done = False
    while not done:
        nxt = next_node_of(env)
        if nxt is None:
            break
        e = env.normalize_edge(env.current_node, nxt)
        _, _, done, _ = env.step(nxt)
        q = env.qos.get(e)
        if q is not None:
            cong = bool(q.get("is_congested", False))
            samples.append((q["delay"], q["loss"], cong))
            log.append((e, len(log), float(q["loss"]), cong))
        if len(env.visited) >= MAX_STEPS:
            break
    return samples, log


def episode_record(env, samples, log):
    """에피소드 하나에서 두 규약을 링크 단위로 나란히 세운다."""
    real = RSC.realized_metrics(samples)
    final = QBH.path_metrics(env, env.visited)
    n_hop = len(log)
    rows = []
    s_real = 1.0
    s_final = 1.0
    for e, idx, loss_t, cong_t in log:
        q = env.qos.get(e)
        loss_f = float(q["loss"]) if q is not None else loss_t
        s_real *= (1.0 - loss_t)
        s_final *= (1.0 - loss_f)
        rows.append({
            "hop_index": idx,
            "elapsed": n_hop - 1 - idx,      # 지난 뒤 끝까지 남은 step 수
            "loss_traverse": loss_t,
            "loss_final": loss_f,
            "gap": loss_f - loss_t,
            "hazard_gap": hazard(loss_f) - hazard(loss_t),
            "ceil_traverse": 1 if loss_t >= LOSS_CEIL - CEIL_EPS else 0,
            "ceil_final": 1 if loss_f >= LOSS_CEIL - CEIL_EPS else 0,
            "cong_traverse": 1 if cong_t else 0,
        })
    rec = {
        "hop": n_hop,
        "c_real_relog": 1.0 - s_real,
        "c_final_relog": 1.0 - s_final,
        "c_real_harness": real["compounded"],
        "c_final_harness": final["compounded"],
        "h_real": sum(hazard(r["loss_traverse"]) for r in rows),
        "h_final": sum(hazard(r["loss_final"]) for r in rows),
        "cong_live": real["cong_live"],
    }
    return rec, rows


# --------------------------------------------------------------- 실행

def run_planner(scenario, seed, episodes, planner):
    np.random.seed(seed)
    random.seed(seed)
    recs, links = [], []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=False, scenario=scenario)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        planned = planner(env)
        if not planned or planned[-1] != DESTINATION:
            continue
        QBH.path_metrics(env, planned)       # plan_snapshot - 난수 소비 없음
        queue = list(planned[1:])

        def next_node_of(_env, queue=queue):
            return queue.pop(0) if queue else None

        samples, log = walk_logging(env, next_node_of)
        if env.current_node != DESTINATION:
            continue
        rec, rows = episode_record(env, samples, log)
        recs.append(rec)
        links.extend(rows)
    return recs, links


def run_risk_dqn(scenario, seed, episodes, model):
    import torch

    np.random.seed(seed)
    random.seed(seed)
    device = next(model.parameters()).device
    recs, links = [], []
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

        samples, log = walk_logging(env, next_node_of)
        if env.current_node != DESTINATION:
            continue
        rec, rows = episode_record(env, samples, log)
        recs.append(rec)
        links.extend(rows)
    return recs, links


def summarize(scenario, policy, seed, recs, links):
    n = float(len(recs)) or 1.0

    def em(k):
        return sum(r[k] for r in recs) / n

    c_f, c_r = em("c_final_harness"), em("c_real_harness")
    s_f, s_r = 1.0 - c_f, 1.0 - c_r
    nl = float(len(links)) or 1.0
    ceil_t = [l for l in links if l["ceil_traverse"]]
    open_t = [l for l in links if not l["ceil_traverse"]]
    cong_t = [l for l in links if l["cong_traverse"]]
    quiet_t = [l for l in links if not l["cong_traverse"]]

    def lm(sub, k):
        return (sum(l[k] for l in sub) / len(sub)) if sub else float("nan")

    return {
        "scenario": scenario, "policy": policy, "seed": seed,
        "episodes": len(recs), "hop": em("hop"),
        "c_final": c_f, "c_real": c_r,
        "rel_pct": (c_r - c_f) / c_f * 100.0,
        "survival_gain_pct": (s_r - s_f) / s_f * 100.0,
        "compression": s_f / c_f,
        "h_final": em("h_final"), "h_real": em("h_real"),
        "hazard_drop": em("h_final") - em("h_real"),
        "cong_live": em("cong_live"),
        "n_links": len(links),
        "ceil_traverse_pct": len(ceil_t) / nl * 100.0,
        "ceil_final_pct": sum(l["ceil_final"] for l in links) / nl * 100.0,
        "cong_traverse_pct": len(cong_t) / nl * 100.0,
        "gap_all": lm(links, "gap"),
        "gap_ceil": lm(ceil_t, "gap"),
        "gap_open": lm(open_t, "gap"),
        "gap_cong": lm(cong_t, "gap"),
        "gap_quiet": lm(quiet_t, "gap"),
        "hazgap_all": lm(links, "hazard_gap"),
        "hazgap_ceil": lm(ceil_t, "hazard_gap"),
        "hazgap_open": lm(open_t, "hazard_gap"),
        "elapsed_all": lm(links, "elapsed"),
        "loss_traverse_all": lm(links, "loss_traverse"),
        "loss_final_all": lm(links, "loss_final"),
        # V1
        "relog_err_real": max(abs(r["c_real_relog"] - r["c_real_harness"])
                              for r in recs) if recs else 0.0,
        "relog_err_final": max(abs(r["c_final_relog"] - r["c_final_harness"])
                               for r in recs) if recs else 0.0,
    }


def link_buckets(scenario, policy, links):
    out = []
    for ceil in (0, 1):
        for cong in (0, 1):
            sub = [l for l in links
                   if l["ceil_traverse"] == ceil and l["cong_traverse"] == cong]
            if not sub:
                continue
            k = float(len(sub))
            out.append({
                "scenario": scenario, "policy": policy,
                "ceil_traverse": ceil, "cong_traverse": cong,
                "n_links": len(sub),
                "share_pct": None,
                "loss_traverse": sum(l["loss_traverse"] for l in sub) / k,
                "loss_final": sum(l["loss_final"] for l in sub) / k,
                "gap": sum(l["gap"] for l in sub) / k,
                "gap_abs": sum(abs(l["gap"]) for l in sub) / k,
                "hazard_gap": sum(l["hazard_gap"] for l in sub) / k,
                "zero_gap_pct": sum(1 for l in sub if l["gap"] == 0.0) / k * 100.0,
                "elapsed": sum(l["elapsed"] for l in sub) / k,
                "ceil_final_pct": sum(l["ceil_final"] for l in sub) / k * 100.0,
            })
    tot = float(sum(r["n_links"] for r in out)) or 1.0
    for r in out:
        r["share_pct"] = r["n_links"] / tot * 100.0
    return out


SUM_FIELDS = ["scenario", "policy", "seed", "episodes", "hop", "c_final",
              "c_real", "rel_pct", "survival_gain_pct", "compression",
              "h_final", "h_real", "hazard_drop", "cong_live", "n_links",
              "ceil_traverse_pct", "ceil_final_pct", "cong_traverse_pct",
              "gap_all", "gap_ceil", "gap_open", "gap_cong", "gap_quiet",
              "hazgap_all", "hazgap_ceil", "hazgap_open", "elapsed_all",
              "loss_traverse_all", "loss_final_all",
              "relog_err_real", "relog_err_final"]
LINK_FIELDS = ["scenario", "policy", "ceil_traverse", "cong_traverse",
               "n_links", "share_pct", "loss_traverse", "loss_final", "gap",
               "gap_abs", "hazard_gap", "zero_gap_pct", "elapsed",
               "ceil_final_pct"]


def measure():
    rows, lrows = [], []
    for scenario in SCENARIOS:
        for name, planner in RSC.PLANNERS:
            pooled = []
            for seed in SEEDS:
                recs, links = run_planner(scenario, seed, EPISODES, planner)
                rows.append(summarize(scenario, name, seed, recs, links))
                pooled.extend(links)
            lrows.extend(link_buckets(scenario, name, pooled))
            print("  {:<11} {:<12} 완료".format(scenario, name), flush=True)

    model = RLC.load_risk_model()
    for scenario in SCENARIOS:
        pooled = []
        for seed in SEEDS:
            recs, links = run_risk_dqn(scenario, seed, EPISODES, model)
            rows.append(summarize(scenario, "risk_dqn", seed, recs, links))
            pooled.extend(links)
        lrows.extend(link_buckets(scenario, "risk_dqn", pooled))
        print("  {:<11} {:<12} 완료".format(scenario, "risk_dqn"), flush=True)

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUM_FIELDS)
        w.writeheader()
        w.writerows(rows)
    with open(LINK_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LINK_FIELDS)
        w.writeheader()
        w.writerows(lrows)
    print("\n저장: {} ({}행)".format(OUT_CSV, len(rows)))
    print("저장: {} ({}행)".format(LINK_CSV, len(lrows)))
    return rows, lrows


def load(path, strkeys):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in list(r.items()):
            if k in strkeys:
                continue
            r[k] = float(v) if v not in ("", "nan") else float("nan")
    return rows


def report(rows, lrows):
    def cell(sc, pol, col):
        return [r[col] for r in rows
                if r["scenario"] == sc and r["policy"] == pol]

    print("\n" + "=" * 78)
    print("V1 - 링크 기록을 다시 더한 값 대 하네스 채점값 (에피소드별 최대 절대차)")
    e1 = max(r["relog_err_real"] for r in rows)
    e2 = max(r["relog_err_final"] for r in rows)
    print("  walk_realized {:.3e} · walk_final {:.3e}  -> {}".format(
        e1, e2,
        "같은 간선·같은 값을 봤다" if max(e1, e2) < 1e-12 else "**어긋난다**"))

    print("\nV2 - 커밋된 scoring_convention.csv 재현 (compounded, 편차 %)")
    ref = load(REF_CSV, {"scenario", "policy", "convention"})
    worst = 0.0
    for sc in SCENARIOS:
        for pol in POLICIES:
            for conv, col in (("walk_final", "c_final"),
                              ("walk_realized", "c_real")):
                r0 = [r["compounded"] for r in ref
                      if r["scenario"] == sc and r["policy"] == pol
                      and r["convention"] == conv]
                mine = cell(sc, pol, col)
                if not r0 or not mine:
                    continue
                a, b = sum(r0) / len(r0), sum(mine) / len(mine)
                worst = max(worst, abs(b - a) / a * 100.0)
    print("  최대 편차 {:.4f}%  (허용 {:.2f}%)  -> {}".format(
        worst, REPRO_TOL_PCT,
        "재현" if worst <= REPRO_TOL_PCT
        else "**재현 실패 - 아래 숫자 쓰지 말 것**"))

    print("\n" + "=" * 78)
    print("[H_den] 상대 이동폭 = -(생존율 상대증가 D) x (압축계수 S/C) -- 항등식")
    print("%-10s %-11s %8s %8s %8s %8s %8s" % (
        "scenario", "policy", "rel %", "D %", "compress", "-D*comp", "hop"))
    for sc in SCENARIOS:
        for pol in POLICIES:
            rel = sum(cell(sc, pol, "rel_pct")) / 5
            d = sum(cell(sc, pol, "survival_gain_pct")) / 5
            cp = sum(cell(sc, pol, "compression")) / 5
            hp = sum(cell(sc, pol, "hop")) / 5
            print("%-10s %-11s %+8.2f %+8.2f %8.4f %+8.2f %8.3f" % (
                sc, pol, rel, d, cp, -d / 100 * cp * 100, hp))
        print()

    print("=" * 78)
    print("[H_ceil] 링크 단위 - 지날 때 천장이면 두 규약이 같은 값을 주는가")
    print("%-10s %-11s %7s %9s %9s %10s %9s" % (
        "scenario", "policy", "천장%", "gap(천장)", "gap(열림)", "0gap(천장)%",
        "경과step"))
    for sc in SCENARIOS:
        for pol in POLICIES:
            ct = sum(cell(sc, pol, "ceil_traverse_pct")) / 5
            gc = [x for x in cell(sc, pol, "gap_ceil") if x == x]
            go = [x for x in cell(sc, pol, "gap_open") if x == x]
            el = sum(cell(sc, pol, "elapsed_all")) / 5
            zb = [r for r in lrows
                  if r["scenario"] == sc and r["policy"] == pol
                  and r["ceil_traverse"] == 1]
            z = (sum(r["zero_gap_pct"] * r["n_links"] for r in zb)
                 / sum(r["n_links"] for r in zb)) if zb else float("nan")
            print("%-10s %-11s %7.2f %+9.5f %+9.5f %10.2f %9.3f" % (
                sc, pol, ct,
                sum(gc) / len(gc) if gc else float("nan"),
                sum(go) / len(go) if go else float("nan"), z, el))
        print()

    print("=" * 78)
    print("[H_len] 경과 스텝·혼잡이 링크 gap 을 얼마나 가르는가 (unstable)")
    print("%-11s %5s %5s %8s %9s %9s %9s %9s" % (
        "policy", "ceil", "cong", "링크%", "loss(지남)", "loss(끝)", "gap",
        "경과step"))
    for r in [x for x in lrows if x["scenario"] == "unstable"]:
        print("%-11s %5d %5d %8.2f %9.4f %9.4f %+9.5f %9.3f" % (
            r["policy"], int(r["ceil_traverse"]), int(r["cong_traverse"]),
            r["share_pct"], r["loss_traverse"], r["loss_final"], r["gap"],
            r["elapsed"]))

    print("\n" + "=" * 78)
    print("[검정] ospf_bw 의 D 가 나머지 셋과 다른가"
          " (Welch, Holm 족 6 = 2시나리오 x 3대조)")
    tests = []
    for sc in ("unstable", "congestion"):
        a = cell(sc, "ospf_bw", "survival_gain_pct")
        for pol in ("ospf_delay", "qos", "risk_dqn"):
            b = cell(sc, pol, "survival_gain_pct")
            _, _, _, p = welch(a, b)
            tests.append(("%s: ospf_bw vs %s" % (sc, pol), p,
                          mean_std(a), mean_std(b)))
    ps = holm_bonferroni([t[1] for t in tests])
    for (name, p, ma, mb), ph in zip(tests, ps):
        print("  %-30s %7.3f±%.3f 대 %7.3f±%.3f  p=%s p_holm=%s %s" % (
            name, ma[0], ma[1], mb[0], mb[1], fmt_p(p), fmt_p(ph),
            "유의" if ph < 0.05 else ""))

    print("\n[검정] 천장 링크의 gap 이 열린 링크보다 작은가 (Welch, Holm 족 4)")
    t2 = []
    for sc in SCENARIOS:
        a, b = [], []
        for pol in POLICIES:
            a += [x for x in cell(sc, pol, "gap_ceil") if x == x]
            b += [x for x in cell(sc, pol, "gap_open") if x == x]
        if len(a) < 2 or len(b) < 2:
            nan2 = (float("nan"), float("nan"))
            t2.append((sc, float("nan"),
                       mean_std(a) if len(a) > 1 else nan2,
                       mean_std(b) if len(b) > 1 else nan2))
            continue
        _, _, _, p = welch(a, b)
        t2.append((sc, p, mean_std(a), mean_std(b)))
    ps2 = holm_bonferroni([x[1] if x[1] == x[1] else 1.0 for x in t2])
    for (sc, p, ma, mb), ph in zip(t2, ps2):
        print("  %-12s 천장 %+.5f±%.5f 대 열림 %+.5f±%.5f  p=%s p_holm=%s" % (
            sc, ma[0], ma[1], mb[0], mb[1], fmt_p(p), fmt_p(ph)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    args = ap.parse_args()
    if args.from_csv:
        rows = load(OUT_CSV, {"scenario", "policy"})
        lrows = load(LINK_CSV, {"scenario", "policy"})
    else:
        rows, lrows = measure()
    return report(rows, lrows)


if __name__ == "__main__":
    raise SystemExit(main())
