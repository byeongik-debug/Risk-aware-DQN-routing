# -*- coding: utf-8 -*-
"""23.8.2절의 '값은 순위에 있다'를 **조작해서** 확인한다 (상관 -> 인과).

23.8.2절은 세 조건의 순위 충실도와 피해가 단조라는 **상관**으로 그 결론을 냈다
(rho 1.00/0.46/0.27 -> 지연 361/412/527). 점이 셋이고 순위를 **직접 조작한
것이 아니므로** 메커니즘을 단정할 수 없다. 그래서 2x2 로 분리한다.

                       값(크기 분포)        순서(엣지 간 등급)
  intact               참                   참
  risk_rankonly        공여 것 (틀림)       **참**
  risk_dimshuffle      **참**               무작위 (틀림)
  risk_frozen          공여 것 (틀림)       공여 것 (틀림)

  rankonly  = 공여 벡터를 **참 벡터의 순서에 맞춰 재배치**.
              -> 값 다중집합은 공여와 **정확히 동일**, 순위는 참과 rho=1.
  dimshuffle = 참 벡터의 칸을 **에피소드마다 한 번 섞음**.
              -> 값 다중집합은 참과 **정확히 동일**, 순위는 무관.

예측 (숫자 보기 전에 적는다):
  순위가 값이면   rankonly ~= intact,  dimshuffle ~= frozen
  크기가 값이면   dimshuffle ~= intact, rankonly ~= frozen
  둘 다면         둘 다 중간

공여는 risk_frozen 과 같은 규약으로 **에피소드마다 하나, 끝까지 고정**이고
평가 시드·모델·에피소드 열이 23.3/23.8절과 같다. 각 조건에서 먹인 위험 벡터와
참 위험 벡터의 스피어만 rho 를 **실측해서** 같이 보고한다(가정하지 않는다).

출력: review_results/rank_vs_value.csv
"""
import csv
import os
import random
import sys

import occlude_state_blocks as OC               # noqa: E402

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

sys.path.append(OC.HERE)
import review_stats as RS                       # noqa: E402

from network_env import NetworkEnv              # noqa: E402
from network_env import DESTINATION             # noqa: E402

LO, HI = 0, 50
EPISODES = int(os.environ.get("RV_EPISODES", "200"))
EVAL_SEED = int(os.environ.get("RV_SEED", str(OC.EVAL_SEED)))
ALL_CONDS = ("intact", "risk_rankonly", "risk_rankpow50",
             "risk_rankpow33", "risk_rankpow05", "risk_rankpow400",
             "risk_rankpow05n", "risk_scale184", "risk_scale054",
             "risk_dimshuffle", "risk_frozen")
# RV_CONDS 로 일부만 돌릴 수 있다(조건별 롤아웃 시드가 독립이라 부분
# 실행 결과를 전체 실행 결과와 합칠 수 있다). intact 는 기준이라 항상 넣는다.
_sel = os.environ.get("RV_CONDS", "").strip()
if _sel:
    _want = set(x.strip() for x in _sel.split(",") if x.strip())
    for _w in _want:
        assert _w in ALL_CONDS, "알 수 없는 조건: %s" % _w
    _want.add("intact")
    CONDS = tuple(c for c in ALL_CONDS if c in _want)
else:
    CONDS = ALL_CONDS
# rankpow: 단조 x**p 재사상 — 순위는 rho=1 로 고정하고 값만 옮긴다.
# p 는 measure_rank_value_geometry.py 의 보정에서 골랐다(모두 rho=1.0000):
#   0.50 → L1 frozen 의 82%  · 코사인 0.9665 · 노름비 1.1721
#   0.33 → L1 frozen 의 139% · 코사인 0.9191 · 노름비 1.3161
#   0.05 → 코사인 0.7664 로 **frozen(0.7636)과 같은 자리**인데 rho=1 이다
#          → '피해는 방향(코사인) 때문'이라는 설명을 직접 반박하는 조건
#   4.00 → 코사인 0.9665 로 pow50 과 **같고** 노름비 0.7865 로 **반대 방향**
#          → 코사인을 고정한 채 노름을 양쪽으로 흔드는 조건
# 주의: p>=10 은 부동소수 동점이 생겨 rho 가 1 미만이 된다. 쓰지 말 것.
POWS = {"risk_rankpow50": 0.50, "risk_rankpow33": 0.33,
        "risk_rankpow05": 0.05, "risk_rankpow400": 4.00}
# pow05 는 rho=1 인데 노름비 1.8409 와 코사인 0.7662 가 **동시에** 움직여
# 피해가 나면 원인을 못 가른다. 축을 하나씩만 움직이는 조건 셋을 더 둔다.
#   rankpow05n : x**0.05 뒤 **원래 블록 노름으로 되돌린다**.
#                양수배는 순서와 코사인을 안 바꾸므로
#                rho 1.0000 · 노름비 **1.0000** · 코사인 **0.7662**.
#                → 노름을 중립으로 묶고 **코사인만** 묻는다.
#   scale184   : 참 블록에 1.8409 를 곱한다.
#                rho 1.0000 · 노름비 **1.8409** · 코사인 **1.0000**.
#                → 코사인을 중립으로 묶고 **노름만** 묻는다.
#   scale054   : 1/1.8409 = 0.5432 배. 노름 축 아래쪽(0.5432).
# 한계: 순수 배율은 값을 원래 [min,max] **밖으로** 내보낸다(pow* 는
# min-max 로 되돌려 범위 안에 있다). 그래서 scale* 가 피해를 내면
# "노름" 과 "범위 밖 입력" 이 섞인다. 반대로 scale* 가 무해하면 둘 다
# 반박된다. rankpow05n 쪽은 양수배라 값이 [min/1.84, max/1.84] 로
# **아래로만** 옮겨 범위 밖 문제가 약하다 — 그래서 **주 조건은 05n** 이다.
POWS_N = {"risk_rankpow05n": 0.05}
SCALES = {"risk_scale184": 1.8409, "risk_scale054": 1.0 / 1.8409}
# [2026-11-14 추가] **부분 섞기** — 50칸 중 k칸만 섞는다.
# 왜: 23.9.6.15 가 `frozen`(165.5) 과 `pow05n`(94.7) 이 `cos_dev`·
# `norm_dev` **두 축이 맞춰진 채로** 1.75배 갈리고, 둘을 가르는 게
# `rho_dev`(0.7505 대 0.0000) 임을 보였다. 그런데 `frozen` 은 **다른
# 모델의 값**을 넣으므로 주변분포도 같이 바뀐다 — 교란이 남는다.
# 부분 섞기는 값 다중집합을 **정확히** 보존하므로(노름비 1.0000,
# 주변분포 동일) 코사인과 rho 만 k 로 움직인다. k=50 은
# `risk_dimshuffle` 과 같다. 코사인이 `pow05n` 과 같아지는 k 를 쓰면
# "코사인·노름·주변분포가 다 같고 순위만 다른" 쌍이 만들어진다.
SHUFK = {"risk_shuf04": 4, "risk_shuf08": 8, "risk_shuf12": 12,
         "risk_shuf16": 16, "risk_shuf20": 20, "risk_shuf24": 24,
         "risk_shuf25": 25, "risk_shuf26": 26, "risk_shuf27": 27,
         "risk_shuf32": 32}


def spearman_np(a, b):
    """50칸짜리라 단순 구현. 동순위는 평균순위."""
    def rank(v):
        idx = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[idx] = np.arange(1, len(v) + 1, dtype=np.float64)
        return r
    ra, rb = rank(np.asarray(a)), rank(np.asarray(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = (np.sqrt((ra ** 2).sum()) * np.sqrt((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def make_fed(state, cond, donor, perm):
    """위험 블록만 바꾼 상태 사본. 다른 블록은 손대지 않는다."""
    out = np.array(state, dtype=np.float32, copy=True)
    true = np.asarray(state[LO:HI], dtype=np.float64)
    if cond == "intact":
        return out
    if cond == "risk_frozen":
        out[LO:HI] = np.asarray(donor[LO:HI], dtype=np.float32)
    elif cond == "risk_rankonly":
        # 공여의 값 다중집합을 참 벡터의 순서에 맞춰 재배치
        order = np.argsort(true, kind="mergesort")
        vals = np.sort(np.asarray(donor[LO:HI], dtype=np.float64))
        blk = np.empty(HI - LO, dtype=np.float64)
        blk[order] = vals
        out[LO:HI] = blk.astype(np.float32)
    elif cond in POWS:
        # min-max 정규화 후 x**p, 다시 원 구간.
        # 엄격하게 단조이므로 순위는 그대로다.
        lo, hi = float(true.min()), float(true.max())
        if hi - lo < 1e-12:
            return out
        x = (true - lo) / (hi - lo)
        out[LO:HI] = (lo + (hi - lo) * (x ** POWS[cond])).astype(np.float32)
    elif cond in POWS_N:
        # pow 재사상 뒤 원래 블록 노름으로 되돌린다.
        # 양수배이므로 순서(rho=1)와 코사인은 그대로다.
        lo, hi = float(true.min()), float(true.max())
        if hi - lo < 1e-12:
            return out
        x = (true - lo) / (hi - lo)
        blk = lo + (hi - lo) * (x ** POWS_N[cond])
        tn = float(np.sqrt((true ** 2).sum()))
        bn = float(np.sqrt((blk ** 2).sum()))
        if bn < 1e-12:
            return out
        out[LO:HI] = (blk * (tn / bn)).astype(np.float32)
    elif cond in SCALES:
        # 순수 배율. 코사인 정확히 1, rho 정확히 1, 노름만 바뀐다.
        out[LO:HI] = (true * SCALES[cond]).astype(np.float32)
    elif cond == "risk_dimshuffle":
        # 참 벡터의 값은 그대로, 칸 위치만 섞음
        out[LO:HI] = true[perm].astype(np.float32)
    elif cond in SHUFK:
        # perm 의 앞 k개를 **고를 칸**으로 쓴다. 그 집합 안에서만
        # 재배치하므로 나머지 칸은 손대지 않는다. src 는 그 집합의
        # 무작위 순서, tgt 는 같은 집합의 정렬 순서이므로 그 k칸에
        # 대한 **균등 무작위 치환**이다(고정점이 생길 수 있다).
        k = min(SHUFK[cond], HI - LO)
        src = np.asarray(perm[:k], dtype=np.int64)
        tgt = np.sort(src)
        blk = true.copy()
        blk[tgt] = true[src]
        out[LO:HI] = blk.astype(np.float32)
    else:
        raise ValueError(cond)
    return out


def run_eval(net, device, cond, donor_pool, episodes=EPISODES,
             seed=EVAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    drng = random.Random(seed + 1)
    prng = np.random.RandomState(seed + 2)

    delays, losses, hops, rhos = [], [], [], []
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        # 조건마다 같은 열을 보도록 두 RNG 는 조건과 무관하게 뽑는다
        donor = donor_pool[drng.randrange(len(donor_pool))]
        perm = prng.permutation(HI - LO)
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            fed = make_fed(state, cond, donor, perm)
            if cond != "intact":
                rhos.append(spearman_np(fed[LO:HI], state[LO:HI]))
            q = OC.qvals(net, fed, device)
            action = OC.masked_argmax_from_q(q, list(valid))
            state, _, done, _ = env.step(action)
        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
    return {"delay": float(np.mean(delays)) if delays else float("nan"),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "hop": float(np.mean(hops)) if hops else float("nan"),
            "rho": float(np.nanmean(rhos)) if rhos else 1.0,
            "success_rate": len(delays) / episodes * 100.0}


def paired(rows, cond, metric):
    sel = [r for r in rows if r["condition"] == cond]
    base = {r["model"]: r for r in rows if r["condition"] == "intact"}
    d = [r[metric] - base[r["model"]][metric] for r in sel
         if r["model"] in base and r[metric] == r[metric]]
    m, sd = OC.mean_std(d)
    se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
    t = m / se if se > 0 else 0.0
    p = RS.t_two_sided_p(t, len(d) - 1) if se > 0 else 1.0
    return m, t, p


def report(rows):
    bar = "=" * 78
    print("")
    print(bar)
    print("[0] 예측 (숫자 보기 전에 정해 둔 것)")
    print("-" * 78)
    print("  순위가 값이면   rankonly ~= intact,  dimshuffle ~= frozen")
    print("  크기가 값이면   dimshuffle ~= intact, rankonly ~= frozen")
    print("  값 다중집합은 rankonly=공여와 동일, dimshuffle=참과 동일이고")
    print("  순위는 rankonly=참과 rho 1, dimshuffle=무관이다 (아래 실측 rho).")
    print("")
    print("[1] 조건별 (모델 10개, %d 에피소드)" % EPISODES)
    print("-" * 78)
    print("%-17s %14s %11s %9s %8s %8s"
          % ("조건", "지연", "손실", "홉", "rho", "ms/홉"))
    for cond in CONDS:
        sel = [r for r in rows if r["condition"] == cond]
        dm, dsd = OC.mean_std([r["delay"] for r in sel])
        lm, _ = OC.mean_std([r["loss"] for r in sel])
        hm, _ = OC.mean_std([r["hop"] for r in sel])
        rm, _ = OC.mean_std([r["rho"] for r in sel])
        print("%-17s %7.2f±%-6.2f %11.5f %9.3f %8.4f %8.2f"
              % (cond, dm, dsd, lm, hm, rm, dm / hm if hm else float("nan")))
    print("")
    print("[2] intact 대비 짝지은 검정 (Holm)")
    print("-" * 78)
    print("%-17s %-7s %10s %9s %10s %10s"
          % ("조건", "지표", "차이", "t", "p", "p_holm"))
    tests = []
    for cond in CONDS[1:]:
        for metric in ("delay", "loss", "hop"):
            m, t, p = paired(rows, cond, metric)
            tests.append((cond, metric, m, t, p))
    holm = RS.holm_bonferroni([x[4] for x in tests])
    for (cond, metric, m, t, p), ph in zip(tests, holm):
        print("%-17s %-7s %10.4f %9.3f %10s %10s"
              % (cond, metric, m, t, RS.fmt_p(p), RS.fmt_p(ph)))
    print("")
    print("[3] 판정 — 순위 몫 대 크기 몫")
    print("-" * 78)
    hm = {(c, mt): m for (c, mt, m, _, _) in tests}
    # 세 조건을 **하드코딩**해서 참조했더니, 그 중 하나라도 없는 부분
    # 실행(RV_CONDS=...)에서 KeyError 로 죽었다. CSV 저장은 이 앞에서
    # 끝나므로 자료 손실은 없었지만 출력이 막힌다. 없으면 건너뛴다.
    want = ("risk_rankonly", "risk_dimshuffle", "risk_frozen")
    missing = [c for c in want if (c, "delay") not in hm]
    if missing:
        print("  [3] 건너뜀 — 이 판정은 %s 가 다 있어야 한다. 없는 것: %s"
              % (", ".join(want), ", ".join(missing)))
        print("  (부분 실행이면 정상이다. [1]·[2]·CSV 는 위에 다 나왔다.)")
        print(bar)
        return
    for metric in ("delay", "loss"):
        dr = hm[("risk_rankonly", metric)]
        dd = hm[("risk_dimshuffle", metric)]
        df = hm[("risk_frozen", metric)]
        print("  %s: D_rankonly %+9.4f · D_dimshuffle %+9.4f · D_frozen %+9.4f"
              % (metric, dr, dd, df))
        if df != 0:
            print("        순서만 살리면 피해의 %.1f%% 가 남는다 (rankonly/frozen)"
                  % (dr / df * 100.0))
            print("        크기만 살리면 피해의 %.1f%% 가 남는다 (dimshuffle/frozen)"
                  % (dd / df * 100.0))
    print("")
    print("  읽는 법: rankonly 가 intact 에 가깝고 dimshuffle 이 frozen 에")
    print("  가까우면 **순위가 값**이다. 반대면 크기가 값이다. 둘 다 frozen")
    print("  근처면 순위와 크기가 **함께** 있어야 하는 것이고, 그때는")
    print("  23.8.2절의 '값은 순위에 있다'를 **약화**해서 써야 한다.")
    print(bar)


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
    print("모델 %d · 공여 %d · 에피소드 %d · 시드 %d"
          % (len(nets), len(donor_pool), EPISODES, EVAL_SEED))
    print("")

    rows = []
    for nm, net in nets:
        line = [nm]
        for cond in CONDS:
            r = run_eval(net, device, cond, donor_pool)
            r.update({"model": nm, "condition": cond})
            rows.append(r)
            line.append("%s %.1f" % (cond.replace("risk_", ""), r["delay"]))
        print("  " + " | ".join(line))

    # RV_CONDS 로 일부만 돌릴 때는 RV_TAG 로 파일을 갈라 저장한다.
    # 안 가르면 전체 실행 결과를 부분 결과가 덮어써 날린다.
    tag = os.environ.get("RV_TAG", "").strip()
    assert tag or not _sel, "부분 실행(RV_CONDS)에는 RV_TAG 가 필요하다"
    name = "rank_vs_value%s.csv" % (("_" + tag) if tag else "")
    path = os.path.join(OC.OUT_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "condition", "delay", "loss", "hop", "rho",
                    "success_rate"])
        for r in rows:
            w.writerow([r["model"], r["condition"], "%.4f" % r["delay"],
                        "%.6f" % r["loss"], "%.4f" % r["hop"],
                        "%.4f" % r["rho"], "%.2f" % r["success_rate"]])
    report(rows)
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
