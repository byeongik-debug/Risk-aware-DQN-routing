# -*- coding: utf-8 -*-
"""23.6절의 빈칸 — risk_mean 의 +51.11 ms 중 '정보 손실' 몫과 '분포 이동' 몫.

23.3절 측정: 위험 블록의 정보만 지우고 크기를 보존한 `risk_mean` 이 지연
+51.11 ms (p_holm=0.0014), 손실 +0.0184 (p_holm=0.0156). 그래서 "정책은 위험
정보를 쓴다"고 썼다. **그런데 이 숫자는 교란되어 있다**: 배치 평균 위험 벡터는
실제로는 **한 번도 발생하지 않는 입력**이다. 따라서 열화의 일부는 정보를 지운
탓이 아니라 **분포 밖 입력을 먹인 탓**일 수 있다.

가르는 조건이 `risk_frozen` 이다 — 에피소드마다 **다른 공여 상태의 실제 위험
블록**을 뽑아 그 에피소드 끝까지 고정한다. 공여 벡터는 **분포 안**에 있고
**시점만** 틀리다. 그래서:

  risk_frozen ≈ intact        → 정책은 '현재' 위험을 안 쓴다. 그럴듯한 위험
                                벡터면 아무거나 된다 → +51.11 의 상당 부분이
                                **분포 이동**이고 23.3절 결론을 좁혀야 한다.
  intact < risk_frozen < mean → 정보 몫과 분포 몫이 섞여 있다. 정보 몫의
                                하한이 risk_frozen 의 열화다.
  risk_frozen ≈ risk_mean     → 시점이 틀린 것만으로도 평균 벡터만큼 나빠진다
                                → +51.11 은 **정보 손실로 읽어도 된다**.

`measure_path_diversity.py` 가 이미 `risk_frozen` 을 돌리지만 **경로와 홉만**
기록하고 지연을 안 남긴다. 경로가 갈리는 것과 경로가 **좋은** 것은 다르므로
품질을 따로 재야 한다. 이 스크립트가 그것만 한다.

조건 넷 (occlusion 본실행과 같은 모델·같은 평가 시드·같은 에피소드 열):
  intact        - 원본
  risk_frozen   - 공여 위험 블록, 에피소드 내 고정 (분포 안, 시점 틀림)
  risk_mean     - 배치 평균 위험 블록 (정보 0, 분포 밖)      [23.3절 재현]
  risk_shuffle  - 공여 위험 블록, 매 스텝 교체 (분포 안, 매번 틀림)  [상한 축척]

출력: review_results/risk_frozen_quality.csv
"""
import csv
import os
import random
import sys

import occlude_state_blocks as OC               # noqa: E402  (sys.path 먼저)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

sys.path.append(OC.HERE)
import review_stats as RS                       # noqa: E402

from network_env import NetworkEnv              # noqa: E402
from network_env import DESTINATION             # noqa: E402

OUT_DIR = OC.OUT_DIR
EPISODES = int(os.environ.get("RFQ_EPISODES", "200"))
EVAL_SEED = int(os.environ.get("RFQ_SEED", str(OC.EVAL_SEED)))

# (이름, 블록, 모드, 에피소드내고정)
CONDS = (("intact", None, None, False),
         ("risk_frozen", "risk", "shuffle", True),
         ("risk_mean", "risk", "mean", False),
         ("risk_shuffle", "risk", "shuffle", False))


def run_eval(net, device, blk, mode, frozen, donor_pool, blockmean,
             episodes=EPISODES, seed=EVAL_SEED):
    """occlude_state_blocks.run_eval_occluded 와 같은 절차 + 에피소드내 고정.

    차이는 `frozen` 하나뿐이다. True 면 공여를 **에피소드 시작 때 한 번** 뽑아
    그 에피소드 전체에 같은 위험 블록을 먹인다. 행동 선택은 RNG 를 쓰지 않는
    인라인 argmax 이고 공여 추출은 전용 RNG 라서, 네 조건이 **같은 평가
    에피소드 열**을 본다 (22.8절의 평가기 교란을 섞지 않기 위한 것).
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

    delays, losses, hops, donors = [], [], [], []
    steps = 0
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=OC.SCENARIO,
                         risk_window=OC.WINDOW)
        env.reset()
        for _ in range(OC.WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        ep_donor = None
        if lo is not None and frozen:
            j = drng.randrange(len(donor_pool))
            donors.append(j)
            ep_donor = donor_pool[j]
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            if lo is None:
                fed = state
            else:
                donor = ep_donor
                if donor is None:
                    donor = donor_pool[drng.randrange(len(donor_pool))]
                fed = OC.occlude(state, lo, hi, mode, donor, blockmean)
            q = OC.qvals(net, fed, device)
            action = OC.masked_argmax_from_q(q, list(valid))
            steps += 1
            state, _, done, _ = env.step(action)
        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
    n_donor = len(set(donors))
    return {"delay": float(np.mean(delays)) if delays else float("nan"),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "hop": float(np.mean(hops)) if hops else float("nan"),
            "success_rate": len(delays) / episodes * 100.0,
            "n_success": len(delays), "steps": steps, "n_donor": n_donor}


def main():
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
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
    blockmean = np.mean(np.stack(donor_pool), axis=0)
    print("모델 %d · 공여 묶음 %d · 에피소드 %d · 평가시드 %d"
          % (len(nets), len(donor_pool), EPISODES, EVAL_SEED))
    print("위험 블록 평균 L2 %.4f · 공여 간 블록 L2 표준편차 %.4f"
          % (float(np.linalg.norm(blockmean[0:50])),
             float(np.std([np.linalg.norm(s[0:50]) for s in donor_pool]))))
    print("")

    rows = []
    for nm, net in nets:
        line = [nm]
        for cond, blk, mode, frozen in CONDS:
            r = run_eval(net, device, blk, mode, frozen, donor_pool,
                         blockmean)
            r.update({"model": nm, "condition": cond})
            rows.append(r)
            line.append("%s %.2f" % (cond, r["delay"]))
        print("  " + " | ".join(line))

    path = os.path.join(OUT_DIR, "risk_frozen_quality.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "condition", "delay", "loss", "hop",
                    "success_rate", "n_success", "steps", "n_donor"])
        for r in rows:
            w.writerow([r["model"], r["condition"], "%.4f" % r["delay"],
                        "%.6f" % r["loss"], "%.4f" % r["hop"],
                        "%.2f" % r["success_rate"], r["n_success"],
                        r["steps"], r["n_donor"]])
    report(rows)
    print("저장: %s" % path)


def paired(rows, cond, metric):
    """같은 모델끼리 intact 와 짝지은 차이의 1표본 t (occlusion 본실행과 동일)."""
    sel = [r for r in rows if r["condition"] == cond]
    base = {r["model"]: r for r in rows if r["condition"] == "intact"}
    d = [r[metric] - base[r["model"]][metric] for r in sel
         if r["model"] in base and r[metric] == r[metric]]
    m, sd = OC.mean_std(d)
    se = sd / len(d) ** 0.5 if len(d) > 1 else 0.0
    t = m / se if se > 0 else 0.0
    p = RS.t_two_sided_p(t, len(d) - 1) if se > 0 else 1.0
    return m, t, p, len(d)


def report(rows):
    bar = "=" * 78
    print("")
    print(bar)
    print("[0] 읽는 규칙 — 숫자를 보기 전에 정한다")
    print("-" * 78)
    print("  설계 당시 예측: risk_frozen 과 risk_mean 은 둘 다 현재 위험에 대한")
    print("  정보가 0 이고(공여는 다른 QoS 실현) 차이는 **분포 안인가**뿐이므로")
    print("  D_mean >= D_frozen 이어야 하고, 그 차가 분포이동 덧돈이다.")
    print("  **탐침에서 이 예측이 꺾였다 (아래 [3]).** 두 조건은 분포 말고도")
    print("  다른 점이 있다: 평균 벡터는 **중립**(모든 엣지가 고르게 중간)이고")
    print("  공여 벡터는 특정 엣지에 대해 **자신 있게 틀렸다**. 따라서:")
    print("    D_frozen = '자신 있게 틀린 위험'의 비용")
    print("    D_mean   = '중립적으로 모르는 위험'의 비용")
    print("  D_frozen ~ 0 이면 정책은 위험 **내용**을 안 읽는 것이고(틀린 내용을")
    print("  먹여도 안 나빠진다), 크면 내용을 읽는 것이다. 홉을 같이 볼 것 —")
    print("  홉이 안 늘고 지연만 나빠지면 **같은 길이에서 나쁜 링크를 고른** 것이다.")
    print("")
    print("[1] 조건별 지표 (모델 10개, %d 에피소드)" % EPISODES)
    print("-" * 78)
    print("%-15s %14s %12s %10s %9s" % ("조건", "지연", "손실", "홉", "성공%"))
    for cond, _, _, _ in CONDS:
        sel = [r for r in rows if r["condition"] == cond]
        dm, dsd = OC.mean_std([r["delay"] for r in sel])
        lm, _ = OC.mean_std([r["loss"] for r in sel])
        hm, _ = OC.mean_std([r["hop"] for r in sel])
        sm, _ = OC.mean_std([r["success_rate"] for r in sel])
        print("%-15s %7.2f±%-6.2f %12.5f %10.3f %9.2f"
              % (cond, dm, dsd, lm, hm, sm))
    nd = [r["n_donor"] for r in rows if r["condition"] == "risk_frozen"]
    print("  (risk_frozen 이 쓴 서로 다른 공여 수: 모델당 중위 %d)"
          % sorted(nd)[len(nd) // 2])
    print("")
    print("[2] intact 대비 짝지은 검정 (Holm 보정)")
    print("-" * 78)
    print("%-15s %-9s %10s %9s %10s %10s"
          % ("조건", "지표", "차이", "t", "p", "p_holm"))
    tests = []
    for cond, _, _, _ in CONDS[1:]:
        for metric in ("delay", "loss", "hop"):
            m, t, p, n = paired(rows, cond, metric)
            tests.append((cond, metric, m, t, p))
    holm = RS.holm_bonferroni([x[4] for x in tests])
    for (cond, metric, m, t, p), ph in zip(tests, holm):
        print("%-15s %-9s %10.4f %9.3f %10s %10s"
              % (cond, metric, m, t, RS.fmt_p(p), RS.fmt_p(ph)))
    print("")
    print("[3] 예측 대조 — 분포이동 분해가 되는가")
    print("-" * 78)
    hm = {(c, mt): (m, ph) for (c, mt, m, _, _), ph
          in zip(tests, holm)}
    for metric in ("delay", "loss", "hop"):
        df, pf = hm[("risk_frozen", metric)]
        dm, pm = hm[("risk_mean", metric)]
        ds, ps = hm[("risk_shuffle", metric)]
        print("  %s:" % metric)
        print("    D_mean    %+10.4f  (p_holm %-9s) 중립·무정보 (분포 밖)"
              % (dm, RS.fmt_p(pm)))
        print("    D_frozen  %+10.4f  (p_holm %-9s) 자신있게 틀림 (분포 안)"
              % (df, RS.fmt_p(pf)))
        print("    D_shuffle %+10.4f  (p_holm %-9s) 매 스텝 틀림"
              % (ds, RS.fmt_p(ps)))
        if dm != 0 and df != 0:
            ok = "예측대로" if dm >= df else "**예측 반대**"
            print("    D_mean >= D_frozen ? %s  (frozen/mean = %.2f배)"
                  % (ok, df / dm))
    print("")
    print("  판정 규칙: D_mean >= D_frozen 이면 그 차를 분포이동 덧돈으로 읽을")
    print("  수 있다. **반대로 나오면 분해 자체가 불가능하다** — 두 조건이 분포")
    print("  말고 '중립 대 자신있게 틀림'에서도 다르기 때문이다. 그 경우 쓸 수")
    print("  있는 결론은 분해가 아니라 **순서**다: 무정보보다 오정보가 더 비싸고,")
    print("  오정보가 비싸다는 것은 정책이 위험 **내용**을 읽는다는 뜻이다.")
    print("")
    print("  한계 둘: (가) 공여 위험 블록은 **주변분포**로는 분포 안이지만 그")
    print("  에피소드의 현재·방문 블록과는 짝이 안 맞는다. (나) 중립성은")
    print("  설계상의 성질이고 따로 재지 않았다 (평균 벡터의 차원간 분산으로")
    print("  확인할 수 있다 — [1] 머리말의 공여 간 블록 L2 표준편차와 대조).")
    print(bar)


if __name__ == "__main__":
    main()
