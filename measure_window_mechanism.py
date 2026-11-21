# -*- coding: utf-8 -*-
"""위험 윈도우 K 가 에이전트가 보는 상태를 실제로 얼마나 바꾸는가.

표 10 K 행의 품질 열은 Holm 보정 후 16칸 전부 비유의였다(window_sweep_tests_final.csv).
"표본이 작아서"인지 "설계상 차이가 날 수 없어서"인지 갈라야 한다. 후자면 런을 더
학습해도 달라질 것이 없다.

K 가 건드리는 곳은 risk_calculator.calculate_link_risk 의 두 항뿐이다.
    delay_instability = min(std(delay_history) / 20.0,  1.0)   가중치 0.15
    loss_instability  = min(std(loss_history)  / 0.03,  1.0)   가중치 0.10
둘 다 len(history) > 1 일 때만 켜지고 1.0 에서 잘린다. 즉 K=1 은 maxlen=1 이라
두 항이 영구히 0 이다 — 시간 항이 아예 없는 모델이다.

에이전트 없이 환경만 굴린다. 같은 시드로 update_dynamic_qos() 를 반복하면
K 말고는 다를 것이 없으므로, 위험 벡터의 차이는 전부 K 의 몫이다.
원본 폴더는 읽기만 한다.
"""
import csv
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, os.pardir, "SDN_DQN_new_env"))
OUT = os.path.join(HERE, "review_results")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from network_env import EDGES, NetworkEnv  # noqa: E402

WINDOWS = [1, 3, 5, 7, 10]
SCENARIO = "dynamic"
SEEDS = list(range(900, 930))   # 궤적 30개
STEPS = 18                      # 워밍업 10 + 전형적 경로 길이 8
DELAY_NORM, LOSS_NORM = 20.0, 0.03
W_DELAY, W_LOSS = 0.15, 0.10


def run_one(window, seed):
    """(위험벡터[step][edge], 지연불안정 생값, 손실불안정 생값, 이용률)"""
    env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=window)
    # network_env 는 stdlib random 만 쓴다 (np.random 호출 0곳). 그것을 시드해야
    # K 말고는 궤적이 같아진다.
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    keys = [env.normalize_edge(u, v) for (u, v) in EDGES]
    risk, d_raw, l_raw, util, hist_len = [], [], [], [], []
    for _ in range(STEPS):
        env.update_dynamic_qos()
        risk.append([env.qos[e]["risk"] for e in keys])
        util.append([env.qos[e]["utilization"] for e in keys])
        dr, lr, hl = [], [], []
        for e in keys:
            dh, lh = env.delay_history[e], env.loss_history[e]
            dr.append(np.std(dh) / DELAY_NORM if len(dh) > 1 else 0.0)
            lr.append(np.std(lh) / LOSS_NORM if len(lh) > 1 else 0.0)
            hl.append(len(dh))
        d_raw.append(dr)
        l_raw.append(lr)
        hist_len.append(hl)
    return (np.array(risk), np.array(d_raw), np.array(l_raw),
            np.array(util), np.array(hist_len))


def main():
    data = {}
    for w in WINDOWS:
        parts = [run_one(w, s) for s in SEEDS]
        data[w] = tuple(np.concatenate([p[i] for p in parts], axis=0) for i in range(5))
        print("K=%-2d 수집 완료  (step x edge = %s)" % (w, data[w][0].shape))

    # 자체 검증: 궤적이 K 간 동일한가 (이용률은 K와 무관해야 한다)
    base_util = data[5][3]
    worst = max(float(np.abs(data[w][3] - base_util).max()) for w in WINDOWS)
    print()
    print("궤적 일치 검증 — 이용률의 K 간 최대 차이: %.3e  %s"
          % (worst, "(동일)" if worst < 1e-12 else "(불일치! 아래 수치 무효)"))
    assert worst < 1e-12, "같은 시드인데 궤적이 갈렸다"

    rows = []
    base_risk = data[5][0]
    print()
    print("=== K 가 두 시간 항을 어떻게 바꾸는가 ===")
    hdr = ("K", "지연항 켜짐%", "지연항 포화%", "손실항 켜짐%", "손실항 포화%",
           "창 꽉찬%", "창평균", "위험평균")
    print("%-3s %12s %12s %12s %12s %10s %8s %9s" % hdr)
    for w in WINDOWS:
        risk, d_raw, l_raw, _util, hl = data[w]
        n = d_raw.size
        row = {
            "window": w,
            "delay_term_on_pct": 100.0 * np.count_nonzero(d_raw > 0) / n,
            "delay_term_sat_pct": 100.0 * np.count_nonzero(d_raw >= 1.0) / n,
            "loss_term_on_pct": 100.0 * np.count_nonzero(l_raw > 0) / n,
            "loss_term_sat_pct": 100.0 * np.count_nonzero(l_raw >= 1.0) / n,
            "window_full_pct": 100.0 * np.count_nonzero(hl >= w) / n,
            "hist_len_mean": float(hl.mean()),
            "risk_mean": float(risk.mean()),
            "delay_raw_mean": float(d_raw.mean()),
            "loss_raw_mean": float(l_raw.mean()),
        }
        rows.append(row)
        print("%-3d %12.2f %12.2f %12.2f %12.2f %10.2f %8.2f %9.5f"
              % (w, row["delay_term_on_pct"], row["delay_term_sat_pct"],
                 row["loss_term_on_pct"], row["loss_term_sat_pct"],
                 row["window_full_pct"], row["hist_len_mean"], row["risk_mean"]))

    print()
    print("=== K 를 바꾸면 에이전트가 보는 50차원 위험 벡터가 얼마나 달라지는가 (기준 K=5) ===")
    print("%-3s %12s %12s %12s %14s %14s" % ("K", "평균|Δrisk|", "최대|Δrisk|",
                                             "동일칸%", "상위10 겹침%", "순위상관 rho"))
    for w, row in zip(WINDOWS, rows):
        risk = data[w][0]
        diff = np.abs(risk - base_risk)
        same = 100.0 * np.count_nonzero(diff < 1e-12) / diff.size
        top = []
        rho = []
        for i in range(risk.shape[0]):
            a = set(np.argsort(-risk[i])[:10])
            b = set(np.argsort(-base_risk[i])[:10])
            top.append(len(a & b) / 10.0)
            ra = np.argsort(np.argsort(risk[i]))
            rb = np.argsort(np.argsort(base_risk[i]))
            sa, sb = ra.std(), rb.std()
            rho.append(float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))
                       if sa and sb else float("nan"))
        row["mean_abs_drisk"] = float(diff.mean())
        row["max_abs_drisk"] = float(diff.max())
        row["identical_cells_pct"] = same
        row["top10_overlap_pct"] = 100.0 * float(np.mean(top))
        row["rank_rho"] = float(np.nanmean(rho))
        print("%-3d %12.6f %12.6f %12.2f %14.2f %14.4f"
              % (w, row["mean_abs_drisk"], row["max_abs_drisk"], same,
                 row["top10_overlap_pct"], row["rank_rho"]))

    # 두 시간 항이 기여하는 위험의 몫
    print()
    print("=== 두 시간 항이 위험 점수에서 차지하는 몫 ===")
    for w, row in zip(WINDOWS, rows):
        risk, d_raw, l_raw, _u, _h = data[w]
        contrib = W_DELAY * np.minimum(d_raw, 1.0) + W_LOSS * np.minimum(l_raw, 1.0)
        row["temporal_contrib_mean"] = float(contrib.mean())
        row["temporal_share_pct"] = 100.0 * float(contrib.mean() / risk.mean())
        print("  K=%-2d 시간항 기여 평균 %.5f / 위험 평균 %.5f = %5.2f%%"
              % (w, contrib.mean(), risk.mean(), row["temporal_share_pct"]))

    # 순위가 안 바뀌는 이유: 시간 항이 나머지 항과 중복인가
    print()
    print("=== 시간 항이 비시간 항과 얼마나 중복인가 (피어슨 r) ===")
    print("%-3s %16s %16s %16s" % ("K", "시간항 vs 이용률", "시간항 vs 비시간위험", "시간항 vs 전체위험"))
    for w, row in zip(WINDOWS, rows):
        risk, d_raw, l_raw, util, _h = data[w]
        contrib = (W_DELAY * np.minimum(d_raw, 1.0)
                   + W_LOSS * np.minimum(l_raw, 1.0)).ravel()
        nontemp = (risk.ravel() - contrib)
        if contrib.std() == 0.0:
            row["r_temporal_util"] = float("nan")
            row["r_temporal_nontemporal"] = float("nan")
            row["r_temporal_risk"] = float("nan")
            print("%-3d %16s %16s %16s" % (w, "n/a (항이 0)", "n/a", "n/a"))
            continue
        for key, other in (("r_temporal_util", util.ravel()),
                           ("r_temporal_nontemporal", nontemp),
                           ("r_temporal_risk", risk.ravel())):
            row[key] = float(np.corrcoef(contrib, other)[0, 1])
        print("%-3d %16.4f %16.4f %16.4f"
              % (w, row["r_temporal_util"], row["r_temporal_nontemporal"],
                 row["r_temporal_risk"]))

    # 시간 항을 떼면 순위가 얼마나 바뀌는가 (K와 무관한 상한)
    print()
    print("=== 시간 항을 아예 떼면 순위가 얼마나 바뀌는가 ===")
    for w, row in zip(WINDOWS, rows):
        risk, d_raw, l_raw, _u, _h = data[w]
        contrib = W_DELAY * np.minimum(d_raw, 1.0) + W_LOSS * np.minimum(l_raw, 1.0)
        stripped = risk - contrib
        ov, rh = [], []
        for i in range(risk.shape[0]):
            a = set(np.argsort(-risk[i])[:10])
            b = set(np.argsort(-stripped[i])[:10])
            ov.append(len(a & b) / 10.0)
            ra = np.argsort(np.argsort(risk[i]))
            rb = np.argsort(np.argsort(stripped[i]))
            if ra.std() and rb.std():
                rh.append(float(np.mean((ra - ra.mean()) * (rb - rb.mean()))
                                / (ra.std() * rb.std())))
        # 출력만 하고 CSV 에 안 담겨 초안의 0.9866 이 출처 없는 값이 돼
        # 있었다(감사 32절). 행에 실어 산출물로 남긴다.
        row["stripped_top10_overlap_pct"] = 100.0 * float(np.mean(ov))
        row["stripped_rank_rho"] = (float(np.mean(rh)) if rh
                                    else float("nan"))
        print("  K=%-2d 시간항 제거 후  상위10 겹침 %6.2f%%  순위상관 rho=%.4f"
              % (w, row["stripped_top10_overlap_pct"],
                 row["stripped_rank_rho"]))

    path = os.path.join(OUT, "window_mechanism.csv")
    cols = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for row in rows:
            wr.writerow(row)
    print()
    print("저장: %s (%d행)" % (path, len(rows)))


if __name__ == "__main__":
    main()
