# -*- coding: utf-8 -*-
"""23.9절 방어용 — rankonly 의 '수치가 가까워서 무해했다' 반론을 수치로 막는다.

왜 필요한가
-----------
`measure_rank_vs_value.py` 에서 `risk_rankonly` 는 공여 값 다중집합을
**참 벡터의 순서에 맞춰 단조 재배치**한다. 재배열 부등식에 따라 그 배치는
그 다중집합의 모든 순열 중 참 벡터와의 L1 거리가 **최소**다. 그러니
"순위를 지켜서 무해"가 아니라 "수치가 제일 가까워서 무해"일 수 있다.
이 스크립트는 조건별 수치 교란 크기를 직접 재서 그 반론의 크기를 적는다.

같이 재는 것
-----------
- `risk_dimshuffle` 은 **참 다중집합을 그대로** 쓴다(값 변화 0, 순서만 파괴).
  그 조건이 피해의 97.6% 를 낸다면 "수치 근접"으로는 설명이 안 된다.
- 참 벡터의 **동점 비율** — rankonly 의 rho 가 1.0000 이 아니라 0.9970 인
  이유 후보. 동점 칸에 서로 다른 공여 값이 들어가면 순위가 어긋난다.

노름도 같이 재는 이유 (23.9.4 의 교란 변수)
-----------
순위를 지키면서 값을 옮기는 `rankpow` 는 p<1 이라 블록 노름을 **키운다**.
그래서 그 조건이 피해를 내더라도 "순위가 부족해서"인지 "노름이 변해서"인지
갈리지 않는다 — 전에는 이걸 말로만 적었다. 여기서 두 축을 숫자로 가른다:
`dimshuffle` 은 노름을 **그대로 두고**(값 다중집합 보존) 순위만 깨는데 이미
피해를 낸다. 두 조건이 교차하므로, rankpow 가 무해하면 노름 변화는 피해의
설명에서 빠진다. 블록 제곱몫은 23.2절 (가)의 57.07% 와 **다른 추정량**이라
(그쪽은 평균 L2 를 먼저 내고 제곱한다) 두 방식을 나란히 적는다.

네트워크 전방계산 없음. `measure_rank_vs_value.py` 와 같은 배치·같은 공여
추출(같은 시드·같은 난수 호출 순서)을 재현해 상태 사본만 만든다.

출력: review_results/rank_value_geometry.csv
"""
import csv
import os
import random
import sys

import occlude_state_blocks as OC               # noqa: E402

# OC 가 임포트 시 HERE 를 sys.path 에서 뺀다(원본 network_env 를 잡기 위해).
sys.path.append(OC.HERE)
import measure_rank_vs_value as RV              # noqa: E402

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

LO, HI = RV.LO, RV.HI
CONDS = ("risk_rankonly", "risk_rankpow50", "risk_rankpow33",
         "risk_rankpow05", "risk_rankpow400",
         "risk_rankpow05n", "risk_scale184", "risk_scale054",
         "risk_dimshuffle", "risk_frozen")
N_DRAW = int(os.environ.get("RVG_DRAW", "2000"))
# [6] 배터리 민감도용. (라벨, 모델, 배터리 종류)
ABL_MODELS = os.path.join(OC.HERE, "models_ablation")
BATTS = (("기준 greedy(window5_run01)", "window5_run01.pth", "greedy"),
         ("random(window5_run01)", "window5_run01.pth", "random"),
         ("greedy(factor_full_run01)", "factor_full_run01.pth", "greedy"),
         ("greedy(window5_run04)", "window5_run04.pth", "greedy"))


def powmap(true, p):
    """순위 보존 단조 재사상. min-max 정규화 후 x**p, 다시 원 구간."""
    lo, hi = float(true.min()), float(true.max())
    if hi - lo < 1e-12:
        return true.copy()
    x = (true - lo) / (hi - lo)
    return lo + (hi - lo) * (x ** p)


def tie_stats(vec):
    """(동점에 속한 칸 수, 서로 다른 값 수)."""
    _, cnt = np.unique(np.asarray(vec, dtype=np.float64),
                       return_counts=True)
    return int(cnt[cnt > 1].sum()), int(len(cnt))


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()

    net = OC.load_net(os.path.join(OC.ORIG_MODELS,
                                   "window5_run01.pth"), device)
    battery = OC.build_battery("greedy", device, net)
    pool = [s for s, _ in battery]
    print("배치 상태 %d · 추출 %d · 블록 칸 %d" % (len(pool), N_DRAW, HI - LO))

    # [7] 범위 밖 정도용. 참 블록이 배터리 전체에서 실제로 가지는 구간.
    # scale* 는 순수 배율이라 값을 이 구간 밖으로 내보낸다 — 그 한계를
    # 말로 적는 대신 크기로 잰다.
    allblk = np.asarray([np.asarray(s[LO:HI], dtype=np.float64)
                         for s in pool], dtype=np.float64)
    gmin, gmax = float(allblk.min()), float(allblk.max())
    grange = gmax - gmin
    print("참 블록 관측 구간 [%.4f, %.4f] (폭 %.4f, 배터리 %d상태 전체)"
          % (gmin, gmax, grange, len(pool)))

    rng = random.Random(RV.EVAL_SEED)
    keys = ("l1", "l2", "rho", "same", "bn", "rt", "sh", "cos",
            "oor", "exc", "sdr", "cvr")
    acc = {c: {k: [] for k in keys} for c in CONDS}
    tru = {"bn": [], "sh": [], "wn": []}
    tie_n, tie_u = [], []
    for _ in range(N_DRAW):
        state = pool[rng.randrange(len(pool))]
        donor = pool[rng.randrange(len(pool))]
        true = np.asarray(state[LO:HI], dtype=np.float64)
        tnorm = float(np.sqrt((true ** 2).sum()))
        whole = float((np.asarray(state, dtype=np.float64) ** 2).sum())
        tru["bn"].append(tnorm)
        tru["sh"].append((tnorm ** 2) / whole if whole else float("nan"))
        tru["wn"].append(float(np.sqrt(whole)))
        t, u = tie_stats(true)
        tie_n.append(t)
        tie_u.append(u)
        perm = np.random.RandomState(rng.randrange(1 << 30)).permutation(
            HI - LO)
        for c in CONDS:
            fed = np.asarray(RV.make_fed(state, c, donor, perm)[LO:HI],
                             dtype=np.float64)
            d = fed - true
            acc[c]["l1"].append(float(np.abs(d).mean()))
            acc[c]["l2"].append(float(np.sqrt((d ** 2).mean())))
            acc[c]["rho"].append(RV.spearman_np(fed, true))
            acc[c]["same"].append(float((np.abs(d) < 1e-9).mean()))
            fnorm = float(np.sqrt((fed ** 2).sum()))
            acc[c]["bn"].append(fnorm)
            acc[c]["rt"].append(fnorm / tnorm if tnorm else float("nan"))
            # 블록을 갈면 상태 전체 제곱노름도 같이 바뀐다.
            w2 = whole - tnorm ** 2 + fnorm ** 2
            acc[c]["sh"].append((fnorm ** 2) / w2 if w2 else float("nan"))
            # 코사인 = 노름을 뺀 방향. 노름 요약이 못 잡는 블록 안 패턴.
            den = fnorm * tnorm
            acc[c]["cos"].append(float(np.dot(fed, true) / den)
                                 if den else float("nan"))
            # 블록 안 표준편차비. 코사인은 방향만 보므로 "값이 평탄해졌다"
            # 를 직접 말하지 못한다. x**0.05 는 (0,1] 을 거의 전부 1 근처로
            # 몰아 블록을 **상수에 가깝게** 만드는데(최소 칸만 lo), 그건
            # path_diversity 의 risk_mean(정확히 상수, 10/10 모델 경로 1개
            # 붕괴)과 같은 모양이다. 그 평탄화를 크기로 적는다.
            tsd = float(true.std())
            acc[c]["sdr"].append(float(fed.std()) / tsd
                                 if tsd > 1e-12 else float("nan"))
            # 표준편차비는 "간격이 지워졌다" 와 "전체가 작아졌다" 를 섞는다
            # — 순수 배율 k 는 간격을 전혀 안 건드리는데 표준편차비가 k 로
            # 나온다. 척도에 안 휘둘리는 변동계수(std/mean)비로 다시 잰다.
            # 순수 배율이면 정확히 1, 값이 한쪽으로 몰리면 0 으로 간다.
            tm, fm = float(true.mean()), float(fed.mean())
            tcv = tsd / tm if tm > 1e-12 else float("nan")
            fcv = float(fed.std()) / fm if fm > 1e-12 else float("nan")
            acc[c]["cvr"].append(fcv / tcv if tcv and tcv == tcv
                                 and tcv > 1e-12 else float("nan"))
            # 범위 밖: 배터리 전체 관측 구간을 벗어난 칸 비율과,
            # 가장 많이 벗어난 폭(구간 폭 기준 배수).
            acc[c]["oor"].append(
                float(((fed < gmin) | (fed > gmax)).mean()))
            over = max(0.0, float(fed.max()) - gmax)
            under = max(0.0, gmin - float(fed.min()))
            acc[c]["exc"].append(max(over, under) / grange
                                 if grange else float("nan"))

    bar = "=" * 78
    print("")
    print(bar)
    print("[1] 참 벡터 자체")
    print("-" * 78)
    print("  동점에 속한 칸 %.2f/%d (%.1f%%) · 서로 다른 값 %.2f 개"
          % (np.mean(tie_n), HI - LO, 100.0 * np.mean(tie_n) / (HI - LO),
             np.mean(tie_u)))
    print("")
    print("[2] 조건별 수치 교란 크기 (참 벡터 대비, %d 추출 평균)" % N_DRAW)
    print("-" * 78)
    print("%-18s %10s %10s %10s %10s" % ("조건", "평균|차|", "RMS",
                                         "rho", "그대로%"))
    for c in CONDS:
        a = acc[c]
        print("%-18s %10.5f %10.5f %10.4f %9.1f%%"
              % (c, np.mean(a["l1"]), np.mean(a["l2"]),
                 np.mean(a["rho"]), 100.0 * np.mean(a["same"])))
    print("")
    print("[3] 판정")
    print("-" * 78)
    r1 = np.mean(acc["risk_rankonly"]["l1"])
    f1 = np.mean(acc["risk_frozen"]["l1"])
    d1 = np.mean(acc["risk_dimshuffle"]["l1"])
    print("  rankonly 와 frozen 은 값 다중집합이 같다(둘 다 공여).")
    print("  단조 정렬 덕에 rankonly 의 평균|차| 가 frozen 의 %.1f%% 다."
          % (100.0 * r1 / f1 if f1 else float("nan")))
    print("  → 'rankonly 는 수치도 가깝다'는 반론은 이 비율만큼 성립한다.")
    print("  그러나 dimshuffle 은 참 값을 그대로 쓰면서 평균|차| %.5f 로"
          % d1)
    print("  frozen 의 %.1f%% 이고, 지연 피해는 frozen 의 97.6%% 다."
          % (100.0 * d1 / f1 if f1 else float("nan")))
    print("  수치 근접도로는 그 쌍이 설명되지 않는다 → 순위 해석이 남는다.")
    print(bar)

    print("")
    print("[4] 순위 보존 + 값 이동 조건의 지수 보정")
    print("-" * 78)
    print("  단조 거듭제곱 재사상 x -> x**p 는 순위를 정확히 보존한다.")
    print("  frozen 의 평균|차| %.5f 에 가장 가까운 p 를 골라 쓴다." % f1)
    print("  코사인·노름비도 같이 적는다 — frozen 의 코사인 0.7636 밑으로")
    print("  내려가면서 rho=1 을 지키는 p 가 있으면 '방향' 가설을 직접 깬다.")
    print("%8s %10s %9s %8s %8s %8s"
          % ("p", "평균|차|", "RMS", "코사인", "노름비", "rho"))
    rng2 = random.Random(RV.EVAL_SEED)
    draws = [pool[rng2.randrange(len(pool))] for _ in range(N_DRAW)]
    best, bestgap = None, None
    SWEEP = (0.05, 0.10, 0.15, 0.2, 0.25, 0.33, 0.5,
             2.0, 3.0, 4.0, 6.0, 10.0, 16.0, 25.0, 40.0)
    for p in SWEEP:
        l1s, l2s, rhos, css, rts = [], [], [], [], []
        for state in draws:
            true = np.asarray(state[LO:HI], dtype=np.float64)
            fed = powmap(true, p)
            d = fed - true
            l1s.append(float(np.abs(d).mean()))
            l2s.append(float(np.sqrt((d ** 2).mean())))
            rhos.append(RV.spearman_np(fed, true))
            tn = float(np.sqrt((true ** 2).sum()))
            fn = float(np.sqrt((fed ** 2).sum()))
            css.append(float(np.dot(fed, true) / (fn * tn))
                       if fn * tn else float("nan"))
            rts.append(fn / tn if tn else float("nan"))
        m1 = float(np.mean(l1s))
        print("%8.2f %10.5f %9.5f %8.4f %8.4f %8.4f"
              % (p, m1, float(np.mean(l2s)), float(np.mean(css)),
                 float(np.mean(rts)), float(np.mean(rhos))))
        gap = abs(m1 - f1)
        # rho 가 1.0000 미만인 p 는 후보에서 뺀다(부동소수 동점으로
        # 순위 보존이 깨진 것이라 조건의 정의가 성립하지 않는다).
        if float(np.mean(rhos)) < 0.99995:
            continue
        if bestgap is None or gap < bestgap:
            best, bestgap = p, gap
    print("  → L1 기준으로 고른 지수 p = %.2f (차 %.5f, rho=1 후보 중)"
          % (best, bestgap))
    print("  이 조건이 무해하면 '값은 순위에 있다'가 저쪽 끝에서도 선다.")
    print("  피해가 L1 에 비례하면 rankonly 의 무해는 교란이 작아서였다.")
    print("")
    print("  p>1 쪽을 왜 같이 보는가: 값을 아래로 눌러 **노름을 줄이므로**")
    print("  노름 교란 변수의 방향이 p<1 과 **반대**다. 두 쪽이 다 무해하면")
    print("  노름 증가도 감소도 피해를 안 낸다는 양면 증거가 된다.")
    print("  그리고 코사인이 frozen(0.7636) 밑으로 내려가는 p 가 있으면,")
    print("  rho=1 · 코사인<frozen 조건이 생겨 '방향' 가설을 직접 깬다.")
    print("  주의: rho 가 1.0000 에서 떨어지는 p 는 쓰면 안 된다(부동소수")
    print("  동점이 생겨 순위 보존이 깨진 것이다) — 표의 rho 열로 확인한다.")
    print(bar)

    print("")
    print("[5] 노름 분해 — 23.9.4 의 교란 변수를 숫자로 가른다")
    print("-" * 78)
    print("  주의: 이 스크립트는 피해(지연)를 재지 않는다. 아래는 교란의")
    print("  기하학뿐이고 피해는 measure_rank_vs_value.py 쪽 숫자다.")
    print("  제곱몫 = 추출별 (블록 제곱노름 / 상태 전체 제곱노름) 의 평균")
    print("%-18s %9s %8s %8s %8s %8s" % ("조건", "블록노름", "참대비",
                                         "제곱몫", "코사인", "rho"))
    print("%-18s %9.5f %8.4f %7.2f%% %8.4f %8.4f"
          % ("(참) intact", float(np.mean(tru["bn"])), 1.0,
             100.0 * float(np.mean(tru["sh"])), 1.0, 1.0))
    for c in CONDS:
        a = acc[c]
        print("%-18s %9.5f %8.4f %7.2f%% %8.4f %8.4f"
              % (c, float(np.mean(a["bn"])), float(np.mean(a["rt"])),
                 100.0 * float(np.mean(a["sh"])), float(np.mean(a["cos"])),
                 float(np.mean(a["rho"]))))
    print("")
    print("")
    print("  23.2절 (가)의 57.07% 와 맞춰본다 — 추정량이 다르다:")
    bm = float(np.mean(tru["bn"]))
    wm = float(np.mean(tru["wn"]))
    print("   . 그쪽 방식(평균 L2 를 먼저 내고 제곱): 블록 %.4f / 전체 %.4f"
          % (bm, wm))
    print("     -> %.2f%%   (보고값 57.07%%, 차 %+.2f%%p)"
          % (100.0 * bm ** 2 / wm ** 2, 100.0 * bm ** 2 / wm ** 2 - 57.07))
    print("   . 이쪽 방식(추출별 비율의 평균)        -> %.2f%%"
          % (100.0 * float(np.mean(tru["sh"]))))
    print("     블록 평균 L2 %.4f 는 23.2절 2.8947 과 %+.4f 차이다."
          % (bm, bm - 2.8947))
    print("     두 방식 차는 Jensen 이고 배치 차이가 아니다(블록 L2 가 맞는다).")
    print("")
    print("  읽는 법 — 두 조건이 교란을 **교차로** 가른다:")
    ds = acc["risk_dimshuffle"]
    print("   . dimshuffle : 노름 참대비 %.4f (값 다중집합 보존) · rho %.4f"
          % (float(np.mean(ds["rt"])), float(np.mean(ds["rho"]))))
    for c in ("risk_rankpow50", "risk_rankpow33"):
        a = acc[c]
        print("   . %-11s: 노름 참대비 %.4f (+%.1f%%) · rho %.4f"
              % (c.replace("risk_", ""), float(np.mean(a["rt"])),
                 100.0 * (float(np.mean(a["rt"])) - 1.0),
                 float(np.mean(a["rho"]))))
    print("  dimshuffle 은 노름을 안 건드리고 순위만 깨는데 피해를 낸다(기측정).")
    print("  그래서 rankpow 가 무해하면 '노름 변화'는 피해를 설명할 수 없다")
    print("  -> 23.9.4 의 교란 변수가 측정으로 빠진다. 반대로 rankpow 가")
    print("  피해를 내면 노름과 순위가 섞여 그 조건만으로는 못 가른다.")
    print("  코사인은 노름을 나눈 **방향**이라 그 한계를 메운다:")
    for c in CONDS:
        print("   . %-14s 코사인 %.4f · rho %8.4f"
              % (c.replace("risk_", ""), float(np.mean(acc[c]["cos"])),
                 float(np.mean(acc[c]["rho"]))))
    print("")
    print("  설계 요지 — 네 축을 따로 흔든다(모두 rho=1 유지):")
    print("   . pow50 대 pow400 : 코사인 같고(0.9665) 노름비 반대")
    print("                       (1.1721 대 0.7865) -> 노름 가설 양면 배제")
    print("   . pow05           : 코사인 0.7664 = frozen 0.7636 자리인데")
    print("                       rho=1 -> 방향 가설 직접 반박")
    print("   . dimshuffle      : 노름·주변분포 보존, 순위만 파괴")
    print("  교란 요약 넷(L1 · 노름비 · 코사인 · rho) 중 피해 순서와 맞는")
    print("  것이 무엇인지 작업 끝나고 대조한다 — L1 과 노름은 이미 틀렸다.")
    print(bar)

    print("")
    print("[6] 배터리 민감도 — 대조쌍 설계의 전제를 확인한다")
    print("-" * 78)
    print("  위 숫자는 window5_run01 의 greedy 배터리 250상태에서 나왔다.")
    print("  그런데 피해는 롤아웃 중 **실제 방문 상태**에서 생기고, 22절은")
    print("  배터리를 random 으로 바꾸면 모델 74개가 전부 구분된다고 쟀다.")
    print("  대조쌍의 '코사인이 같다'가 배터리 산물이면 설계가 무너진다.")
    print("")
    print("%-26s %8s %8s %8s %8s"
          % ("배터리", "pow50", "pow400", "차", "pow05-frozen"))
    sens = []
    for lab, model, kind in BATTS:
        try:
            mp = os.path.join(OC.ORIG_MODELS, model)
            if not os.path.exists(mp):
                mp = os.path.join(ABL_MODELS, model)
            n2 = OC.load_net(mp, device)
            b2 = OC.build_battery(kind, device, n2)
        except Exception as exc:                     # noqa: BLE001
            print("%-26s  건너뜀 (%s)" % (lab, type(exc).__name__))
            continue
        pool2 = [st for st, _ in b2]
        r2 = random.Random(RV.EVAL_SEED)
        got = {c: [] for c in ("risk_rankpow50", "risk_rankpow400",
                               "risk_rankpow05", "risk_frozen")}
        for _ in range(400):
            st = pool2[r2.randrange(len(pool2))]
            dn = pool2[r2.randrange(len(pool2))]
            tv = np.asarray(st[LO:HI], dtype=np.float64)
            tn = float(np.sqrt((tv ** 2).sum()))
            pm = np.random.RandomState(r2.randrange(1 << 30)).permutation(
                HI - LO)
            for c in got:
                fv = np.asarray(RV.make_fed(st, c, dn, pm)[LO:HI],
                                dtype=np.float64)
                fn = float(np.sqrt((fv ** 2).sum()))
                got[c].append(float(np.dot(fv, tv) / (fn * tn))
                              if fn * tn else float("nan"))
        c50 = float(np.mean(got["risk_rankpow50"]))
        c400 = float(np.mean(got["risk_rankpow400"]))
        c05 = float(np.mean(got["risk_rankpow05"]))
        cfz = float(np.mean(got["risk_frozen"]))
        print("%-26s %8.4f %8.4f %8.4f %8.4f"
              % (lab, c50, c400, abs(c50 - c400), abs(c05 - cfz)))
        sens.append((lab, c50, c400, c05, cfz))
    if sens:
        g1 = max(abs(a - b) for _, a, b, _, _ in sens)
        g2 = max(abs(a - b) for _, _, _, a, b in sens)
        print("")
        print("  배터리 전체에서 쌍 (가) 코사인 차 최대 %.4f, "
              "쌍 (나) 최대 %.4f" % (g1, g2))
        print("  읽는 법: 두 값이 작게 유지되면 '한 축만 다르다'가 배터리와")
        print("  무관하다. 커지면 그 배터리에서는 대조쌍 판정을 쓸 수 없다.")
        print("  한계: 이것도 고정 배터리다. 롤아웃의 방문 분포는 조건마다")
        print("  달라지므로(경로가 바뀐다) 실제 교란은 여기서 직접 안 잰다.")
    print(bar)

    print("")
    print(bar)
    print("[7] 범위 밖 정도 — scale* 한계를 말이 아니라 크기로 적는다")
    print("-" * 78)
    print("  참 블록 관측 구간 [%.4f, %.4f] (폭 %.4f). 순수 배율은 값을"
          % (gmin, gmax, grange))
    print("  이 구간 밖으로 내보내므로 scale* 가 피해를 내면 '노름' 과")
    print("  '범위 밖 입력' 이 섞인다. pow* 는 min-max 로 되돌리므로")
    print("  구간 안에 있어야 한다(0.0% 가 나오면 설계대로다).")
    print("")
    print("%-18s %12s %14s" % ("조건", "범위밖 칸%", "최대 초과(폭배수)"))
    for c in CONDS:
        print("%-18s %11.2f%% %14.4f"
              % (c, 100.0 * np.mean(acc[c]["oor"]),
                 np.mean(acc[c]["exc"])))
    print("")
    print("  읽는 법: 범위밖 0.00% 이면 그 조건의 피해는 범위 밖 입력으로")
    print("  설명되지 않는다. scale184 가 크게 벗어나면 그 조건은 보조")
    print("  근거로만 쓰고, 주 조건은 rankpow05n(배율이 1 아래라 벗어남이")
    print("  작은 쪽)으로 읽어야 한다.")
    print(bar)

    print("")
    print(bar)
    print("[8] 블록 안 평탄화 — 코사인이 말하지 못하는 축")
    print("-" * 78)
    print("  표준편차비 = std(교란 블록)/std(참 블록). 1 보다 훨씬 작으면")
    print("  칸들 사이 차이가 **지워졌다**는 뜻이다. 코사인은 노름을 빼고")
    print("  방향만 보므로 '평탄해졌다' 를 직접 말하지 못한다.")
    print("  대조군: risk_mean(정확히 상수)은 표준편차비 0 이고, 경로")
    print("  다양성 측정에서 10/10 모델을 경로 1개로 붕괴시켰다.")
    print("")
    print("%-18s %9s %8s %8s %8s"
          % ("조건", "변동계수비", "표준편차비", "코사인", "노름비"))
    for c in CONDS:
        print("%-18s %9.4f %8.4f %8.4f %8.4f"
              % (c, np.mean(acc[c]["cvr"]), np.mean(acc[c]["sdr"]),
                 np.mean(acc[c]["cos"]), np.mean(acc[c]["rt"])))
    print("")
    print("  읽는 법: rankpow05/05n 이 0 에 가까우면 그 조건의 피해는")
    print("  '순위를 지킨 채 간격을 지운' risk_mean 쪽 메커니즘이고,")
    print("  frozen/dimshuffle(표준편차비 ~1)과는 **다른 경로**로 같은")
    print("  피해를 낸 것이다. 그러면 단일 축 설명은 애초에 없다.")
    print(bar)

    path = os.path.join(OC.OUT_DIR, "rank_value_geometry.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "mean_abs_diff", "rms_diff", "mean_rho",
                    "frac_unchanged", "block_norm", "norm_ratio_vs_true",
                    "block_sqnorm_share", "cosine", "n_draw",
                    "frac_out_of_range", "max_excursion_x_range",
                    "block_std_ratio", "block_cv_ratio"])
        w.writerow(["_estimator_check", "", "", "", "",
                    "%.6f" % np.mean(tru["wn"]),
                    "%.6f" % (np.mean(tru["bn"]) ** 2
                              / np.mean(tru["wn"]) ** 2),
                    "ratio_of_means_vs_reported_57.07", "", N_DRAW,
                    "", "", "", ""])
        w.writerow(["true_intact", "0.000000", "0.000000", "1.0000",
                    "1.0000", "%.6f" % np.mean(tru["bn"]), "1.0000",
                    "%.6f" % np.mean(tru["sh"]), "1.0000", N_DRAW,
                    "0.0000", "0.0000", "1.0000", "1.0000"])
        for c in CONDS:
            a = acc[c]
            w.writerow([c, "%.6f" % np.mean(a["l1"]),
                        "%.6f" % np.mean(a["l2"]),
                        "%.4f" % np.mean(a["rho"]),
                        "%.4f" % np.mean(a["same"]),
                        "%.6f" % np.mean(a["bn"]),
                        "%.4f" % np.mean(a["rt"]),
                        "%.6f" % np.mean(a["sh"]),
                        "%.6f" % np.mean(a["cos"]), N_DRAW,
                        "%.4f" % np.mean(a["oor"]),
                        "%.4f" % np.mean(a["exc"]),
                        "%.4f" % np.mean(a["sdr"]),
                        "%.4f" % np.mean(a["cvr"])])
        w.writerow(["true_vector_ties", "%.4f" % np.mean(tie_n), "",
                    "", "%.4f" % np.mean(tie_u), "", "", "", "",
                    N_DRAW, "", "", "", ""])
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
