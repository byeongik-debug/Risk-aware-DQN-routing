# -*- coding: utf-8 -*-
"""25절 보강 — 정책 '클래스' 대신 **행동 일치율**로 런 간 거리를 잰다.

왜 필요한가
-----------
22.3절의 정책 클래스는 배치 250상태에서 행동 벡터가 **완전히 같은지**로
묶는다. 그래서 250칸 중 한 칸만 달라도 250칸 전부 다른 것과 같게 취급되고,
`factor_util`·`weight_temporal` 은 다섯 런이 전부 단독 클래스가 되어
25절의 다수-클래스 열이 **비어 버린다**. 그런데 `factor_util` 의 지연은
눈으로도 {377,371,375} 대 {440,437} 로 갈린다 — 도구가 못 잡는 것이다.

무엇을 재는가
-------------
(1) 같은 팔 안 모델 쌍의 **행동 일치율**(250상태 중 같은 행동 비율).
(2) 같은 쌍의 **지연 차이** |delay_i - delay_j|.
(3) 둘의 상관 — 일치율이 높을수록 지연이 비슷한가. 팔 안에서, 그리고
    팔 평균을 뺀 뒤 전체에서.
(4) `factor_util` 처럼 클래스가 못 가른 팔에서 일치율이 지연 분할과
    같은 쪽으로 갈리는지.

한계 (미리 적는다)
------------------
일치율은 **상관**이다. 배치는 `window5_run01` 의 그리디 궤적이라 기준
정책 근처에 편향돼 있고, 배포 시 각 팔은 위험 블록을 다르게 계산한다.
그래서 "일치율이 낮아서 느리다"는 인과 진술로 쓰면 안 된다.

재학습·롤아웃 없음. 원본은 읽기만.
출력: review_results/policy_distance.csv
"""
import csv
import os

import cluster_policy_classes as CP             # noqa: E402

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

ORIG_RESULTS = os.path.abspath(
    os.path.join(CP.HERE, "..", "SDN_DQN_new_env", "results"))
OUT_DIR = CP.OUT_DIR
# (팔, 결과 CSV 접두, 결과 디렉터리) — 25절과 같은 11개 팔
ARMS = (
    ("factor_cong", "factor_cong_dynamic", ORIG_RESULTS),
    ("factor_delay_inst", "factor_delay_inst_dynamic", ORIG_RESULTS),
    ("factor_link_type", "factor_link_type_dynamic", ORIG_RESULTS),
    ("factor_loss_inst", "factor_loss_inst_dynamic", ORIG_RESULTS),
    ("factor_util", "factor_util_dynamic", ORIG_RESULTS),
    ("weight_equal", "weight_equal_dynamic", ORIG_RESULTS),
    ("weight_temporal", "weight_temporal_dynamic", ORIG_RESULTS),
    ("window1", "window1_dynamic", ORIG_RESULTS),
    ("window5", "window5_dynamic", ORIG_RESULTS),
    ("factor_full", "factor_full_dynamic", OUT_DIR),
    ("factor_temporal", "factor_temporal_dynamic", OUT_DIR),
)
RUNS = (1, 2, 3, 4, 5)


def read_metric(d, prefix, run, key="avg_delay"):
    """런별 결과 CSV 에서 한 지표를 읽는다. 없으면 None."""
    path = os.path.join(d, "%s_run%02d_result.csv" % (prefix, run))
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None
    for k in (key, "delay", "Delay", "avg_delay"):
        if k in rows[0]:
            return float(rows[0][k])
    return None


def pearson(xs, ys):
    """표본 상관. n<3 이면 nan."""
    n = len(xs)
    if n < 3:
        return float("nan")
    a = np.asarray(xs, dtype=np.float64)
    b = np.asarray(ys, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def main():
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    ref = CP.load_net(CP.REFERENCE, device)
    battery = CP.build_battery("greedy", device, ref_net=ref)
    print("배치 상태 %d (window5_run01 그리디) · 팔 %d · 런 %d"
          % (len(battery), len(ARMS), len(RUNS)))

    inv = {(a, r): p for a, r, p in CP.model_inventory()}
    acts, delays = {}, {}
    for arm, prefix, d in ARMS:
        for r in RUNS:
            key = (arm, r)
            if key not in inv:
                continue
            net = CP.load_net(inv[key], device)
            acts[key] = np.asarray(CP.action_vector(net, battery, device))
            dl = read_metric(d, prefix, r)
            if dl is not None:
                delays[key] = dl
    print("모델 %d · 지연 읽힌 모델 %d" % (len(acts), len(delays)))

    rows, pooled_a, pooled_d = [], [], []
    bar = "=" * 78
    print("")
    print(bar)
    print("[1] 팔별 — 쌍 행동일치율 대 쌍 지연차")
    print("-" * 78)
    print("%-19s %6s %9s %9s %9s %8s"
          % ("팔", "쌍수", "일치 최소", "일치 최대", "지연차 폭", "상관 r"))
    for arm, _, _ in ARMS:
        ags, dds = [], []
        for i in range(len(RUNS)):
            for j in range(i + 1, len(RUNS)):
                ki, kj = (arm, RUNS[i]), (arm, RUNS[j])
                if ki not in acts or kj not in acts:
                    continue
                if ki not in delays or kj not in delays:
                    continue
                ag = float((acts[ki] == acts[kj]).mean())
                dd = abs(delays[ki] - delays[kj])
                ags.append(ag)
                dds.append(dd)
                rows.append([arm, RUNS[i], RUNS[j], "%.4f" % ag,
                             "%.4f" % dd])
        if not ags:
            continue
        r = pearson(ags, dds)
        print("%-19s %6d %9.3f %9.3f %9.2f %8.4f"
              % (arm, len(ags), min(ags), max(ags), max(dds), r))
        ma, md = float(np.mean(ags)), float(np.mean(dds))
        pooled_a += [a - ma for a in ags]
        pooled_d += [x - md for x in dds]

    print("")
    print("[2] 팔 평균을 뺀 전체 (쌍 %d)" % len(pooled_a))
    print("-" * 78)
    rp = pearson(pooled_a, pooled_d)
    print("  일치율 대 지연차 상관 r = %.4f" % rp)
    print("  (음수여야 한다: 행동이 비슷하면 지연도 비슷하다)")

    print("")
    print("[3] 클래스가 못 가른 팔 — 일치율은 가르는가")
    print("-" * 78)
    for arm in ("factor_util", "weight_temporal"):
        print("  %s  지연: %s" % (arm, " ".join(
            "%.1f" % delays[(arm, r)] for r in RUNS if (arm, r) in delays)))
        print("      %s" % "  ".join(
            "r%d" % r for r in RUNS if (arm, r) in acts))
        for i in RUNS:
            if (arm, i) not in acts:
                continue
            cells = []
            for j in RUNS:
                if (arm, j) not in acts:
                    continue
                if i == j:
                    cells.append("  -  ")
                else:
                    cells.append("%.3f" % float(
                        (acts[(arm, i)] == acts[(arm, j)]).mean()))
            print("  r%d  %s" % (i, "  ".join(cells)))
        print("")
    print("  읽는 법: 지연이 비슷한 런끼리 일치율이 높게 뭉치면 분지 설명이")
    print("  이 팔에도 살아난다. 일치율이 고르게 낮으면 이 배치로는 이 팔의")
    print("  런 차이를 못 본다는 뜻이고, 25절의 '계산 불가'가 유지된다.")
    print(bar)

    print("")
    print("[4] 상관 대신 — 두 도구가 같은 런을 이상치로 지목하는가")
    print("-" * 78)
    print("  주의: [1]의 r 이 -0.99 인 팔은 쌍 10개가 두 덩이로 갈려서다")
    print("  (한 런이 두 지표 모두에서 이상치면 r 은 거의 자동으로 붙는다).")
    print("  그래서 더 약한 형태로 센다: 평균 일치율이 가장 낮은 런이")
    print("  지연이 가장 큰 런과 같은가.")
    print("%-19s %10s %10s %7s" % ("팔", "최저일치 런", "최대지연 런", "같나"))
    hit = tot = 0
    for arm, _, _ in ARMS:
        ks = [(arm, r) for r in RUNS if (arm, r) in acts and (arm, r) in delays]
        if len(ks) < 3:
            continue
        mean_ag = {}
        for ki in ks:
            vs = [float((acts[ki] == acts[kj]).mean())
                  for kj in ks if kj != ki]
            mean_ag[ki] = float(np.mean(vs))
        lo = min(ks, key=lambda k: mean_ag[k])
        hi = max(ks, key=lambda k: delays[k])
        same = (lo == hi)
        hit += 1 if same else 0
        tot += 1
        print("%-19s %10s %10s %7s"
              % (arm, "r%d (%.3f)" % (lo[1], mean_ag[lo]),
                 "r%d (%.1f)" % (hi[1], delays[hi]),
                 "O" if same else "X"))
    print("  일치: %d/%d" % (hit, tot))
    print("")
    print("[5] factor_util 반례를 명시한다")
    print("-" * 78)
    fu = [(r, delays[("factor_util", r)]) for r in RUNS
          if ("factor_util", r) in delays]
    slow = sorted(fu, key=lambda t: -t[1])[:2]
    if len(slow) == 2:
        a1 = float((acts[("factor_util", slow[0][0])]
                    == acts[("factor_util", slow[1][0])]).mean())
        allp = []
        for i in range(len(RUNS)):
            for j in range(i + 1, len(RUNS)):
                ki, kj = ("factor_util", RUNS[i]), ("factor_util", RUNS[j])
                if ki in acts and kj in acts:
                    allp.append(float((acts[ki] == acts[kj]).mean()))
        print("  느린 두 런 r%d(%.1f)·r%d(%.1f) 의 일치율 %.3f"
              % (slow[0][0], slow[0][1], slow[1][0], slow[1][1], a1))
        print("  이 팔 쌍 일치율 범위 %.3f~%.3f · 중위 %.3f"
              % (min(allp), max(allp), float(np.median(allp))))
        if a1 <= min(allp) + 1e-9:
            print("  → 느린 두 런이 이 팔에서 **가장 다른 쌍**이다.")
            print("     분지 설명은 이 팔에서 틀렸다. 25.5절에 그대로 적을 것.")

    path = os.path.join(OUT_DIR, "policy_distance.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "run_i", "run_j", "action_agreement",
                    "abs_delay_diff"])
        w.writerows(rows)
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
