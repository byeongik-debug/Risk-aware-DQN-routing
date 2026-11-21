# -*- coding: utf-8 -*-
"""런간 분산이 '연속 잡음'인가 '정책 분지 추첨'인가 — 재학습 없이 기존 값만 다시 묶는다.

왜 필요한가
-----------
14절은 "Holm 후 0/21, 전부 MDE 미달"로 끝났고 그 근거는 **런간 분산이 크다**였다.
23.10절에서 모델 중복을 재면서 다른 그림이 보였다: `factor_full` 다섯 런의 지연이
350.83 / 355.02 / 351.19 / 351.79 / **386.98** 이다. 넷은 **4.2 ms 안**에 모여
있고 하나가 **36 ms** 떨어져 있다. 그 하나가 sd ±15.64 를 거의 혼자 만든다
(r05 빼면 sd 1.90). 그리고 22.3절의 greedy 정책 클래스가 r01~r04 를 한 부류,
r05 를 다른 부류로 이미 갈라놨다 — **두 측정이 같은 분할을 가리킨다.**

그렇다면 "런간 분산"은 연속 잡음이 아니라 **학습이 어느 정책 분지에 떨어지는가**의
추첨이고, n=5 로 재는 것은 평균이 아니라 **분지 빈도**다. 이건 14절의 결론을
뒤집지 않지만 **이유를 바꾼다** — "잡음이 커서 못 가른다"가 아니라 "팔들이
같은 정책 몇 개를 다른 비율로 뽑는다"가 된다.

측정할 것
---------
(1) 팔마다: 다섯 런의 지연 · sd · greedy 클래스 소속 · 소수 클래스 런을 뺀 sd.
    지연의 이상치 런이 **클래스의 이상치 런과 같은가**를 본다.
(2) 전체 풀: 지연 분산 중 **클래스 소속이 설명하는 몫**(eta^2). 크면 "런 잡음"의
    상당 부분이 실은 분지 식별이다.

주의 — 이 측정의 한계를 먼저 적는다
-----------------------------------
클래스는 `window5_run01` 의 그리디 경로 250상태에서 행동이 같은지로 묶은 것이다
(22.2절). 기준 궤적 한 개에 의존하므로 **기준 정책에 가까운 팔이 유리하다**.
그리고 인자 절제 팔들은 배포 환경에서 위험 블록을 **다르게 계산**하므로, 공통
배치에서 같은 행동을 한다는 것이 배포 성능이 같다는 뜻은 아니다. 그래서 (2)의
eta^2 는 **상관**이고 인과가 아니다. 재지 않은 것을 단정하지 않기 위해 적어둔다.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import review_stats as RS                            # noqa: E402

ORIG_RESULTS = os.path.abspath(
    os.path.join(HERE, "..", "SDN_DQN_new_env", "results"))
OUT_DIR = os.path.join(HERE, "review_results")

# (팔 이름, 결과파일 접두, 결과 디렉터리). dynamic 시나리오로 통일한다.
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


def load_classes():
    """policy_classes.csv greedy 행 -> {(팔, 런): 클래스}."""
    out = {}
    path = os.path.join(OUT_DIR, "policy_classes.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["battery"] == "greedy":
                out[(row["arm"], int(row["run_id"]))] = int(row["class_id"])
    return out


def load_runs(prefix, folder):
    """런별 (run_id, delay, loss, hop). 없으면 건너뛴다."""
    out = []
    for i in range(1, 6):
        path = os.path.join(folder, "%s_run%02d_result.csv" % (prefix, i))
        if not os.path.exists(path):
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.append((int(row["run_id"]), float(row["delay"]),
                            float(row["loss"]), float(row["hop"])))
    return sorted(out)


def eta_squared(groups):
    """{그룹: [값]} -> (eta^2, 그룹수, n). 집단간 제곱합 / 전체 제곱합."""
    vals = [v for g in groups.values() for v in g]
    n = len(vals)
    gm = sum(vals) / n
    ss_tot = sum((v - gm) ** 2 for v in vals)
    ss_bet = sum(len(g) * (sum(g) / len(g) - gm) ** 2 for g in groups.values())
    return (ss_bet / ss_tot if ss_tot else 0.0), len(groups), n


def main():
    cls = load_classes()
    rows = []
    pool, hpool, agree = {}, {}, []

    print("[0] 재는 것")
    print("  (1) 팔마다 다섯 런의 지연 이상치가 greedy 클래스 이상치와 같은가")
    print("  (2) 전체 풀에서 지연 분산 중 클래스가 설명하는 몫(eta^2)")
    print("  한계: 클래스는 window5_run01 그리디 궤적 250상태 기준이다(22.2절).")
    print("        따라서 (2)는 상관이고 인과가 아니다.")
    print("")
    print("=" * 96)
    print("[1] 팔별 — 런 지연과 정책 클래스")
    print("%-18s %-30s %7s %7s | %-14s %7s"
          % ("팔", "런별 지연(run1..5)", "sd", "폭", "클래스", "다수sd"))

    for arm, prefix, folder in ARMS:
        runs = load_runs(prefix, folder)
        if len(runs) < 3:
            print("%-18s (런 결과 파일 없음: %s)" % (arm, prefix))
            continue
        delays = [d for _r, d, _l, _h in runs]
        losses = [ll for _r, _d, ll, _h in runs]
        hops = [h for _r, _d, _l, h in runs]
        ids = [r for r, _d, _l, _h in runs]
        klass = [cls.get((arm, r)) for r in ids]
        _m, sd = RS.mean_std(delays)
        span = max(delays) - min(delays)

        # 다수 클래스만 남긴 sd
        counts = {}
        for k in klass:
            counts[k] = counts.get(k, 0) + 1
        major = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        keep = [d for d, k in zip(delays, klass) if k == major]
        sd_major = RS.mean_std(keep)[1] if len(keep) >= 2 else float("nan")

        for r, d, h, k in zip(ids, delays, hops, klass):
            pool.setdefault(k, []).append(d)
            hpool.setdefault(k, []).append(h)
            rows.append([arm, r, d, h, k, sd, span, sd_major, len(keep)])

        print("%-18s %-30s %7.2f %7.2f | %-14s %7.2f %s"
              % (arm,
                 " ".join("%.1f" % d for d in delays),
                 sd, span,
                 ",".join(str(k) for k in klass),
                 sd_major, "n=%d" % len(keep)))

        # 지연 이상치 런이 홉 이상치 런과 같은가
        imax = delays.index(max(delays))
        hmax = hops.index(max(hops))
        lmin = losses.index(min(losses))
        dm = sum(delays) / len(delays)
        hm = sum(hops) / len(hops)
        mspz = dm / hm
        if delays[imax] != dm:
            hop_share = (hops[imax] - hm) * mspz / (delays[imax] - dm)
        else:
            hop_share = float("nan")
        agree.append((arm, ids[imax], ids[hmax], imax == hmax,
                      hops[imax], hm, ids[lmin], imax == lmin,
                      losses[imax], sum(losses) / len(losses), hop_share))

    print("")
    print("[2] 가장 느린 런이 (가) 가장 긴 런인가 (나) 손실이 가장 낮은 런인가")
    print("  (가)가 11/11 이면 그 자체로는 거의 동어반복이다 — 지연은 홉에 걸쳐")
    print("  더해지는 양이므로. 그래서 (다) 홉 몫으로 지연 초과가 얼마나")
    print("  설명되는지, 그리고 (나) 손실이 어디로 가는지를 같이 본다.")
    print("")
    print("%-18s %-14s %-16s %9s %9s %8s"
          % ("팔", "지연최대=홉최대", "지연최대=손실최소", "그 런 손실",
             "팔 평균", "홉 몫"))
    nag = sum(1 for a in agree if a[3])
    nlo = sum(1 for a in agree if a[7])
    for (arm, _ri, _rh, ok, _hi, _hm, _rl, oklo,
         li, lmean, share) in agree:
        print("%-18s %-14s %-16s %9.5f %9.5f %7.0f%%"
              % (arm, "같다" if ok else "다르다",
                 "같다" if oklo else "다르다", li, lmean, share * 100.0))
    print("")
    print("  (가) 지연최대 = 홉최대: %d/%d 팔" % (nag, len(agree)))
    print("  (나) 지연최대 = 손실최소: %d/%d 팔" % (nlo, len(agree)))
    print("  (다) 홉 몫 = (그 런 홉 - 팔평균홉) x (팔평균 ms/홉) / (지연 초과).")
    print("      100%% 를 넘으면 그 런은 길지만 홉당 품질은 오히려 좋았다는 뜻이다.")

    print("")
    print("=" * 96)
    print("[3] 전체 풀 — 클래스가 설명하는 분산 몫")
    e_d, g_d, n_d = eta_squared(pool)
    e_h, g_h, n_h = eta_squared(hpool)
    print("  지연: eta^2 = %.4f (클래스 %d개 · 모델 %d개)" % (e_d, g_d, n_d))
    print("  홉  : eta^2 = %.4f (클래스 %d개 · 모델 %d개)" % (e_h, g_h, n_h))
    multi = dict((k, v) for k, v in pool.items() if len(v) >= 2)
    if multi:
        e2, g2, n2 = eta_squared(multi)
        print("  단독 클래스를 뺀 지연: eta^2 = %.4f (클래스 %d · 모델 %d)"
              % (e2, g2, n2))
        print("  (단독 클래스는 자기 혼자로 완벽히 설명되므로 eta^2 를"
              " 위로 부풀린다. 이 줄이 정직한 쪽이다.)")

    print("")
    print("[4] 판정")
    helped = [r for r in rows if r[8] >= 2]
    arms_seen = {}
    for r in rows:
        arms_seen[r[0]] = (r[5], r[7], r[8])
    better = [a for a, (sd, sdm, nk) in arms_seen.items()
              if nk >= 2 and sdm < sd]
    worse = [a for a, (sd, sdm, nk) in arms_seen.items()
             if nk >= 2 and sdm >= sd]
    print("  다수 클래스만 남기면 sd 가 **작아지는** 팔: %d개 — %s"
          % (len(better), ", ".join(sorted(better))))
    print("  오히려 **커지는** 팔: %d개 — %s"
          % (len(worse), ", ".join(sorted(worse))))
    print("  계산 불가(다섯 런이 전부 단독 클래스): %s"
          % ", ".join(sorted(a for a, (_s, _m, nk) in arms_seen.items()
                             if nk < 2)))
    print("")
    print("  읽는 법: '런간 분산 = 분지 추첨'이 맞으면 다수 클래스만 남긴 sd 가")
    print("  크게 줄어야 한다. 줄지 않는 팔이 있으면 그 팔에서는 클래스가")
    print("  성능 차이를 못 잡는 것이고, 그때는 분지 설명을 그 팔에 쓰면 안 된다.")
    del helped

    out = os.path.join(OUT_DIR, "run_variance_basins.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "run_id", "delay", "hop", "greedy_class",
                    "arm_sd", "arm_span", "major_class_sd", "major_class_n"])
        for r in rows:
            w.writerow(r)
    print("")
    print("저장: %s" % out)
    return rows, pool


if __name__ == "__main__":
    main()
