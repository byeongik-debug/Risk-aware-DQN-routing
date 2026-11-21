# -*- coding: utf-8 -*-
"""표 10 의 유의성을 **내 하네스 안에서만** 닫는다.

왜 새 파일인가
--------------
`analyze_table10_pairwise.py` 는 원본
`SDN_DQN_new_env/results/factor_*_result.csv`
(각 학습 스크립트 자신의 평가기가 쓴 것)를 읽는다. 그 자체로는 정당한 분석이다.
문제는 `table10_fill.csv` 가 **평균은 내 롤아웃**(`table10_risky.csv`)에서,
**p 는 저 파일**에서 가져다 한 행에 올렸다는 것이다. 실제로 Full 대 -Util 지연차가
원본 하네스에서 -40.89, 내 롤아웃에서 -44.34 로 다르다 - 표시된 Δ 와 p 가
서로 다른 자료의 것이었다. 이 파일은 그 혼합을 없애려고 **한 하네스만** 읽는다.

세 가지를 같이 적는다
---------------------
1) **Welch** - (평균, 표준편차, n) 만의 함수다. 원자료에서 계산한 값과 요약통계
   에서 계산한 값은 근사가 아니라 정확히 같다(`_selfcheck()` 가 확인한다).
2) **정확 순열검정** - 모델별 값이 있을 때. K=3 팔에서 독립 학습 10개가 고작
   세 가지 지연값만 냈다(354.294 x3, 388.202 x5, 389.693 x1). 결과가 '경로 집합
   몇 개 사이의 추첨'이면 n=5 에서 정규성을 쓰는 Welch 는 약하다. 순열검정은
   분포 가정이 없고 C(10,5)=252 분할이라 전수로 돈다.
3) **도달 가능한 최소 p** - n=5 대 n=5 의 양측 정확 p 최솟값은 2/252=0.0079 다.
   족 30 에 Holm 을 걸면 0.0079x30=0.238 - **효과가 아무리 커도 유의가 나올 수
   없다**. 표 10 이 '유의한 칸 0개'인 것의 일부는 효과가 없어서가 아니라 설계가
   그만큼의 검정력을 갖고 있지 않아서다. 족마다 이 값을 찍는다.

짝 검정은 금지다: `run_id` 는 시드(42+run_id)만 공유하고, 8000 에피소드 학습을
거치는 동안 RNG 스트림이 갈라지므로 평가 조건을 짝지어 주지 않는다. 실측 근거:
같은 시드·같은 설정으로 학습한 factor_full 과 window5 의 가중치 max|diff| 가
0.82~1.47 이다 - 시드는 재현을 보장하지 않는다.

족을 하나로 안 섞는 이유
------------------------
학습 스크립트가 다르면 같은 족에 넣을 때 또 다른 혼합이 된다. 다섯으로 가른다.
  A 절제        기준 Full      train_factor_ablation.py (Full·-VD,VL 은 내 주입 드라이버)
  B K 스윕      기준 K=5       train_window_dqn.py
  C 가중치 내부 기준 W-Equal   train_weight_compare.py
  D 재현 바닥   Full 대 K=5    같은 설정·같은 시드 규약의 독립 재학습 두 묶음
                               (단 학습 스레드 수가 다르다 - 아래 단서)
  E 교차 참고   Full 대 W-*    스크립트 경계를 넘는다 -> D 가 허용할 때만 읽는다

D 가 이 파일의 핵심이다. window5 는 ZEROED_FACTORS=[] · risk_window=5 ·
DEFAULT_WEIGHTS 로 factor_full 과 설정이 **완전히 같다**(risk_calculator 의
기본값이 [] 이고 train_window_dqn.py 는 risk_calculator 를 건드리지 않는다.
학습 하이퍼파라미터·신경망 구조·타깃 계산·시나리오 샘플링·시드 규약도 전부
같다 - 세 학습 스크립트를 줄 단위로 맞춰 봤다). 그러므로 둘의 차이는 "아무
설정도 바꾸지 않았을 때 나오는 차이" = 재현 바닥이다. A 의 효과가 그 바닥보다
작으면 절제 효과라고 읽을 수 없다.

단서 (FINDINGS 26.9): 한 가지는 같지 않다 - **학습 시 torch 스레드 수**다.
`factor_full`/`factor_temporal` 은 내 스크립트가 1 로 고정해 학습했고,
`SDN_DQN_new_env/` 안에는 `set_num_threads` 를 부르는 스크립트가 하나도
없어 원본 모델군은 기계 기본값(이 기계 12)으로 돌았다. 스레드 수를 고정하면
학습은 비트 단위로 재현되지만(1:1 과 12:12 모두 max|diff| = 0) 수를 바꾸면
텐서 10개가 전부 갈라진다(1:12 에서 0.4247, 300 에피소드). 그러므로 족 D 가
재는 값은 '재학습 바닥' 이 아니라 **재학습 + 스레드 수**다. 족 A 도 같은
경계를 넘는다(기준 Full 은 고정1, 절제 팔 다섯은 비고정). 다만 이 혼입이
절제 패턴을 만들지는 못한다 - `-Cong`(비고정)이 기준에서 0.05 ms 거리이고
두 체제 모델 23개가 같은 결과값에 함께 들어간다(26.9.7).

Holm 은 족 안에서 매긴다. 족 크기를 바꾸면 유의성이 뒤집힌 전례가 두 번
있었으므로(FINDINGS 23.9.6.16 [5]) 전체를 한 족(pooled)으로 묶은 값도 같이
적어 둔다 - 인용할 때 **어느 족인지 반드시 병기**해야 한다.
"""
import csv
import io
import itertools
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import review_stats as RS                       # noqa: E402

RESULT_DIR = os.path.join(HERE, "review_results")
RISKY_CSV = os.path.join(RESULT_DIR, "table10_risky.csv")
EXTRA_CSV = os.path.join(RESULT_DIR, "table10_extra.csv")
FACTOR_RUNS = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
EXTRA_RUNS = os.path.join(RESULT_DIR, "table10_extra_runs.csv")
OUT_CSV = os.path.join(RESULT_DIR, "table10_welch.csv")

# (열 이름, 표시 이름, 서식)
METRICS = [
    ("delay", "지연(ms)", "%9.2f"),
    ("loss", "손실(홉당)", "%9.5f"),
    ("loss_total", "손실(종단간합)", "%9.5f"),
    ("hop", "홉", "%9.3f"),
    ("risky_pct", "Risky%", "%9.2f"),
]

FAMILIES = [
    ("A", "절제 (기준 = Full)", "Full",
     ["-Util", "-Cong", "-dDelay", "-dLoss", "-Type", "-VD,VL"]),
    ("B", "K 스윕 (기준 = K=5)", "K=5", ["K=3", "K=7", "K=10"]),
    ("C", "가중치 내부 (기준 = W-Equal)", "W-Equal", ["W-Temporal"]),
    ("D", "재현 바닥 (같은 설정, 독립 재학습)", "Full", ["K=5"]),
    ("E", "교차 스크립트 참고 (기준 = Full)", "Full",
     ["W-Equal", "W-Temporal"]),
]

PERM_CAP = 500000          # 분할 수가 이보다 크면 전수를 포기한다


def welch_summary(m1, s1, n1, m2, s2, n2):
    """(평균, 표준편차, n) 에서 Welch 를 정확히 계산한다.

    반환 (차이, t, Satterthwaite df, 양측 p). n<2 면 전부 nan 이다.
    """
    nan = float("nan")
    diff = m1 - m2
    if n1 < 2 or n2 < 2:
        return (diff, nan, nan, nan)
    a = (s1 * s1) / n1
    b = (s2 * s2) / n2
    se = math.sqrt(a + b)
    if se == 0.0:
        return (diff, nan, nan, nan)
    t = diff / se
    df = (a + b) ** 2 / (a * a / (n1 - 1) + b * b / (n2 - 1))
    return (diff, t, df, RS.t_two_sided_p(t, df))


def perm_exact(xs, ys):
    """두 독립 표본의 **정확** 순열검정 (통계량 = 평균차, 양측).

    반환 (p, 분할 수, 도달 가능한 최소 p). 분할이 PERM_CAP 을 넘으면
    (nan, 분할 수, 최소 p) 를 준다 - 표본으로 때우지 않고 못 했다고 적는다.
    """
    n1, n2 = len(xs), len(ys)
    nan = float("nan")
    if n1 < 1 or n2 < 1:
        return (nan, 0, nan)
    total = math.factorial(n1 + n2) // (math.factorial(n1)
                                        * math.factorial(n2))
    pmin = 2.0 / total if total > 1 else nan
    if n1 < 2 or n2 < 2:
        return (nan, total, pmin)
    if total > PERM_CAP:
        return (nan, total, pmin)
    pool = list(xs) + list(ys)
    obs = abs(sum(xs) / n1 - sum(ys) / n2)
    s_all = sum(pool)
    hit = 0
    for comb in itertools.combinations(range(n1 + n2), n1):
        sa = 0.0
        for i in comb:
            sa += pool[i]
        d = abs(sa / n1 - (s_all - sa) / n2)
        if d >= obs - 1e-12:
            hit += 1
    return (float(hit) / total, total, pmin)


def _f(row, key):
    v = row.get(key, "")
    if v is None or v == "":
        return float("nan")
    return float(v)


def load_summary():
    """두 집계 CSV 에서 팔 -> 지표 -> (평균, 표준편차, n)."""
    arms, src = {}, {}
    for path, keycol in ((RISKY_CSV, "factor"), (EXTRA_CSV, "arm")):
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                label = row[keycol]
                n = int(float(row["n"]))
                arms[label] = dict(
                    (m, (_f(row, m), _f(row, m + "_std"), n))
                    for m, _d, _g in METRICS)
                src[label] = os.path.basename(path)
    return arms, src


def load_raw():
    """두 모델별 CSV 에서 팔 -> 지표 -> [값...]."""
    raw, src = {}, {}
    for path in (FACTOR_RUNS, EXTRA_RUNS):
        if not os.path.exists(path):
            continue
        with io.open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                label = row["arm"]
                d = raw.setdefault(label, dict((m, [])
                                               for m, _x, _y in METRICS))
                for m, _x, _y in METRICS:
                    d[m].append(float(row[m]))
                src[label] = os.path.basename(path)
    return raw, src


def _selfcheck(raw):
    """원자료 경로와 요약통계 경로가 같은 Welch 를 주는지 확인한다."""
    usable = sorted(a for a in raw if len(raw[a]["delay"]) >= 2)
    if len(usable) < 2:
        print("  (원자료가 n>=2 인 팔이 %d 개뿐이라 자체 점검을 건너뛴다)"
              % len(usable))
        return
    a, b = usable[0], usable[1]
    worst = 0.0
    for m, _d, _g in METRICS:
        xs, ys = raw[a][m], raw[b][m]
        d1, t1, f1, p1 = RS.welch(xs, ys)
        mx, sx = RS.mean_std(xs)
        my, sy = RS.mean_std(ys)
        d2, t2, f2, p2 = welch_summary(mx, sx, len(xs), my, sy, len(ys))
        for u, v in ((d1, d2), (t1, t2), (f1, f2), (p1, p2)):
            if u == u and v == v:
                worst = max(worst, abs(u - v))
    print("  자체 점검: %s 대 %s, 원자료 경로와 요약통계 경로의 최대 차이 %.3g"
          % (a, b, worst))
    if worst > 1e-9:
        sys.exit("!! 두 경로가 어긋났다 (%.6g)" % worst)
    print("  -> 일치. 요약통계 Welch 는 원자료 계산과 같은 값이다.")


def crosscheck(raw, summ):
    """모델별 자료의 평균이 집계 CSV 와 맞는지 본다 (출처 혼합 탐지)."""
    print("")
    print("[출처 점검] 모델별 자료의 평균 대 집계 CSV")
    print("-" * 96)
    bad = 0
    for label in sorted(raw):
        if label not in summ:
            print("  %-10s 집계에 없다" % label)
            continue
        m, _s = RS.mean_std(raw[label]["delay"])
        got = summ[label]["delay"][0]
        d = m - got
        flag = "" if abs(d) < 1e-4 else "   <-- 어긋남"
        if abs(d) >= 1e-4:
            bad += 1
        print("  %-10s 모델별 평균 %10.4f   집계 %10.4f   차이 %+.6f%s"
              % (label, m, got, d, flag))
    if bad:
        print("  !! %d 개 팔이 어긋났다. 두 자료가 같은 측정이 아니다." % bad)
    else:
        print("  -> 전부 일치. 같은 측정의 자료다.")
    return bad


def _C(n, k):
    if k < 0 or k > n:
        return 0
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def fisher_2x2(a, b, c, d):
    """2x2 정확검정. 행 = 팔, 열 = (최빈결과 도달, 미도달).

    양측 p = 관측 표의 확률 이하인 모든 표의 확률 합(초기하 분포).
    scipy 가 이 환경에서 ABI 깨져 있어 직접 계산한다.
    """
    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    tot = _C(n, c1)
    if tot == 0 or r1 == 0 or r2 == 0:
        return float("nan"), float("nan")

    def pr(x):
        return _C(r1, x) * _C(r2, c1 - x) / float(tot)

    p_obs = pr(a)
    lo, hi = max(0, c1 - r2), min(r1, c1)
    p2 = sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs + 1e-12)
    # 같은 n 에서 **완전분리**가 났을 때의 양측 p (도달 가능한 최소)
    sep = 2.0 / _C(n, min(r1, r2)) if _C(n, min(r1, r2)) else float("nan")
    return min(p2, 1.0), sep


def attractor(raw, baseline="Full"):
    """모델이 **같은 정책 결과**로 수렴하는지 센다.

    왜: Full 5개 모델 중 4개가 delay·loss·hop·risky 네 지표 전부 소수
    4자리까지 같다. 200 에피소드 평균이 네 지표 다 같다는 건 경로 선택이
    같다는 뜻이다(경로 자체를 따로 확인하지 않았으므로 '네 지표가 같다'
    까지만 주장한다). 그러면 팔의 평균은 '연속적 성능' 이 아니라 '몇 개
    결과 중 어디에 몇 개가 떨어졌나' 의 가중평균이고, ± 는 그 추첨의
    산포다. 평균차 검정보다 **도달 비율**이 이 자료에 맞는 통계다.
    """
    print("")
    print("[수렴] 모델들이 같은 결과로 수렴하는가 (네 지표 동시 비교)")
    print("-" * 96)
    keys = [m for m, _d, _g in METRICS if m in ("delay", "loss", "hop",
                                                "risky_pct")]
    sig = {}
    for arm, d in raw.items():
        n = len(d["delay"])
        sig[arm] = [tuple(round(d[k][i], 4) for k in keys)
                    for i in range(n)]

    cross = {}
    for arm, ss in sig.items():
        for s in ss:
            cross.setdefault(s, {}).setdefault(arm, 0)
            cross[s][arm] += 1
    shared = sorted(cross.items(), key=lambda kv: -sum(kv[1].values()))

    print("  팔을 가로질러 반복되는 결과 (모델 2개 이상):")
    any_shared = False
    for s, who in shared:
        tot = sum(who.values())
        if tot < 2:
            continue
        any_shared = True
        print("    지연 %9.4f  홉 %6.3f  손실 %.6f  Risky%% %7.4f  -> 모델 %d개"
              % (s[0], s[2], s[1], s[3], tot))
        print("        %s" % ", ".join("%s x%d" % (a, c)
                                       for a, c in sorted(who.items())))
    if not any_shared:
        print("    없다 - 모든 모델이 서로 다른 결과를 냈다.")
        return []

    # 최빈 결과 = 가장 많은 모델이 떨어진 자리
    best_sig, best_who = shared[0]
    nw = sum(best_who.values())
    print("")
    print("  최빈 결과: 지연 %.4f (전체 모델 %d개 중 %d개). 이 결과를 내는 팔:"
          % (best_sig[0], sum(len(v) for v in sig.values()), nw))
    print("    %s" % ", ".join(sorted(best_who)))
    nonw = sorted(a for a in sig if a not in best_who)
    print("  한 번도 내지 않은 팔: %s" % (", ".join(nonw) or "(없음)"))
    print("")
    print("  읽는 법: 가중치를 바꾼 팔(W-*)은 상태의 위험값 자체가 달라지므로")
    print("  같은 결과가 나올 이유가 없다. 반면 K 를 바꾼 팔은 기본 가중치라")
    print("  같은 결과가 나올 수 있고 **실제로 K=3,5,7,10 전부 낸다** - 즉")
    print("  시간 윈도는 정책 결과를 가르지 못한다(기존 rho>=0.986 측정과")
    print("  일관). 절제 팔 중에서 갈리는 것이 설계 효과다.")

    if baseline not in sig:
        return []
    rows = []
    nb = len(sig[baseline])
    ab = sum(1 for s in sig[baseline] if s == best_sig)
    print("")
    print("  [도달 비율 정확검정] 기준 %s = %d/%d" % (baseline, ab, nb))
    print("  %-11s %9s %11s %11s %11s" % ("팔", "도달", "p_fisher",
                                          "완전분리시", "같은가중치"))
    for arm in sorted(sig):
        if arm == baseline:
            continue
        na = len(sig[arm])
        aa = sum(1 for s in sig[arm] if s == best_sig)
        p, sep = fisher_2x2(ab, nb - ab, aa, na - aa)
        same = "아니다(W)" if arm.startswith("W-") else "그렇다"
        rows.append(dict(arm=arm, reached=aa, n=na, p_fisher=p, p_sep=sep))
        print("  %-11s %4d/%-4d %11.4f %11.4f %11s"
              % (arm, aa, na, p, sep, same))
    print("")
    print("  * p_fisher 는 보정 전 값이다. 족 크기를 걸면 아래 Holm 과 같은")
    print("    문제가 생긴다 - 절제 6팔을 한 족으로 보면 x6 이다.")
    print("  * '완전분리시' 는 이 n 으로 **가능한 최소 양측 p** 다. 그 값에")
    print("    족 크기를 곱한 게 0.05 를 넘으면, 효과가 아무리 커도 그 족에서")
    print("    유의가 나올 수 없다 - 자료가 아니라 설계의 한계다.")
    return rows


def main():
    summ, ssrc = load_summary()
    raw, rsrc = load_raw()
    print("=" * 96)
    print("표 10 유의성 - 한 하네스(내 롤아웃) 안에서만")
    print("=" * 96)
    print("집계 팔 %d: %s" % (len(summ), ", ".join(sorted(summ))))
    print("원자료 팔 %d: %s" % (len(raw), ", ".join(sorted(raw))))
    _selfcheck(raw)
    if raw:
        crosscheck(raw, summ)

    # 서로 다른 결과가 몇 가지인지 - ± 를 어떻게 읽을지 바꾸는 특성
    if raw:
        print("")
        print("[분포 특성] 모델이 서로 다른 결과를 몇 가지 내는가")
        print("-" * 96)
        print("%-10s %4s %9s %9s %s"
              % ("팔", "n", "지연가지", "홉가지", "지연값(가지별 개수)"))
        for label in sorted(raw):
            ds = [round(v, 3) for v in raw[label]["delay"]]
            hs = [round(v, 3) for v in raw[label]["hop"]]
            cnt = {}
            for v in ds:
                cnt[v] = cnt.get(v, 0) + 1
            txt = ", ".join("%.1f x%d" % (k, cnt[k]) for k in sorted(cnt))
            print("%-10s %4d %9d %9d %s"
                  % (label, len(ds), len(set(ds)), len(set(hs)), txt))
        print("  읽기: 가지 수가 n 보다 훨씬 적으면 ± 는 정규 오차가 아니라")
        print("  **경로 집합 몇 개 사이의 추첨 폭**이다. 그래서 Welch 와 함께")
        print("  분포 가정이 없는 정확 순열검정을 나란히 적는다.")

    rows = []
    for fam, title, base, tests in FAMILIES:
        if base not in summ:
            print("\n[%s] %s -> 기준 팔 '%s' 이 없다. 건너뛴다."
                  % (fam, title, base))
            continue
        for arm in tests:
            if arm not in summ:
                print("\n[%s] %s -> 비교 팔 '%s' 이 없다. 건너뛴다."
                      % (fam, title, arm))
                continue
            for m, disp, _g in METRICS:
                m1, s1, n1 = summ[base][m]
                m2, s2, n2 = summ[arm][m]
                diff, t, df, p = welch_summary(m1, s1, n1, m2, s2, n2)
                pp, nsp, pmin = float("nan"), 0, float("nan")
                if base in raw and arm in raw:
                    pp, nsp, pmin = perm_exact(raw[base][m], raw[arm][m])
                rows.append(dict(
                    family=fam, family_title=title, baseline=base, arm=arm,
                    metric=m, metric_disp=disp,
                    mean_base=m1, std_base=s1, n_base=n1,
                    mean_arm=m2, std_arm=s2, n_arm=n2,
                    diff=diff, welch_t=t, welch_df=df, p_welch=p,
                    p_perm=pp, n_splits=nsp, p_min_attainable=pmin,
                    src_base=ssrc.get(base, ""), src_arm=ssrc.get(arm, ""),
                    raw_base=rsrc.get(base, ""), raw_arm=rsrc.get(arm, "")))

    for fam, _t, _b, _x in FAMILIES:
        idx = [i for i, r in enumerate(rows) if r["family"] == fam]
        if not idx:
            continue
        for src_key, dst_key in (("p_welch", "p_holm_welch"),
                                 ("p_perm", "p_holm_perm")):
            ps = [rows[i][src_key] for i in idx]
            adj = RS.holm_bonferroni(ps)
            for i, a in zip(idx, adj):
                rows[i][dst_key] = a
        size = sum(1 for i in idx if rows[i]["p_welch"] == rows[i]["p_welch"])
        for i in idx:
            rows[i]["family_size"] = size
    for src_key, dst_key in (("p_welch", "p_holm_welch_pooled"),
                             ("p_perm", "p_holm_perm_pooled")):
        adj = RS.holm_bonferroni([r[src_key] for r in rows])
        for r, a in zip(rows, adj):
            r[dst_key] = a
    psize = sum(1 for r in rows if r["p_welch"] == r["p_welch"])
    for r in rows:
        r["pooled_size"] = psize

    for fam, title, base, _tests in FAMILIES:
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        size = sub[0]["family_size"]
        print("")
        print("[%s] %s    (Holm 족 크기 %d)" % (fam, title, size))
        print("-" * 96)
        print("%-11s %-14s %10s %10s %10s %8s %8s %8s %8s"
              % ("비교", "지표", "기준", "비교", "차이", "p(W)",
                 "p(순열)", "pH(W)", "pH(순열)"))
        for r in sub:
            g = [g for m, _d, g in METRICS if m == r["metric"]][0]
            mark = ""
            ph = r.get("p_holm_perm")
            pw = r.get("p_holm_welch")
            if ph == ph and ph < 0.05:
                mark += "  **순열유의"
            if pw == pw and pw < 0.05:
                mark += "  **Welch유의"
            if r["n_arm"] < 2 or r["n_base"] < 2:
                mark = "  (n<2, 검정 불가)"
            print("%-11s %-14s %s %s %s %8s %8s %8s %8s%s"
                  % (r["arm"], r["metric_disp"],
                     g % r["mean_base"], g % r["mean_arm"], g % r["diff"],
                     RS.fmt_p(r["p_welch"]), RS.fmt_p(r["p_perm"]),
                     RS.fmt_p(pw), RS.fmt_p(ph), mark))
        # 이 족에서 도달 가능한 최소 p - 검정력의 상한
        pm = [r["p_min_attainable"] for r in sub
              if r["p_min_attainable"] == r["p_min_attainable"]]
        if pm:
            best = min(pm)
            print("  검정력 상한: 이 족에서 순열검정이 낼 수 있는 최소 양측 p 는")
            print("  %.4f 다(분할 %d). Holm 족 %d 을 걸면 보정 후 최소 %.4f -> "
                  % (best, min(r["n_splits"] for r in sub
                               if r["n_splits"]), size, best * size), end="")
            if best * size >= 0.05:
                print("**효과 크기와 무관하게 0.05 를 넘을 수 없다**.")
                print("  이 족의 '유의한 칸 0개' 는 효과가 없다는 뜻이 아니라")
                print("  표본(모델 n)이 그만큼의 검정력을 안 가졌다는 뜻이다.")
            else:
                print("유의가 나올 수는 있다.")
        for r in sub:
            if r["n_arm"] < 2 or r["n_base"] < 2:
                print("  ! %s 는 n=%d 라 검정이 성립하지 않는다 - 표에 n 을 "
                      "병기할 것." % (r["arm"], r["n_arm"]))
                break

    print("")
    print("=" * 96)
    print("읽기 - D(재현 바닥) 를 먼저 보고 A(절제) 를 그 바닥과 견준다")
    print("=" * 96)
    dmap = dict((r["metric"], r) for r in rows if r["family"] == "D")
    amap = {}
    for r in rows:
        if r["family"] == "A":
            amap.setdefault(r["metric"], []).append(r)
    if not dmap:
        print("D 족이 비어 있다(K=5 팔이 아직 없다).")
    for m, disp, g in METRICS:
        d = dmap.get(m)
        if d is None:
            continue
        floor = abs(d["diff"])
        print("")
        print("%s: 재현 바닥 |Full - K=5| = %s  (p 생=%s, 순열=%s)"
              % (disp, (g % floor).strip(), RS.fmt_p(d["p_welch"]),
                 RS.fmt_p(d["p_perm"])))
        for r in sorted(amap.get(m, []), key=lambda x: -abs(x["diff"])):
            rel = abs(r["diff"]) / floor if floor > 0 else float("inf")
            if rel < 1.0:
                tag = "바닥보다 작다 (%.2f배)" % rel
            else:
                tag = "바닥의 %.2f배" % rel
            print("    %-9s 차이 %s  %s"
                  % (r["arm"], (g % r["diff"]).strip(), tag))

    if raw:
        attractor(raw)

    fields = ["family", "family_title", "baseline", "arm", "metric",
              "metric_disp", "mean_base", "std_base", "n_base",
              "mean_arm", "std_arm", "n_arm", "diff", "welch_t", "welch_df",
              "p_welch", "p_perm", "n_splits", "p_min_attainable",
              "p_holm_welch", "p_holm_perm", "family_size",
              "p_holm_welch_pooled", "p_holm_perm_pooled", "pooled_size",
              "src_base", "src_arm", "raw_base", "raw_arm"]
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(dict((k, r.get(k, "")) for k in fields))
    print("")
    print("-> %s (%d 행)" % (OUT_CSV, len(rows)))
    print("주의 1: 짝 검정은 쓰지 않았다. run_id 는 시드만 공유하고 평가 조건을")
    print("        짝지어 주지 않는다.")
    print("주의 2: Holm 값을 인용할 때 족 크기를 반드시 병기할 것(전체 족 %d)."
          % psize)
    print("주의 3: 표에 올린 평균과 p 는 모두 내 롤아웃 하나에서 나온 것이다 -")
    print("        원본 harness 의 factor_*_result.csv 는 섞지 않았다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
