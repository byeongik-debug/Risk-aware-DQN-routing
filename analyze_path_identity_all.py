# -*- coding: utf-8 -*-
"""표 10 모수 전수(69개)의 경로 동일성을 **팔내 대 팔간**으로 가른다.

왜 이 집계가 필요한가
---------------------
26.7절·22.5절에 적은 "팔을 가로질러 경로가 100% 같고 `-Util` 만 31% 로
갈린다" 는 두 가지 문제가 있다.

  (1) **선택 순환** — 그 측정(`path_identity_modal_*.csv`)은 *최빈 결과값에
      속한 모델만* 대상으로 했다. 네 지표가 같은 모델을 골라 놓고 경로가
      같은지 물었으므로, 100% 는 상당 부분 선택 기준의 반복이다.
  (2) **분모가 없다** — "A 팔과 B 팔이 31% 만 일치한다" 는 B 팔 *안에서*
      런끼리 몇 % 일치하는지를 모르면 해석할 수 없다. 팔내 일치율이 이미
      낮으면 팔간 불일치는 인자의 효과가 아니라 **학습 시드의 효과**다.

그래서 전수(69모델)의 쌍별 일치율을 받아, 팔마다
  - 팔내 일치율(그 팔 런끼리)  = 잡음 바닥
  - 팔간 일치율(기준 팔과)      = 신호 + 잡음
을 같이 내고, **라벨 교환 정확 순열검정**으로 "팔 라벨이 경로를 가르는가"
를 판정한다. 통계량은

    S = mean(팔간 일치율) - mean(팔내 일치율)

이고(음수일수록 라벨이 가른다), 두 팔의 런을 합쳐 가능한 모든 분할
C(nA+nB, nA) 를 전수 열거해 S 의 귀무분포를 만든다. 근사 없음·표본추출
없음이라 n=5 에서도 정확하다. 경로 일치율은 스레드 수·부동소수에 흔들리지
않으므로(지연 평균과 달리) 이 판정은 Welch MDE 약 30 ms 한계에 걸리지
않는다.

한계를 미리 적는다
------------------
  - 런이 1개인 팔(K=10)은 팔내 일치율이 정의되지 않고 순열 수가 6개뿐이라
    최소 p 가 1/6 = 0.1667 이다 → **원리적으로 유의할 수 없다**.
  - 일치율은 '에피소드마다 노드 열이 완전히 같은가' 의 비율이므로 한 홉만
    달라도 0 으로 센다. 즉 "정책이 다르다" 에 민감하고 "얼마나 다른가" 는
    말하지 않는다.
  - 검정은 팔 라벨이 경로를 가르는지만 본다. 가른다고 해서 그 인자가
    **성능**을 바꾼다는 뜻은 아니다(지연 쪽은 84족 Holm 에서 0/84).

사용법:
    python analyze_path_identity_all.py
    python analyze_path_identity_all.py --base Full
"""
import argparse
import csv
import io
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULT_DIR = os.path.join(HERE, "review_results")
PAIRS = os.path.join(RESULT_DIR, "path_identity_all_pairs.csv")
ARMS = os.path.join(RESULT_DIR, "path_identity_all_arms.csv")
OUT_WITHIN = os.path.join(RESULT_DIR, "path_identity_all_within.csv")
OUT_TEST = os.path.join(RESULT_DIR, "path_identity_all_test.csv")


def arm_of(label):
    """'-VD,VL-r03' -> '-VD,VL' ; 'K=3-r10' -> 'K=3'"""
    i = label.rfind("-r")
    return label[:i] if i > 0 else label


def load_pairs():
    """라벨 쌍 -> 일치율(%) 과 비교 에피소드 수."""
    if not os.path.exists(PAIRS):
        print("!! %s 가 없다. check_path_identity.py --all 을 먼저 돌려라."
              % PAIRS)
        return None, None
    pct, eps, labels = {}, {}, set()
    with io.open(PAIRS, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = row["a"], row["b"]
            key = (a, b) if a < b else (b, a)
            pct[key] = float(row["same_pct"])
            eps[key] = int(float(row["episodes"]))
            labels.add(a)
            labels.add(b)
    globals()["EPISODES"] = (max(eps.values()) if eps
                             else 0)
    return pct, labels


def get(pct, a, b):
    return pct.get((a, b) if a < b else (b, a))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def within_stats(pct, runs):
    """한 팔 안의 런끼리 일치율."""
    vs = [get(pct, a, b) for a, b in itertools.combinations(sorted(runs), 2)]
    vs = [v for v in vs if v is not None]
    return vs


def cross_stats(pct, ra, rb):
    vs = [get(pct, a, b) for a in ra for b in rb]
    return [v for v in vs if v is not None]


def exact_perm_p(pct, ra, rb):
    """라벨 교환 정확 순열검정.

    통계량 S = mean(팔간) - mean(팔내 둘을 합친 것). 관측 S 이하가 나오는
    분할 비율이 p (단측: '라벨이 가른다' 쪽).
    """
    pool = sorted(ra) + sorted(rb)
    na = len(ra)
    if na == 0 or len(rb) == 0:
        return float("nan"), float("nan"), 0

    def stat(group_a):
        ga = set(group_a)
        gb = [x for x in pool if x not in ga]
        ga = list(ga)
        w = within_stats(pct, ga) + within_stats(pct, gb)
        c = cross_stats(pct, ga, gb)
        if not c or not w:
            return float("nan")
        return mean(c) - mean(w)

    obs = stat(sorted(ra))
    splits = list(itertools.combinations(pool, na))
    vals = [stat(list(s)) for s in splits]
    vals = [v for v in vals if v == v]           # NaN 제거
    if not vals or obs != obs:
        return obs, float("nan"), len(splits)
    le = sum(1 for v in vals if v <= obs + 1e-12)
    return obs, le / float(len(vals)), len(vals)


def run_of(label):
    """'K=3-r10' -> 10 ; 학습 스크립트 넷이 seed = 42 + run_id 를 쓰므로
    run_id 가 곧 학습 시드다."""
    i = label.rfind("-r")
    try:
        return int(label[i + 2:])
    except ValueError:
        return -1


def seed_vs_factor(pct, arms, n_shuffle=10000, seed=20261115):
    """팔간 쌍에서 '같은 시드' 가 일치율을 올리는가.

    통계량 D = mean(같은 run_id 팔간 쌍) - mean(다른 run_id 팔간 쌍).
    귀무: 팔 안에서 run_id 라벨을 섞는다(팔 소속은 그대로).
    """
    import random
    members = dict((k, sorted(arms[k])) for k in arms)
    cross = []
    for (x, y) in sorted(pct.keys()):
        if arm_of(x) != arm_of(y):
            cross.append((x, y, pct[(x, y)]))
    if not cross:
        return (float("nan"),) * 3 + (0, 0, 0)

    def stat(rid):
        s = [v for (x, y, v) in cross if rid[x] == rid[y]]
        d = [v for (x, y, v) in cross if rid[x] != rid[y]]
        if not s or not d:
            return float("nan"), len(s), len(d)
        return mean(s) - mean(d), len(s), len(d)

    obs_rid = {}
    for k in members:
        for x in members[k]:
            obs_rid[x] = run_of(x)
    obs, n_same, n_diff = stat(obs_rid)

    rnd = random.Random(seed)
    ge = 0
    for _ in range(n_shuffle):
        rid = {}
        for k in members:
            ids = [run_of(x) for x in members[k]]
            rnd.shuffle(ids)
            for x, v in zip(members[k], ids):
                rid[x] = v
        v, _a, _b = stat(rid)
        if v == v and v >= obs - 1e-12:
            ge += 1
    p = (ge + 1.0) / (n_shuffle + 1.0)
    return obs, p, n_shuffle, n_same, n_diff, len(cross)


def perm_floor(na, nb, family):
    """런 수만으로 정해지는 최소 p 와 최소 p_holm.

    분할 수는 C(na+nb, na). 통계량이 두 묶음 라벨 교환에 대칭이므로
    na == nb 일 때만 여집합이 같은 값을 내 2중 축퇴가 생긴다.
    """
    from math import comb
    n = comb(na + nb, na)
    deg = 2 if na == nb else 1
    minp = deg / float(n)
    return n, minp, min(1.0, minp * family)


def policy_classes(pct, labels):
    """'경로 100% 일치' 동치류로 모델을 묶는다 = 서로 다른 정책의 수.

    추이성 검사를 같이 한다: 같은 묶음에 든 두 모델이 100% 가 아니면
    측정이 모순이라는 뜻이므로 그 쌍을 돌려준다.
    """
    labels = sorted(labels)
    parent = dict((x, x) for x in labels)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, x in enumerate(labels):
        for y in labels[i + 1:]:
            v = get(pct, x, y)
            if v is not None and v >= 99.9999:
                rx, ry = find(x), find(y)
                if rx != ry:
                    parent[rx] = ry
    groups = {}
    for x in labels:
        groups.setdefault(find(x), []).append(x)
    classes = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    viol = []
    for g in classes:
        for i, x in enumerate(sorted(g)):
            for y in sorted(g)[i + 1:]:
                v = get(pct, x, y)
                if v is None or v < 99.9999:
                    viol.append((x, y, v))
    return classes, viol


def global_perm_test(pct, arms, n_shuffle=10000, seed=20261115):
    """팔 소속이 경로 일치를 예측하는가 — 검정 **한 개**.

    통계량 G = mean(팔내 쌍 일치율) - mean(팔간 쌍 일치율).
    69개 모델의 팔 라벨을 섞어(팔 크기는 유지) G 의 귀무분포를 만든다.
    쌍이 2346개라 분해능이 충분하고, 검정이 하나이므로 Holm 보정이
    필요 없다. 몬테카를로이므로 p 의 하한은 1/(n_shuffle+1) 이다.
    """
    import random
    labels = sorted(set([x for k in arms for x in arms[k]]))
    sizes = [(k, len(arms[k])) for k in sorted(arms)]
    pairs = sorted(pct.keys())

    def stat(assign):
        w, c = [], []
        for (x, y) in pairs:
            v = pct[(x, y)]
            if assign[x] == assign[y]:
                w.append(v)
            else:
                c.append(v)
        if not w or not c:
            return float("nan")
        return mean(w) - mean(c)

    obs_assign = {}
    for k in arms:
        for x in arms[k]:
            obs_assign[x] = k
    obs = stat(obs_assign)

    rnd = random.Random(seed)
    ge = 0
    perm = list(labels)
    for _ in range(n_shuffle):
        rnd.shuffle(perm)
        asg, i = {}, 0
        for k, n in sizes:
            for x in perm[i:i + n]:
                asg[x] = k
            i += n
        v = stat(asg)
        if v == v and v >= obs - 1e-12:
            ge += 1
    p = (ge + 1.0) / (n_shuffle + 1.0)
    return obs, p, n_shuffle


def holm(ps, alpha=0.05):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    m, out, prev = len(ps), [0.0] * len(ps), 0.0
    for r, i in enumerate(idx):
        adj = min(1.0, max(prev, (m - r) * ps[i]))
        out[i] = adj
        prev = adj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Full", help="기준 팔")
    a = ap.parse_args()

    pct, labels = load_pairs()
    if pct is None:
        return 1
    arms = {}
    for lb in sorted(labels):
        arms.setdefault(arm_of(lb), []).append(lb)

    print("=" * 92)
    print("표 10 모수 전수 경로 동일성 — 팔내(잡음 바닥) 대 팔간(신호)")
    print("팔 %d개 / 모델 %d개 / 쌍 %d개"
          % (len(arms), len(labels), len(pct)))
    print("=" * 92)

    # ── ① 팔내 일치율 ─────────────────────────────────────────────────
    print("\n[팔내] 같은 설정·다른 시드의 런끼리 경로가 같은가")
    print("-" * 92)
    wrows = []
    for arm in sorted(arms, key=lambda k: -len(arms[k])):
        runs = arms[arm]
        vs = within_stats(pct, runs)
        n100 = sum(1 for v in vs if v >= 99.9999)
        row = dict(arm=arm, n_runs=len(runs), n_pairs=len(vs),
                   within_mean=("%.4f" % mean(vs)) if vs else "",
                   within_min=("%.4f" % min(vs)) if vs else "",
                   within_max=("%.4f" % max(vs)) if vs else "",
                   pairs_100pct=n100)
        wrows.append(row)
        if vs:
            print("  %-11s 런 %2d  쌍 %2d   평균 %6.2f%%  최저 %6.2f%%  "
                  "최고 %6.2f%%   100%%쌍 %d/%d"
                  % (arm, len(runs), len(vs), mean(vs), min(vs), max(vs),
                     n100, len(vs)))
        else:
            print("  %-11s 런 %2d  쌍  0   (팔내 비교 불가)"
                  % (arm, len(runs)))

    labels = sorted(labels)
    allw = []
    for arm in arms:
        allw += within_stats(pct, arms[arm])
    print("\n  전체 팔내 쌍 %d개: 평균 %.2f%%, 100%% 일치 %d개 (%.1f%%)"
          % (len(allw), mean(allw),
             sum(1 for v in allw if v >= 99.9999),
             100.0 * sum(1 for v in allw if v >= 99.9999) / len(allw)))

    # 정책 동치류 전수조사
    classes, viol = policy_classes(pct, labels)
    print("\n[정책 전수조사] '%d에피소드 경로가 전부 같다' 로 묶은 동치류" % EPISODES)
    print("-" * 92)
    print("  모델 %d개 -> 서로 다른 정책 **%d개**" % (len(labels),
                                                     len(classes)))
    if viol:
        print("  !! 추이성 위반 %d쌍 — 묶음을 믿으면 안 된다. 예: %s"
              % (len(viol), viol[:3]))
    singles = sum(1 for g in classes if len(g) == 1)
    print("  가장 큰 묶음 %d개 모델 / 혼자인 정책 %d개"
          % (len(classes[0]), singles))
    crows = []
    for ci, g in enumerate(classes, 1):
        garms = sorted(set(arm_of(x) for x in g))
        crows.append(dict(policy_id=ci, n_models=len(g),
                          n_arms=len(garms), arms=" ".join(garms),
                          models=" ".join(sorted(g))))
        if ci <= 8 or len(g) > 1:
            print("   P%-2d 모델 %2d  팔 %2d  [%s]"
                  % (ci, len(g), len(garms), " ".join(garms)))
    import csv as _csv
    cp = os.path.join(RESULT_DIR, "path_identity_all_classes.csv")
    with io.open(cp, "w", encoding="utf-8", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=["policy_id", "n_models",
                                            "n_arms", "arms", "models"])
        w.writeheader()
        w.writerows(crows)
    print("  -> %s (%d행)" % (cp, len(crows)))

    # 교란 검정: 같은 학습 시드(run_id)가 팔간 일치를 올리는가
    s_obs, s_p, s_n, n_same, n_diff, n_cross = seed_vs_factor(pct, arms)
    print("\n[교란 검정] 팔간 쌍에서 '같은 학습 시드' 가 경로 일치를 올리는가")
    print("-" * 92)
    print("  학습 스크립트 넷이 seed = 42 + run_id 이므로 run_id = 시드다.")
    print("  팔간 쌍 %d개 = 같은 시드 %d + 다른 시드 %d"
          % (n_cross, n_same, n_diff))
    print("  D = 같은시드평균 - 다른시드평균 = %+.4f%%p   섞기 %d회  p = %.6g"
          % (s_obs, s_n, s_p))
    print("  (D 가 크고 p 가 작으면 경로를 가르는 것은 인자보다 **시드**다.)")

    # 전역 검정(한 개): 팔 소속이 경로 일치를 예측하는가
    g_obs, g_p, g_n = global_perm_test(pct, arms)
    print("\n[전역 검정 1개] 팔 소속이 경로 일치를 예측하는가")
    print("-" * 92)
    print("  G = 팔내평균 - 팔간평균 = %+.4f%%p   섞기 %d회  p = %.6g"
          % (g_obs, g_n, g_p))
    print("  (검정이 하나이므로 Holm 불필요. p 하한 %.2g)"
          % (1.0 / (g_n + 1.0)))

    # ── ② 기준 팔과의 팔간 비교 + 정확 순열검정 ───────────────────────
    base = a.base
    if base not in arms:
        print("\n!! 기준 팔 %s 이 없다. 있는 팔: %s"
              % (base, ", ".join(sorted(arms))))
        return 1
    print("\n[팔간] 기준 %s 과 비교 — 라벨 교환 정확 순열검정" % base)
    print("-" * 92)
    others = [k for k in sorted(arms) if k != base]
    trows = []
    for arm in others:
        cs = cross_stats(pct, arms[base], arms[arm])
        wb = within_stats(pct, arms[base])
        wa = within_stats(pct, arms[arm])
        obs, p, nperm = exact_perm_p(pct, arms[base], arms[arm])
        _ns, _mp, _mh = perm_floor(len(arms[base]), len(arms[arm]),
                                   len(others))
        trows.append(dict(min_p=("%.6g" % _mp), min_p_holm=("%.4g" % _mh),
                          reachable=("가능" if _mh < 0.05 else "불가"),
                          base=base, arm=arm, n_runs=len(arms[arm]),
                          cross_mean="%.4f" % mean(cs),
                          within_base="%.4f" % mean(wb),
                          within_arm=("%.4f" % mean(wa)) if wa else "",
                          stat="%.4f" % obs if obs == obs else "",
                          p_perm="%.6g" % p if p == p else "",
                          n_perm=nperm))
    ps = [float(r["p_perm"]) for r in trows if r["p_perm"]]
    hp = holm(ps)
    k = 0
    for r in trows:
        if r["p_perm"]:
            r["p_holm"] = "%.6g" % hp[k]
            k += 1
        else:
            r["p_holm"] = ""
    print("  %-11s %4s %9s %9s %9s %9s %9s %8s %8s %5s"
          % ("팔", "런", "팔간평균", "팔내기준", "팔내팔",
             "S", "p", "p_holm", "하한pH", "도달"))
    for r in trows:
        print("  %-11s %4s %8s%% %8s%% %8s%% %9s %9s %8s %8s %5s"
              % (r["arm"], r["n_runs"], r["cross_mean"], r["within_base"],
                 r["within_arm"] or "  -  ", r["stat"], r["p_perm"],
                 r["p_holm"], r["min_p_holm"], r["reachable"]))
    nope = [r["arm"] for r in trows if r.get("reachable") == "불가"]
    print("\n  원리적으로 Holm 유의 **불가** %d/%d 팔 (런이 적어 순열이 모자람):"
          % (len(nope), len(trows)))
    print("    %s" % ", ".join(nope))
    print("  필요한 런 수(기준 %d런·족 %d 기준):" % (len(arms[base]),
                                                    len(others)))
    for nb in range(1, 11):
        _n, _p, _h = perm_floor(len(arms[base]), nb, len(others))
        print("    비교팔 %2d런: 분할 %5d  최소 p %.6f  최소 p_holm %.4f %s"
              % (nb, _n, _p, _h, "가능" if _h < 0.05 else "불가"))
    trows.append(dict(base="(교란)", arm="SAME-seed-vs-diff",
                      n_runs=n_cross, cross_mean="", within_base="",
                      within_arm="", stat="%.4f" % s_obs,
                      p_perm="%.6g" % s_p, p_holm="(보정불필요)",
                      n_perm=s_n, min_p="", min_p_holm="",
                      reachable=""))
    trows.append(dict(base="(전역)", arm="ALL-arms-shuffle",
                      n_runs=len(labels), cross_mean="", within_base="",
                      within_arm="", stat="%.4f" % g_obs,
                      p_perm="%.6g" % g_p, p_holm="(보정불필요)",
                      n_perm=g_n, min_p="", min_p_holm="",
                      reachable=""))
    sig = [r for r in trows
           if r["p_holm"] and r["p_holm"] != "(보정불필요)"
           and float(r["p_holm"]) < 0.05]
    print("\n  Holm(족 %d) 유의: %s"
          % (len(ps), ", ".join(r["arm"] for r in sig) if sig else "없음"))

    for path, rows, flds in (
            (OUT_WITHIN, wrows, ["arm", "n_runs", "n_pairs", "within_mean",
                                 "within_min", "within_max",
                                 "pairs_100pct"]),
            (OUT_TEST, trows, ["base", "arm", "n_runs", "cross_mean",
                               "within_base", "within_arm", "stat",
                               "p_perm", "p_holm", "n_perm",
                               "min_p", "min_p_holm", "reachable"])):
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print("-> %s (%d행)" % (path, len(rows)))

    print("\n주의: 5런 대 5런은 순열 252개이고 통계량이 라벨 교환에")
    print("대칭이라 최소 p = 2/252 = 0.00794, Holm 족 12 면 하한이")
    print("0.0952 다. 즉 5런 팔은 **원리적으로 Holm 유의할 수 없다** —")
    print("p_holm 이 크다고 '차이 없음' 이 아니다. 판정은 전역 검정으로.")
    print("\n읽는 법: 팔내 평균이 낮으면 그 팔의 '정책' 자체가 시드마다")
    print("다르다는 뜻이고, 그 팔에 대한 팔간 불일치는 인자의 효과로 읽을")
    print("수 없다. S 가 음수이고 p 가 작을 때만 '이 인자가 경로를 바꾼다'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
