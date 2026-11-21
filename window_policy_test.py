# -*- coding: utf-8 -*-
"""윈도우 값(K=3/5/7/10)이 '정책'을 가르는가 — 라벨 순열 검정.

왜 새로 재는가. 전수조사(FINDINGS 29.4)는 각 팔을 `Full` 과 **쌍으로만**
비교했다. 항목 7 의 주장은 그게 아니라 '윈도우 값들끼리 구분되는가' 이므로
윈도우 네 값을 한꺼번에 놓고 4수준으로 재야 한다.

통계량 S = mean(같은 윈도우 쌍 일치율) - mean(다른 윈도우 쌍 일치율).
귀무가설: 모델 24개에 붙은 윈도우 라벨이 교환 가능하다.
순열: 라벨을 군 크기(10/5/8/1)를 유지한 채 섞어 S 를 다시 계산한다.

입력은 이미 측정된 경로 일치 행렬뿐이다 — 추가 롤아웃/학습 없음
(`review_results/path_identity_all_pairs.csv`, 69모델 2346쌍,
각 쌍은 200 에피소드의 노드 열을 글자 단위로 비교한 값).
"""
from __future__ import print_function

import argparse
import csv
import itertools
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review_results")
PAIRS = os.path.join(OUT, "path_identity_all_pairs.csv")
CLASSES = os.path.join(OUT, "path_identity_all_classes.csv")
WITHIN = os.path.join(OUT, "path_identity_all_within.csv")

FIELDS = ["test", "n_models", "n_within", "n_between", "within_mean",
          "between_mean", "stat", "p_perm", "n_perm",
          "p_holm_famA", "p_holm_famB"]


def load_pairs():
    """{(a,b): same_pct} 양방향."""
    m, eps = {}, set()
    with open(PAIRS) as f:
        for r in csv.DictReader(f):
            v = float(r["same_pct"])
            m[(r["a"], r["b"])] = v
            m[(r["b"], r["a"])] = v
            eps.add(int(r["episodes"]))
    assert len(eps) == 1, u"에피소드 수가 섞였다: %r" % (eps,)
    return m, eps.pop()


def arm_of(model):
    return model.rsplit("-r", 1)[0]


def models_of(pairmap, arms):
    out = set()
    for a, b in pairmap:
        for x in (a, b):
            if arm_of(x) in arms:
                out.add(x)
    return sorted(out)


def stat(pairmap, labels):
    """labels: {model: group}. S = 군내평균 - 군간평균."""
    ms = sorted(labels.keys())
    win, bet = [], []
    for a, b in itertools.combinations(ms, 2):
        v = pairmap[(a, b)]
        if labels[a] == labels[b]:
            win.append(v)
        else:
            bet.append(v)
    if not win or not bet:
        return None, len(win), len(bet), None, None
    wm = sum(win) / float(len(win))
    bm = sum(bet) / float(len(bet))
    return wm - bm, len(win), len(bet), wm, bm


def perm_test(pairmap, labels, n_perm, seed):
    obs, nw, nb, wm, bm = stat(pairmap, labels)
    ms = sorted(labels.keys())
    groups = [labels[m] for m in ms]
    rng = random.Random(seed)
    ge = 1          # 관측 자신을 포함한다 (보수적)
    for _ in range(n_perm):
        rng.shuffle(groups)
        s = stat(pairmap, dict(zip(ms, groups)))[0]
        if s >= obs - 1e-12:
            ge += 1
    return obs, ge / float(n_perm + 1), nw, nb, wm, bm


def run(pairmap, name, labels, rows, n_perm, seed, note=""):
    obs, p, nw, nb, wm, bm = perm_test(pairmap, labels, n_perm, seed)
    print(u"\n[%s]  모델 %d개" % (name, len(labels)))
    print(u"    군내 %4d쌍 평균 %8.4f%%   군간 %4d쌍 평균 %8.4f%%"
          % (nw, wm, nb, bm))
    print(u"    S = %+8.4f %%p    p_perm = %.5f  (순열 %d)"
          % (obs, p, n_perm))
    if note:
        print(u"    %s" % note)
    rows.append(dict(test=name, n_models=len(labels), n_within=nw,
                     n_between=nb, within_mean="%.4f" % wm,
                     between_mean="%.4f" % bm, stat="%+.4f" % obs,
                     p_perm="%.5f" % p, n_perm=n_perm))
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261115)
    a = ap.parse_args()

    pairmap, episodes = load_pairs()
    print(u"경로 일치 행렬: %d 쌍, 에피소드 %d (추가 롤아웃 없음)"
          % (len(pairmap) // 2, episodes))

    rows = []

    # (1) 윈도우 네 값 4수준
    KARMS = {"K=3": 3, "K=5": 5, "K=7": 7, "K=10": 10}
    ms = models_of(pairmap, set(KARMS))
    lab = dict((m, KARMS[arm_of(m)]) for m in ms)
    cnt = {}
    for m in ms:
        cnt[lab[m]] = cnt.get(lab[m], 0) + 1
    run(pairmap, u"윈도우 4수준 K=3/5/7/10", lab, rows, a.perm, a.seed,
        note=u"군 크기 " + ", ".join("K=%d:%d" % (k, cnt[k])
                                    for k in sorted(cnt)))

    # (1b) **배치 교란을 뺀다.** `Full`(factor_full) 과 `K=5`(window5) 는
    #      같은 설정(윈도우 5·제거 없음)의 **다른 학습 배치**다. 그래서
    #      윈도우 5 에는 독립 학습군이 두 벌 있다. (1) 은 Full 을 빼고
    #      돌렸으므로 '같은 팔' 과 '같은 윈도우' 를 구분하지 못한다.
    WARMS = dict(KARMS)
    WARMS["Full"] = 5
    msw = models_of(pairmap, set(WARMS))
    labw = dict((m, WARMS[arm_of(m)]) for m in msw)
    cw = {}
    for m in msw:
        cw[labw[m]] = cw.get(labw[m], 0) + 1
    run(pairmap, u"윈도우 4수준 + Full 포함(w5 는 배치 2벌)", labw, rows,
        a.perm, a.seed + 10,
        note=u"군 크기 " + ", ".join("w%d:%d" % (k, cw[k])
                                    for k in sorted(cw))
             + u"  (w5 = Full 5 + K=5 5)")

    # (1c) **배치 없는 비교** — 같은 팔 쌍을 전부 버리고, 팔이 다른 쌍만
    #      쓴다. 그러면 양쪽이 모두 '다른 배치' 비교이고 차이는 윈도우
    #      값이 같은지뿐이다. 정확검정: 다섯 팔에 라벨 [3,5,5,7,10] 을
    #      붙이는 서로 다른 배정 5!/2! = 60 가지를 전수.
    print(u"\n" + "-" * 74)
    print(u"배치를 뺀 비교 — 팔이 다른 쌍만 쓴다(양쪽 다 독립 학습 배치)")
    print("-" * 74)
    warms = sorted(WARMS)
    armpair = {}
    bym = {}
    for m in msw:
        bym.setdefault(arm_of(m), []).append(m)
    for i in range(len(warms)):
        for j in range(i + 1, len(warms)):
            A, B = warms[i], warms[j]
            armpair[(A, B)] = [pairmap[(x, y)]
                               for x in bym[A] for y in bym[B]]

    def s_batchfree(assign):
        same, diff = [], []
        for (A, B), vs in armpair.items():
            (same if assign[A] == assign[B] else diff).extend(vs)
        if not same or not diff:
            return None, 0, 0, None, None
        sm = sum(same) / float(len(same))
        dm = sum(diff) / float(len(diff))
        return sm - dm, len(same), len(diff), sm, dm

    obs_bf, nsame, ndiff, sm, dm = s_batchfree(
        dict((x, WARMS[x]) for x in warms))
    print(u"  같은 윈도우·다른 배치  %3d쌍  평균 %8.4f%%  (= Full x K=5)"
          % (nsame, sm))
    print(u"  다른 윈도우·다른 배치  %3d쌍  평균 %8.4f%%" % (ndiff, dm))
    print(u"  S_batchfree = %+.4f %%p" % obs_bf)
    labels_ms = [WARMS[x] for x in warms]
    seen, ge, tot = set(), 0, 0
    for perm in itertools.permutations(labels_ms):
        if perm in seen:
            continue
        seen.add(perm)
        tot += 1
        s = s_batchfree(dict(zip(warms, perm)))[0]
        if s is not None and s >= obs_bf - 1e-12:
            ge += 1
    p_bf = ge / float(tot)
    print(u"  정확검정: 서로 다른 라벨 배정 %d가지 중 %d가지가 관측 이상"
          u"  -> p = %.4f  (최소 가능 p = %.4f)"
          % (tot, ge, p_bf, 1.0 / tot))
    rows.append(dict(test=u"배치 제외: 같은윈도우·다른배치 대 다른윈도우",
                     n_models=len(msw), n_within=nsame, n_between=ndiff,
                     within_mean="%.4f" % sm, between_mean="%.4f" % dm,
                     stat="%+.4f" % obs_bf, p_perm="%.4f" % p_bf,
                     n_perm=tot))

    # (2) K=10 은 런이 1개뿐 -> 군내 쌍 0. 빼고 3수준
    lab3 = dict((m, lab[m]) for m in ms if lab[m] != 10)
    run(pairmap, u"윈도우 3수준 K=3/5/7 (K=10 제외)", lab3, rows,
        a.perm, a.seed + 1,
        note=u"K=10 은 런 1개라 군내 쌍에 기여하지 못한다")

    # (3) 기계 검증 — 13팔 전부를 라벨로 쓰면 29.3 전역 검정과 같아야 한다
    allms = sorted(set(x for k in pairmap for x in k))
    laba = dict((m, arm_of(m)) for m in allms)
    run(pairmap, u"대조(기계검증): 13팔 전부 = 29.3 전역", laba, rows,
        a.perm, a.seed + 2,
        note=u"FINDINGS 29.3 의 +11.4706 %p / p 9.999e-05 와 같아야 한다")

    # (4) 시간 가중치 제거 / 균등 가중치 — 같은 척도에 올린다
    for nm, arms in ((u"W-Temporal 대 Full", {"W-Temporal": 0, "Full": 1}),
                     (u"W-Equal 대 Full", {"W-Equal": 0, "Full": 1})):
        mm = models_of(pairmap, set(arms))
        run(pairmap, nm, dict((m, arms[arm_of(m)]) for m in mm), rows,
            a.perm, a.seed + 3)

    # (4b) Holm — 족 크기를 두 가지로 병기한다 (족이 소속을 뒤집은 전례가 있다)
    import review_stats as RS
    subs = [r for r in rows if not r["test"].startswith(u"대조")]
    ps = [float(r["p_perm"]) for r in subs]
    print(u"\n" + "-" * 74)
    print(u"Holm 보정 — 족 정의를 병기한다")
    print("-" * 74)
    for fam, idx in ((u"족A = 실질 검정 전부", range(len(subs))),
                     (u"족B = 중첩 검정(3수준) 제외",
                      [i for i, r in enumerate(subs)
                       if u"3수준" not in r["test"]])):
        sel = [ps[i] for i in idx]
        adj = RS.holm_bonferroni(sel)
        print(u"  %s" % fam)
        for i, pa in zip(idx, adj):
            print(u"    %-34s p=%.5f -> p_holm=%.5f %s"
                  % (subs[i]["test"], ps[i], pa,
                     u"유의" if pa < 0.05 else u"기각"))
        for i, pa in zip(idx, adj):
            subs[i].setdefault("p_holm", {})
            subs[i]["p_holm"][fam] = pa
    for r in subs:
        d = r.pop("p_holm", {})
        r["p_holm_famA"] = "%.5f" % d.get(u"족A = 실질 검정 전부",
                                          float("nan"))
        v = d.get(u"족B = 중첩 검정(3수준) 제외")
        r["p_holm_famB"] = "" if v is None else "%.5f" % v
    for r in rows:
        r.setdefault("p_holm_famA", "")
        r.setdefault("p_holm_famB", "")

    # (4c) 거리-일치 관계 — 윈도우 '값' 이 의미가 있으면 가까운 값끼리
    #      더 일치해야 한다. 안 그러면 '팔이 뭉친다' 일 뿐이다.
    print(u"\n" + "-" * 74)
    print(u"윈도우 거리 대 경로 일치 — 값이 의미가 있는가")
    print("-" * 74)
    karms = sorted(KARMS, key=lambda s: KARMS[s])
    bymodel = {}
    for m in ms:
        bymodel.setdefault(arm_of(m), []).append(m)
    dmat, dist, agr = [], [], []
    for i in range(len(karms)):
        for j in range(i + 1, len(karms)):
            A, B = karms[i], karms[j]
            vs = [pairmap[(x, y)] for x in bymodel[A] for y in bymodel[B]]
            mu = sum(vs) / float(len(vs))
            d = abs(KARMS[A] - KARMS[B])
            dmat.append((A, B, d, len(vs), mu))
            dist.append(d)
            agr.append(mu)
    print(u"  팔쌍        |dK|  쌍수   평균일치율")
    for A, B, d, n, mu in dmat:
        print(u"  %-5s-%-5s  %2d   %4d   %8.4f%%" % (A, B, d, n, mu))
    print(u"  군내(참고)  %-13s %8.4f%%"
          % ("", sum(float(r["within_mean"]) for r in rows[:1])))

    def pearson(xs, ys):
        n = float(len(xs))
        mx = sum(xs) / n
        my = sum(ys) / n
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        sxx = sum((x - mx) ** 2 for x in xs)
        syy = sum((y - my) ** 2 for y in ys)
        if sxx <= 0 or syy <= 0:
            return float("nan")
        return sxy / (sxx * syy) ** 0.5

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk

    r_p = pearson(dist, agr)
    r_s = pearson(rank(dist), rank(agr))
    with open(os.path.join(OUT, "window_policy_distance.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["arm_a", "arm_b", "abs_dK", "n_pairs", "agree_pct"])
        for A, B, d, n, mu in dmat:
            w.writerow([A, B, d, n, "%.4f" % mu])
        w.writerow(["(pearson_r)", "", "", "", "%+.4f" % r_p])
        w.writerow(["(spearman_rho)", "", "", "", "%+.4f" % r_s])
    print(u"  Pearson r(|dK|, 일치율) = %+.4f   Spearman rho = %+.4f"
          u"   (팔쌍 %d개)" % (r_p, r_s, len(dmat)))
    print(u"  -> 음수여야 '윈도우 값이 가까우면 정책도 가깝다' 가 선다.")

    # (4d) 팔 하나 빼기 — K=3 의 10런이 S 를 끌고 가는지
    print(u"\n" + "-" * 74)
    print(u"팔 하나 빼기 (leave-one-arm-out) — S 가 한 팔에 기대는가")
    print("-" * 74)
    loo = []
    for drop in karms:
        sub = dict((m, lab[m]) for m in ms if arm_of(m) != drop)
        gs = set(sub.values())
        if len(gs) < 2:
            continue
        o, nw, nb, wm, bm = stat(pairmap, sub)
        if o is None:
            print(u"  -%-5s  군내 쌍 0 -> 계산 불가" % drop)
            continue
        print(u"  -%-5s  모델 %2d  군내 %3d쌍 %8.4f%%  군간 %3d쌍 %8.4f%%"
              u"  S = %+8.4f" % (drop, len(sub), nw, wm, nb, bm, o))
        loo.append([drop, len(sub), nw, "%.4f" % wm, nb, "%.4f" % bm,
                    "%+.4f" % o])
    with open(os.path.join(OUT, "window_policy_loo.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["dropped_arm", "n_models", "n_within", "within_mean",
                    "n_between", "between_mean", "stat"])
        for r in loo:
            w.writerow(r)

    # (5) 최빈 정책 안의 윈도우 구성
    print(u"\n" + "-" * 74)
    print(u"최빈 정책 안의 윈도우 구성 — 윈도우 값이 정책을 결정하지"
          u" 않는다는 직접 근거")
    print("-" * 74)
    with open(CLASSES) as f:
        cl = list(csv.DictReader(f))
    big = max(cl, key=lambda r: int(r["n_models"]))
    byarm = {}
    for m in big["models"].split():
        byarm.setdefault(arm_of(m), []).append(m)
    tot = {}
    with open(WITHIN) as f:
        for r in csv.DictReader(f):
            tot[r["arm"]] = int(r["n_runs"])
    print(u"  정책 #%s: 모델 %s개, 팔 %s개"
          % (big["policy_id"], big["n_models"], big["n_arms"]))
    for arm in sorted(byarm):
        print(u"    %-10s %d / %d 런" % (arm, len(byarm[arm]),
                                         tot.get(arm, -1)))
    kin = sorted([x for x in byarm if x.startswith("K=")],
                 key=lambda s: int(s[2:]))
    print(u"  -> 이 정책 하나에 윈도우 %s 가 **함께** 들어 있다."
          % ", ".join(kin))

    # (6) 각 윈도우 팔의 런들이 어떤 정책 묶음에 떨어지나 —
    #     '팔내 응집' 이 그 팔 고유의 것인지, 팔을 가로지르는 묶음인지.
    print(u"\n" + "-" * 74)
    print(u"윈도우 팔의 응집은 그 팔 고유인가 — 정책 묶음 구성")
    print("-" * 74)
    cls_of, members_of = {}, {}
    for r in cl:
        mm2 = r["models"].split()
        members_of[r["policy_id"]] = mm2
        for m in mm2:
            cls_of[m] = r["policy_id"]
    for arm in karms:
        here = sorted(bymodel[arm])
        groups = {}
        for m in here:
            groups.setdefault(cls_of[m], []).append(m)
        parts = []
        for pid in sorted(groups, key=lambda p: -len(groups[p])):
            others = set(arm_of(x) for x in members_of[pid]) - set([arm])
            parts.append(u"#%s(%d런%s)"
                         % (pid, len(groups[pid]),
                            u", 공유: " + " ".join(sorted(others))
                            if others else u", 단독"))
        print(u"  %-5s %2d런 -> 정책 %d개: %s"
              % (arm, len(here), len(groups), "  ".join(parts)))

    path = os.path.join(OUT, "window_policy_test.csv")
    with open(path, "w") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(u"\n저장: %s (%d행)" % (path, len(rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
