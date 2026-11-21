# -*- coding: utf-8 -*-
"""31.10 의 마지막 한계 한 줄을 재려고 만들었다.

    경로 일치는 '같다/다르다' 의 이진 비교다. 다른 경로가 **얼마나** 다른지는
    재지 않았다.

왜 중요한가. 그 이진 비교가 초안 본문에 들어가 있다
(`DRAFT_4_6_1_temporal.md` 83~91행, 그리고 120행에서 "같은 정책" 의 정의를
공저자 확인 항목으로 올려 뒀다):

    이력 길이 3·5·7·10 으로 학습된 모델들이 **같은 정책 하나**에 함께 속한다
    ... 이력 길이가 같고 학습 배치가 다른 쌍의 경로 일치율이 ... 낮다
    (정확검정 p = 0.80).

이진화는 정보를 버린다. 같은 윈도우 쌍이 **달라도 덜 다르다**면, 31.4 의
-2.68 %p 영가설은 등급 효과를 가린 것일 수 있다 - 즉 이건 검정력 문제이고,
등급 거리로 다시 재면 원고 문장이 유지되거나 **고쳐야 한다**. 어느 쪽이든
이진 비교만으로는 알 수 없다.

재는 것. 69개 모델(표 10 모수 전수)의 평가 200 에피소드 노드 열을
`check_path_identity.rollout_paths` 로 그대로 받아, 쌍마다 이진 일치율 대신
**등급 거리 넷**을 센다:

    d_edit        노드 열 사이 편집거리(Levenshtein). 이진값은 1[d_edit>0] 의
                  평균이므로 이게 그 일반화다.
    d_jacc_edge   간선 집합 Jaccard 거리 - 순서를 버리고 '어느 링크를 썼나' 만.
    d_hop         |홉수 차| - 길이만.
    shared_prefix 갈라지기 전까지 공유한 노드 수(같은 출발점에서 시작하므로
                  '어디서 결정이 갈렸나'다). 거리가 아니라 유사도다.

방향 규약. 31.4 의 통계량은 S = 군내평균 - 군간평균이고 단측 검정이
`s >= obs` 다(일치율이므로 군내가 **클** 때 양수). 거리는 부호가 반대이므로
**유사도 = -거리**로 넣어 같은 방향을 유지한다. 그래야 같은 코드
(`window_policy_test.perm_test`)를 재사용할 수 있다.

검증 둘을 통과해야 아래 숫자를 쓴다:
  V1 내 롤아웃으로 다시 센 2346쌍의 이진 `same_pct` 가 커밋된
     `path_identity_all_pairs.csv` 와 **정확히** 같아야 한다(같은 롤아웃을
     보고 있다는 뜻 - 다르면 거리도 다른 정책을 재고 있다).
  V2 미도달 자리(None)를 쌍에서 빼는 규약이 이진 쪽과 같아야 한다
     (`check_path_identity` 는 `pa[i] is not None and pa[i] == pb[i]` 로
     세므로, 둘 다 미도달이면 '같다' 로 세지 않는다).

원본은 읽기만 한다. 롤아웃은 비싸므로 `policy_paths_all.csv` 에 캐시하고
`--from-csv` 로 거리만 다시 계산할 수 있게 했다.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_policy_path_distance.py
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_policy_path_distance.py --from-csv
"""
from __future__ import print_function

import argparse
import csv
import io
import itertools
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "SDN_DQN_new_env"))

RESULT_DIR = os.path.join(HERE, "review_results")
PATHS_CSV = os.path.join(RESULT_DIR, "policy_paths_all.csv")
PAIR_CSV = os.path.join(RESULT_DIR, "policy_path_distance_pairs.csv")
ARM_CSV = os.path.join(RESULT_DIR, "policy_path_distance_arms.csv")
TEST_CSV = os.path.join(RESULT_DIR, "policy_path_distance_tests.csv")
REF_PAIRS = os.path.join(RESULT_DIR, "path_identity_all_pairs.csv")

EPISODES = 200

# 등급 거리 넷 + 이진 하나. similarity = -distance 로 넣는 것은 SIGN 에서 정한다.
METRICS = ("same_pct", "d_edit", "d_jacc_edge", "d_hop", "shared_prefix")
SIGN = {"same_pct": +1.0, "d_edit": -1.0, "d_jacc_edge": -1.0,
        "d_hop": -1.0, "shared_prefix": +1.0}


# --------------------------------------------------------------- 거리

def edit_distance(a, b):
    """노드 열 사이 Levenshtein. 길이가 25 이하라 단순 DP 로 충분하다."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[lb]


def edge_set(p):
    return set(frozenset((p[i], p[i + 1])) for i in range(len(p) - 1))


def shared_prefix(a, b):
    n = min(len(a), len(b))
    k = 0
    while k < n and a[k] == b[k]:
        k += 1
    return k


def pair_distance(pa, pb):
    """두 모델의 200 에피소드를 같은 자리끼리 비교한다.

    `check_path_identity` 와 같은 규약: 자리 i 는 둘 다 도달했을 때만 센다.
    이진 일치율은 그 스크립트와 **정확히** 같은 식으로 센다(V1 용).
    """
    n = min(len(pa), len(pb))
    same = 0
    acc = {"d_edit": 0.0, "d_jacc_edge": 0.0, "d_hop": 0.0,
           "shared_prefix": 0.0}
    cacc = dict(acc)
    valid = 0
    diff = 0
    for i in range(n):
        a, b = pa[i], pb[i]
        if a is not None and a == b:
            same += 1
        if a is None or b is None:
            continue
        valid += 1
        ed = edit_distance(a, b)
        ea, eb = edge_set(a), edge_set(b)
        un = len(ea | eb)
        jd = 1.0 - (len(ea & eb) / float(un)) if un else 0.0
        hd = abs(len(a) - len(b))
        sp = shared_prefix(a, b)
        acc["d_edit"] += ed
        acc["d_jacc_edge"] += jd
        acc["d_hop"] += hd
        acc["shared_prefix"] += sp
        if ed > 0:
            diff += 1
            cacc["d_edit"] += ed
            cacc["d_jacc_edge"] += jd
            cacc["d_hop"] += hd
            cacc["shared_prefix"] += sp
    out = {"episodes": n, "same": same,
           "same_pct": (100.0 * same / n) if n else float("nan"),
           "n_valid": valid, "n_diff": diff,
           "diff_pct": (100.0 * diff / valid) if valid else float("nan")}
    for k in acc:
        out[k] = (acc[k] / valid) if valid else float("nan")
        out["cond_" + k] = (cacc[k] / diff) if diff else float("nan")
    return out


# --------------------------------------------------------------- 롤아웃

def rollout_all(episodes):
    import check_path_identity as CPI

    targets = CPI.build_targets_all()
    print("대상 모델 %d개 (표 10 모수 전수), 에피소드 %d"
          % (len(targets), episodes), flush=True)
    store = {}
    for n, (label, path, rw, wkey, zeroed, _key) in enumerate(targets, 1):
        if not os.path.exists(path):
            print("  [skip] %-14s 모델 없음" % label, flush=True)
            continue
        paths, _means = CPI.rollout_paths(path, rw, wkey, zeroed, episodes)
        store[label] = paths
        uniq = len(set(p for p in paths if p is not None))
        miss = sum(1 for p in paths if p is None)
        print("  [%2d/%d] %-14s 고유경로 %2d  미도달 %d"
              % (n, len(targets), label, uniq, miss), flush=True)
    return store


def save_paths(store):
    with io.open(PATHS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "ep", "path"])
        for label in sorted(store):
            for i, p in enumerate(store[label]):
                w.writerow([label, i,
                            "" if p is None else "-".join(str(x) for x in p)])
    print("\n저장: %s" % PATHS_CSV)


def load_paths():
    store = {}
    with io.open(PATHS_CSV, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            s = r["path"]
            p = None if s == "" else tuple(int(x) for x in s.split("-"))
            store.setdefault(r["label"], []).append((int(r["ep"]), p))
    out = {}
    for k, v in store.items():
        v.sort()
        out[k] = [p for _i, p in v]
    return out


# --------------------------------------------------------------- 집계

def arm_of(label):
    return label.rsplit("-r", 1)[0]


def build_pairs(store):
    labels = sorted(store)
    rows = []
    for a, b in itertools.combinations(labels, 2):
        d = pair_distance(store[a], store[b])
        d["a"], d["b"] = a, b
        d["arm_a"], d["arm_b"] = arm_of(a), arm_of(b)
        d["same_arm"] = 1 if d["arm_a"] == d["arm_b"] else 0
        rows.append(d)
    return rows


PAIR_FIELDS = ["a", "b", "arm_a", "arm_b", "same_arm", "episodes", "same",
               "same_pct", "n_valid", "n_diff", "diff_pct",
               "d_edit", "d_jacc_edge", "d_hop", "shared_prefix",
               "cond_d_edit", "cond_d_jacc_edge", "cond_d_hop",
               "cond_shared_prefix"]


def verify_binary(rows):
    ref = {}
    with io.open(REF_PAIRS, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            ref[(r["a"], r["b"])] = (int(r["same"]), float(r["same_pct"]))
    checked, bad = 0, []
    for r in rows:
        k = (r["a"], r["b"])
        if k not in ref:
            k = (r["b"], r["a"])
        if k not in ref:
            continue
        checked += 1
        s0, p0 = ref[k]
        if s0 != r["same"] or abs(p0 - r["same_pct"]) > 5e-5:
            bad.append((k, s0, r["same"], p0, r["same_pct"]))
    return checked, bad


def arm_summary(rows):
    out = []
    arms = sorted(set(r["arm_a"] for r in rows) | set(r["arm_b"] for r in rows))
    for arm in arms:
        win = [r for r in rows if r["arm_a"] == arm and r["arm_b"] == arm]
        bet = [r for r in rows
               if (r["arm_a"] == arm) != (r["arm_b"] == arm)]
        if not win:
            continue

        def mn(sub, k):
            vs = [r[k] for r in sub if r[k] == r[k]]
            return sum(vs) / len(vs) if vs else float("nan")

        out.append({
            "arm": arm, "n_within": len(win), "n_between": len(bet),
            "within_same_pct": mn(win, "same_pct"),
            "between_same_pct": mn(bet, "same_pct"),
            "within_d_edit": mn(win, "d_edit"),
            "between_d_edit": mn(bet, "d_edit"),
            "within_cond_d_edit": mn(win, "cond_d_edit"),
            "between_cond_d_edit": mn(bet, "cond_d_edit"),
            "within_d_jacc": mn(win, "d_jacc_edge"),
            "between_d_jacc": mn(bet, "d_jacc_edge"),
            "within_shared_prefix": mn(win, "shared_prefix"),
            "between_shared_prefix": mn(bet, "shared_prefix"),
        })
    return out


ARM_FIELDS = ["arm", "n_within", "n_between", "within_same_pct",
              "between_same_pct", "within_d_edit", "between_d_edit",
              "within_cond_d_edit", "between_cond_d_edit", "within_d_jacc",
              "between_d_jacc", "within_shared_prefix",
              "between_shared_prefix"]


# --------------------------------------------------------------- 검정

KARMS = {"K=3": 3, "K=5": 5, "K=7": 7, "K=10": 10}


def pairmap_of(rows, metric):
    """similarity = SIGN * value. 31.4 의 단측 방향을 유지한다."""
    m = {}
    s = SIGN[metric]
    for r in rows:
        v = r[metric]
        if v != v:
            continue
        m[(r["a"], r["b"])] = s * v
        m[(r["b"], r["a"])] = s * v
    return m


def models_in(store, arms):
    return sorted(m for m in store if arm_of(m) in arms)


def batchfree_exact(pairmap, models, warms):
    """31.4 의 '배치 제외' 정확검정 - 같은 윈도우·다른 배치 대 다른 윈도우.

    `window_policy_test.main` 안의 `s_batchfree` 와 같은 설계다. 거기서는 중첩
    함수라 import 할 수 없어 같은 식을 다시 적었다(설계는 바꾸지 않았다).
    """
    bym = {}
    for m in models:
        bym.setdefault(arm_of(m), []).append(m)
    arms = sorted(warms)
    armpair = {}
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            A, B = arms[i], arms[j]
            vs = [pairmap[(x, y)] for x in bym[A] for y in bym[B]
                  if (x, y) in pairmap]
            if vs:
                armpair[(A, B)] = vs

    def s_of(assign):
        same, diff = [], []
        for (A, B), vs in armpair.items():
            (same if assign[A] == assign[B] else diff).extend(vs)
        if not same or not diff:
            return None, 0, 0, None, None
        sm = sum(same) / float(len(same))
        dm = sum(diff) / float(len(diff))
        return sm - dm, len(same), len(diff), sm, dm

    obs, nsame, ndiff, sm, dm = s_of(dict((x, warms[x]) for x in arms))
    labels = [warms[x] for x in arms]
    seen, ge, tot = set(), 0, 0
    for perm in itertools.permutations(labels):
        if perm in seen:
            continue
        seen.add(perm)
        tot += 1
        s = s_of(dict(zip(arms, perm)))[0]
        if s is not None and s >= obs - 1e-12:
            ge += 1
    return obs, ge / float(tot), nsame, ndiff, sm, dm, tot


def report(rows, store):
    import window_policy_test as WPT
    from review_stats import holm_bonferroni

    print("\n" + "=" * 78)
    print("V1 - 이진 same_pct 가 커밋된 path_identity_all_pairs.csv 를 재현하는가")
    checked, bad = verify_binary(rows)
    print("  대조 %d쌍 · 불일치 %d쌍 -> %s"
          % (checked, len(bad),
             "같은 롤아웃을 보고 있다" if not bad else "**어긋난다**"))
    for k, s0, s1, p0, p1 in bad[:5]:
        print("    %s: same %d 대 %d, pct %.4f 대 %.4f" % (k, s0, s1, p0, p1))

    print("\n" + "=" * 78)
    print("[1] 팔별 - 이진 일치율과 등급 거리를 나란히")
    print("%-12s %5s %9s %9s %9s %9s %9s" % (
        "arm", "n_in", "일치%(내)", "일치%(간)", "편집(내)", "편집(간)",
        "조건부(내)"))
    for r in arm_summary(rows):
        print("%-12s %5d %9.2f %9.2f %9.4f %9.4f %9.4f" % (
            r["arm"], r["n_within"], r["within_same_pct"],
            r["between_same_pct"], r["within_d_edit"], r["between_d_edit"],
            r["within_cond_d_edit"]))

    print("\n" + "=" * 78)
    print("[2] 다른 경로는 얼마나 다른가 - 불일치 에피소드만 (전체 쌍 풀)")
    cd = [r for r in rows if r["n_diff"] > 0]
    for k, lab in (("cond_d_edit", "편집거리"),
                   ("cond_d_jacc_edge", "간선 Jaccard"),
                   ("cond_d_hop", "홉수 차"),
                   ("cond_shared_prefix", "공유 접두 길이")):
        vs = sorted(r[k] for r in cd if r[k] == r[k])
        n = len(vs)
        print("  %-14s n=%4d  평균 %7.4f  중앙 %7.4f  최소 %7.4f  최대 %7.4f"
              % (lab, n, sum(vs) / n, vs[n // 2], vs[0], vs[-1]))
    allv = [r["diff_pct"] for r in rows if r["diff_pct"] == r["diff_pct"]]
    print("  불일치 에피소드 비율: 평균 %.2f%% (쌍 %d개)"
          % (sum(allv) / len(allv), len(allv)))

    print("\n" + "=" * 78)
    print("[3] 31.4 와 같은 설계, 등급 거리로 - 윈도우 4수준 라벨 순열")
    msw = models_in(store, set(KARMS))
    labw = dict((m, KARMS[arm_of(m)]) for m in msw)
    trows, ps, names = [], [], []
    for metric in METRICS:
        pm = pairmap_of(rows, metric)
        obs, p, nw, nb, wm, bm = WPT.perm_test(pm, labw, 10000, 20261120)
        print("  %-14s 군내 %3d쌍 %+9.4f · 군간 %3d쌍 %+9.4f · S=%+8.4f"
              "  p_perm=%.4f" % (metric, nw, wm, nb, bm, obs, p))
        trows.append(dict(test="윈도우 4수준", metric=metric, n_models=len(msw),
                          n_within=nw, n_between=nb,
                          within_mean="%.6f" % wm, between_mean="%.6f" % bm,
                          stat="%+.6f" % obs, p_perm="%.5f" % p,
                          n_perm=10000, family_size=len(METRICS)))
        ps.append(p)
        names.append("4수준 " + metric)

    print("\n[4] 31.4 의 배치 제외 정확검정, 등급 거리로"
          " (같은 윈도우·다른 배치 = Full x K=5)")
    warms = dict(KARMS)
    warms["Full"] = 5
    msb = models_in(store, set(warms))
    for metric in METRICS:
        pm = pairmap_of(rows, metric)
        obs, p, ns, nd, sm, dm, tot = batchfree_exact(pm, msb, warms)
        print("  %-14s 같은윈도우 %3d쌍 %+9.4f · 다른윈도우 %3d쌍 %+9.4f"
              "  S=%+8.4f  p=%.4f (배정 %d, 최소 p %.4f)"
              % (metric, ns, sm, nd, dm, obs, p, tot, 1.0 / tot))
        trows.append(dict(test="배치 제외 정확검정", metric=metric,
                          n_models=len(msb), n_within=ns, n_between=nd,
                          within_mean="%.6f" % sm, between_mean="%.6f" % dm,
                          stat="%+.6f" % obs, p_perm="%.5f" % p,
                          n_perm=tot, family_size=len(METRICS)))
        ps.append(p)
        names.append("배치제외 " + metric)

    print("\n[5] Holm - 족 %d (검정 %d개, 지표 %d개 x 설계 2개)"
          % (len(ps), len(ps), len(METRICS)))
    adj = holm_bonferroni(ps)
    for name, p, ph in zip(names, ps, adj):
        print("  %-26s p=%.4f  p_holm=%.4f %s"
              % (name, p, ph, "유의" if ph < 0.05 else ""))

    with io.open(TEST_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "test", "metric", "n_models", "n_within", "n_between",
            "within_mean", "between_mean", "stat", "p_perm", "n_perm",
            "family_size", "p_holm"])
        w.writeheader()
        for r, ph in zip(trows, adj):
            r["p_holm"] = "%.5f" % ph
            w.writerow(r)
    print("\n저장: %s (%d행)" % (TEST_CSV, len(trows)))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--episodes", type=int, default=EPISODES)
    a = ap.parse_args()

    if a.from_csv:
        store = load_paths()
        print("캐시에서 읽음: 모델 %d개" % len(store))
    else:
        store = rollout_all(a.episodes)
        save_paths(store)

    rows = build_pairs(store)
    with io.open(PAIR_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=PAIR_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with io.open(ARM_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=ARM_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in arm_summary(rows):
            w.writerow(r)
    print("저장: %s (%d쌍)" % (PAIR_CSV, len(rows)))
    print("저장: %s" % ARM_CSV)
    return report(rows, store)


if __name__ == "__main__":
    raise SystemExit(main())
