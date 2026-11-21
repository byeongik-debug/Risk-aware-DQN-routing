# -*- coding: utf-8 -*-
"""순위 가설이 이 교란 족에서 **식별 가능한가**를 잰다 (롤아웃 없음).

왜 필요한가
-----------
23.9.6.16 은 설계한 쌍(`shuf25` 대 `pow05n`)으로 순위 교란을 가르려다
**양방향 모두 실패**했다. 코사인·노름·주변분포는 맞췄지만 `l1`(0.1358 대
0.2685)과 건드린 칸(46.8% 대 100.0%)이 남았고, 그 둘만으로 관측된 방향
(섞은 쪽이 **덜** 망가짐)이 설명되기 때문이다. 그래서 한계로 "코사인과
l1 을 둘 다 맞추고 순위만 다른 조건이 필요하다(미정)" 를 적었다.

이 스크립트는 그 **미정을 닫는다** — 그런 조건이 **만들어질 수 있는가**를
교란의 기하만으로 묻는다. 정책도 모델도 필요 없다(피해를 재는 게 아니다).

무엇을 재는가
-------------
`[1] m=2 전수조사` — 칸 두 개만 건드리는 **최소** 교란. 불변량 넷
(노름비 1, 코사인, l1, 건드린 칸 수)을 **정확히** 고정하면 남는 자유도가
무엇인지 전수로 센다. 대수적으로는 이렇게 좁혀진다:

  d 가 칸 k,l 에서만 α,β 이고  α²+β² = 2g²  (노름비 1 + 코사인 고정)
  이면서                      |α|+|β| = 2g  (l1 고정)
  → (|α|−|β|)² = 2(α²+β²) − (|α|+|β|)² = 0 → |α| = |β| = g.

즉 부호와 위치만 남는다. 거기에 <d,v> = −g² 를 걸고 **전수로** 센 뒤,
각 해가 (ㄱ) 순위를 지키는지 (ㄴ) 값 범위를 벗어나지 않는지 본다.
가설이 식별 가능하려면 "순위를 지키면서 불변량 넷이 같은" 해가 **하나는**
있어야 한다.

`[2] m>2 닫힌형 탐색` — 칸을 더 많이 건드리면 자유도가 생긴다. 그래서
**순위를 지키는** 교란 중에서 코사인·노름을 shuffle 과 **정확히** 맞춘
것들을 모아 `l1` 이 어디까지 갈 수 있는지 구간으로 잰다. 순위를 지키는
방향은 다음으로 만든다:

  u 를 v 의 오름차순 자리에 **비내림차순**으로 놓고  d = a·u + b·v
  (a ≥ 0, b ≥ −1) → v + d = (1+b)v + a·u 는 **같은 순서**를 유지한다.

제약 둘을 걸면 a 가 닫힌형으로 떨어진다(교차항이 지워진다):

  <d,v> = P,  |d|² = D  →  a² = (D − P²/|v|²) / (|u|² − <u,v>²/|v|²)
  b = (P − a<u,v>) / |v|²

이 족을 N_DIR 번 추출해 `l1` 의 도달 범위를 재고, shuffle 의 `l1` 이 그
안에 있는지 본다. 안에 있으면 **조건을 만들 수 있다**(다음 실험이 생긴다).
없으면 이 족에서 순위는 **식별 불가**다(가설을 접어야 한다).

한계
----
- `[1]` 은 건드린 칸이 **정확히 2** 인 교란에 대한 전수조사다. 칸이 더
  많으면 결론이 그대로 가지 않는다 — 그래서 `[2]` 를 따로 둔다.
- `[2]` 는 건드린 칸 수를 **맞추지 않는다**(a·u + b·v 는 50칸을 전부
  건드린다). 맞춘 축은 코사인·노름이고, 재는 축은 `l1` 이다. 건드린 칸은
  **측정해서 같이 적는다**(맞춘 게 아니다).
- `[2]` 의 `l1` 범위는 **추출 하한**이다(도달 가능 집합의 최소/최대를
  증명한 게 아니라 N_DIR 번 추출해서 본 것이다).
- 순위 보존은 **엄격한 쌍 순서 일치**로 판정한다(동점으로 가는 것도
  순위 변화로 센다). v 안에 이미 있는 동점 쌍은 분모에서 뺀다.

출력: review_results/rank_identifiability.csv
환경변수: RID_STATES(기본 20) · RID_PAIRS(기본 30) · RID_DIRS(기본 4000)
"""
import csv
import io
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402

sys.path.append(OC.HERE)
import measure_rank_vs_value as RV  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LO, HI = RV.LO, RV.HI
NSLOT = HI - LO
N_STATE = int(os.environ.get("RID_STATES", "20"))
N_PAIR = int(os.environ.get("RID_PAIRS", "30"))
N_DIR = int(os.environ.get("RID_DIRS", "4000"))
KS = (16, 25, 32)
# 불변량 일치 판정 허용오차. 위험값은 0~1 규모라 1e-9 는 배정도에서
# 값 차이를 못 숨긴다(값 자체가 1e-9 보다 훨씬 크게 벌어져 있다).
TOL = 1e-9
# [4] 의 사이 구간. [3] 이 "목표를 넘기는 최소 지지크기 = 중앙 40칸" 을
# 냈으므로 33~49 를 훑는다(49 는 50칸 바로 아래, 33 은 [3] 상한이 거의
# 안 통과하는 쪽).
MGRID4 = tuple(int(x) for x in os.environ.get(
    "RID_M4", "33,36,40,44,49").split(","))
N_SUP4 = int(os.environ.get("RID_M4_SUP", "12"))
N_DIR4 = int(os.environ.get("RID_M4_DIR", "16"))


def n_inv(v, w):
    """순서가 **깨진** 쌍 수와 분모. v 의 동점 쌍은 분모에서 뺀다."""
    dv = np.sign(v[:, None] - v[None, :])
    dw = np.sign(w[:, None] - w[None, :])
    up = np.triu(np.ones((len(v), len(v)), dtype=bool), 1)
    live = up & (dv != 0)
    return int((live & (dv != dw)).sum()), int(live.sum())


def feasible_m2(v, g):
    """칸 2개 교란 중 불변량 넷이 전부 맞는 것을 **전수로** 찾는다.

    |α| = |β| = g 는 대수적으로 강제되므로(문서 참고) 남은 것은 위치
    (k, l) 와 부호 둘이다. 조건은 <d,v> = −g², 즉 sa·v_k + sb·v_l = −g.
    """
    out = []
    for sa in (1.0, -1.0):
        for sb in (1.0, -1.0):
            m = sa * v[:, None] + sb * v[None, :] + g
            hit = np.argwhere(np.triu(np.abs(m) < TOL, 1))
            for k, ll in hit:
                out.append((int(k), int(ll), sa, sb))
    return out


def room_ub(v, sup):
    """지지집합 `sup` 에서 순위를 지키며 갈 수 있는 |d|² 의 **상한**.

    정렬 순서에서 `sup` 안의 칸들이 연달아 붙어 있으면 그 덩어리는 함께
    움직일 수 있으므로, 덩어리를 감싸는 **`sup` 밖** 이웃 둘이 경계다
    (끝이면 0 과 1). 덩어리 안의 각 칸은 그 경계까지밖에 못 가므로
    칸별 최대 이동거리의 제곱합이 |d|² 의 상한이 된다. 칸별로 독립하게
    최대를 줘서 느슨하게 잡은 값이므로 **상한임이 보장된다**.
    """
    order = np.argsort(v, kind="stable")
    inset = np.zeros(NSLOT, dtype=bool)
    inset[np.asarray(sup, dtype=np.int64)] = True
    tot = 0.0
    p = 0
    while p < NSLOT:
        if not inset[order[p]]:
            p += 1
            continue
        q = p
        while q + 1 < NSLOT and inset[order[q + 1]]:
            q += 1
        lo = float(v[order[p - 1]]) if p > 0 else 0.0
        hi = float(v[order[q + 1]]) if q + 1 < NSLOT else 1.0
        for t in range(p, q + 1):
            vi = float(v[order[t]])
            tot += max(vi - lo, hi - vi) ** 2
        p = q + 1
    return tot


def mono_dir(v, rng):
    """v 의 오름차순 자리에 비내림차순으로 놓은 비음수 방향 u."""
    order = np.argsort(v, kind="stable")
    raw = np.sort(rng.random(NSLOT) ** rng.uniform(0.2, 5.0))
    u = np.empty(NSLOT, dtype=np.float64)
    u[order] = raw
    return u


def solve_ab(u, v, nv, pp, dd):
    """<d,v>=pp, |d|²=dd 를 만족하는 d = a·u + b·v 의 (a, b) 목록.

    교차항이 지워져 a² 가 닫힌형으로 떨어진다. 부호 두 뿌리를 **둘 다**
    돌려준다. `b >= -1` 로 걸러내지 **않는다** — 그건 순위 보존의
    *충분* 조건일 뿐이고, 그걸로 걸러 버리면 k 가 커질 때 표본이 말라
    (12101 -> 229) `l1` 도달 범위를 아래로 편향시킨다. 순위는 호출한
    쪽에서 `n_inv` 로 **직접 확인**한다.
    """
    s = float(np.dot(u, v))
    uu = float(np.dot(u, u))
    den = uu - (s * s) / nv
    num = dd - (pp * pp) / nv
    if den <= 0.0 or num < 0.0:
        return []
    a0 = float(np.sqrt(num / den))
    out = []
    for a in (a0, -a0):
        out.append((a, (pp - a * s) / nv))
    return out


def l1_on_sup(v, u, sup, nvs, pp, dd, root=0):
    """지지집합 `sup` 에 가둔 d = a·u_S + b·v_S 의 (l1, d). 없으면 None.

    제약 <d,v> = pp · |d|² = dd 는 d 가 S 밖에서 0 이므로 S 위의 내적만
    쓴다 → `solve_ab` 를 u_S, v_S, nv = |v_S|² 로 그대로 부른다.
    """
    us = np.zeros(NSLOT, dtype=np.float64)
    vs = np.zeros(NSLOT, dtype=np.float64)
    us[sup] = u[sup]
    vs[sup] = v[sup]
    ab = solve_ab(us, vs, nvs, pp, dd)
    if len(ab) <= root:
        return None
    a, b = ab[root]
    d = a * us + b * vs
    return float(np.abs(d).mean()), d


def bisect_l1(v, u1, u2, sup, nvs, pp, dd, tgt, root=0, it=48):
    """u1~u2 를 잇는 방향 족에서 l1 = tgt 인 해를 이분으로 찾는다.

    u(t) = (1−t)·u1 + t·u2 는 비음수 선형보간이라 족 안에 머문다. l1(t) 는
    연속이므로 양 끝이 목표를 끼면 중간값 정리로 해가 **있다**. 찾은 해에서
    순위와 값 범위를 다시 확인한다(연속성은 l1 만 보장한다).
    """
    lo, hi = 0.0, 1.0
    r1 = l1_on_sup(v, u1, sup, nvs, pp, dd, root)
    r2 = l1_on_sup(v, u2, sup, nvs, pp, dd, root)
    if r1 is None or r2 is None:
        return None
    if (r1[0] - tgt) * (r2[0] - tgt) > 0.0:
        return None
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        rm = l1_on_sup(v, (1.0 - mid) * u1 + mid * u2, sup, nvs, pp, dd,
                       root)
        if rm is None:
            return None
        if (r1[0] - tgt) * (rm[0] - tgt) <= 0.0:
            hi = mid
        else:
            lo = mid
            r1 = rm
    rf = l1_on_sup(v, (1.0 - hi) * u1 + hi * u2, sup, nvs, pp, dd, root)
    return rf


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()
    net = OC.load_net(os.path.join(OC.ORIG_MODELS, "window5_run01.pth"),
                      device)
    pool = [s for s, _ in OC.build_battery("greedy", device, net)]
    bar = "=" * 92
    print(bar)
    print("순위 가설의 **식별 가능성** — 교란 기하만으로 묻는다")
    print("  배터리 상태 %d · 블록 칸 %d · 상태표본 %d"
          % (len(pool), NSLOT, N_STATE))
    print(bar)

    rng = random.Random(RV.EVAL_SEED)
    nprng = np.random.default_rng(RV.EVAL_SEED)
    states = [np.asarray(pool[rng.randrange(len(pool))][LO:HI],
                         dtype=np.float64) for _ in range(N_STATE)]
    allv = np.concatenate(states)
    print("")
    print("  위험값 범위 %.6f ~ %.6f (평균 %.6f)"
          % (allv.min(), allv.max(), allv.mean()))
    rows = []

    print("")
    print("[1] 칸 2개 교란 — 불변량 넷을 고정하고 **전수조사**")
    print("-" * 92)
    print("  고정: 노름비 1 · 코사인 · l1 · 건드린 칸 수 = 2")
    print("  묻는 것: 그 넷이 같은데 **순위를 지키는** 해가 있는가")
    tot = okrank = okdom = okboth = swaps = 0
    for v in states:
        idx = [(i, j) for i in range(NSLOT) for j in range(i + 1, NSLOT)
               if abs(v[i] - v[j]) > TOL]
        rng.shuffle(idx)
        for (i, j) in idx[:N_PAIR]:
            g = abs(v[i] - v[j])
            for (k, ll, sa, sb) in feasible_m2(v, g):
                d = np.zeros(NSLOT)
                d[k] = sa * g
                d[ll] = sb * g
                w = v + d
                bad, _den = n_inv(v, w)
                dom = bool(w.min() >= -TOL)
                tot += 1
                if bad == 0:
                    okrank += 1
                if dom:
                    okdom += 1
                if bad == 0 and dom:
                    okboth += 1
                # 교환인가? 두 칸의 값이 서로 바뀌었는지로 판정한다.
                if (abs(w[k] - v[ll]) < TOL and abs(w[ll] - v[k]) < TOL):
                    swaps += 1
    print("")
    print("  불변량 넷이 맞는 해 **%d개** (상태 %d × 목표쌍 %d)"
          % (tot, N_STATE, N_PAIR))
    print("    그중 값 범위 안(위험 ≥ 0)      : %d" % okdom)
    print("    그중 두 칸 값의 **교환**        : %d" % swaps)
    print("    그중 순위를 지키는 것           : %d" % okrank)
    print("    범위 안 **그리고** 순위 지킴    : **%d**" % okboth)
    if okboth == 0:
        print("")
        print("  → 칸 2개에서는 불변량 넷을 맞추는 순간 해가 **교환뿐**")
        print("    이고 교환은 반드시 순위를 깬다. 이 규모에서 순위는")
        print("    **식별 불가**다(검정력 부족이 아니라 구성상 없다).")
    else:
        print("")
        print("  → 순위를 지키는 해가 **있다**. 조건을 만들 수 있다.")
    rows.append(["m2_exhaustive", "", tot, okdom, swaps, okrank, okboth,
                 N_STATE, N_PAIR, ""])

    print("")
    print("[2] 칸을 더 건드리면 — 순위 지키며 코사인·노름 맞추고 l1 은")
    print("    어디까지 가는가")
    print("-" * 92)
    print("  **상태별로** 판정한다: 상태 s 의 도달 구간이 그 상태 자신의")
    print("  shuffle `l1` 을 품는가. 상태를 모아 min/max 를 찍으면 어떤")
    print("  상태의 목표가 그 상태 범위 밖인 걸 가릴 수 있다.")
    print("")
    print("%4s %10s %22s %7s %9s %8s"
          % ("k", "목표 l1 중앙", "상태별 도달구간(중앙)", "품은 상태",
             "해/시도", "순위깨짐"))
    for kk in KS:
        cond = "risk_shuf%02d" % kk
        if cond not in RV.SHUFK:
            continue
        los, his, tgts, inside_n = [], [], [], 0
        nb = nsol = ntry = 0
        for v in states:
            st = np.zeros(HI, dtype=np.float32)
            st[LO:HI] = v.astype(np.float32)
            perm = np.random.RandomState(
                rng.randrange(1 << 30)).permutation(NSLOT)
            fed = np.asarray(RV.make_fed(st, cond, st, perm)[LO:HI],
                             dtype=np.float64)
            dsh = fed - v
            nv = float(np.dot(v, v))
            pp = float(np.dot(dsh, v))
            dd = float(np.dot(dsh, dsh))
            if dd <= 0.0:
                continue
            tgt_s = float(np.abs(dsh).mean())
            cand = []
            for _ in range(N_DIR):
                u = mono_dir(v, nprng)
                ntry += 1
                for a, b in solve_ab(u, v, nv, pp, dd):
                    nsol += 1
                    d = a * u + b * v
                    bad, _den = n_inv(v, v + d)
                    if bad:
                        nb += 1
                        continue
                    cand.append(float(np.abs(d).mean()))
            if not cand:
                continue
            lo_s, hi_s = min(cand), max(cand)
            tgts.append(tgt_s)
            los.append(lo_s)
            his.append(hi_s)
            if lo_s - TOL <= tgt_s <= hi_s + TOL:
                inside_n += 1
        if not tgts:
            continue
        print("%4d %10.6f   [%.6f, %.6f] %5d/%-3d %9s %8d"
              % (kk, float(np.median(tgts)), float(np.median(los)),
                 float(np.median(his)), inside_n, len(tgts),
                 "%d/%d" % (nsol, ntry), nb))
        rows.append(["l1_range_per_state", cond,
                     "%.6f" % float(np.median(tgts)),
                     "%.6f" % float(np.median(los)),
                     "%.6f" % float(np.median(his)),
                     inside_n, len(tgts), nsol, nb,
                     "모든상태품음" if inside_n == len(tgts) else "일부밖"])

    print("")
    print("  읽기: 코사인과 노름은 **상태마다 정확히** 맞춰진다. 막는")
    print("  것은 l1 이다 — '품은 상태' 가 0 이면 도달 구간이 그 상태")
    print("  자신의 목표를 못 품는다는 뜻이고, 위 표에서 도달 구간은")
    print("  목표보다 **위**에 있다(고정 L2 에서 칸을 넓게 쓰면 L1 이")
    print("  커진다, L1 ≤ √n·L2). 여기서 건드린 칸 수는 **맞춰지지**")
    print("  **않는다**(50칸을 전부 건드린다) — 그 축을 포기해도 l1 이")
    print("  막는다는 게 이 블록의 결과다. 지지집합을 좁히는 쪽은 [4].")
    print("")
    print("  주의(자기 교정): 상태를 **모아서** min/max 를 찍으면 어떤")
    print("  상태의 최소와 다른 상태의 최대가 이어 붙어 '품는다' 는")
    print("  착시가 생긴다. 그래서 위 판정은 **상태별**이다.")

    print("")
    print("[3] 희소 교란은 어떤가 — **노름을 못 채운다**(표본 없이 상한)")
    print("-" * 92)
    print("  [1] 은 칸 2개, [2] 는 50칸을 봤다. 남은 건 3~49칸이다.")
    print("  여기선 추출하지 않고 **상한**을 계산한다: 순위를 지키려면")
    print("  칸 i 의 새 값은 정렬 이웃 사이에 머물러야 한다. 지지집합 S")
    print("  안에서 연달아 붙은 칸들은 함께 움직일 수 있으므로, 그 **덩어리**")
    print("  를 감싸는 S 밖 이웃 둘이 경계다. 그 경계까지가 각 칸이 갈 수")
    print("  있는 최대 거리이고, 제곱합이 |d|² 의 **엄격한 상한**이다.")
    print("  (노름비가 1 로 고정돼 있으니 코사인을 맞추는 것 = |d|² 를")
    print("  맞추는 것이다. 상한이 목표보다 작으면 **그 크기에서는")
    print("  코사인부터 못 맞춘다**.)")
    print("")
    print("%4s %12s %14s %10s %28s"
          % ("k", "shuf |d|²", "같은지지 상한", "채우나?",
             "목표를 넘는 최소 지지크기"))
    mgrid = (4, 8, 12, 16, 20, 25, 32, 40, 50)
    for kk in KS:
        cond = "risk_shuf%02d" % kk
        if cond not in RV.SHUFK:
            continue
        ddl, ubl, okn, mmin = [], [], 0, []
        for v in states:
            st = np.zeros(HI, dtype=np.float32)
            st[LO:HI] = v.astype(np.float32)
            perm = np.random.RandomState(
                rng.randrange(1 << 30)).permutation(NSLOT)
            fed = np.asarray(RV.make_fed(st, cond, st, perm)[LO:HI],
                             dtype=np.float64)
            dsh = fed - v
            dd = float(np.dot(dsh, dsh))
            if dd <= 0.0:
                continue
            ddl.append(dd)
            sup = np.flatnonzero(np.abs(dsh) > TOL)
            ub_same = room_ub(v, sup)
            ubl.append(ub_same)
            if ub_same >= dd:
                okn += 1
            got = None
            for m in mgrid:
                best = 0.0
                for _ in range(24):
                    s2 = nprng.choice(NSLOT, size=m, replace=False)
                    best = max(best, room_ub(v, s2))
                if best >= dd:
                    got = m
                    break
            mmin.append(got if got else NSLOT + 1)
        if not ddl:
            continue
        mm = float(np.median(mmin))
        print("%4d %12.6f %14.6f %10s %28s"
              % (kk, float(np.median(ddl)), float(np.median(ubl)),
                 "%d/%d" % (okn, len(ddl)),
                 ("중앙 %d칸 (섞은 칸 %d개)" % (mm, kk)) if mm <= NSLOT
                 else "50칸으로도 못 채움"))
        rows.append(["norm_room_ub", cond, "%.6f" % float(np.median(ddl)),
                     "%.6f" % float(np.median(ubl)), okn, len(ddl),
                     "%.1f" % mm, "", "",
                     "같은지지로는 코사인 불가" if okn == 0 else "일부가능"])
    print("")
    print("  읽기: 같은 지지집합(섞인 그 칸들)에서는 순위를 지키는 교란이")
    print("  shuffle 의 |d|² 에 **못 미친다**. 목표를 넘기려면 지지집합을")
    print("  훨씬 넓혀야 하는데, |d|² 가 고정된 상태에서 칸을 넓히면 l1 이")
    print("  같이 커진다 — 그게 [2] 에서 0/19·0/8·0/4 로 나온 바로 그")
    print("  현상이다. **두 끝이 서로 막는다**: 좁으면 코사인을 못 맞추고,")
    print("  넓히면 l1 이 넘친다.")

    print("")
    print("[4] 사이 구간(지지집합 %d~%d칸) — 셋을 **동시에** 맞출 수 있나"
          % (min(MGRID4), max(MGRID4)))
    print("-" * 92)
    print("  [3] 은 섞인 그 칸들만 쓰는 경우를 상한으로 막았고, [2] 는")
    print("  50칸에서 l1 이 넘치는 것을 봤다. 그 **사이**가 비어 있었다.")
    print("  여기선 d 를 크기 m 의 지지집합에 가두고 코사인·노름을 정확히")
    print("  맞춘 뒤(그래서 건드린 칸 수도 m 으로 맞는다), 방향을 움직여")
    print("  l1 이 목표를 **아래·위로 끼면 이분**해서 정확히 맞춘다.")
    print("  끼운 해에서 순위(n_inv=0)와 값 범위를 **다시** 확인한다.")
    print("")
    print("%4s %4s %9s %22s %9s %9s"
          % ("k", "m", "상한통과", "l1 도달구간(중앙)", "목표를 끼움",
             "**정확해**"))
    for kk in KS:
        cond = "risk_shuf%02d" % kk
        if cond not in RV.SHUFK:
            continue
        for m in MGRID4:
            n_ub = n_brk = n_hit = n_st = 0
            lol, hil, tgl = [], [], []
            for v in states:
                st = np.zeros(HI, dtype=np.float32)
                st[LO:HI] = v.astype(np.float32)
                perm = np.random.RandomState(
                    rng.randrange(1 << 30)).permutation(NSLOT)
                fed = np.asarray(RV.make_fed(st, cond, st, perm)[LO:HI],
                                 dtype=np.float64)
                dsh = fed - v
                dd = float(np.dot(dsh, dsh))
                if dd <= 0.0:
                    continue
                pp = float(np.dot(dsh, v))
                tgt = float(np.abs(dsh).mean())
                n_st += 1
                tgl.append(tgt)
                got_ub = brk = hit = False
                cand, keep = [], []
                for _ in range(N_SUP4):
                    sup = np.sort(nprng.choice(NSLOT, size=m,
                                               replace=False))
                    if room_ub(v, sup) < dd:
                        continue
                    got_ub = True
                    nvs = float(np.dot(v[sup], v[sup]))
                    if nvs <= 0.0:
                        continue
                    us = []
                    for _ in range(N_DIR4):
                        u = mono_dir(v, nprng)
                        r = l1_on_sup(v, u, sup, nvs, pp, dd)
                        if r is None:
                            continue
                        bad, _den = n_inv(v, v + r[1])
                        w = v + r[1]
                        ok = (bad == 0 and w.min() >= -TOL
                              and w.max() <= 1.0 + TOL)
                        us.append((u, r[0], ok))
                        if ok:
                            cand.append(r[0])
                    below = [x for x in us if x[1] < tgt]
                    above = [x for x in us if x[1] > tgt]
                    if below and above:
                        brk = True
                        for (ua, _la, _oa) in above[:3]:
                            for (ub, _lb, _ob) in below[:3]:
                                rf = bisect_l1(v, ub, ua, sup, nvs, pp,
                                               dd, tgt)
                                if rf is None:
                                    continue
                                if abs(rf[0] - tgt) > 1e-7:
                                    continue
                                bad, _den = n_inv(v, v + rf[1])
                                w = v + rf[1]
                                nnz = int((np.abs(rf[1]) > TOL).sum())
                                if (bad == 0 and w.min() >= -TOL
                                        and w.max() <= 1.0 + TOL
                                        and nnz == m):
                                    hit = True
                                    keep.append(rf[0])
                if got_ub:
                    n_ub += 1
                if brk:
                    n_brk += 1
                if hit:
                    n_hit += 1
                if cand:
                    lol.append(min(cand))
                    hil.append(max(cand))
            if not n_st:
                continue
            rng_s = ("[%.6f, %.6f]" % (float(np.median(lol)),
                                       float(np.median(hil)))
                     if lol else "(순위보존 해 없음)")
            print("%4d %4d %6d/%-2d %22s %6d/%-2d %6s%d/%-2d"
                  % (kk, m, n_ub, n_st, rng_s, n_brk, n_st,
                     "**", n_hit, n_st))
            rows.append(["mid_support", "%s_m%02d" % (cond, m),
                         "%.6f" % float(np.median(tgl)),
                         ("%.6f" % float(np.median(lol))) if lol else "",
                         ("%.6f" % float(np.median(hil))) if hil else "",
                         n_ub, n_brk, n_hit, n_st,
                         "정확해있음" if n_hit else "없음"])
    print("")
    print("  읽기: '상한통과' 는 그 m 에서 |d|² 를 채울 지지집합이 **있다**")
    print("  는 뜻이고(없으면 코사인부터 불가), '목표를 끼움' 은 l1 이")
    print("  목표를 사이에 둔다는 뜻이다.")
    print("")
    print("  **이 블록의 0 을 '불가능' 으로 읽으면 안 된다.** 여기서는")
    print("  지지집합을 **무작위로** 잡는데, 그러면 지지집합 **밖** 칸이")
    print("  안 움직이므로 안쪽 칸이 밖쪽 칸을 넘어가 순위가 깨진다 —")
    print("  m=33~44 에서 '순위보존 해 없음' 이 나온 게 그 증거다(상한은")
    print("  통과하는데도 그렇다). 즉 이 0 은 **내 구성 족이 그 규모에서")
    print("  순위를 지키게 만들어져 있지 않다**는 뜻이고 조건의 부재가")
    print("  **아니다**. 순위가 지켜질 수 있는 족으로 다시 묻는 게 [5].")

    print("")
    print("[5] 정렬 **연속** 지지집합 — 순위가 지켜질 수 있는 족에서")
    print("-" * 92)
    print("  [4] 의 무작위 지지집합은 m<50 에서 순위를 깬다. 여기선 v 의")
    print("  정렬 순서에서 **연속한 덩어리**를 지지집합으로 쓴다. 덩어리")
    print("  안의 값은 덩어리를 감싸는 밖 이웃 둘 사이에만 머물면 전체")
    print("  순서가 유지된다([3] 의 상한이 전제한 구조). 연속 덩어리는")
    print("  m 마다 %d-m+1 개이므로 **전수**로 훑고, solve_ab 의 두 뿌리"
          % NSLOT)
    print("  (a=+a0, -a0)도 **둘 다** 쓴다 — 족을 좁게 잡아 놓고 '없다' 고")
    print("  말하지 않기 위해서다.")
    print("")
    print("%4s %4s %9s %22s %9s %9s"
          % ("k", "m", "상한통과", "l1 도달구간(중앙)", "목표를 끼움",
             "**정확해**"))
    for kk in KS:
        cond = "risk_shuf%02d" % kk
        if cond not in RV.SHUFK:
            continue
        for m in MGRID4:
            n_ub = n_brk = n_hit = n_st = n_sol = 0
            lol, hil, tgl = [], [], []
            for v in states:
                st = np.zeros(HI, dtype=np.float32)
                st[LO:HI] = v.astype(np.float32)
                perm = np.random.RandomState(
                    rng.randrange(1 << 30)).permutation(NSLOT)
                fed = np.asarray(RV.make_fed(st, cond, st, perm)[LO:HI],
                                 dtype=np.float64)
                dsh = fed - v
                dd = float(np.dot(dsh, dsh))
                if dd <= 0.0:
                    continue
                pp = float(np.dot(dsh, v))
                tgt = float(np.abs(dsh).mean())
                n_st += 1
                tgl.append(tgt)
                order = np.argsort(v, kind="stable")
                got_ub = brk = hit = False
                cand = []
                for s0 in range(0, NSLOT - m + 1):
                    sup = np.sort(order[s0:s0 + m])
                    if room_ub(v, sup) < dd:
                        continue
                    got_ub = True
                    nvs = float(np.dot(v[sup], v[sup]))
                    if nvs <= 0.0:
                        continue
                    dirs = [mono_dir(v, nprng) for _ in range(N_DIR4)]
                    for root in (0, 1):
                        us = []
                        for u in dirs:
                            r = l1_on_sup(v, u, sup, nvs, pp, dd, root)
                            if r is None:
                                continue
                            bad, _den = n_inv(v, v + r[1])
                            w = v + r[1]
                            if (bad == 0 and w.min() >= -TOL
                                    and w.max() <= 1.0 + TOL):
                                n_sol += 1
                                cand.append(r[0])
                                us.append((u, r[0]))
                        below = [x for x in us if x[1] < tgt]
                        above = [x for x in us if x[1] > tgt]
                        if not (below and above):
                            continue
                        brk = True
                        for (ua, _la) in above[:3]:
                            for (ub, _lb) in below[:3]:
                                rf = bisect_l1(v, ub, ua, sup, nvs, pp,
                                               dd, tgt, root)
                                if rf is None or abs(rf[0] - tgt) > 1e-7:
                                    continue
                                bad, _den = n_inv(v, v + rf[1])
                                w = v + rf[1]
                                nnz = int((np.abs(rf[1]) > TOL).sum())
                                if (bad == 0 and w.min() >= -TOL
                                        and w.max() <= 1.0 + TOL
                                        and nnz == m):
                                    hit = True
                if got_ub:
                    n_ub += 1
                if brk:
                    n_brk += 1
                if hit:
                    n_hit += 1
                if cand:
                    lol.append(min(cand))
                    hil.append(max(cand))
            if not n_st:
                continue
            rng_s = ("[%.6f, %.6f]" % (float(np.median(lol)),
                                       float(np.median(hil)))
                     if lol else "(순위보존 해 없음)")
            print("%4d %4d %6d/%-2d %22s %6d/%-2d %6s%d/%-2d"
                  % (kk, m, n_ub, n_st, rng_s, n_brk, n_st,
                     "**", n_hit, n_st))
            rows.append(["contig_support", "%s_m%02d" % (cond, m),
                         "%.6f" % float(np.median(tgl)),
                         ("%.6f" % float(np.median(lol))) if lol else "",
                         ("%.6f" % float(np.median(hil))) if hil else "",
                         n_ub, n_brk, n_hit, n_st,
                         "정확해있음" if n_hit else "없음(해 %d개)" % n_sol])
    print("")
    print("  읽기: 이 족에서는 순위 보존이 **구조적으로 가능**하다. 그래도")
    print("  **정확해** 가 0 이면, 순위를 지키면서 코사인·노름을 맞춘 해의")
    print("  l1 이 목표를 **못 끼운다**는 뜻이다(위 도달구간과 목표를")
    print("  비교하면 방향이 보인다). 그러면 [3](좁은 쪽 노름 불가)과")
    print("  [2](넓은 쪽 l1 초과) 사이가 **이 족 안에서** 메워진다.")
    print("  여전히 '모든 교란에 대한 불가능 증명' 은 아니다 — 증명이 된")
    print("  것은 [1](칸 2개 전수)과 [3](같은 지지집합 상한)뿐이다.")

    path = os.path.join(OC.OUT_DIR, "rank_identifiability.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "condition", "v1", "v2", "v3", "v4", "v5",
                    "v6", "v7", "note"])
        for r in rows:
            w.writerow(r)
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
