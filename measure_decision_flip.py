# -*- coding: utf-8 -*-
"""조건이 **결정을 뒤집는 비율**을 잰다 — 기하와 피해 사이의 빠진 고리.

왜 필요한가
-----------
23.9.6 의 기하 요약 네 개(l1/cos/rho/노름비)는 모두 **50칸 전역** 평균이다.
그런데 정책은 한 번의 결정에서 **현재 노드에 인접한 칸만** 읽는다. 전역
기하가 흔들려도 **결정이 안 바뀌면** 경로도 안 바뀌고 피해도 없다. 그래서
"전역 기하가 피해를 설명한다" 에는 빠진 고리가 하나 있다.

이 측정의 특별한 점: **배터리를 모든 조건이 공유한다**. 조건마다 **같은
상태 목록**을 보므로 23.9.6.14 가 쟀던 조건 간 환경 갈라짐이 **구조적으로
없다**. 23.9.6 의 조건 간 비교에 매번 붙던 전제가 여기서는 필요 없다.

무엇을 재는가 (모델 10개 × 조건마다, 배터리 상태 전부)
- `flip`: 참 상태의 `masked_argmax` 와 먹인 상태의 그것이 다른 비율.
- `loc_argmin`: 결정에 쓰이는 칸(현재 노드 - 유효 이웃) 중 **위험 최소**
  엣지가 바뀌는 비율.
- `loc_inv`: 그 칸들 안의 **불일치 쌍 비율**(국소 순위 교란).
- `loc_cos` / `glob_cos`: 국소·전역 코사인.

현재 노드는 상태 **50:75 의 argmax** 로 복원한다. `get_state` 가
[50 위험][25 현재][25 목적지][25 방문] 순이고(`network_env.py`), 위험 블록은
`EDGES` 순서 그대로이며 `normalize_edge` 는 `tuple(sorted(...))` 다.

참 벡터는 `np.float32` 로 맞춰 비교한다. `build_battery` 가 float32 를
돌려주고(`occlude_state_blocks.py:138`) `make_fed` 도 float32 사본을
돌려주므로, `intact` 는 **비트 단위로 같다** — 즉 아래 예측 (1) 은
반올림 때문에 깨질 수 없다.

숫자 보기 전 예측(이 프로젝트 규약 — 적어 놓고 본다)
-----------------------------------------------------
1. `intact` 의 `flip` 은 **정확히 0**. 아니면 측정이 틀렸다.
2. `risk_rankpow05n`·`risk_scale*`·`risk_rankpow05` 는 **단조**변환이라
   국소 순위가 완벽히 보존된다 → `loc_inv` = 0 · `loc_argmin` = 0 이
   **정확히** 나와야 한다. 그런데도 피해가 +94.72 / +53.26 ms
   (23.9.6.15 [4])라면, 피해는 순위가 아니라 **크기 재배치가 Q 를
   흔드는 경로**로 들어온다.
3. `risk_scale184`/`054` 는 코사인이 **정확히 1**(배율은 방향을 안 바꿈)
   이다. 그 조건의 `flip` 이 0 이 아니면 "코사인이 피해를 설명한다" 에
   직접 반례가 된다 — 23.9.6.15 가 `scale184` 에서 본 **음의 부호**
   (-12.01 ms)와 맞춰 읽어야 한다.

한계
----
- 배터리는 `intact` 정책이 걸은 경로에서 모은 상태다. 손상된 조건은
  **다른 경로**를 걷기 때문에 실제로 보는 상태 분포가 다르다. 그래서
  이 `flip` 은 "같은 상태를 줬을 때 첫 결정이 바뀌는 비율" 이고,
  "경로 전체가 갈라지는 양" 이 아니다. 후자는 23.9.6.15 의 경로 동일성
  열(`=risk` 69.8%→1.1%)이 이미 재고 있다.
- 조건 n 이 작아 [3] 의 순위상관은 **서술용**이다(검정 아님).

출력: review_results/decision_flip.csv
환경변수: 없음 (배터리는 OC.BATTERY_N 기본값을 그대로 쓴다)
"""
import csv
import io
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402

sys.path.append(OC.ORIG)
# OC 는 import 시 sys.path 에서 HERE 를 빼낸다(occlude_state_blocks.py:36).
sys.path.append(HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import analyze_perturb_vs_damage as AP  # noqa: E402
import measure_rank_vs_value as RV  # noqa: E402
import reanalyze_cluster_n as RC  # noqa: E402
import review_stats as RS  # noqa: E402
from topology import EDGES, NUM_NODES  # noqa: E402

LO, HI = RV.LO, RV.HI
# 기본값은 23.9.6.16 이 인용하는 **그 11개**다. `DF_CONDS` 로 덮어쓸 수
# 있게만 열어 둔다 — 인자 없이 돌리면 전과 동일하다. 조건을 바꿔 볼 때는
# `DF_OUT` 도 같이 바꿔서 `decision_flip.csv` 를 건드리지 말 것
# ([B] 의 Holm 족 1 과 [E] 의 rho n=10 이 그 파일에서 나온다).
_CONDS_DEF = ("intact", "risk_scale184", "risk_scale054", "risk_rankpow05",
              "risk_rankpow05n", "risk_frozen", "risk_dimshuffle",
              "risk_shuf08", "risk_shuf16", "risk_shuf25", "risk_shuf32")
_CONDS_ENV = os.environ.get("DF_CONDS", "").strip()
CONDS = (tuple(c.strip() for c in _CONDS_ENV.split(",") if c.strip())
         if _CONDS_ENV else _CONDS_DEF)
OUT_NAME = os.environ.get("DF_OUT", "decision_flip.csv")
# 상태 레이아웃: [위험 50][현재 25][목적지 25][방문 25]
CUR0 = HI
EIDX = dict((tuple(sorted(e)), i) for i, e in enumerate(EDGES))
assert len(EIDX) == HI - LO, "엣지 50개 가정이 깨졌다"
DAMAGE_CSV = os.path.join(OC.OUT_DIR, "damage_magnitude.csv")
# 23.9.6.15 [4] 의 세 척도 중 '초과'. 원시 지연은 조건별 기준이 달라
# 비교에 안 쓰고, frac 은 Jensen 격차 때문에 비율평균을 써야 한다.
DAMAGE_SCALE = "지연 초과"
QTYS = ("flip", "loc_argmin", "loc_inv", "loc_cos", "glob_cos")


def local_slots(state, valid):
    """결정에 쓰이는 칸 번호. 현재 노드와 유효 이웃을 잇는 엣지들."""
    cur = int(np.argmax(np.asarray(state[CUR0:CUR0 + NUM_NODES])))
    out = []
    for v in valid:
        k = tuple(sorted((cur, int(v))))
        if k in EIDX:
            out.append(EIDX[k])
    return out


def inv_frac(tv, fv):
    """불일치 쌍 비율. 참 쪽 동점 쌍은 분모에서 뺀다."""
    n = len(tv)
    tot = bad = 0
    for i in range(n):
        for j in range(i + 1, n):
            if tv[i] == tv[j]:
                continue
            tot += 1
            if (tv[i] < tv[j]) != (fv[i] < fv[j]):
                bad += 1
    return (float(bad) / tot) if tot else float("nan")


def cos_of(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    den = np.sqrt((a ** 2).sum()) * np.sqrt((b ** 2).sum())
    return float(np.dot(a, b) / den) if den > 0 else float("nan")


def load_damage():
    """23.9.6.15 [4] 의 조건별 피해 추정치. 없으면 빈 dict."""
    out = {}
    if not os.path.exists(DAMAGE_CSV):
        return out
    with io.open(DAMAGE_CSV, encoding="utf-8") as fh:
        rd = csv.reader(fh)
        hdr = None
        for row in rd:
            if not row:
                hdr = None
                continue
            if row[0] == "block":
                hdr = row
                continue
            if hdr is None or len(row) != len(hdr):
                continue
            r = dict(zip(hdr, row))
            if r.get("block") != "cond_vs_intact":
                continue
            sc = r.get("scale") or r.get("b") or ""
            if DAMAGE_SCALE not in sc:
                continue
            key = r.get("condition") or r.get("a") or ""
            try:
                out[key] = (float(r.get("estimate")),
                            float(r.get("p_holm")))
            except (TypeError, ValueError):
                pass
    return out


def measure(net, device, battery, base, donors, perms, cond):
    """조건 하나. 배터리 전체 평균을 돌려준다."""
    acc = dict((k, 0.0) for k in QTYS)
    nb = len(battery)
    nd = ninv = 0
    for i, (st, valid) in enumerate(battery):
        fed = RV.make_fed(st, cond, donors[i], perms[i])
        a = OC.masked_argmax_from_q(OC.qvals(net, fed, device), list(valid))
        if a != base[i]:
            acc["flip"] += 1.0
        acc["glob_cos"] += cos_of(fed[LO:HI], st[LO:HI])
        sl = local_slots(st, valid)
        if len(sl) < 2:
            continue
        nd += 1
        tv = [float(st[j]) for j in sl]
        fv = [float(fed[j]) for j in sl]
        if int(np.argmin(fv)) != int(np.argmin(tv)):
            acc["loc_argmin"] += 1.0
        acc["loc_cos"] += cos_of(fv, tv)
        iv = inv_frac(tv, fv)
        if iv == iv:
            ninv += 1
            acc["loc_inv"] += iv
    dn = max(nd, 1)
    return {
        "flip": acc["flip"] / nb,
        "glob_cos": acc["glob_cos"] / nb,
        "loc_argmin": acc["loc_argmin"] / dn,
        "loc_cos": acc["loc_cos"] / dn,
        "loc_inv": acc["loc_inv"] / max(ninv, 1),
        "ndec": nd,
    }


def dump_csv(res, mnames, battery, rows, bar):
    """per_model 섹션 + (있으면) 검정 섹션을 OUT_NAME 으로 쓴다."""
    path = os.path.join(OC.OUT_DIR, OUT_NAME)
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["block", "condition", "model", "flip", "loc_argmin",
                    "loc_inv", "loc_cos", "glob_cos", "n_decisions",
                    "battery_n"])
        for c in CONDS:
            for n in mnames:
                r = res[c][n]
                w.writerow(["per_model", c, n, "%.6f" % r["flip"],
                            "%.6f" % r["loc_argmin"], "%.6f" % r["loc_inv"],
                            "%.6f" % r["loc_cos"], "%.6f" % r["glob_cos"],
                            int(r["ndec"]), len(battery)])
        w.writerow([])
        w.writerow(["block", "quantity", "estimate", "t", "p_holm",
                    "holm_family", "n"])
        w.writerows(rows)
    print("")
    print("저장: %s" % path)
    print(bar)


def main():
    device = torch.device("cpu")
    OC.check_env()
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    models = []
    for arm, d, pat in OC.ARMS:
        for r in OC.RUNS:
            q = os.path.join(d, pat % r)
            if os.path.exists(q):
                models.append(("%s_run%02d" % (arm, r), q))
    if len(models) != 10:
        raise SystemExit("엄격 비교군 10개가 아니다: %d" % len(models))
    ref = OC.load_net(models[0][1], device)
    battery = OC.build_battery("greedy", device, ref)
    pool = [s for s, _ in battery]
    bar = "=" * 92
    print(bar)
    print("결정 전환율 — 배터리를 **모든 조건이 공유**한다")
    print("  배터리 %d 상태 · 모델 %d · 조건 %d"
          % (len(battery), len(models), len(CONDS)))
    print("  조건 간 환경 차이가 **구조적으로 0** 이다(23.9.6.14 무관)")
    print(bar)

    # 공여·치환을 상태마다 하나씩 뽑되 **조건 간 동일**하게 쓴다.
    # measure_shuffle_geometry.py 와 같은 패턴·같은 시드.
    rng = random.Random(RV.EVAL_SEED)
    donors, perms = [], []
    for _ in battery:
        donors.append(pool[rng.randrange(len(pool))])
        perms.append(np.random.RandomState(
            rng.randrange(1 << 30)).permutation(HI - LO))

    res = dict((c, {}) for c in CONDS)
    for name, ckpt in models:
        net = OC.load_net(ckpt, device)
        base = [OC.masked_argmax_from_q(OC.qvals(net, st, device),
                                        list(valid))
                for st, valid in battery]
        for cond in CONDS:
            res[cond][name] = measure(net, device, battery, base,
                                      donors, perms, cond)
        print("  %-20s 완료 (결정칸>=2 인 상태 %d)"
              % (name, res["intact"][name]["ndec"]))

    mnames = [n for n, _ in models]

    def cm(cond, k):
        return sum(res[cond][n][k] for n in mnames) / len(mnames)

    dmg = load_damage()
    print("")
    print("[1] 조건별 — 모델 10개 평균")
    print("-" * 92)
    print("%-20s %8s %11s %9s %9s %9s %10s %9s"
          % ("조건", "flip", "loc_argmin", "loc_inv", "loc_cos",
             "glob_cos", "피해(ms)", "p_holm"))
    for c in CONDS:
        dv = dmg.get(c)
        print("%-20s %7.2f%% %10.2f%% %8.4f %9.4f %9.4f %10s %9s"
              % (c, 100 * cm(c, "flip"), 100 * cm(c, "loc_argmin"),
                 cm(c, "loc_inv"), cm(c, "loc_cos"), cm(c, "glob_cos"),
                 ("%+.2f" % dv[0]) if dv else "-",
                 RS.fmt_p(dv[1]) if dv else "-"))
    if not dmg:
        print("  (피해 열 비었다 — damage_magnitude.csv 의 척도명을"
              " 확인할 것)")

    if not set(_CONDS_DEF) <= set(CONDS):
        print("")
        print("  [주의] `DF_CONDS` 로 조건을 **바꿔서** 돌렸다. [2]~[5]")
        print("  는 기본 11조건(단조 4개·배율 2개·frozen·pow05n)을 이름")
        print("  으로 찾으므로 **건너뛴다**. 이 실행은 [1] 의 기술 통계와")
        print("  per_model CSV 만 낸다 — 23.9.6.16 이 인용하는 검정값은")
        print("  **기본 실행에서만** 나온다(족 크기가 조건 수에 걸린다).")
        dump_csv(res, mnames, battery, [], bar)
        return

    print("")
    print("[2] 예측 점검 — 숫자 보기 전에 적어 둔 세 가지")
    print("-" * 92)
    f0 = cm("intact", "flip")
    print("  (1) intact flip = %.6f%%  →  %s"
          % (100 * f0, "통과" if f0 == 0.0 else "**실패 — 측정 결함**"))
    mono = ("risk_rankpow05n", "risk_rankpow05", "risk_scale184",
            "risk_scale054")
    ok2 = True
    for c in mono:
        li, la = cm(c, "loc_inv"), cm(c, "loc_argmin")
        good = (li == 0.0 and la == 0.0)
        ok2 = ok2 and good
        print("  (2) %-17s loc_inv %.6f · loc_argmin %.6f%% · flip"
              " %5.2f%% · 피해 %8s  %s"
              % (c, li, 100 * la, 100 * cm(c, "flip"),
                 ("%+.2f" % dmg[c][0]) if c in dmg else "-",
                 "" if good else "**예측 위반**"))
    if ok2:
        print("      → 단조 조건 넷 모두 국소 순위가 **완전히 보존된다**.")
        print("        그런데도 결정이 바뀐다면, 피해는 순위가 아니라")
        print("        **크기 재배치가 Q 를 흔드는 경로**로 들어온다.")
    s1, s2 = cm("risk_scale184", "glob_cos"), cm("risk_scale054", "glob_cos")
    print("  (3) scale184/054 glob_cos = %.6f / %.6f (배율은 방향 불변)"
          % (s1, s2))
    print("      그 조건의 flip = %.2f%% / %.2f%%"
          % (100 * cm("risk_scale184", "flip"),
             100 * cm("risk_scale054", "flip")))

    print("")
    print("[3] 어느 양이 피해를 따라가는가 — 조건 수준 순위상관")
    print("-" * 92)
    have = [c for c in CONDS if c in dmg]
    print("  피해를 아는 조건 %d개: %s" % (len(have), ", ".join(have)))
    if len(have) >= 4:
        ys = [dmg[c][0] for c in have]
        for k in QTYS:
            print("    rho(%-11s, 피해) = %+.4f"
                  % (k, AP.spearman([cm(c, k) for c in have], ys)))
        print("  주의: 조건 n=%d 이라 **서술용**이다(검정 아님)." % len(have))
    else:
        print("  → 4점 미만이라 생략.")

    print("")
    print("[3b] 근거를 줄여 다시 — 노이즈 조건 배제와 **동점 구조**")
    print("-" * 92)
    print("  [3] 은 피해가 **유의하지 않은** 조건까지 줄 사이에 넣었다.")
    print("  그런 조건의 점추정치는 노이즈다. 떼고 다시 본다.")
    sig = [c for c in have if dmg[c][1] < 0.05]
    print("  유의한 조건 %d개: %s" % (len(sig), ", ".join(sig)))
    if len(sig) >= 4:
        ys = [dmg[c][0] for c in sig]
        for k in QTYS:
            print("    rho(%-11s, 피해) = %+.4f"
                  % (k, AP.spearman([cm(c, k) for c in sig], ys)))
    else:
        print("    → 4점 미만이라 생략.")
    print("")
    print("  동점 구조 — 어떤 양이 어떤 조건 쌍을 **원래 못 가르는가**.")
    print("  지표가 두 조건에서 같은 값이면 그 쌍의 피해 상하는")
    print("  원리적으로 맞힐 수 없다(표본이 아니라 구조적 한계다).")
    for k in QTYS:
        ties = []
        for i in range(len(have)):
            for j in range(i + 1, len(have)):
                a, b = have[i], have[j]
                if abs(cm(a, k) - cm(b, k)) < 1e-9:
                    gap = abs(dmg[a][0] - dmg[b][0])
                    ties.append("%s=%s(피해간격 %.1f)"
                                % (a.replace("risk_", ""),
                                   b.replace("risk_", ""), gap))
        print("    %-11s 동점 %d쌍%s"
              % (k, len(ties), ("  " + " · ".join(ties)) if ties else ""))

    print("")
    print("[4] frozen 대 pow05n — 모델 대응차")
    print("-" * 92)
    print("  23.9.6.15 [5b] 는 이 쌍을 **피해**로 못 갈랐다(p_holm 0.4817).")
    print("  같은 쌍을 **결정 전환**으로 갈라 본다. 배터리가 공유되므로")
    print("  이 비교엔 환경 교란이 아예 없다.")
    print("")
    # 어떤 양이 **모델에 의지하는가**를 먼지 재고
    # 거기에만 검정을 둔다. 기하 양 넷은 상태+교란만의
    # 함수라 모델 간 편차가 0 이다 — 그런 양에 t 를 돌리면
    # sd=0 이라 t=inf 가 나온다. 가정이 아니라 **재서** 가른다.
    spread = {}
    for k in QTYS:
        sp = 0.0
        for c in CONDS:
            vs = [res[c][n][k] for n in mnames]
            sp = max(sp, max(vs) - min(vs))
        spread[k] = sp
    tested = [k for k in QTYS if spread[k] > 0.0]
    const = [k for k in QTYS if spread[k] == 0.0]
    print("  모델 간 편차(전 조건 중 최대): "
          + " · ".join("%s %.8f" % (k, spread[k]) for k in QTYS))
    print("  → 검정 대상 %d개(%s) · 모델 무관 상수 %d개(%s)"
          % (len(tested), ", ".join(tested) or "-",
             len(const), ", ".join(const) or "-"))
    print("    상수 쪽에는 표본변동이 **없다**. p 를 적지 않는다.")
    print("")
    rows = []
    if tested:
        ps, st2 = [], {}
        for k in tested:
            d = [res["risk_frozen"][n][k] - res["risk_rankpow05n"][n][k]
                 for n in mnames]
            mu, _sd, t, _df, pv = RC.one_sample(d)
            ps.append(pv)
            st2[k] = (mu, t, sum(1 for v in d if v > 0), len(d))
        hh = RS.holm_bonferroni(ps)
        print("  검정 — Holm 족 %d" % len(ps))
        print("  %-14s %12s %9s %11s %8s"
              % ("양", "차", "t", "p_holm", "부호"))
        for k, ph in zip(tested, hh):
            mu, t, npos, nn = st2[k]
            print("%s %-14s %12.6f %9.3f %11s  %d/%d"
                  % ("**" if ph < 0.05 else "  ", k, mu, t, RS.fmt_p(ph),
                     npos, nn))
            rows.append(["frozen_vs_pow05n", k, "%.6f" % mu, "%.4f" % t,
                         "%.6f" % ph, len(ps), nn])
    if const:
        print("")
        print("  모델 무관 상수 — 정확한 값, 검정 아님")
        print("  %-14s %12s" % ("양", "차"))
        for k in const:
            mu = (res["risk_frozen"][mnames[0]][k]
                  - res["risk_rankpow05n"][mnames[0]][k])
            print("     %-14s %12.6f" % (k, mu))
            rows.append(["frozen_vs_pow05n_const", k, "%.6f" % mu,
                         "", "", 0, 1])
    if "flip" in tested:
        d = [res["risk_frozen"][n]["flip"]
             - res["risk_rankpow05n"][n]["flip"] for n in mnames]
        npos = sum(1 for v in d if v > 0)
        print("")
        print("  읽기: 모델에 의지하는 양은 `flip` 하나다. 그걸로는")
        print("  이 쌍이 **여전히 어긋나지 않는다**"
              "(차 %+.2f%%p · 부호 %d/10)."
              % (100 * sum(d) / len(d), npos))
        print("  즉 설계한 쌍은 **환경 교란이 0 인** 상황에서도 전환")
        print("  으로 갈리지 않는다(23.9.6.16 [A]: 피해 방향이 반대였다).")

    dump_csv(res, mnames, battery, rows, bar)


if __name__ == "__main__":
    main()
