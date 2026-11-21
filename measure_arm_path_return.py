# -*- coding: utf-8 -*-
"""28.4.1 이 남긴 "안 쟀다" 하나를 닫는다.

    이 분석은 **고정 경로끼리의** 비교다. 환경을 보는 팔(Full)은 에피소드마다
    경로를 바꿀 수 있으므로 같은 후보 집합에 올려 비교하지 않았다.

올릴 수 있다. `policy_paths_all.csv` 는 표 10 모수 **69개 전부**가 평가
200에피소드에서 실제로 걸은 경로를 담고 있고(`check_path_identity.rollout_paths`),
그 평가 규약은 내 스냅샷 규약과 **같다**(dynamic, EVAL_SEED+i, 워밍업 10,
epsilon 0). 그리고 13,800개 경로 전부가 0 에서 출발해 24 에 도착하고 **재방문이
없다** — 그래서 보상 함수의 재방문 항(-8.0)과 MAX_STEPS 항(-15.0)이 걸리지
않고, 고정 경로에 쓴 것과 **같은 식**으로 정확히 채점된다.

재는 것:
  (1) 환경을 보는 팔이 **환경 전맹 고정 경로**(0-4-8-14-15-21-22-24)보다 나은가.
      — 상태를 보는 값이 보상 단위로 얼마인가.
  (2) 환경을 보는 팔이 **평가 분포에서 가장 좋은 고정 경로**보다 나은가.
      — 적응이 '최선의 열린 루프' 를 이기는가. 이게 28.4.1 의 빈 칸이다.
  (3) 참조 천장: 에피소드마다 246 후보 중 최선을 고르는 **신탁**(oracle).

주의 — (2)의 '가장 좋은 고정 경로' 는 **같은 200에피소드에서 뽑았다.** 즉 그
경로는 이 표본에 맞춰 고른 것이고, 팔들은 그 표본을 못 봤다. 그래서 (2)는
팔에게 불리한 비교다. 결론을 쓸 때 그 비대칭을 같이 적는다.

후보 집합은 홉 <= 8(246개)로 둔다. 팔들의 경로는 7~12홉이므로 **후보 집합 밖의
경로도 채점한다**(스냅샷은 12홉까지 쌓는다). 후보 집합은 '최선의 고정 경로' 를
고르는 데만 쓴다.

원본은 읽기만 한다. 산출은 `review_results/` 로.

재현:
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 python -X utf8 \
        measure_arm_path_return.py
"""
from __future__ import print_function

import argparse
import collections
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import review_stats as RS                                   # noqa: E402
from measure_blind_path_optimality import (                  # noqa: E402
    EVAL_SEED, EVAL_WARMUP, OBSERVED, RESULT_DIR, RISK_WINDOW,
    build_adj, enumerate_simple_paths, hop_dist_to_dest, snapshots)
from measure_blind_path_paired import (                      # noqa: E402
    GAMMAS, episode_return, one_sample_p)

PATHS_CSV = os.path.join(RESULT_DIR, "policy_paths_all.csv")
OUT_MODELS = os.path.join(RESULT_DIR, "arm_path_return_models.csv")
OUT_TESTS = os.path.join(RESULT_DIR, "arm_path_return_tests.csv")


def load_paths():
    rows = list(csv.DictReader(io.open(PATHS_CSV, encoding="utf-8")))
    byrun = collections.defaultdict(dict)       # label -> ep -> path tuple
    maxhop = 0
    for r in rows:
        p = tuple(int(x) for x in r["path"].split("-"))
        assert p[0] == 0 and p[-1] == 24, r
        assert len(set(p)) == len(p), ("재방문", r)
        byrun[r["label"]][int(r["ep"])] = p
        maxhop = max(maxhop, len(p) - 1)
    return byrun, maxhop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--cand-hops", type=int, default=8)
    a = ap.parse_args()

    byrun, maxhop = load_paths()
    labels = sorted(byrun)
    eps = sorted(set(e for d in byrun.values() for e in d))
    eps = [e for e in eps if e < a.episodes]
    print("모수 %d개 · 에피소드 %d개 · 실제 경로 최대 %d홉"
          % (len(labels), len(eps), maxhop), flush=True)

    adj = build_adj()
    dist = hop_dist_to_dest(adj)
    cands = enumerate_simple_paths(adj, a.cand_hops)
    assert OBSERVED in cands
    print("고정 후보 %d개 (홉<=%d) · 스냅샷은 %d홉까지 쌓는다"
          % (len(cands), a.cand_hops, maxhop), flush=True)

    # ---- 에피소드별 스냅샷 (한 번만, 가장 긴 경로까지)
    #
    # 워밍업을 하나 앞에서 끊어 **관측 스냅샷**(에이전트가 홉 0에서 실제로 보는
    # 값 = 워밍업 10회 직후)과 **보상 스냅샷**(홉 k 의 보상을 정하는 값 =
    # 11+k 회 직후)을 한 번에 얻는다. `eval_ablation_risky` 는 `get_state()` 를
    # `step()` **앞에서** 부르고 `step()` 은 갱신을 **먼저** 하므로, 관측은
    # 보상보다 정확히 한 갱신 낡았다. 에이전트는 실현값을 못 본다.
    # 보상 스냅샷이 기존 규약과 같은지 아래에서 비트로 확인한다.
    snaps = {}
    obs = {}
    for e in eps:
        s = snapshots("dynamic", EVAL_SEED + e, EVAL_WARMUP - 1, maxhop + 1,
                      RISK_WINDOW)
        obs[e] = s[0]
        snaps[e] = s[1:]
    ref = snapshots("dynamic", EVAL_SEED + eps[0], EVAL_WARMUP, maxhop,
                    RISK_WINDOW)
    dmax = max(abs(a[k][i] - b[k][i]) for a, b in zip(snaps[eps[0]], ref)
               for k in a for i in (0, 1, 2))
    assert dmax == 0.0, dmax
    print("스냅샷 %d개 완료 · 보상 스냅샷 자기점검 |차| 최대 %.3g" % (len(snaps), dmax),
          flush=True)

    # ---- 고정 후보 채점 + 평가 최적 고정 경로 + 신탁
    fixed = dict((g, {}) for g in GAMMAS)
    for g in GAMMAS:
        for p in cands:
            fixed[g][p] = [episode_return(p, snaps[e], "dynamic", dist, g)
                           for e in eps]
    best_fixed = {}
    best_fixed_mean = {}
    oracle = {}
    for g in GAMMAS:
        means = dict((p, sum(v) / len(v)) for p, v in fixed[g].items())
        bp = max(means, key=lambda p: means[p])
        best_fixed[g] = bp
        best_fixed_mean[g] = means[bp]
        oracle[g] = [max(fixed[g][p][i] for p in cands)
                     for i in range(len(eps))]
        print("gamma %.2f · 평가 최적 고정 경로 %s (%+.4f) · "
              "전맹 경로 %+.4f · 신탁 %+.4f"
              % (g, "-".join(str(x) for x in bp), means[bp],
                 means[OBSERVED], sum(oracle[g]) / len(oracle[g])),
              flush=True)

    # ---- 모수별 실제 경로 채점
    model_rows = []
    series = dict((g, {}) for g in GAMMAS)   # label -> per-episode return
    for lab in labels:
        d = byrun[lab]
        for g in GAMMAS:
            v = [episode_return(d[e], snaps[e], "dynamic", dist, g)
                 for e in eps]
            series[g][lab] = v
        nuniq = len(set(d[e] for e in eps))
        hopm = sum(len(d[e]) - 1 for e in eps) / float(len(eps))
        model_rows.append(dict(
            label=lab, arm=lab.rsplit("-r", 1)[0], n_unique_paths=nuniq,
            mean_hop="%.4f" % hopm,
            mean_return_g100="%.6f" % (sum(series[1.00][lab]) / len(eps)),
            mean_return_g099="%.6f" % (sum(series[0.99][lab]) / len(eps))))

    arms = collections.OrderedDict()
    for r in model_rows:
        arms.setdefault(r["arm"], []).append(r["label"])

    print("\n" + "=" * 78)
    print("[팔별] 실제 경로의 정확 보상 (평가 200에피소드)")
    print("  %-12s %3s %7s %7s | %10s %10s"
          % ("팔", "런", "고유경로", "평균홉", "보상 g=1.00", "g=0.99"))
    arm_series = dict((g, {}) for g in GAMMAS)
    for arm, labs in arms.items():
        for g in GAMMAS:
            arm_series[g][arm] = [
                sum(series[g][l][i] for l in labs) / float(len(labs))
                for i in range(len(eps))]
        nu = sum(len(set(byrun[l][e] for e in eps)) for l in labs) / len(labs)
        hm = sum(sum(len(byrun[l][e]) - 1 for e in eps) / float(len(eps))
                 for l in labs) / len(labs)
        print("  %-12s %3d %7.1f %7.3f | %+10.4f %+10.4f"
              % (arm, len(labs), nu, hm,
                 sum(arm_series[1.00][arm]) / len(eps),
                 sum(arm_series[0.99][arm]) / len(eps)))

    # ---- 쌍체 검정: 팔 대 두 기준
    test_rows = []
    for g in GAMMAS:
        refs = [("blind_fixed", fixed[g][OBSERVED]),
                ("best_fixed", fixed[g][best_fixed[g]]),
                ("oracle_246", oracle[g])]
        stats = []
        for rname, rv in refs:
            for arm in arms:
                dv = [x - y for x, y in zip(arm_series[g][arm], rv)]
                m, se, z, pv = one_sample_p(dv)
                stats.append((rname, arm, m, se, z, pv))
        adj = RS.holm_bonferroni([s[5] for s in stats])
        print("\n" + "-" * 78)
        print("[쌍체] gamma=%.2f · 팔 %d개 x 기준 3개 = 족 %d (Holm)"
              % (g, len(arms), len(stats)))
        for rname, _ in [(r[0], 0) for r in refs]:
            sel = [(s, aj) for s, aj in zip(stats, adj) if s[0] == rname]
            win = sum(1 for s, aj in sel if aj == aj and aj < 0.05 and s[2] > 0)
            los = sum(1 for s, aj in sel if aj == aj and aj < 0.05 and s[2] < 0)
            print("  기준 %-12s : 유의하게 **나은** 팔 %2d개 · 나쁜 팔 %2d개 · "
                  "못 가름 %2d개" % (rname, win, los, len(sel) - win - los))
            sel.sort(key=lambda x: -x[0][2])
            for s, aj in sel[:3] + sel[-2:]:
                print("      %-12s 차 %+9.4f ± %.4f  p_holm %s"
                      % (s[1], s[2], s[3],
                         ("%.3g" % aj) if aj == aj and aj < 1e-4
                         else RS.fmt_p(aj)))
        for s, aj in zip(stats, adj):
            test_rows.append(dict(
                gamma="%.2f" % g, reference=s[0], arm=s[1],
                diff_arm_minus_ref="%.6f" % s[2], se="%.6f" % s[3],
                z="%.4f" % s[4], p_raw="%.6g" % s[5],
                p_holm=("%.6g" % aj) if aj == aj else "nan"))

    # ---- 왜 좋은가: '어느 경로를 쓰는가' 와 '바꿔 타는가' 를 나눈다
    #
    # 팔 하나가 쓰는 경로는 3~5개뿐인데도 246후보 신탁에 0.05~0.17 까지 붙는다.
    # 그게 정말 '상태를 보고 갈아타서' 인지, 아니면 '애초에 좋은 경로 하나를
    # 쓰는데 고정 경로 최적(-16.1466)이 그보다 나빠서' 인지는 다른 주장이다.
    # 그래서 모수마다 자기 메뉴의 **단일 최선 경로**(열린 루프)와 실제 값을
    # 쌍체로 비교한다. 메뉴 신탁은 그 메뉴로 가능한 천장이다.
    print("\n" + "=" * 78)
    print("[분해] 경로 선택 대 갈아타기 (gamma=1.00, 모수 69개)")
    dec_rows = []
    pv_sw = []
    for lab in labels:
        d = byrun[lab]
        menu = sorted(set(d[e] for e in eps))
        mser = dict((p, [episode_return(p, snaps[e], "dynamic", dist, 1.00)
                         for e in eps]) for p in menu)
        bestp = max(menu, key=lambda p: sum(mser[p]) / len(eps))
        act = series[1.00][lab]
        mor = [max(mser[p][i] for p in menu) for i in range(len(eps))]
        hit_menu = sum(1 for i in range(len(eps))
                       if abs(act[i] - mor[i]) < 1e-9)
        hit_all = sum(1 for i, e in enumerate(eps)
                      if d[e] in fixed[1.00]
                      and abs(act[i] - max(fixed[1.00][p][i] for p in cands))
                      < 1e-9)
        m, se, z, pv = one_sample_p([x - y for x, y in zip(act, mser[bestp])])
        pv_sw.append(pv)
        dec_rows.append(dict(
            label=lab, arm=lab.rsplit("-r", 1)[0], menu_size=len(menu),
            best_single="-".join(str(x) for x in bestp),
            best_single_return="%.6f" % (sum(mser[bestp]) / len(eps)),
            actual_return="%.6f" % (sum(act) / len(eps)),
            menu_oracle="%.6f" % (sum(mor) / len(eps)),
            switch_gain="%.6f" % m, switch_se="%.6f" % se,
            hit_menu_argmax_pct="%.1f" % (100.0 * hit_menu / len(eps)),
            hit_246_argmax_pct="%.1f" % (100.0 * hit_all / len(eps)),
            p_raw="%.6g" % pv))
    adj_sw = RS.holm_bonferroni(pv_sw)
    for r, aj in zip(dec_rows, adj_sw):
        r["p_holm"] = ("%.6g" % aj) if aj == aj else "nan"
    nwin = sum(1 for r, aj in zip(dec_rows, adj_sw)
               if aj == aj and aj < 0.05 and float(r["switch_gain"]) > 0)
    nlos = sum(1 for r, aj in zip(dec_rows, adj_sw)
               if aj == aj and aj < 0.05 and float(r["switch_gain"]) < 0)
    print("  갈아타기 이득 > 0 유의 %d개 · < 0 유의 %d개 · 못 가름 %d개 "
          "(족 %d, Holm)" % (nwin, nlos, len(dec_rows) - nwin - nlos,
                             len(dec_rows)))
    print("  %-12s %4s %10s %10s %10s %8s %8s"
          % ("팔", "메뉴", "단일최선", "실제", "메뉴천장", "메뉴적중", "246적중"))
    for arm, labs in arms.items():
        sel = [r for r in dec_rows if r["label"] in set(labs)]
        f = lambda k: sum(float(r[k]) for r in sel) / len(sel)   # noqa: E731
        print("  %-12s %4.1f %+10.4f %+10.4f %+10.4f %7.1f%% %7.1f%%"
              % (arm, f("menu_size"), f("best_single_return"),
                 f("actual_return"), f("menu_oracle"),
                 f("hit_menu_argmax_pct"), f("hit_246_argmax_pct")))
    sb = sum(float(r["best_single_return"]) for r in dec_rows) / len(dec_rows)
    sa = sum(float(r["actual_return"]) for r in dec_rows) / len(dec_rows)
    bfm = best_fixed_mean[1.00]
    print("  전체 평균: 단일최선 %+.4f -> 실제 %+.4f (갈아타기 몫 %+.4f) · "
          "고정최적 %+.4f 대비 '경로 선택' 몫 %+.4f"
          % (sb, sa, sa - sb, bfm, sb - bfm))
    # ---- 그 갈아타기가 어려운 일인가: 낡은 관측 하나로 되는가
    #
    # 메뉴천장에 98~99% 붙는다고 '정교한 가치함수' 라고는 못 쓴다. 관측이 한
    # 갱신 낡았을 뿐이고 링크 값은 자기상관이 있으므로, **낡은 관측을 그대로
    # 얼려** 메뉴를 점수내 고르는 근시 규칙이 같은 일을 할 수 있다. 그 규칙과
    # 비교한다. 통제군은 **다른 에피소드의 관측**으로 고르는 것이다 —
    # 이득이 '에피소드별 정보' 에서 오는지 '메뉴 모양' 에서 오는지 가른다.
    print("\n" + "=" * 78)
    print("[통제] 낡은 관측 근시 규칙 대 에이전트 (gamma=1.00)")

    def stale_pick(menu, src):
        frozen = [src] * (maxhop + 1)
        return max(menu, key=lambda p: episode_return(p, frozen, "dynamic",
                                                      dist, 1.00))

    st_rows = []
    for lab in labels:
        d = byrun[lab]
        menu = sorted(set(d[e] for e in eps))
        mser = dict((p, [episode_return(p, snaps[e], "dynamic", dist, 1.00)
                         for e in eps]) for p in menu)
        act = series[1.00][lab]
        mor = [max(mser[p][i] for p in menu) for i in range(len(eps))]
        stv, wrv, hs = [], [], 0
        for i, e in enumerate(eps):
            ps = stale_pick(menu, obs[e])
            stv.append(mser[ps][i])
            if abs(mser[ps][i] - mor[i]) < 1e-9:
                hs += 1
            wrv.append(mser[stale_pick(menu, obs[eps[(i + 37) % len(eps)]])][i])
        m, se, z, pv = one_sample_p([x - y for x, y in zip(act, stv)])
        st_rows.append(dict(
            label=lab, arm=lab.rsplit("-r", 1)[0], menu_size=len(menu),
            actual="%.6f" % (sum(act) / len(eps)),
            stale_greedy="%.6f" % (sum(stv) / len(eps)),
            wrong_episode="%.6f" % (sum(wrv) / len(eps)),
            menu_oracle="%.6f" % (sum(mor) / len(eps)),
            stale_hit_pct="%.1f" % (100.0 * hs / len(eps)),
            agent_minus_stale="%.6f" % m, se="%.6f" % se, p_raw="%.6g" % pv))
    adj_st = RS.holm_bonferroni([float(r["p_raw"]) for r in st_rows])
    for r, aj in zip(st_rows, adj_st):
        r["p_holm"] = ("%.6g" % aj) if aj == aj else "nan"
    nb = sum(1 for r, aj in zip(st_rows, adj_st)
             if aj == aj and aj < 0.05 and float(r["agent_minus_stale"]) > 0)
    nw = sum(1 for r, aj in zip(st_rows, adj_st)
             if aj == aj and aj < 0.05 and float(r["agent_minus_stale"]) < 0)
    print("  %-12s %9s %9s %9s %9s %8s"
          % ("팔", "에이전트", "낡은그리디", "엉뚱관측", "메뉴천장", "낡은적중"))
    nbetter = 0
    for arm, labs in arms.items():
        sel = [r for r in st_rows if r["label"] in set(labs)]
        f = lambda k: sum(float(r[k]) for r in sel) / len(sel)   # noqa: E731
        if f("actual") > f("stale_greedy"):
            nbetter += 1
        print("  %-12s %+9.4f %+9.4f %+9.4f %+9.4f %7.1f%%"
              % (arm, f("actual"), f("stale_greedy"), f("wrong_episode"),
                 f("menu_oracle"), f("stale_hit_pct")))
    print("  낡은 그리디의 메뉴천장 적중률(모수 평균) %.1f%% · "
          "팔 평균에서 에이전트가 더 나은 팔 %d/%d"
          % (sum(float(r["stale_hit_pct"]) for r in st_rows) / len(st_rows),
             nbetter, len(arms)))
    print("  모수별 쌍체(족 %d, Holm): 에이전트 유의하게 나음 %d개 · "
          "낡은 그리디가 나음 %d개 · 못 가름 %d개"
          % (len(st_rows), nb, nw, len(st_rows) - nb - nw))

    # ---- '메뉴를 만든 것' 의 값: 근시 규칙을 후보 246개 전체에 풀어놓는다
    #
    # 앞 통제군은 근시 규칙을 **팔의 메뉴 안에서만** 돌렸다. 메뉴 제약을 풀면
    # 두 갈래다. (ㄱ) 더 좋아진다 -> 학습 정책은 근시 착취 이상을 안 했고
    # 메뉴는 오히려 **족쇄**다. (ㄴ) 더 나빠진다 -> 메뉴 자체가 값이다(근시
    # 점수가 246개 안에서 길을 잃는다). 어느 쪽인지는 재야 안다.
    print("\n" + "=" * 78)
    print("[메뉴 값] 근시 규칙을 후보 246개 전체에 풀면 (gamma=1.00)")
    full_pick, full_ser = [], []
    for i, e in enumerate(eps):
        p = stale_pick(cands, obs[e])
        full_pick.append(p)
        full_ser.append(fixed[1.00][p][i])
    fm = sum(full_ser) / len(eps)
    nuq = len(set(full_pick))
    hopm = sum(len(p) - 1 for p in full_pick) / float(len(eps))
    hit = sum(1 for i in range(len(eps))
              if abs(full_ser[i] - oracle[1.00][i]) < 1e-9)
    print("  246 근시: %+.4f · 고유경로 %d개 · 평균 %.3f홉 · 신탁 적중 %.1f%%"
          % (fm, nuq, hopm, 100.0 * hit / len(eps)))
    print("  (참고) 신탁 %+.4f · 평가 최적 고정 %+.4f · 전맹 고정 %+.4f"
          % (sum(oracle[1.00]) / len(eps), best_fixed_mean[1.00],
             sum(fixed[1.00][OBSERVED]) / len(eps)))
    fp, frows = [], []
    for arm in arms:
        dv = [x - y for x, y in zip(arm_series[1.00][arm], full_ser)]
        m, se, z, pv = one_sample_p(dv)
        fp.append(pv)
        frows.append((arm, m, se, pv))
    fadj = RS.holm_bonferroni(fp)
    nb = sum(1 for r, aj in zip(frows, fadj)
             if aj == aj and aj < 0.05 and r[1] > 0)
    nw = sum(1 for r, aj in zip(frows, fadj)
             if aj == aj and aj < 0.05 and r[1] < 0)
    print("  팔 대 246근시 쌍체(족 %d, Holm): 팔이 유의하게 나음 %d개 · "
          "246근시가 나음 %d개 · 못 가름 %d개"
          % (len(frows), nb, nw, len(frows) - nb - nw))
    for (arm, m, se, _pv), aj in sorted(zip(frows, fadj), key=lambda x: -x[0][1])[:3]:
        print("      %-12s 차 %+8.4f ± %.4f  p_holm %s"
              % (arm, m, se, ("%.3g" % aj) if aj == aj and aj < 1e-4
                 else RS.fmt_p(aj)))
    with io.open(os.path.join(RESULT_DIR, "arm_path_menu_value.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        w.writerow(["stale246_return_g100", "%.6f" % fm])
        w.writerow(["stale246_unique_paths", nuq])
        w.writerow(["stale246_mean_hop", "%.4f" % hopm])
        w.writerow(["stale246_oracle_hit_pct", "%.1f" % (100.0 * hit / len(eps))])
        w.writerow(["oracle246_g100", "%.6f" % (sum(oracle[1.00]) / len(eps))])
        w.writerow(["best_fixed_g100", "%.6f" % best_fixed_mean[1.00]])
        w.writerow(["blind_fixed_g100",
                    "%.6f" % (sum(fixed[1.00][OBSERVED]) / len(eps))])
        w.writerow(["n_arms_better", nb])
        w.writerow(["n_stale246_better", nw])
        for (arm, m, se, pv), aj in zip(frows, fadj):
            w.writerow(["arm_minus_stale246:" + arm,
                        "%.6f|se=%.6f|p_raw=%.6g|p_holm=%s"
                        % (m, se, pv, ("%.6g" % aj) if aj == aj else "nan")])

    # ---- 같은 정책이 몇 개인가 (팔 간 독립성)
    sig = collections.defaultdict(list)
    for lab in labels:
        sig[tuple(byrun[lab][e] for e in eps)].append(lab)
    big = max(sig.values(), key=len)
    nspan = len(set(l.rsplit("-r", 1)[0] for l in big))
    print("\n[독립성] 모수 %d개 -> 서로 다른 경로수열 %d개 · 가장 큰 묶음 %d개"
          "(팔 %d개에 걸침)" % (len(labels), len(sig), len(big), nspan))
    print("  -> 위 쌍체의 '팔 13개' 는 서로 독립인 증거 13개가 아니다.")

    with io.open(os.path.join(RESULT_DIR, "arm_path_stale_control.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "label", "arm", "menu_size", "actual", "stale_greedy",
            "wrong_episode", "menu_oracle", "stale_hit_pct",
            "agent_minus_stale", "se", "p_raw", "p_holm"])
        w.writeheader()
        for r in st_rows:
            w.writerow(r)
    with io.open(os.path.join(RESULT_DIR, "arm_path_policy_groups.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "n_models", "n_arms", "members"])
        for i, (_k, v) in enumerate(sorted(sig.items(), key=lambda kv: -len(kv[1]))):
            w.writerow([i + 1, len(v),
                        len(set(l.rsplit("-r", 1)[0] for l in v)),
                        " ".join(sorted(v))])

    with io.open(os.path.join(RESULT_DIR, "arm_path_switch_gain.csv"), "w",
                 encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "label", "arm", "menu_size", "best_single", "best_single_return",
            "actual_return", "menu_oracle", "switch_gain", "switch_se",
            "hit_menu_argmax_pct", "hit_246_argmax_pct", "p_raw", "p_holm"])
        w.writeheader()
        for r in dec_rows:
            w.writerow(r)

    with io.open(OUT_MODELS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "label", "arm", "n_unique_paths", "mean_hop",
            "mean_return_g100", "mean_return_g099"])
        w.writeheader()
        for r in model_rows:
            w.writerow(r)
    with io.open(OUT_TESTS, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "gamma", "reference", "arm", "diff_arm_minus_ref", "se", "z",
            "p_raw", "p_holm"])
        w.writeheader()
        for r in test_rows:
            w.writerow(r)
    print("\n저장: %s (%d행) · %s (%d행)"
          % (OUT_MODELS, len(model_rows), OUT_TESTS, len(test_rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
