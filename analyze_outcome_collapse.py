# -*- coding: utf-8 -*-
"""독립 학습한 모델들이 **같은 결과값**으로 몰리는 정도를 파일로 남긴다.

왜 따로 만드는가
----------------
`analyze_table10_welch.attractor()` 가 이 집계를 하지만 **화면에만** 찍는다.
그래서 26.7·26.8절이 인용할 근거 파일이 디스크에 없다. 검정 함수와 적재
함수는 그 파일에서 **임포트해서** 쓴다(베끼면 하네스가 갈라진다).

내는 것 셋
----------
  outcome_collapse.csv    서로 다른 결과값(지연·손실·홉·Risky% 4자리) 별로
                          모델 몇 개가 거기 들어갔는지, 어느 팔에서 왔는지
  outcome_membership.csv  모델 하나하나가 어느 결과값에 속하는지 (감사용)
  reach_rate_fisher.csv   팔별 '최빈 결과 도달률' 2x2 Fisher 정확검정과
                          그 표본수에서 **도달 가능한 최소 p**

읽을 때 지켜야 하는 것
----------------------
최빈 결과는 **최빈**이고 **최적이 아니다**. 전체 모델 중 가장 좋은 지연이
최빈 결과가 아닌 데서 나오면 `best_delay_is_modal=0` 으로 찍힌다. 그
경우 "수렴 지점이 더 좋다" 고 쓰면 측정과 어긋난다.
"""
import argparse
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import analyze_table10_welch as W            # noqa: E402

RESULT_DIR = W.RESULT_DIR
OUT_COLLAPSE = os.path.join(RESULT_DIR, "outcome_collapse.csv")
OUT_MEMBER = os.path.join(RESULT_DIR, "outcome_membership.csv")
OUT_FISHER = os.path.join(RESULT_DIR, "reach_rate_fisher.csv")

SIG_KEYS = ("delay", "loss", "hop", "risky_pct")


def load_models():
    """모델 하나당 한 줄. (arm, run, 지표dict) 리스트."""
    out = []
    for path in (W.FACTOR_RUNS, W.EXTRA_RUNS):
        if not os.path.exists(path):
            continue
        base = os.path.basename(path)
        with io.open(path, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                out.append(dict(
                    arm=row["arm"], run=int(float(row["run"])),
                    model=row.get("model", ""), src=base,
                    vals=dict((k, float(row[k])) for k in SIG_KEYS)))
    return out


def signature(vals, nd=4):
    return tuple(round(vals[k], nd) for k in SIG_KEYS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="Full")
    ap.add_argument("--round", type=int, default=4,
                    help="결과값을 몇 자리로 묶을지 (기본 4)")
    args = ap.parse_args()

    models = load_models()
    if not models:
        print("모델별 CSV 가 없다. eval_factor_runs / eval_table10_extra 를"
              " 먼저 돌려라.")
        return 1

    groups = {}
    for m in models:
        sig = signature(m["vals"], args.round)
        groups.setdefault(sig, []).append(m)

    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0][0]))
    modal_sig, modal_members = order[0]

    print("=" * 94)
    print("독립 학습 모델의 결과값 붕괴 - 모델 %d개, 서로 다른 결과 %d가지"
          % (len(models), len(groups)))
    print("(결과값 = 지연·손실·홉·Risky%% 를 소수 %d자리로 묶은 것)"
          % args.round)
    print("=" * 94)
    print("%-4s %10s %8s %7s %9s  %5s  %s"
          % ("id", "지연", "손실", "홉", "Risky%", "모델수", "팔(개수)"))
    coll_rows, mem_rows = [], []
    for i, (sig, mem) in enumerate(order, 1):
        cnt = {}
        for m in mem:
            cnt[m["arm"]] = cnt.get(m["arm"], 0) + 1
        armtxt = ", ".join("%s x%d" % (a, c) if c > 1 else a
                           for a, c in sorted(cnt.items(),
                                              key=lambda kv: (-kv[1], kv[0])))
        if i <= 12 or len(mem) > 1:
            print("%-4d %10.4f %8.6f %7.3f %9.4f  %5d  %s"
                  % (i, sig[0], sig[1], sig[2], sig[3], len(mem), armtxt))
        coll_rows.append(dict(
            outcome_id=i, delay="%.6f" % sig[0], loss="%.6f" % sig[1],
            hop="%.6f" % sig[2], risky_pct="%.6f" % sig[3],
            n_models=len(mem), n_arms=len(cnt), arms=armtxt,
            is_modal=int(sig == modal_sig)))
        for m in mem:
            mem_rows.append(dict(
                outcome_id=i, arm=m["arm"], run=m["run"], model=m["model"],
                src=m["src"], is_modal=int(sig == modal_sig),
                **dict((k, "%.6f" % m["vals"][k]) for k in SIG_KEYS)))

    # 최빈이 최적인가 - 아니면 '수렴 지점이 좋다' 고 쓸 수 없다
    best = min(models, key=lambda m: m["vals"]["delay"])
    best_is_modal = signature(best["vals"], args.round) == modal_sig
    print("")
    print("최빈 결과: 지연 %.4f, 모델 %d/%d 개, 팔 %d 개"
          % (modal_sig[0], len(modal_members), len(models),
             len(set(m["arm"] for m in modal_members))))
    print("전체 최고(최소 지연): %.4f  (%s run%02d)"
          % (best["vals"]["delay"], best["arm"], best["run"]))
    if best_is_modal:
        print("-> 최빈 = 최저 지연이다.")
    else:
        print("-> **최빈은 최적이 아니다**. %s 의 비도달 모델이 최빈(%.4f)보다"
              % (best["arm"], modal_sig[0]))
        print("   %.4f ms 좋다. '수렴 지점이 더 좋다' 고 쓰면 안 된다."
              % (modal_sig[0] - best["vals"]["delay"]))

    # 팔별 도달률 + Fisher
    arms = {}
    for m in models:
        a = arms.setdefault(m["arm"], dict(n=0, reach=0))
        a["n"] += 1
        a["reach"] += int(signature(m["vals"], args.round) == modal_sig)
    # 팔 하나가 여러 족에 들어갈 수 있다(K=5 는 족 B 기준이면서 족 D 의 팔,
    # W-Equal 은 족 C 기준이면서 족 E 의 팔). 그래서 전부 적는다.
    fam = {}
    for code, _title, _base, members in W.FAMILIES:
        for arm in members:
            fam.setdefault(arm, []).append(code)
    fam = dict((a, "/".join(v)) for a, v in fam.items())

    b = arms.get(args.baseline)
    print("")
    print("[최빈 결과 도달률] 기준 %s %d/%d"
          % (args.baseline, (b or {}).get("reach", 0), (b or {}).get("n", 0)))
    print("-" * 94)
    fisher_rows = []
    for arm in sorted(arms, key=lambda a: (-arms[a]["reach"] / arms[a]["n"],
                                           a)):
        a = arms[arm]
        if arm == args.baseline or b is None:
            p, sep = float("nan"), float("nan")
        else:
            p, sep = W.fisher_2x2(b["reach"], b["n"] - b["reach"],
                                  a["reach"], a["n"] - a["reach"])
        fisher_rows.append(dict(
            arm=arm, family=fam.get(arm, ""), n=a["n"], reach=a["reach"],
            reach_pct="%.4f" % (100.0 * a["reach"] / a["n"]),
            baseline=args.baseline,
            baseline_n=(b or {}).get("n", ""),
            baseline_reach=(b or {}).get("reach", ""),
            fisher_p=("" if p != p else "%.6f" % p),
            p_min_attainable=("" if sep != sep else "%.6f" % sep),
            modal_delay="%.6f" % modal_sig[0],
            best_delay="%.6f" % best["vals"]["delay"],
            best_arm="%s run%02d" % (best["arm"], best["run"]),
            best_delay_is_modal=int(best_is_modal),
            _p=p, _sep=sep))

    # Holm. 이 표에서 **검정한 팔의 수**가 족 크기다(기준 Full 과 검정
    # 불가인 팔은 뺀다). 보정 전 p 로 판정하면 족 크기 함정을 또 밟는다.
    tested = [r for r in fisher_rows if r["_p"] == r["_p"]]
    fsize = len(tested)
    srt = sorted(tested, key=lambda r: r["_p"])
    run_max = 0.0
    for i, r in enumerate(srt):
        adj = min(1.0, r["_p"] * (fsize - i))
        run_max = max(run_max, adj)          # Holm 은 단조 증가로 고친다
        r["p_holm"] = run_max
    for r in tested:
        r["_hsep"] = (r["_sep"] * fsize if r["_sep"] == r["_sep"]
                      else float("nan"))
    ok = [r for r in tested if r["_hsep"] == r["_hsep"] and r["_hsep"] < 0.05]
    print("족 크기 = 검정한 팔 %d개(기준 Full 제외). Holm 보정 후 0.05 에"
          % fsize)
    print("닿을 수 있는 팔: %d/%d  %s"
          % (len(ok), fsize,
             "(" + ", ".join(r["arm"] for r in ok) + ")" if ok else "(없음)"))
    print("-> 나머지 팔은 **완전히 갈려도** 보정 후 0.05 를 넘을 수 없다.")
    print("   팔별 '최소가능p x 족크기' 를 열에 적었다(p_holm_min).")
    # 족 12 를 유지한 채 Holm 을 통과하려면 팔당 몇 개를 학습해야 하는가.
    # 완전 분리(n/n 대 0/n) 를 가정한 상한선이다 - 실제로는 더 필요하다.
    need = None
    for nn in range(2, 41):
        if 2.0 / W._C(2 * nn, nn) * fsize <= 0.05:
            need = nn
            break
    if need:
        print("   같은 족 크기에서 통과 가능해지는 최소 팔당 모델 수 = %d"
              % need)
        print("   (완전 분리 n/n 대 0/n 을 가정한 **하한**이다. 지금은 5개다.)")
    print("-" * 94)
    print("%-12s %8s %8s %10s %10s %10s %10s %s"
          % ("팔", "도달/n", "비율", "Fisher p", "Holm p", "최소가능p",
             "p_holm_min", "판정"))
    for r in fisher_rows:
        p, sep = r["_p"], r["_sep"]
        ph = r.get("p_holm", float("nan"))
        if r["arm"] == args.baseline:
            verdict = "기준"
        elif p != p:
            verdict = "검정불가(n=%d)" % r["n"]
        elif ph < 0.05:
            verdict = "도달률 다름"
        elif p < 0.05:
            verdict = "보정 전만 유의 -> 못 씀"
        else:
            verdict = "구분안됨"
        r["verdict"] = verdict
        r["p_holm"] = "" if ph != ph else "%.6f" % ph
        r["family_size"] = fsize
        r["min_n_for_holm"] = need if need else ""
        hs = r.get("_hsep", float("nan"))
        r["p_holm_min"] = "" if hs != hs else "%.6f" % hs
        print("%-12s %8s %8.1f%% %10s %10s %10s %10s %s"
              % (r["arm"], "%d/%d" % (r["reach"], r["n"]),
                 float(r["reach_pct"]),
                 "—" if p != p else "%.4f" % p,
                 "—" if ph != ph else "%.4f" % ph,
                 "—" if sep != sep else "%.4f" % sep,
                 "—" if hs != hs else "%.4f" % hs, verdict))
    for r in fisher_rows:
        r.pop("_p", None)
        r.pop("_sep", None)
        r.pop("_hsep", None)

    # 팔 안에서의 집중도. 이게 없으면 '0/5 도달' 을 '더 나쁘다' 로 잘못
    # 읽는다. W-Equal 은 최빈 결과에 한 번도 안 가지만 평균이 가장 좋다.
    print("")
    print("[팔 안에서의 집중도] 자기 팔의 모델들이 몇 가지 결과로 몰리는가")
    print("-" * 94)
    print("%-12s %5s %9s %9s %11s %11s %9s"
          % ("팔", "n", "서로다름", "자기최빈", "자기최빈값", "평균지연",
             "최빈도달"))
    for arm in sorted(arms):
        mm = [m for m in models if m["arm"] == arm]
        sigs = {}
        for m in mm:
            sigs.setdefault(signature(m["vals"], args.round), []).append(m)
        own = max(sigs.items(), key=lambda kv: (len(kv[1]), -kv[0][0]))
        mean_d = sum(m["vals"]["delay"] for m in mm) / len(mm)
        print("%-12s %5d %9d %9s %11.4f %11.4f %9s"
              % (arm, len(mm), len(sigs),
                 "%d/%d" % (len(own[1]), len(mm)), own[0][0], mean_d,
                 "%d/%d" % (arms[arm]["reach"], len(mm))))
        for r in fisher_rows:
            if r["arm"] == arm:
                r["n_distinct_own"] = len(sigs)
                r["own_modal_n"] = len(own[1])
                r["own_modal_delay"] = "%.6f" % own[0][0]
                r["mean_delay"] = "%.6f" % mean_d
    # 값 동일성과 산포는 다른 것이다. W-Equal 은 같은 값이 하나도 없는데
    # sd 가 전체 최소다. 그래서 정렬한 지연값의 **간격 구조**를 따로 잰다:
    # 최빈 결과에 가는 팔은 '붙어 있다가 하나가 멀리 튄다' 로 분산이 생기고,
    # 안 가는 팔은 '다 다른데 좁게 모여 있다' 일 수 있다.
    print("")
    print("[간격 구조] 정렬한 지연값에서 가장 큰 칸 간격 vs 전체 폭")
    print("-" * 94)
    print("%-12s %5s %9s %9s %9s %9s %s"
          % ("팔", "n", "최빈도달", "폭", "최대간격", "sd", "모양"))
    gap_rows = []
    for arm in sorted(arms, key=lambda a: -arms[a]["reach"] / arms[a]["n"]):
        mm = sorted(m["vals"]["delay"] for m in models if m["arm"] == arm)
        if len(mm) < 2:
            print("%-12s %5d %9s %9s %9s %9s %s"
                  % (arm, len(mm), "%d/%d" % (arms[arm]["reach"], len(mm)),
                     "—", "—", "—", "모델 1개 - 판단 불가"))
            continue
        rng = mm[-1] - mm[0]
        gaps = [mm[i + 1] - mm[i] for i in range(len(mm) - 1)]
        mg = max(gaps)
        mean_d = sum(mm) / len(mm)
        sd = (sum((x - mean_d) ** 2 for x in mm) / (len(mm) - 1)) ** 0.5
        shape = ("한 덩이 + 멀리 튄 것" if mg > 0.5 * rng and mg > 15.0
                 else "고르게 퍼짐")
        print("%-12s %5d %9s %9.2f %9.2f %9.2f %s"
              % (arm, len(mm), "%d/%d" % (arms[arm]["reach"], len(mm)),
                 rng, mg, sd, shape))
        gap_rows.append(dict(arm=arm, n=len(mm), reach=arms[arm]["reach"],
                             delay_range="%.4f" % rng,
                             max_gap="%.4f" % mg, sd="%.4f" % sd,
                             gap_over_range="%.4f" % (mg / rng if rng else 0),
                             shape=shape))
        for r in fisher_rows:
            if r["arm"] == arm:
                r["delay_range"] = "%.4f" % rng
                r["max_gap"] = "%.4f" % mg
                r["shape"] = shape

    print("")
    print("읽는 법: '자기최빈' 이 1/n 이면 그 팔은 수렴 지점이 없다(모델마다")
    print("다른 해). n/n 이면 그 팔만의 수렴 지점이 따로 있다는 뜻이다.")
    print("최빈도달 0/n 은 '다른 정책' 이라는 뜻이고 '더 나쁘다' 가 아니다.")

    for path, rows, flds in (
            (OUT_COLLAPSE, coll_rows,
             ["outcome_id", "delay", "loss", "hop", "risky_pct", "n_models",
              "n_arms", "arms", "is_modal"]),
            (OUT_MEMBER, mem_rows,
             ["outcome_id", "arm", "run", "model", "src", "delay", "loss",
              "hop", "risky_pct", "is_modal"]),
            (OUT_FISHER, fisher_rows,
             ["arm", "family", "n", "reach", "reach_pct", "baseline",
              "baseline_n", "baseline_reach", "fisher_p",
              "p_min_attainable", "p_holm", "p_holm_min", "family_size",
              "min_n_for_holm", "verdict",
              "n_distinct_own",
              "own_modal_n", "own_modal_delay", "mean_delay",
              "delay_range", "max_gap", "shape",
              "modal_delay", "best_delay", "best_arm",
              "best_delay_is_modal"])):
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("-> %s (%d 행)" % (path, len(rows)))

    print("")
    print("주의: 도달률 Fisher 는 '같은 정책으로 수렴하는 비율이 다른가' 를")
    print("묻는다. 평균 차이 검정과 묻는 게 다르므로 둘을 같은 가설로 섞어")
    print("세면 안 된다(족 크기 계산이 틀어진다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
