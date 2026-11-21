# -*- coding: utf-8 -*-
"""절제 7팔의 **모델별** 값을 전정밀도로 다시 받는다.

왜 필요한가
-----------
`table10_risky.csv` 에는 집계(평균·표준편차·n)만 있고 모델별 값이 없다.
`eval_ablation_risky.py` 는 모델별 줄을 **화면에 찍기는** 했지만 그 로그가
남아 있지 않고, 찍힌 자리수도 깎여 있다(delay=%.1f).

모델별 값이 필요한 이유 둘:
  1) K=3 팔에서 독립 학습 10개가 고작 세 가지 값만 냈다(354.294 x3,
     388.202 x5, 389.693 x1). 모델 간 분산이 연속 잡음이 아니라 **경로 집합
     몇 개 사이의 추첨**이면 n=5 에서 정규성을 쓰는 Welch 가 약하다.
     원자료가 있으면 **정확 순열검정**(C(10,5)=252 분할 전수)을 쓸 수 있다.
  2) 서로 다른 결과가 몇 가지인지(distinct outcome) 세는 것 자체가 표 10 의
     ± 를 어떻게 읽어야 하는지 바꾼다.

하는 일
-------
`eval_ablation_risky` 를 임포트해 그 안의 `probe_hook`·`evaluate` 를 **그대로**
부른다(평가 코드를 베끼지 않는다 - 베끼면 또 하네스가 갈라진다). 절제 매핑도
그 파일의 main 과 같은 식으로 구한다: `T.FACTOR_MAP.get(key, TEMPORAL)`.

출력은 **새 파일**이다. `table10_risky.csv` 는 읽기만 하고 덮지 않는다:
  review_results/table10_factor_runs.csv   모델별 (순열검정용)
  review_results/table10_factor_agg.csv    집계 + 기존 집계와의 차이

끝에 기존 `table10_risky.csv` 와 맞춰 본다. 같은 값이 나오면 하네스가
확인되고, 어긋나면 그 자리에서 드러난다.
"""
import argparse
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

import eval_ablation_risky as E                 # noqa: E402

RS = E.RS
risk_calculator = E.risk_calculator
T = E.T
RESULT_DIR = E.RESULT_DIR
FIELDS = E.FIELDS
EVAL_SEED = E.EVAL_SEED

RUNS_CSV = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
AGG_CSV = os.path.join(RESULT_DIR, "table10_factor_agg.csv")
OLD_CSV = os.path.join(RESULT_DIR, "table10_risky.csv")


def zeroed_for(key):
    """eval_ablation_risky.main 과 **같은** 규칙으로 절제 목록을 만든다."""
    return T.FACTOR_MAP.get(key, E.TEMPORAL if key == "temporal" else None)


def load_old():
    """기존 집계를 읽는다 (label -> 지표 -> 값)."""
    if not os.path.exists(OLD_CSV):
        return {}
    out = {}
    with io.open(OLD_CSV, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["factor"]] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=T.EVAL_EPISODES)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--only", default="",
                    help="쉼표로 구분한 라벨 (예: 'Full,-Util')")
    args = ap.parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()]

    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    print("=" * 92)
    print("절제 7팔 모델별 재측정 - eval_ablation_risky 의 평가 함수를 그대로 쓴다")
    print("EVAL_SEED=%d, 에피소드/모델=%d" % (EVAL_SEED, args.episodes))
    print("=" * 92)

    per_rows = []
    agg_rows = []
    for key, label, mdir in E.SETTINGS:
        if only and label not in only:
            continue
        zeroed = zeroed_for(key)
        if zeroed is None:
            print("[skip] %s: FACTOR_MAP 에 없다" % key)
            continue
        paths = [os.path.join(mdir, "factor_%s_run%02d.pth" % (key, r))
                 for r in range(1, args.runs + 1)]
        have = [p for p in paths if os.path.exists(p)]
        if not have:
            print("[skip] %-7s 모델 없음 (%s)" % (label, mdir))
            continue
        b, g = E.probe_hook(zeroed)
        print("")
        print("[%s] 모델 %d개, zeroed=%s, 탐침 %.6f -> %.6f"
              % (label, len(have), ",".join(zeroed) or "(없음)", b, g))
        print("   디렉터리: %s" % mdir)
        per = []
        for i, p in enumerate(have, 1):
            r = E.evaluate(p, zeroed, args.episodes)
            per.append(r)
            row = dict(arm=label, key=key, run=i,
                       model=os.path.basename(p), model_dir=mdir,
                       zeroed=",".join(zeroed))
            for k in FIELDS + ("success_rate",):
                row[k] = "%.6f" % r[k]
            per_rows.append(row)
            print("   run%02d  delay %9.4f  loss %.6f  hop %.4f  "
                  "risky%% %7.4f  도달 %5.1f%%"
                  % (i, r["delay"], r["loss"], r["hop"], r["risky_pct"],
                     r["success_rate"]))
        agg = dict(arm=label, key=key, n=len(per), model_dir=mdir,
                   zeroed=",".join(zeroed))
        for k in FIELDS + ("success_rate",):
            m, s = RS.mean_std([x[k] for x in per])
            agg[k] = "%.6f" % m
            agg[k + "_std"] = "%.6f" % s
        agg_rows.append(agg)
        uniq = sorted(set(round(x["delay"], 3) for x in per))
        print("   -> 평균 delay %.4f +- %s   서로 다른 지연값 %d/%d 가지: %s"
              % (float(agg["delay"]),
                 agg["delay_std"], len(uniq), len(per),
                 ", ".join("%.3f" % u for u in uniq)))

    if not per_rows:
        print("\n평가한 모델이 없다.")
        return 1

    pf = (["arm", "key", "run", "model", "model_dir", "zeroed"]
          + list(FIELDS) + ["success_rate"])
    with io.open(RUNS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=pf, extrasaction="ignore")
        w.writeheader()
        for r in per_rows:
            w.writerow(r)
    af = ["arm", "key", "n", "model_dir", "zeroed"]
    for k in FIELDS + ("success_rate",):
        af += [k, k + "_std"]
    with io.open(AGG_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=af, extrasaction="ignore")
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    # 기존 집계와 맞춰 본다 - 재현되면 하네스 확인, 어긋나면 드러난다
    old = load_old()
    print("")
    print("=" * 92)
    print("기존 table10_risky.csv 와 맞춰 보기 (덮어쓰지 않았다)")
    print("=" * 92)
    if args.episodes != T.EVAL_EPISODES:
        print("에피소드 %d 로 돌렸다(기존은 %d). 비교해도 의미가 없으므로"
              % (args.episodes, T.EVAL_EPISODES))
        print("재현 판정을 하지 않는다.")
        old = {}
    if not old:
        print("기존 집계와 비교하지 않았다.")
    else:
        print("%-8s %12s %12s %12s %12s %10s"
              % ("팔", "지연(새)", "지연(기존)", "차이", "홉 차이", "판정"))
        worst = 0.0
        for a in agg_rows:
            o = old.get(a["arm"])
            if o is None:
                print("%-8s %12s %12s %12s %12s %10s"
                      % (a["arm"], a["delay"], "(없음)", "-", "-", "신규"))
                continue
            dd = float(a["delay"]) - float(o["delay"])
            dh = float(a["hop"]) - float(o["hop"])
            worst = max(worst, abs(dd))
            verdict = "동일" if abs(dd) < 1e-6 else "다름"
            print("%-8s %12.4f %12.4f %12.6f %12.6f %10s"
                  % (a["arm"], float(a["delay"]), float(o["delay"]),
                     dd, dh, verdict))
        print("")
        if worst < 1e-6:
            print("-> 전부 재현됐다. 모델별 값은 기존 집계와 같은 측정의 것이므로")
            print("   순열검정에 바로 쓸 수 있다.")
        else:
            print("-> 최대 %.6f ms 어긋났다. 같은 코드·같은 시드인데 값이 달라졌다면"
                  % worst)
            print("   평가 경로에 시드로 통제되지 않는 요소가 있다는 뜻이다.")
            print("   (학습에서는 이미 확인됐다: 같은 시드로 학습한 factor_full 과")
            print("   window5 의 가중치 max|diff| 0.82~1.47)")

    print("")
    print("-> %s (%d 행)" % (RUNS_CSV, len(per_rows)))
    print("-> %s (%d 행)" % (AGG_CSV, len(agg_rows)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
