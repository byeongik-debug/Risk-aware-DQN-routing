# -*- coding: utf-8 -*-
"""기준 행 `Full` 에도 EVAL_SEED 표본 치우침이 걸리는지 직접 잰다.

`analyze_eval_seed_offset.py` 는 절제 팔 세 개에서 표 10 값이 그 모델의
15회 평균보다 +1.16~1.40% 높다는 것을 보였다. 기준 행 `Full` 에 같은
양이 걸린다는 건 거기선 **추정**이다. 표 10 의 모든 비교가 `Full` 을
기준으로 하므로 이 칸을 직접 재야 한다.

`factor_full_run01`(최빈 결과값에 들어가는 런)과 `factor_full_run05`
(혼자 387.03 으로 튄 런) 둘을 같은 15 시드로 11월식 평가한다.
절제 없음(ZEROED_FACTORS=[]).

출력 `review_results/full_seed_offset.csv`. 원본은 읽기만 한다.
"""

import argparse
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "review_results")
OUT = os.path.join(RESULT_DIR, "full_seed_offset.csv")

sys.path.insert(0, HERE)
import measure_eval_protocol_noise as M          # noqa: E402
import review_stats as RS                        # noqa: E402

MODELS = [
    ("Full-r01", "factor_full_run01.pth", 1),
    ("Full-r05", "factor_full_run05.pth", 5),
]


def t10_full():
    p = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
    out = {}
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "Full":
                out[int(float(r["run"]))] = float(r["delay"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=15)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()

    mdir = os.path.join(HERE, "models_ablation")
    ref = t10_full()
    rows, summ = [], []

    for label, fname, run in MODELS:
        path = os.path.join(mdir, fname)
        if not os.path.exists(path):
            print("  [없음] %s" % path)
            continue
        vals = []
        for d in range(a.draws):
            seed = a.seed0 + d * 7919
            r = M.rollout(path, [], a.episodes, seed, "nov", 5)
            vals.append(r["delay"])
            rows.append({"label": label, "model": fname, "run": run,
                         "draw": d + 1, "seed": seed,
                         "episodes": a.episodes,
                         "reach_pct": "%.4f" % r["reach_pct"],
                         "delay": "%.6f" % r["delay"],
                         "loss": "%.8f" % r["loss"],
                         "hop": "%.6f" % r["hop"]})
            print("  nov  %-9s draw%02d seed=%-7d delay=%10.4f reach=%.1f%%"
                  % (label, d + 1, seed, r["delay"], r["reach_pct"]))
        mu, sd = RS.mean_std(vals)
        v = ref.get(run)
        summ.append((label, len(vals), mu, sd, v))

    os.makedirs(RESULT_DIR, exist_ok=True)
    if rows:
        with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("\n기록: %s (%d행)" % (OUT, len(rows)))

    print("\n" + "=" * 76)
    print("EVAL_SEED=20261114 표본이 `Full` 에도 같은 쪽으로 치우쳤는가")
    print("=" * 76)
    hdr = "%-9s %3s %11s %8s | %11s %9s %8s %6s"
    print(hdr % ("모델", "R", "11월평균", "sd", "표10값", "차ms", "차%", "z"))
    print("-" * 76)
    pcts = []
    for label, n, mu, sd, v in summ:
        if v is None:
            print("  [표10 값 없음] %s" % label)
            continue
        diff = v - mu
        pct = diff / mu * 100.0
        z = diff / sd if sd > 0 else 0.0
        pcts.append(pct)
        print(hdr % (label, n, "%.4f" % mu, "%.4f" % sd, "%.4f" % v,
                     "%+.4f" % diff, "%+.3f%%" % pct, "%+.2f" % z))
    print("-" * 76)
    if pcts:
        mp = sum(pcts) / len(pcts)
        print("`Full` 치우침 평균 %+.3f%% (절제 팔 세 개는 +1.317%%)" % mp)
        print()
        if min(pcts) > 0:
            print("  -> 기준 행도 같은 쪽으로 치우쳤다. 그러면 그 치우침은")
            print("     팔 간 **차**에서는 대부분 지워지고, 재채움 열의")
            print("     **절대값**에만 남는다. 표 10 의 절대 수준을 인용할 때는")
            print("     '평가 표본 하나 기준' 이라고 적어야 한다.")
            print("     기준 행 보정 추정: %.4f -> %.4f ms"
                  % (summ[0][4], summ[0][4] / (1.0 + mp / 100.0)))
        else:
            print("  -> 기준 행은 같은 쪽으로 안 치우쳤다. 그러면 표본 치우침이")
            print("     팔 간 차에서 **지워지지 않는다**. 재채움 열의 절제 효과")
            print("     크기를 그만큼 의심해야 한다.")
    print()
    print("  주의: 홉·손실은 CSV 에만 넣었다. 여기서 판단하는 것은 지연이다.")


if __name__ == "__main__":
    main()
