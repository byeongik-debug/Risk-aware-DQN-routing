# -*- coding: utf-8 -*-
"""26.10.4 의 계통 +1.249% 가 규약 탓인지, **표본 하나** 탓인지 가른다.

11월 하네스는 모든 모델이 `EVAL_SEED=20261114` 의 같은 200 에피소드를
겪는다(공통난수). 그 표본 하나가 평균보다 어렵다면 25개 모델의 11월
숫자가 **같이** 올라가고, 7월 숫자는 모델마다 독립 추첨이라 상쇄된다.
그러면 +1.249% 와 22/25 한쪽 부호가 규약 메커니즘 없이도 설명된다.

재는 법: 모델 고정 15회 측정(`eval_protocol_noise*`)의 11월식 평균을
그 모델의 표 10 값(= EVAL_SEED 표본 하나)과 견준다. 세 모델에서 같은
방향·같은 크기면 표본 쪽이다.

새 롤아웃 없음. 읽기만 한다. 출력 `review_results/eval_seed_offset.csv`.
"""

import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "review_results")
OUT = os.path.join(RESULT_DIR, "eval_seed_offset.csv")

sys.path.insert(0, HERE)
import review_stats as RS          # noqa: E402

# 표 10 행 이름 -> (arm, run)
TARGETS = {
    "-Cong-r03": ("-Cong", 3),
    "-dLoss-r03": ("-dLoss", 3),
    "-Util-r03": ("-Util", 3),
}


def read(name, required=True):
    p = os.path.join(RESULT_DIR, name)
    if not os.path.exists(p):
        if required:
            raise SystemExit("[없음] %s" % p)
        return None
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_draws():
    """11월식·7월식 추첨값. 온전한 CSV 가 있으면 그걸, 없으면 로그 복구본."""
    rows = read("eval_protocol_noise.csv", required=False)
    src = "eval_protocol_noise.csv"
    if not rows:
        rows = read("eval_protocol_noise_from_log.csv")
        src = "eval_protocol_noise_from_log.csv"
    out = {}
    for r in rows:
        out.setdefault((r["protocol"], r["label"]), []).append(float(r["delay"]))
    return out, src


def main():
    draws, src = load_draws()
    t10 = {}
    for r in read("table10_factor_runs.csv"):
        t10[(r["arm"], int(float(r["run"])))] = float(r["delay"])

    print("추첨 출처: %s" % src)
    print()
    hdr = "%-11s %4s %10s %8s | %10s %9s %7s %7s"
    print(hdr % ("모델", "R", "11월평균", "sd", "표10값", "차ms", "차%", "z"))
    print("-" * 78)
    rows, pcts, zs = [], [], []
    for label, (arm, run) in sorted(TARGETS.items()):
        d = draws.get(("nov", label))
        v = t10.get((arm, run))
        if not d or v is None:
            print("  [없음] %s" % label)
            continue
        mu, sd = RS.mean_std(d)
        diff = v - mu
        pct = diff / mu * 100.0
        z = diff / sd if sd > 0 else 0.0
        print(hdr % (label, len(d), "%.4f" % mu, "%.4f" % sd,
                     "%.4f" % v, "%+.4f" % diff, "%+.3f%%" % pct,
                     "%+.2f" % z))
        pcts.append(pct)
        zs.append(z)
        rows.append({"label": label, "arm": arm, "run": run,
                     "nov_draws": len(d), "nov_mean": "%.6f" % mu,
                     "nov_sd": "%.6f" % sd, "table10_value": "%.6f" % v,
                     "diff_ms": "%+.6f" % diff, "diff_pct": "%+.4f" % pct,
                     "z": "%+.4f" % z})

    if not rows:
        raise SystemExit("대상 없음")

    os.makedirs(RESULT_DIR, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    mp = sum(pcts) / len(pcts)
    mz = sum(zs) / len(zs)
    print("-" * 78)
    print("EVAL_SEED=20261114 표본의 치우침: 평균 %+.3f%% (z %+.2f), "
          "셋 다 부호 %s" % (mp, mz, "같음" if min(zs) * max(zs) > 0 else "갈림"))
    print("기록: %s" % OUT)
    print()

    jn = read("july_nov_protocol.csv", required=False)
    if jn:
        obs = [float(r["delay_pct"]) for r in jn]
        mo, so = RS.mean_std(obs)
        pos = sum(1 for x in obs if x > 0)
        print("26.10.4 의 관측: 같은 .pth %d개에서 11월 - 7월 = %+.3f%% "
              "(sd %.3f, %d/%d 양수)" % (len(obs), mo, so, pos, len(obs)))
        print("이 표본 치우침만으로 기대되는 값: %+.3f%%" % mp)
        print("남는 부분: %+.3f%%" % (mo - mp))
        print()
        if abs(mo - mp) < so:
            print("  -> 관측 치우침이 **표본 하나의 치우침 안**에 들어간다.")
            print("     11월 숫자 전부가 같은 200 에피소드를 공유하므로 25개가")
            print("     **같이** 올라간다. 7월 숫자는 독립 추첨이라 상쇄된다.")
            print("     즉 +%.3f%% 와 %d/%d 한쪽 부호는 **평가 규약의 메커니즘을"
                  % (mo, pos, len(obs)))
            print("     불러오지 않고** 설명된다. 26.10.4 의 '이 치우침은 규약에서")
            print("     온다' 는 **좁혀야 한다** — 규약이 만드는 것은 '모든 팔이")
            print("     한 표본을 공유한다' 는 구조이고, 치우침의 크기는 그")
            print("     표본이 어디 떨어졌는지다.")
        else:
            print("  -> 표본 치우침으로 다 설명되지 않는다. 남는 %+.3f%% 는"
                  % (mo - mp))
            print("     다른 데서 온다(미측정).")
        print()
        print("  주의: 표본 치우침은 세 모델에서만 쟀다(25개 전부가 아니다).")
        print("  세 값이 %+.3f ~ %+.3f%% 로 모여 있다는 것이 근거다."
              % (min(pcts), max(pcts)))

    # 7월식 대 11월식 — 같은 시드 묶음에서 두 규약의 평균이 같은가
    print()
    print("덤 — 같은 시드 묶음에서 두 규약의 평균 차 (규약 자체의 몫)")
    ds = []
    for label in sorted(TARGETS):
        j, n = draws.get(("july", label)), draws.get(("nov", label))
        if not j or not n:
            continue
        mj = RS.mean_std(j)[0]
        mn = RS.mean_std(n)[0]
        ds.append((mn - mj) / mj * 100.0)
        print("  %-11s 7월식 %.4f  11월식 %.4f  차 %+.4f ms (%+.3f%%)"
              % (label, mj, mn, mn - mj, (mn - mj) / mj * 100.0))
    if ds:
        print("  평균 %+.3f%% — 관측 %s 와 견주면 규약 자체의 몫은 작다."
              % (sum(ds) / len(ds),
                 ("%+.3f%%" % mo) if jn else "26.10.4"))


if __name__ == "__main__":
    main()
