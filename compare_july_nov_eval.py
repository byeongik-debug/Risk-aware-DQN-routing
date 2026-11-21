# -*- coding: utf-8 -*-
"""같은 `.pth` 25개를 7월 측정값과 11월 하네스 값으로 맞춘다.

7월 값의 출처는 `train_factor_ablation.py` 가 학습 직후 스스로 쓴
`SDN_DQN_new_env/results/factor_*_dynamic_run*_result.csv` 다(`.pth` 보다
약 1분 뒤 기록). 11월 값은 `review_results/table10_factor_runs.csv`
(= `eval_ablation_risky.evaluate`).

두 평가 루프는 지표 계산이 글자 그대로 같고 **시드만** 다르다.
7월은 200 에피소드 앞에 시드를 안 박아 학습이 남긴 난수 상태에서 이어
돈다(모델마다 다른 표본). 11월은 에피소드마다 `EVAL_SEED + i` 를 박아
모든 모델이 같은 표본을 겪는다(공통난수).

그래서 이 비교는 **환경 코드가 7월 이후 바뀌었는지**를 본다. 바뀌었다면
같은 가중치의 지연이 크게 움직이고 팔 사이 순위도 흐트러진다.

출력: `review_results/july_nov_protocol.csv`
원본(`SDN_DQN_new_env/`)은 읽기만 한다.
"""

import csv
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RESULT_DIR = os.path.join(HERE, "review_results")
OUT_CSV = os.path.join(RESULT_DIR, "july_nov_protocol.csv")

sys.path.insert(0, HERE)
import review_stats as RS          # noqa: E402

KEY2ARM = {"util": "-Util", "cong": "-Cong", "delay_inst": "-dDelay",
           "loss_inst": "-dLoss", "link_type": "-Type"}


def load_july():
    out = {}
    pat = os.path.join(ORIG, "results", "factor_*_dynamic_run*_result.csv")
    for p in sorted(glob.glob(pat)):
        m = re.match(r"factor_(.+)_dynamic_run(\d+)_result\.csv",
                     os.path.basename(p))
        if not m:
            continue
        key, run = m.group(1), int(m.group(2))
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            row = list(csv.DictReader(fh))[0]
        out[(KEY2ARM.get(key, key), run)] = {
            "delay": float(row["delay"]), "loss": float(row["loss"]),
            "hop": float(row["hop"]),
            "reach": float(row["success_rate"]),
            "model": "factor_%s_run%02d.pth" % (key, run),
            "src": os.path.basename(p),
        }
    return out


def load_nov():
    out = {}
    p = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            out[(row["arm"], int(float(row["run"])))] = row
    return out


def sign_test(diffs):
    """부호검정 — 양수 개수가 치우쳤는지. 정확 이항 양측."""
    n = sum(1 for d in diffs if d != 0.0)
    k = sum(1 for d in diffs if d > 0.0)
    k = max(k, n - k)

    def comb(a, b):
        r = 1
        for i in range(b):
            r = r * (a - i) // (i + 1)
        return r

    tail = sum(comb(n, i) for i in range(k, n + 1))
    p = 2.0 * tail / float(2 ** n)
    return n, k, min(p, 1.0)


def main():
    jul, nov = load_july(), load_nov()
    rows = []
    for k in sorted(jul, key=lambda x: (x[0], x[1])):
        if k not in nov:
            continue
        j, n = jul[k], nov[k]
        dj, dn = j["delay"], float(n["delay"])
        rows.append({
            "arm": k[0], "run": k[1], "model": j["model"],
            "july_src": j["src"],
            "july_delay": "%.6f" % dj, "nov_delay": "%.6f" % dn,
            "delay_diff": "%+.6f" % (dn - dj),
            "delay_pct": "%+.4f" % ((dn - dj) / dj * 100.0),
            "july_hop": "%.4f" % j["hop"], "nov_hop": "%.4f" % float(n["hop"]),
            "july_loss": "%.8f" % j["loss"],
            "nov_loss": "%.8f" % float(n["loss"]),
            "july_reach": "%.2f" % j["reach"],
        })

    os.makedirs(RESULT_DIR, exist_ok=True)
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    pct = [float(r["delay_pct"]) for r in rows]
    ms = [float(r["delay_diff"]) for r in rows]
    hopd = [abs(float(r["nov_hop"]) - float(r["july_hop"])) for r in rows]
    n, k, p = sign_test(ms)
    mu, sd = RS.mean_std(pct)

    print("같은 .pth %d개 — 7월 측정 대 11월 하네스" % len(rows))
    print("기록: %s" % OUT_CSV)
    print()
    hdr = "%-8s %4s | %12s %12s %10s %9s | %8s %8s"
    print(hdr % ("arm", "run", "7월 지연", "11월 지연", "차이ms", "차이%",
                 "7월hop", "11월hop"))
    print("-" * 84)
    for r in rows:
        print(hdr % (r["arm"], r["run"], r["july_delay"], r["nov_delay"],
                     r["delay_diff"], r["delay_pct"] + "%",
                     r["july_hop"], r["nov_hop"]))
    print("-" * 84)
    print("차이%%  최소 %+.3f / 최대 %+.3f / 평균 %+.3f / 절대평균 %.3f "
          "(표준편차 %.3f)"
          % (min(pct), max(pct), mu, sum(abs(x) for x in pct) / len(pct), sd))
    print("차이ms 최소 %+.3f / 최대 %+.3f" % (min(ms), max(ms)))
    print("홉 차이 최대 %.3f" % max(hopd))
    print("부호검정: %d개 중 %d개가 한쪽 부호 -> p = %s"
          % (n, k, RS.fmt_p(p)))
    print()
    if p < 0.05:
        print("  -> 한쪽으로 치우쳤다. 표본 잡음만으로는 설명 안 된다.")
        print("     두 루프는 지표 계산이 같고 시드 규약만 다르므로")
        print("     이 치우침은 **규약**에서 온다. 어느 부분인지는 아직")
        print("     측정하지 않았다(measure_eval_protocol_noise.py 가 가린다).")
    else:
        print("  -> 치우침 없음. 표본 잡음과 구분되지 않는다.")
    print()
    print("  환경 코드 변경 여부: 같은 가중치의 지연이 %.1f%% 안에서"
          % max(abs(x) for x in pct))
    print("  움직였고 홉은 %.3f 안이다. 스레드 수 하나가 26.13 ms(=약 7%%)"
          % max(hopd))
    print("  를 움직였던 것(26.9.2)과 견주면 작다. 환경 코드가 7월 이후")
    print("  바뀌었다는 신호는 이 비교에 없다.")


if __name__ == "__main__":
    main()
