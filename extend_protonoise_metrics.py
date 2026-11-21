# -*- coding: utf-8 -*-
"""26.10.6~.7 의 '규약 산포' 판정을 **지연 말고 다른 지표**에서도 돌린다.

26.10.6~.7 은 전부 `delay` 하나로 세워졌다. `measure_eval_protocol_noise.py`
재실행본(`review_results/eval_protocol_noise.csv`)에는 `loss`·`hop`·`risky_pct`
열이 같이 들어 있으므로, 같은 세 비교를 지표별로 다시 한다.

A. 모델 고정 산포 — 한 모델 × 15 추첨. 정책 차이가 0 이므로 전부 규약 몫이다.
B. 최빈 6개 모델의 7월 CSV 산포 — 경로가 100% 같다고 판정된 모델들
   (`review_results/path_identity_modal_arms.csv`), 모델마다 다른 표본.
C. 비 = B / A. 1 근처면 '그 차이는 전부 규약 몫'. 1 보다 크게 크면
   경로 동일 판정과 모순이므로 **그 지표에서는 결론을 다시 봐야 한다**.

덤: 11월식에서 경로 동일 두 모델(`-Cong-r03`/`-dLoss-r03`)이 15 추첨 전부에서
    세 지표 모두 같은 값인지 — 지연만 같고 손실·홉이 다르면 '경로 동일' 이
    아니다.

7월 CSV 에는 `risky` 열이 없어 B 는 `delay`/`loss`/`hop` 세 지표만 선다.
원본(`SDN_DQN_new_env/`)은 읽기만 한다.
출력: `review_results/protonoise_metrics.csv`
"""

import csv
import glob
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RD = os.path.join(HERE, "review_results")
OUT = os.path.join(RD, "protonoise_metrics.csv")

sys.path.insert(0, HERE)
import review_stats as RS          # noqa: E402

K2F = {"util": "-Util", "cong": "-Cong", "delay_inst": "-dDelay",
       "loss_inst": "-dLoss", "link_type": "-Type"}
METRICS = ["delay", "loss", "hop", "risky_pct"]


def load_rerun():
    """(protocol, label) -> {metric: [15 values in draw order]}"""
    out = {}
    with io.open(os.path.join(RD, "eval_protocol_noise.csv"),
                 encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            k = (r["protocol"], r["label"])
            d = out.setdefault(k, {m: [] for m in METRICS})
            d.setdefault("_draw", []).append(int(float(r["draw"])))
            for m in METRICS:
                d[m].append(float(r[m]))
    return out


def modal_members():
    keep = set()
    p = os.path.join(RD, "path_identity_modal_arms.csv")
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["label"].startswith("*"):
                continue
            keep.add(row["model"])
    return keep


def july_modal():
    """최빈 구성원들의 7월 CSV 값 — 지표별."""
    keep = modal_members()
    vals = []
    pat = os.path.join(ORIG, "results", "factor_*_dynamic_run*_result.csv")
    for p in sorted(glob.glob(pat)):
        m = re.match(r"factor_(.+)_dynamic_run(\d+)_result\.csv",
                     os.path.basename(p))
        if not m:
            continue
        key, run = m.group(1), int(m.group(2))
        model = "factor_%s_run%02d.pth" % (key, run)
        if model not in keep:
            continue
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            row = list(csv.DictReader(fh))[0]
        vals.append({
            "label": "%s-r%02d" % (K2F.get(key, key), run),
            "delay": float(row["delay"]),
            "loss": float(row["loss"]),
            "hop": float(row["hop"]),
        })
    return vals


def spread(xs):
    mu, sd = RS.mean_std(xs)
    return mu, sd, min(xs), max(xs), max(xs) - min(xs)


def main():
    rr = load_rerun()
    jm = july_modal()
    rows = []

    print("=" * 78)
    print("A. 모델 고정 산포 — 지표별 (정책 차이 = 0, 전부 규약 몫)")
    print("=" * 78)
    hdr = "%-6s %-12s %-10s %12s %10s %10s %9s"
    print(hdr % ("규약", "모델", "지표", "평균", "표준편차", "범위", "변동계수%"))
    print("-" * 78)
    fixed = {}
    for k in sorted(rr):
        for m in METRICS:
            xs = rr[k][m]
            mu, sd, lo, hi, rng = spread(xs)
            cv = sd / mu * 100.0 if mu else 0.0
            fixed[(k[0], k[1], m)] = (mu, sd, rng, cv)
            print(hdr % (k[0], k[1], m, "%.6f" % mu, "%.6f" % sd,
                         "%.6f" % rng, "%.3f" % cv))
            rows.append({"block": "A_model_fixed", "protocol": k[0],
                         "label": k[1], "metric": m, "n": len(xs),
                         "mean": "%.8f" % mu, "sd": "%.8f" % sd,
                         "min": "%.8f" % lo, "max": "%.8f" % hi,
                         "range": "%.8f" % rng, "cv_pct": "%.5f" % cv})
        print("-" * 78)

    print()
    print("=" * 78)
    print("B. 최빈 %d개 모델의 7월 CSV 산포 (경로 100%% 동일 판정)" % len(jm))
    print("=" * 78)
    for v in sorted(jm, key=lambda x: x["delay"]):
        print("  %-12s  delay %12.4f  loss %.8f  hop %.4f"
              % (v["label"], v["delay"], v["loss"], v["hop"]))
    print("-" * 78)
    b = {}
    for m in ["delay", "loss", "hop"]:
        xs = [v[m] for v in jm]
        mu, sd, lo, hi, rng = spread(xs)
        cv = sd / mu * 100.0 if mu else 0.0
        b[m] = (mu, sd, rng, cv)
        print("  %-6s n=%d  평균 %.8f  표준편차 %.8f  범위 %.8f  변동계수 %.3f%%"
              % (m, len(xs), mu, sd, rng, cv))
        rows.append({"block": "B_july_modal", "protocol": "july",
                     "label": "modal_%d_models" % len(xs), "metric": m,
                     "n": len(xs), "mean": "%.8f" % mu, "sd": "%.8f" % sd,
                     "min": "%.8f" % lo, "max": "%.8f" % hi,
                     "range": "%.8f" % rng, "cv_pct": "%.5f" % cv})

    print()
    print("=" * 78)
    print("C. 예측 검정 — 비 = B(6개 모델) / A(모델 고정, july -Cong-r03)")
    print("=" * 78)
    ref = ("july", "-Cong-r03")
    print("%-8s %14s %14s %9s %14s %14s %9s"
          % ("지표", "A_sd", "B_sd", "비_sd", "A_범위", "B_범위", "비_범위"))
    print("-" * 78)
    for m in ["delay", "loss", "hop"]:
        a_mu, a_sd, a_rng, _ = fixed[(ref[0], ref[1], m)]
        b_mu, b_sd, b_rng, _ = b[m]
        r_sd = b_sd / a_sd if a_sd else float("nan")
        r_rng = b_rng / a_rng if a_rng else float("nan")
        print("%-8s %14.8f %14.8f %8.2f배 %14.8f %14.8f %8.2f배"
              % (m, a_sd, b_sd, r_sd, a_rng, b_rng, r_rng))
        rows.append({"block": "C_ratio", "protocol": "july",
                     "label": "B_over_A_%s" % ref[1], "metric": m, "n": 0,
                     "mean": "", "sd": "%.8f" % r_sd, "min": "", "max": "",
                     "range": "%.8f" % r_rng, "cv_pct": ""})
    print("-" * 78)
    print("  비가 1 근처면 6개 모델 차이는 전부 규약 몫이다.")
    print("  1 보다 크게 크면 그 지표에서는 '경로 동일' 과 모순이므로")
    print("  결론을 다시 봐야 한다.")

    print()
    print("=" * 78)
    print("D. 11월식 경로 동일 두 모델이 세 지표 **전부** 같은가")
    print("=" * 78)
    pairs = [(("nov", "-Cong-r03"), ("nov", "-dLoss-r03")),
             (("july", "-Cong-r03"), ("july", "-dLoss-r03"))]
    for ka, kb in pairs:
        if ka not in rr or kb not in rr:
            continue
        print("  %s %s 대 %s" % (ka[0], ka[1], kb[1]))
        for m in METRICS:
            xa, xb = rr[ka][m], rr[kb][m]
            same = sum(1 for i in range(len(xa))
                       if abs(xa[i] - xb[i]) < 1e-9)
            mx = max(abs(xa[i] - xb[i]) for i in range(len(xa)))
            print("    %-10s 같은 추첨 %2d/%2d   최대 차 %.10f"
                  % (m, same, len(xa), mx))
            rows.append({"block": "D_pair_identity", "protocol": ka[0],
                         "label": "%s_vs_%s" % (ka[1], kb[1]), "metric": m,
                         "n": len(xa), "mean": "", "sd": "",
                         "min": "", "max": "%.10f" % mx,
                         "range": "", "cv_pct": "%d" % same})
        print()

    os.makedirs(RD, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("기록: %s (%d행)" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
