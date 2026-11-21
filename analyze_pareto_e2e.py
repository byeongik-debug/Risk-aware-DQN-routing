# -*- coding: utf-8 -*-
"""논문의 "Pareto-optimal" 주장을 종단간 손실과 교체 기준선으로 다시 판정한다.

원고 주장(4.6절 인자 절제): "어떤 단일 절제 모델도 Full 을 delay 와 loss 양
지표에서 동시에 능가하지 못한다" → 그래서 Full 설계가 Pareto 최적이다.

그 주장의 입력 둘이 지금은 **둘 다 틀린 것으로 밝혀져 있다.**

(1) 기준선. 발표된 Full(368.80)은 인자 절제 실험 값이 아니라 **시간 절제(w=5)
    요약에서 옮겨 온 값**이다(21.10절, 네 모멘트 전부 자리수까지 일치). 같은
    스크립트로 재학습한 Full 은 359.16 이다. 기준선을 내리면 `-Cong`(354.92)와
    `-ΔLoss`(354.98)가 **이미 Full 보다 빠르다** → 지연 쪽 지배 조건이 성립한다.
(2) 손실 규약. `loss` 는 종단간이 아니라 **홉당 평균 링크 손실률**이다
    (FINDINGS_loss_metric.md). 홉이 길어지면 분모가 커져 내려가므로 경로가 긴
    방법에 유리하고, 실제로 런 69개에서 분산의 88%가 홉 수로 설명되며 **부호가
    반대**다. 즉 "loss 가 낮다"가 "링크를 더 잘 골랐다"를 뜻하지 않는다.

그래서 2×2 로 판정한다 — 기준선 {발표, 재학습} × 손실 {홉당평균, 종단간}.
Pareto 주장이 규약에 의존하는지, 아니면 어느 규약에서도 버티는지가 답이다.

**종단간 규약 둘, 정확도가 다르므로 구분해서 적는다:**
  additive   = p_bar x hop  — `loss` 정의가 홉당 평균이므로 **정확하다**(합의 복원).
  compounded = 1 - (1-p_bar)^hop — 런별 평균만 있고 링크별 값이 없으므로
               **근사다**(링크 분산을 무시). 주 지표는 additive 로 쓴다.

비교 가능성: 절제 다섯 팔과 재학습 Full 은 **같은 스크립트**
(`train_factor_ablation.py`)가 학습·평가했으므로 22.8절의 평가기 교란이 섞이지
않는다. 반면 발표 Full(window5)은 `train_window_dqn.py` 산출이라 평가기가 다르다
— 그 열의 비교는 그래서 **참고용**이고, 판정은 재학습 기준선으로 한다.

원본 코드·결과 수정 없음(읽기만). 출력: review_results/pareto_e2e.csv
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import review_stats as RS                        # noqa: E402

ORIG_RES = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env",
                                        "results"))
OUT_DIR = os.path.join(HERE, "review_results")
OUT = os.path.join(OUT_DIR, "pareto_e2e.csv")

# (라벨, 디렉터리, 파일패턴) — 절제 다섯 팔은 원본, 재학습 Full 은 review_results
ARMS = (
    ("Full_retrained", OUT_DIR, "factor_full_dynamic_run%02d_result.csv"),
    ("Full_published", ORIG_RES, "window5_dynamic_run%02d_result.csv"),
    ("-Util", ORIG_RES, "factor_util_dynamic_run%02d_result.csv"),
    ("-Cong", ORIG_RES, "factor_cong_dynamic_run%02d_result.csv"),
    ("-dDelay", ORIG_RES, "factor_delay_inst_dynamic_run%02d_result.csv"),
    ("-dLoss", ORIG_RES, "factor_loss_inst_dynamic_run%02d_result.csv"),
    ("-Type", ORIG_RES, "factor_link_type_dynamic_run%02d_result.csv"),
)
RUNS = (1, 2, 3, 4, 5)


def read_runs(d, pat):
    """런별 (delay, loss_perhop, hop, success) 을 읽는다."""
    out = []
    for r in RUNS:
        p = os.path.join(d, pat % r)
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1:
            raise SystemExit("%s 행이 1개가 아니다: %d" % (p, len(rows)))
        row = rows[0]
        out.append({"run": r,
                    "delay": float(row["delay"]),
                    "loss_perhop": float(row["loss"]),
                    "hop": float(row["hop"]),
                    "success": float(row["success_rate"])})
    return out


def add_e2e(runs):
    for x in runs:
        p, h = x["loss_perhop"], x["hop"]
        # 정확: loss 정의가 홉당 평균이므로 합은 p*h 로 복원된다
        x["loss_additive"] = p * h
        # 근사: 링크별 값이 없어 평균으로 대체(링크 분산 무시)
        x["loss_compounded"] = 1.0 - (1.0 - p) ** h
    return runs


def col(runs, k):
    return [x[k] for x in runs]


def verdict(arm, base, metric):
    """arm 이 base 를 (지연, 해당 손실규약) 양쪽에서 능가하는가."""
    dd, td, _, pd_ = RS.welch(col(arm, "delay"), col(base, "delay"))
    dl, tl, _, pl = RS.welch(col(arm, metric), col(base, metric))
    both_mean = (dd < 0) and (dl < 0)
    both_sig = both_mean and (pd_ < 0.05) and (pl < 0.05)
    return {"d_delay": dd, "p_delay": pd_, "d_loss": dl, "p_loss": pl,
            "dominates_mean": "Y" if both_mean else "N",
            "dominates_sig": "Y" if both_sig else "N"}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {}
    for label, d, pat in ARMS:
        runs = add_e2e(read_runs(d, pat))
        if len(runs) != 5:
            raise SystemExit("%s 런이 5개가 아니다: %d" % (label, len(runs)))
        data[label] = runs

    print("=" * 78)
    print("[1] 런 5개 평균 (dynamic, n=5)")
    print("%-15s %9s %9s %7s %11s %12s"
          % ("팔", "지연", "홉당손실", "홉", "종단간(합)", "종단간(누적)"))
    for label, _, _ in ARMS:
        r = data[label]
        print("%-15s %9.2f %9.5f %7.3f %11.4f %12.4f"
              % (label,
                 RS.mean_std(col(r, "delay"))[0],
                 RS.mean_std(col(r, "loss_perhop"))[0],
                 RS.mean_std(col(r, "hop"))[0],
                 RS.mean_std(col(r, "loss_additive"))[0],
                 RS.mean_std(col(r, "loss_compounded"))[0]))

    abl = [a for a, _, _ in ARMS if a.startswith("-")]
    rows = []
    print("")
    print("[2] Pareto 판정 2x2 — 기준선 {발표, 재학습} x 손실 {홉당, 종단간}")
    for base in ("Full_published", "Full_retrained"):
        for metric, mname in (("loss_perhop", "홉당평균"),
                              ("loss_additive", "종단간(합)")):
            print("")
            print("  기준선 %-15s · 손실 %s" % (base, mname))
            print("  %-9s %10s %9s %11s %9s   %s"
                  % ("팔", "지연차", "p", "손실차", "p", "지배"))
            praw = []
            vs = []
            for a in abl:
                v = verdict(data[a], data[base], metric)
                v.update({"base": base, "metric": metric, "arm": a})
                vs.append(v)
                praw.extend([v["p_delay"], v["p_loss"]])
            padj = RS.holm_bonferroni(praw)
            for i, v in enumerate(vs):
                v["p_delay_holm"] = padj[2 * i]
                v["p_loss_holm"] = padj[2 * i + 1]
                print("  %-9s %+10.2f %9s %+11.4f %9s   %s"
                      % (v["arm"], v["d_delay"], RS.fmt_p(v["p_delay"]),
                         v["d_loss"], RS.fmt_p(v["p_loss"]),
                         "평균지배" if v["dominates_mean"] == "Y" else "-"))
                rows.append(v)
            win = [v["arm"] for v in vs if v["dominates_mean"] == "Y"]
            print("  → 평균으로 Full 을 양쪽에서 능가하는 팔: %s"
                  % (", ".join(win) if win else "없음"))

    fields = ["base", "metric", "arm", "d_delay", "p_delay", "p_delay_holm",
              "d_loss", "p_loss", "p_loss_holm", "dominates_mean",
              "dominates_sig"]
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})
    report(data, rows)
    print("저장: %s" % OUT)


def report(data, rows):
    print("")
    print("=" * 78)
    print("[3] 읽는 법 — 무엇이 주장을 뒤집고 무엇이 안 뒤집는가")
    cells = {}
    for r in rows:
        cells.setdefault((r["base"], r["metric"]), [])
        if r["dominates_mean"] == "Y":
            cells[(r["base"], r["metric"])].append(r["arm"])
    for base in ("Full_published", "Full_retrained"):
        for metric in ("loss_perhop", "loss_additive"):
            w = cells.get((base, metric), [])
            print("  %-15s %-14s → 지배 팔 %d개 %s"
                  % (base, metric, len(w),
                     ("(%s)" % ", ".join(w)) if w else ""))
    print("")
    sig = [r["arm"] for r in rows if r["dominates_sig"] == "Y"]
    print("  **유의하게** 지배하는 팔(보정 전 양쪽 p<0.05): %s"
          % (", ".join(sorted(set(sig))) if sig else "없음"))
    print("  → n=5 에서는 어느 칸에서도 통계적 지배가 성립하지 않는다")
    print("    (14절의 Holm 후 0/21 과 같은 사정: 런간 분산이 지배).")
    print("")
    print("  그러므로 쓸 수 있는 문장은 '어떤 절제도 Full 을 능가하지 못한다'가")
    print("  **아니라** '5런으로는 Full 과 절제들을 구분할 수 없다'다.")
    print("  Pareto 최적성은 **입증된 바 없고, 반증된 바도 없다** —")
    print("  주장 자체를 철회하는 것이 정직하다.")
    print("")
    print("  경고: 홉당 손실로 '지배'로 보이는 팔이 있으면 그것은 희석일 수")
    print("  있다. 같은 팔의 홉 수를 함께 볼 것(아래).")
    print("  %-9s %7s %11s %11s" % ("팔", "홉", "홉당손실", "종단간(합)"))
    for label, _, _ in ARMS:
        r = data[label]
        print("  %-9s %7.3f %11.5f %11.4f"
              % (label, RS.mean_std(col(r, "hop"))[0],
                 RS.mean_std(col(r, "loss_perhop"))[0],
                 RS.mean_std(col(r, "loss_additive"))[0]))
    print("=" * 78)


if __name__ == "__main__":
    main()
