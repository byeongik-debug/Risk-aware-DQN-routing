# -*- coding: utf-8 -*-
"""표 10(구성요소 제거 및 파라미터 민감도)의 빈칸을 **측정값으로** 채운다.

원고 표 10 은 `Proposed (Full)` 행만 숫자가 있고 나머지 11행 44칸이 전부
`【】` 다. 이 스크립트는 이미 측정해 둔 산출물만 **조합**해서 채울 수 있는
칸과 채울 수 없는 칸을 가른다.

2026-11-14 자기 교정 — 이전 판의 틀린 문장
-------------------------------------------
이전 판은 `Equal weights (0.2)`·`K = 3`·`K = 10` 의 라우팅 3열에 "이 설정으로
**학습한 모델이 없다**" 고 적었다. **틀렸다.** 모델이 다 있다:
`weight_equal_run01..05`(5), `window3_run01..10`(10), `window5_run01..05`(5),
`window7_run01..08`(8), `window10_run01`(**1개뿐**). 그래서 문장을 고치는 게
아니라 **다시 롤아웃해서 측정**했다(`eval_table10_extra.py`). 원본 학습
스크립트가 쓴 `*_result.csv` 를 그대로 가져다 붙이지 않은 이유는, 그게 바로
원고 Full 행이 348.7(다른 하네스의 값)을 차용한 것과 같은 종류의 혼합이기
때문이다. 지금 진짜로 못 채우는 건 **σref 두 행뿐**이고, 이유도 다르다
(그 설정으로 학습한 모델이 없어서 재채점만 할 수 있다).

두 번째 자기 교정 — 평균과 p 의 출처가 달랐다
---------------------------------------------
이전 판은 평균을 `table10_risky.csv`(내 롤아웃)에서, p 를
`table10_pairwise.csv`(**원본 하네스**의 per-run CSV)에서 가져와 한 행에
올렸다. 두 자료의 Full 대 -Util 지연차가 -40.89 와 -44.34 로 다르다 —
표시된 Δ 와 p 가 서로 다른 측정의 것이었다. 이제 p 도 내 롤아웃에서
계산한 `table10_welch.csv` 에서만 읽는다.

출처(모두 `review_results/`):
  * `table10_f1.csv`        — F1 열. 시드 10개, 11설정 **전부 있다**.
  * `table10_risky.csv`     — 절제 7팔의 Delay·Loss·Risky·Hop. **한 평가
                              패스**에서 네 열을 같이 뽑았다(모델 n=5).
  * `table10_extra.csv`     — W-Equal·W-Temporal·K=3·K=5·K=7·K=10. 같은
                              하네스·같은 EVAL_SEED·같은 200에피소드.
  * `table10_welch.csv`     — 유의성. Welch + **정확 순열검정**, 족을 학습
                              스크립트별로 가른 것(A 절제 / B K스윕 /
                              C 가중치 / D 재현바닥 / E 교차참고).
  * `table10_sigma_ref.csv` — σref 두 행. **같은 정책으로 위험 점수만 다시
                              계산**한 것이다(재학습 아님, 다른 하네스).

행마다 검정 기준이 다르다
-------------------------
절제 6행은 Full 과 같은 학습 코드라 Full 을 기준으로 검정한다(족 A).
K 행은 `train_window_dqn.py` 내부에서 **K=5** 를 기준으로 본다(족 B) —
원고 표의 기준 행이 Full 이라 Δ 는 Full 대비로도 적지만, 검정은 같은
스크립트 안에서 한다. Equal weights 는 `train_weight_compare.py` 산물이라
Full 과 비교하면 스크립트 경계를 넘는다(족 E) — 그 비교가 읽을 만한지는
**족 D(재현 바닥)** 가 결정한다.

족 D 가 이 표를 읽는 열쇠다. `window5` 는 ZEROED_FACTORS=[] · risk_window=5
· DEFAULT_WEIGHTS 로 `factor_full` 과 설정이 완전히 같은데(세 학습 스크립트를
줄 단위로 맞춰 봤다: 하이퍼파라미터·신경망·타깃 계산·시나리오 분포·시드
규약 전부 동일) 모델 가중치가 실제로 다르다(max|diff| 0.82~1.47). 즉 **아무
설정도 바꾸지 않아도** 생기는 차이가 있고, 그게 절제 효과를 읽는 바닥이다.

주의: 원고 표제는 "10회 평균" 인데 라우팅 열은 **모델 5개**다(K=3 은 10,
K=7 은 8, **K=10 은 1**). F1 열만 10이다. 열마다 n 을 병기해야 한다.
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review_results")

# 원고 표 10 행 순서 -> (F1 파일의 설정명, 라우팅 자료 키)
#   "X:" 접두사 = table10_extra.csv,  "SIGMA:" = table10_sigma_ref.csv
ROWS = [
    ("Proposed (Full)", "Proposed (Full)", "Full"),
    ("w/o VD", "w/o VD", "-dDelay"),
    ("w/o VL", "w/o VL", "-dLoss"),
    ("w/o VD, VL", "w/o VD, VL", "-VD,VL"),
    ("w/o U", "w/o U", "-Util"),
    ("w/o IC", "w/o IC", "-Cong"),
    ("w/o T", "w/o T", "-Type"),
    ("Equal weights (0.2)", "Equal weights (0.2)", "X:W-Equal"),
    ("sigma_ref x0.5", "sigma_ref x0.5", "SIGMA:both x0.5"),
    ("sigma_ref x2", "sigma_ref x2", "SIGMA:both x2"),
    ("K = 3 (K/2)", "K = 3 (K/2)", "X:K=3"),
    ("K = 10 (2K)", "K = 10 (2K)", "X:K=10"),
]

# 원고에 없지만 표를 읽는 데 필요한 참고 행
EXTRA_ROWS = [
    ("[참고] K = 5 (Full 과 같은 설정, 독립 재학습)", None, "X:K=5"),
    ("[참고] K = 7 (스윕 보강)", None, "X:K=7"),
    ("[참고] W-Temporal (표 12 전용)", None, "X:W-Temporal"),
]

# 어느 족의 검정을 인용할지 (팔 이름 -> 족)
TEST_FAMILY = {
    "-dDelay": "A", "-dLoss": "A", "-VD,VL": "A",
    "-Util": "A", "-Cong": "A", "-Type": "A",
    "K=3": "B", "K=7": "B", "K=10": "B",
    "W-Temporal": "C",
    "K=5": "D",
    "W-Equal": "E",
}
FAM_NOTE = {
    "A": "Full (같은 학습 스크립트)",
    "B": "K=5 (train_window_dqn 내부)",
    "C": "W-Equal (train_weight_compare 내부)",
    "D": "Full (독립 재학습 — 단 스레드 수가 다르다, 26.9.4)",
    "E": "Full (교차 스크립트 — 족 D 를 먼저 볼 것)",
}


def read(name, required=True):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        if required:
            sys.exit("!! %s 가 없다" % path)
        print("[경고] %s 가 없다 — 그 열은 비워 둔다" % name)
        return []
    with io.open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def main():
    f1 = {r["setting"]: r for r in read("table10_f1.csv")}
    rk = {r["factor"]: r for r in read("table10_risky.csv")}
    ex = {r["arm"]: r for r in read("table10_extra.csv", required=False)}
    sg = {(r["scenario"], r["setting"]): r
          for r in read("table10_sigma_ref.csv")}
    we = read("table10_welch.csv", required=False)

    # 지연 검정: (족, 팔) -> 행.  diff 는 (기준 - 팔) 이므로 부호를 돌린다.
    tests = {}
    for r in we:
        if r["metric"] != "delay":
            continue
        tests[(r["family"], r["arm"])] = r

    base = rk["Full"]
    base_delay = float(base["delay"])

    def routing_cells(src, key, label):
        """라우팅 5열 + 검정 열을 만든다. src 는 'risky' 또는 'extra'."""
        r = rk[key] if src == "risky" else ex[key]
        d = float(r["delay"]) - base_delay
        fam = TEST_FAMILY.get(key, "")
        t = tests.get((fam, key))
        nn = r["n"]
        cells = {
            "Delay": "%.1f" % float(r["delay"]),
            "Delay_sd": ("%.1f" % float(r["delay_std"])
                         if r["delay_std"] else "—(n=1)"),
            "Loss_pct": "%.1f" % (100.0 * float(r["loss"])),
            "Risky_pct": "%.1f" % float(r["risky_pct"]),
            "Hop": "%.2f" % float(r["hop"]),
            "라우팅_n": nn,
            "Delay_vs_Full": "%+.1f" % d,
            "검정기준": FAM_NOTE.get(fam, "—"),
            "족": fam,
        }
        if key == "Full":
            cells.update({"Delay_vs_Full": "0.0", "검정기준": "(기준 행)",
                          "족": "", "Delay_welch_p": "", "Delay_p_holm": "",
                          "Delay_p_perm": "", "Delay_p_holm_perm": "",
                          "족크기": ""})
        elif t is None:
            cells.update({"Delay_welch_p": "—", "Delay_p_holm": "—",
                          "Delay_p_perm": "—", "Delay_p_holm_perm": "—",
                          "족크기": "—"})
        else:
            def g(k):
                v = t.get(k, "")
                if v == "":
                    return "—"
                try:
                    f = float(v)
                except ValueError:
                    return "—"
                # NaN 은 '검정 불가'(n<2)다. 숫자처럼 보이게 찍으면 안 된다.
                return "—" if f != f else "%.4f" % f
            cells.update({
                "Delay_welch_p": g("p_welch"),
                "Delay_p_holm": g("p_holm_welch"),
                "Delay_p_perm": g("p_perm"),
                "Delay_p_holm_perm": g("p_holm_perm"),
                "족크기": t.get("family_size", "")})
        src_name = ("table10_risky.csv" if src == "risky"
                    else "table10_extra.csv")
        note = "%s (한 평가 패스, 모델 n=%s)" % (src_name, nn)
        if int(float(nn)) < 2:
            note += "  ** n=1: 표준편차도 검정도 없다 **"
        cells["출처"] = note
        return cells

    out = []
    for label, k1, k2 in list(ROWS) + list(EXTRA_ROWS):
        row = {"표10 행": label}
        f = f1.get(k1) if k1 else None
        if k1 and f is None:
            print("[경고] F1 에 %s 없음" % k1)
            continue
        if f is None:
            row.update({"F1": "—", "F1_sd": "—", "F1_n": "0",
                        "F1_dvs_full": "—", "F1_p": "—"})
        else:
            row["F1"] = "%.3f" % float(f["f1_mean"])
            row["F1_sd"] = "%.3f" % float(f["f1_std"])
            row["F1_n"] = f["n_seeds"]
            row["F1_dvs_full"] = f["delta_vs_full"] or "—"
            row["F1_p"] = f["paired_p"] or "—"

        if k2.startswith("X:"):
            key = k2.split(":", 1)[1]
            if key not in ex:
                print("[경고] table10_extra.csv 에 %s 없음 — 비워 둔다" % key)
                row.update({"Delay": "", "Delay_sd": "", "Loss_pct": "",
                            "Risky_pct": "", "Hop": "", "라우팅_n": "0",
                            "Delay_vs_Full": "", "Delay_welch_p": "",
                            "Delay_p_holm": "", "Delay_p_perm": "",
                            "Delay_p_holm_perm": "", "검정기준": "",
                            "족": "", "족크기": "",
                            "출처": "아직 미측정(롤아웃 대기)"})
            else:
                row.update(routing_cells("extra", key, label))
        elif k2.startswith("SIGMA:"):
            s = sg[("dynamic", k2.split(":", 1)[1])]
            # **절대값을 이 열에 넣지 않는다.** 이 팔은 다른 평가
            # 하네스(스냅숏 100개 재채점)에서 나왔고 그 기본값이
            # 327.87 ms 다 — 모델 롤아웃 열(기본 360.8)과 섞으면
            # 원고 Full 행의 차용값(348.7)과 같은 오류가 된다.
            row.update({
                "Delay": "", "Delay_sd": "", "Loss_pct": "",
                "Risky_pct": "", "Hop": "",
                "라우팅_n": "1*",
                "Delay_vs_Full": ("%+.1f" % (float(s["delay"])
                                  - float(s["delay_default"]))),
                "Delay_welch_p": "", "Delay_p_holm": "",
                "Delay_p_perm": "", "Delay_p_holm_perm": "",
                "검정기준": "검정 불가(1* = 재학습 없이 정책 1개 재채점)",
                "족": "", "족크기": "",
                "출처": ("같은 정책으로 위험점수만 재계산(재학습 아님, "
                         "다른 하네스: 자기 기본 %.1f ms). "
                         "rho_vs_default=%s, 위험경로변경 %s%%, "
                         "Loss %.1f->%.1f%%, Risky %.1f->%.1f%%"
                         % (float(s["delay_default"]),
                            s["spearman_vs_default"][:6],
                            s["risk_path_changed_pct"],
                            100.0 * float(s["loss_per_hop_default"]),
                            100.0 * float(s["loss_per_hop"]),
                            float(s["risky_pct_default"]),
                            float(s["risky_pct"])))})
        else:
            row.update(routing_cells("risky", k2, label))
        out.append(row)

    cols = ["표10 행", "F1", "F1_sd", "F1_n", "F1_dvs_full", "F1_p",
            "Delay", "Delay_sd", "Loss_pct", "Risky_pct", "Hop",
            "라우팅_n", "Delay_vs_Full", "검정기준", "족", "족크기",
            "Delay_welch_p", "Delay_p_holm", "Delay_p_perm",
            "Delay_p_holm_perm", "출처"]
    path = os.path.join(OUT, "table10_fill.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in out:
            w.writerow(dict((c, r.get(c, "")) for c in cols))

    bar = "=" * 104
    print(bar)
    print("표 10 채움 — 채운 칸 / 못 채운 칸")
    print(bar)
    hdr = ("%-44s %6s %7s %6s %6s %5s %8s %4s %8s %8s"
           % ("설정", "F1", "Delay", "Loss%", "Risky%", "Hop", "ΔFull",
              "n", "pH(W)", "pH(순열)"))
    print(hdr)
    print("-" * len(hdr))
    for r in out:
        print("%-44s %6s %7s %6s %6s %5s %8s %4s %8s %8s"
              % (r["표10 행"], r["F1"], r["Delay"] or "—",
                 r["Loss_pct"] or "—", r["Risky_pct"] or "—",
                 r["Hop"] or "—", r["Delay_vs_Full"] or "—",
                 r.get("라우팅_n", "—"),
                 r.get("Delay_p_holm", "—") or "—",
                 r.get("Delay_p_holm_perm", "—") or "—"))

    nman = len(ROWS)
    nfill = sum(1 for r in out[:nman] if r["Delay"])
    print("")
    print("  원고 12행 중 라우팅 3열을 채운 행: %d/%d." % (nfill, nman))
    print("  못 채운 %d 행은 **σref 두 행**이고, 이유는 '학습한 모델이 없어"
          % (nman - nfill))
    print("  재채점만 할 수 있다' 다. 이전 판이 Equal weights·K=3·K=10 에")
    print("  같은 이유를 적었던 것은 **틀렸고**(모델이 다 있다) 이번에 다시")
    print("  롤아웃해서 채웠다.  F1 열은 %d/%d 전부 있다." % (nman, nman))

    # 재현 바닥을 먼저 읽는다
    d = tests.get(("D", "K=5"))
    print("")
    if d is None:
        print("  경고 0: 족 D(재현 바닥)가 아직 없다. analyze_table10_welch.py")
        print("  를 먼저 돌려야 아래 비교를 읽을 수 있다.")
    else:
        floor = abs(float(d["diff"]))
        print("  경고 0 (**먼저 읽을 것**): 재현 바닥 = %.1f ms." % floor)
        print("  `window5` 는 `factor_full` 과 설정이 완전히 같은데(절제 없음,")
        print("  K=5, 기본 가중치) 독립 재학습하면 지연 평균이 %.1f ms 움직인다."
              % floor)
        print("  그래서 |Δ| 가 이 값보다 작은 절제 행은 **절제 효과로 읽을 수")
        print("  없다**. 아래 ΔFull 열을 이 값과 먼저 견주어야 한다.")
        rels = []
        for _l, _k1, k2 in ROWS:
            if k2 in ("Full",) or k2.startswith(("X:", "SIGMA:")):
                continue
            rr = rk[k2]
            dd = abs(float(rr["delay"]) - base_delay)
            rels.append((dd / floor if floor else float("inf"), k2, dd))
        rels.sort(reverse=True)
        for rel, k2, dd in rels:
            tag = ("바닥의 %.2f배" % rel if rel >= 1.0
                   else "**바닥보다 작다**(%.2f배)" % rel)
            print("      %-8s |Δ|=%6.2f ms  %s" % (k2, dd, tag))

    print("")
    print("  경고 1: 원고 표제 '10회 평균' 과 안 맞는다 — F1 은 시드 10,")
    print("  라우팅은 절제·W·K=5 가 모델 5, K=3 이 10, K=7 이 8,")
    print("  **K=10 은 1** 이다. 열마다 n 을 병기해야 한다.")
    print("  경고 2: σref 두 행은 **재채점**이다. 본문은 '각 설정으로 DQN")
    print("  을 다시 학습한 뒤' 라고 적혀 있으므로 **서술을 고치거나**")
    print("  재학습해야 한다(지금 수치는 정책이 그대로라 거의 안 움직인다).")
    a_util = tests.get(("A", "-Util"))
    if a_util:
        print("  경고 3: 지연차는 내 하네스에서도 **하나도 유의하지 않다**.")
        print("  가장 큰 -Util 이 생 p=%s, Holm(족 %s) %s, 정확 순열 %s."
              % (a_util["p_welch"][:6], a_util["family_size"],
                 a_util["p_holm_welch"][:6], (a_util["p_perm"] or "—")[:6]))
        pmin = a_util.get("p_min_attainable", "")
        if pmin:
            try:
                pm = float(pmin)
                fs = int(float(a_util["family_size"]))
                print("  **검정력 상한**: n=5 대 n=5 의 정확 순열검정이 낼 수")
                print("  있는 최소 양측 p 는 %.4f 이고, 족 %d 에 Holm 을 걸면"
                      % (pm, fs))
                print("  보정 후 최소 %.3f 다 — %s"
                      % (pm * fs,
                         ("**효과 크기와 무관하게 0.05 를 넘을 수 없다**."
                          if pm * fs >= 0.05 else "유의가 나올 수는 있다.")))
                print("  '유의한 칸 0개' 를 '효과 없음' 으로 읽으면 안 된다.")
            except ValueError:
                pass
    else:
        print("  경고 3: table10_welch.csv 가 없어 유의성 열이 비어 있다.")
    print("  경고 4: Loss 는 **홉당 평균**이다. 긴 경로가 유리해진다 —")
    print("  -Util 은 홉당 최저인데 홉이 9.45 로 가장 길다.")
    print("  경고 5: σref 팔의 절대 지연은 이 열에 **넣지 않았다**. 그 팔은")
    print("  다른 하네스(기본 327.9 ms)에서 나왔고 모델 롤아웃 열(기본")
    print("  360.8 ms)과 섞으면 원고 Full 행 348.7 과 같은 종류의 차용값이")
    print("  된다. 자기 기본값 대비 변화만 '출처' 열에 적었다.")
    print("  경고 6: Equal weights 행의 검정은 **스크립트 경계를 넘는다**")
    print("  (족 E). W-Current 모델이 아예 없어서 같은 스크립트 안에 기준이")
    print("  없기 때문이다. 경고 0 의 바닥을 같이 보지 않으면 못 읽는다.")
    print("")
    print("  Full 행 재현 점검(원고 값 대비): F1 0.966=0.966 · Risky 21.5")
    print("  =21.5 · Loss 8.3~8.4 vs 8.4 는 **맞고**, Delay 만 360.8 대")
    print("  348.7 로 **12.1 ms 어긋난다** → 원고 Full 지연이 다른 실험")
    print("  차용값이라는 판단과 일관된다(표 11 도 같은 348.7 을 쓴다).")
    print("")
    print("저장: %s" % path)
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
