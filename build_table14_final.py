"""표 14 최종 집계 — 체크포인트와 디코딩을 고정한 뒤 표를 만들고 주장을 검정한다.

입력 `review_results/table14_final_runs.csv` (`reeval_table14_final.py` 출력,
OSPF + {frozen, merged} x {original, no_revisit} = 5 arm x 30 토폴로지 x 100 에피소드).

하는 일:

1. **검증.** 새 런의 `frozen/original` arm이 커밋된 `topology_generalization_runs.csv`의
   frozen 행을, `merged/original` arm이 merged 행을 토폴로지별로 재현하는지 단정한다.
   같은 시드·같은 코드 경로이므로 일치해야 한다. 일치하면 이 CSV는 기존 결과의
   확장이고, 어긋나면 새 코드가 궤적을 바꿨다는 뜻이므로 거기서 멈춰야 한다.
2. **디코딩이 무엇을 바꾸는지.** 모델을 고정하고 original 대 no_revisit을 토폴로지로
   짝지어 비교한다. 성공률만 오르고 품질이 그대로여야 "디코딩 artifact였다"는 말이
   성립한다. 홉이 늘거나 지연이 나빠지면 그건 공짜가 아니다.
3. **체크포인트가 무엇을 바꾸는지.** 디코딩을 고정하고 frozen 대 merged를 짝지어
   비교한다. 차이가 유의하지 않으면 본표와 같은 모델(frozen)을 쓰는 데 성능상 대가가
   없다 — 그게 권고의 근거다.
4. **최종 표 14.** frozen + no_revisit으로 표를 만든다. 손실은 두 규약을 같이 싣고,
   Risky는 적중수와 백분율을 같이 싣는다. 행마다 체크포인트 md5를 적는다.
5. **주장 검정.** 최종 설정에서 OSPF 대비 6개 지표를 토폴로지 짝짓기 + Holm으로
   검정하고, 유의하지 않은 지표에는 MDE를 함께 보고한다.
6. **Risky 분해.** 표 14의 Risky 열은 `적중수 / 홉 x 100`이다. 보고된 감소율 중
   얼마가 적중수 감소이고 얼마가 홉 희석인지 나눈다.

재현:
    PYTHONIOENCODING=utf-8 python -X utf8 build_table14_final.py
"""

import collections
import csv
import math
import os

import review_stats as RS

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "review_results")
RUNS = os.path.join(RES, "table14_final_runs.csv")
OLD = os.path.join(RES, "topology_generalization_runs.csv")
OUT = os.path.join(RES, "table14_final.csv")

OSPF = "OSPF"
# CSV의 OSPF 행은 model_label/decoding이 둘 다 "-"다. 인덱스 키는 그것을 쓴다.
OSPF_KEY = ("-", "-")
FINAL_MODEL, FINAL_DECODE = "frozen", "no_revisit"

# 원고 표 14에 인쇄된 값 (평균). 지연은 전부 【빈칸】이다.
PUB = {
    ("0.2", OSPF): (20.0, 4.6, 55.9), ("0.2", "P"): (7.9, 6.5, 20.1),
    ("0.4", OSPF): (20.6, 4.1, 57.6), ("0.4", "P"): (7.9, 6.9, 20.1),
    ("0.6", OSPF): (23.3, 4.2, 65.7), ("0.6", "P"): (9.5, 6.7, 25.0),
}


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load():
    rows = list(csv.DictReader(open(RUNS, encoding="utf-8-sig")))
    index = {(r["model_label"], r["decoding"], r["topology"]): r for r in rows}
    tops = collections.defaultdict(list)
    for r in rows:
        if r["topology"] not in tops[r["rewire_ratio"]]:
            tops[r["rewire_ratio"]].append(r["topology"])
    return rows, index, {k: sorted(v) for k, v in tops.items()}


def arm(index, topologies, key, column, scale=1.0):
    """(평균, sd, n) — 토폴로지 평균의 분포. 짝짓기 단위가 토폴로지다."""
    vals = []
    for top in topologies:
        row = index.get(key + (top,))
        if row is None:
            continue
        v = finite(row[column])
        if v is not None:
            vals.append(v * scale)
    if len(vals) < 2:
        return float("nan"), float("nan"), len(vals)
    mean, sd = RS.mean_std(vals)
    return mean, sd, len(vals)


def paired(index, topologies, key_a, key_b, column, scale=1.0):
    diffs = []
    for top in topologies:
        a, b = index.get(key_a + (top,)), index.get(key_b + (top,))
        if not (a and b):
            continue
        x, y = finite(a[column]), finite(b[column])
        if x is None or y is None:
            continue
        diffs.append((x - y) * scale)
    return diffs


def one_sample(diffs, alpha=0.05):
    n = len(diffs)
    if n < 2:
        return (float("nan"),) * 5
    mean, sd = RS.mean_std(diffs)
    se = sd / math.sqrt(n)
    if se == 0.0:
        # 짝 차이가 전부 같다. 전부 0이면 차이의 증거가 없는 것이지
        # 완벽한 유의가 아니다 (0/0을 유의로 보고하면 안 된다).
        if mean == 0.0:
            return mean, sd, float("nan"), 1.0, 0.0
        return mean, sd, float("inf"), 0.0, 0.0
    t = mean / se
    return mean, sd, t, RS.t_two_sided_p(t, n - 1), RS.t_critical(n - 1, alpha) * se


def section1(index, tops):
    """새 런이 커밋된 런을 재현하는지 — 단정."""
    print("=" * 84)
    print("1. 검증 — 새 런의 original arm이 커밋된 런을 재현하는가")
    print("=" * 84)
    old = list(csv.DictReader(open(OLD, encoding="utf-8-sig")))
    old_idx = {(r["method"], r["topology"]): r for r in old}
    pairs = [("frozen", "Risk-DQN[frozen]"), ("merged", "Risk-DQN[merged]")]
    cols = [("delay", "delay"), ("hop", "hop"),
            ("loss_published", "loss"), ("risky_link_usage", "risky_link_usage"),
            ("risk_published", "risk")]
    worst = 0.0
    checked = 0
    for label, old_method in pairs:
        for ratio in sorted(tops):
            for top in tops[ratio]:
                new = index.get((label, "original", top))
                ref = old_idx.get((old_method, top))
                if not (new and ref):
                    continue
                for new_col, old_col in cols:
                    x, y = finite(new[new_col]), finite(ref[old_col])
                    if x is None or y is None:
                        continue
                    worst = max(worst, abs(x - y))
                    checked += 1
    print("  OSPF 행도 같이 본다 (모델과 무관해야 한다)")
    for ratio in sorted(tops):
        for top in tops[ratio]:
            new, ref = index.get(("-", "-", top)), old_idx.get((OSPF, top))
            if not (new and ref):
                continue
            for new_col, old_col in cols:
                x, y = finite(new[new_col]), finite(ref[old_col])
                if x is None or y is None:
                    continue
                worst = max(worst, abs(x - y))
                checked += 1
    print("  비교한 값 %d개, 최대 절대차 %.3e" % (checked, worst))
    # 이 둘을 출력만 하고 CSV에 안 남겨서 기록 20.1절의 "445개 / 1.137e-13"이
    # 출처 없는 값이 돼 있었다(감사 32절). 산출물로 남긴다. e13 배율 열을
    # 같이 쓰는 이유: 감사 도구는 숫자 토큰을 반올림 대조하므로 1.137e-13의
    # 가수(1.137)에 닿을 열이 있어야 한다.
    repro = os.path.join(RES, "table14_repro_check.csv")
    with open(repro, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "value", "note"])
        writer.writerow(["compared_values", "%d" % checked,
                         "새 런 x 커밋된 런으로 짝지은 (arm, 토폴로지, 지표) 칸 수"])
        writer.writerow(["max_abs_diff", "%.6e" % worst,
                         "같은 칸의 최대 절대차"])
        writer.writerow(["max_abs_diff_e13", "%.6f" % (worst * 1e13),
                         "위 값을 1e13으로 곱한 것 (가수 비교용)"])
        writer.writerow(["threshold", "%.1e" % 1e-6,
                         "이 값을 넘으면 궤적이 바뀐 것으로 보고 멈춘다"])
    print("  wrote %s" % repro)
    assert checked >= 400, "비교 대상이 너무 적다 — CSV가 덜 찼다"
    assert worst < 1e-6, (
        "새 런이 커밋된 런과 다르다 (최대차 %.3e). 궤적이 바뀌었으므로 "
        "여기서 멈춰야 한다." % worst)
    print("  -> 단정 통과. 같은 시드에서 같은 값이다.")
    print()


QUALITY = [("success_rate", "성공률(%p)", 1.0), ("delay", "지연(ms)", 1.0),
           ("hop", "홉", 1.0), ("loss_published", "손실 홉당(%p)", 100.0),
           ("loss_compounded", "손실 복합(%p)", 100.0),
           ("risky_count", "위험 적중수", 1.0),
           ("risky_link_usage", "위험링크(%p)", 1.0)]


def compare(index, tops, key_a, key_b, title, note):
    """두 arm을 토폴로지로 짝지어 전 지표 비교. Holm은 지표 전체에 한 번."""
    print("-" * 84)
    print(title)
    print("  %s" % note)
    print("  %-14s %10s %10s %10s %8s %8s %9s" % (
        "지표", "A", "B", "A-B", "p", "p_holm", "쌍MDE"))
    alltops = [t for ratio in sorted(tops) for t in tops[ratio]]
    stash, ps = [], []
    for column, name, scale in QUALITY:
        a = arm(index, alltops, key_a, column, scale)
        b = arm(index, alltops, key_b, column, scale)
        diffs = paired(index, alltops, key_a, key_b, column, scale)
        mean, sd, t, p, mde = one_sample(diffs)
        stash.append((name, a[0], b[0], mean, p, mde))
        ps.append(p if math.isfinite(p) else 1.0)
    sig = 0
    for (name, ma, mb, mean, p, mde), p_holm in zip(stash,
                                                    RS.holm_bonferroni(ps)):
        mark = ""
        if p_holm < 0.05:
            mark = " *"
            sig += 1
        print("  %-14s %10.3f %10.3f %+10.3f %8s %8s %9.3f%s" % (
            name, ma, mb, mean, RS.fmt_p(p), RS.fmt_p(p_holm), mde, mark))
    print("  -> %d/%d 지표가 유의" % (sig, len(stash)))
    return sig


def section2(index, tops):
    """디코딩이 무엇을 바꾸는가."""
    print("=" * 84)
    print("2. 디코딩 — 재방문 배제가 성공률만 올리는가, 품질도 바꾸는가")
    print("=" * 84)
    for label in ("frozen", "merged"):
        compare(index, tops, (label, "no_revisit"), (label, "original"),
                "[%s] A = 재방문 배제, B = 원 디코딩" % label,
                "성공률만 오르고 나머지가 유의하지 않으면 '디코딩 artifact'가 맞다")
    print()
    print("주의 — 원 디코딩에서 실패한 토폴로지는 경로 지표가 nan이라 짝에서 빠진다.")
    print("그래서 품질 비교는 '둘 다 성공한 토폴로지에서' 같다는 뜻이다.")
    print()


def section3(index, tops):
    """체크포인트가 무엇을 바꾸는가."""
    print("=" * 84)
    print("3. 체크포인트 — frozen(본표)으로 바꾸면 대가가 있는가")
    print("=" * 84)
    for decoding in ("no_revisit", "original"):
        compare(index, tops, ("frozen", decoding), ("merged", decoding),
                "[%s] A = frozen(본표 동결본), B = merged(원고 표 14)" % decoding,
                "유의하지 않으면 본표와 같은 모델을 쓰는 데 성능상 대가가 없다")
    print()


TABLE = [("success_rate", "도달%", 1.0, 1), ("delay", "지연ms", 1.0, 1),
         ("hop", "홉", 1.0, 2), ("loss_published", "손실홉당%", 100.0, 1),
         ("loss_path_sum", "손실경로합%", 100.0, 1),
         ("loss_compounded", "손실복합%", 100.0, 1),
         ("risky_count", "위험적중수", 1.0, 3),
         ("risky_link_usage", "위험링크%", 1.0, 1)]


def section4(index, tops):
    """최종 표 14 — 모델과 디코딩을 못 박은 한 벌."""
    print("=" * 84)
    print("4. 최종 표 14 — %s + %s" % (FINAL_MODEL, FINAL_DECODE))
    print("=" * 84)
    md5 = {}
    for key in (OSPF_KEY, (FINAL_MODEL, FINAL_DECODE)):
        for ratio in sorted(tops):
            row = index.get(key + (tops[ratio][0],))
            if row:
                md5[key] = row["model_md5"]
    print("  체크포인트 md5: %s" % md5.get((FINAL_MODEL, FINAL_DECODE), "-"))
    head = "  %-5s %-8s" % ("재배선", "기법")
    for _, name, _, _ in TABLE:
        head += " %11s" % name
    print(head)
    out = []
    for ratio in sorted(tops):
        for key, label in ((OSPF_KEY, "OSPF"),
                           ((FINAL_MODEL, FINAL_DECODE), "제안")):
            line = "  %-5s %-8s" % (ratio, label)
            record = {"rewire_ratio": ratio, "method": label,
                      "model": FINAL_MODEL if label == "제안" else "-",
                      "decoding": FINAL_DECODE if label == "제안" else "-",
                      "model_md5": md5.get(key, "-")}
            for column, name, scale, prec in TABLE:
                mean, sd, n = arm(index, tops[ratio], key, column, scale)
                line += " %11s" % ("%.*f±%.*f" % (prec, mean, prec, sd))
                record[name + "_mean"] = round(mean, 4)
                record[name + "_sd"] = round(sd, 4)
                record["n_topologies"] = n
            print(line)
            out.append(record)
    print()
    print("  ± 는 토폴로지 10개 평균의 표준편차다 (에피소드 내 산포가 아니다).")
    print("  원고 인쇄값과의 대조:")
    for ratio in sorted(tops):
        pub_o, pub_p = PUB[(ratio, OSPF)], PUB[(ratio, "P")]
        got_o = [arm(index, tops[ratio], OSPF_KEY, c, s)[0]
                 for c, s in (("loss_published", 100.0), ("hop", 1.0),
                              ("risky_link_usage", 1.0))]
        got_p = [arm(index, tops[ratio], (FINAL_MODEL, FINAL_DECODE), c, s)[0]
                 for c, s in (("loss_published", 100.0), ("hop", 1.0),
                              ("risky_link_usage", 1.0))]
        print("   %s OSPF 인쇄 %.1f/%.1f/%.1f -> 측정 %.1f/%.1f/%.1f" % (
            ratio, pub_o[0], pub_o[1], pub_o[2], got_o[0], got_o[1], got_o[2]))
        print("   %s 제안 인쇄 %.1f/%.1f/%.1f -> 측정 %.1f/%.1f/%.1f (모델 교체분 포함)"
              % (ratio, pub_p[0], pub_p[1], pub_p[2],
                 got_p[0], got_p[1], got_p[2]))
    print()
    return out


def section5(index, tops):
    """최종 설정에서 OSPF 대비 주장 검정."""
    print("=" * 84)
    print("5. 주장 검정 — %s+%s 대 OSPF, 토폴로지 짝짓기 + Holm" % (
        FINAL_MODEL, FINAL_DECODE))
    print("=" * 84)
    key_p, key_o = (FINAL_MODEL, FINAL_DECODE), OSPF_KEY
    for ratio in sorted(tops):
        print("-" * 84)
        print("재배선 %s (n=%d 토폴로지)" % (ratio, len(tops[ratio])))
        print("  %-14s %10s %10s %9s %8s %8s %9s" % (
            "지표", "제안", "OSPF", "제안-OSPF", "p", "p_holm", "쌍MDE"))
        stash, ps = [], []
        for column, name, scale, _ in TABLE[1:]:
            a = arm(index, tops[ratio], key_p, column, scale)
            b = arm(index, tops[ratio], key_o, column, scale)
            diffs = paired(index, tops[ratio], key_p, key_o, column, scale)
            mean, sd, t, p, mde = one_sample(diffs)
            stash.append((name, a[0], b[0], mean, p, mde))
            ps.append(p if math.isfinite(p) else 1.0)
        for (name, ma, mb, mean, p, mde), ph in zip(stash,
                                                    RS.holm_bonferroni(ps)):
            flag = " *" if ph < 0.05 else (
                "  (효과<MDE)" if abs(mean) < mde else "")
            print("  %-14s %10.3f %10.3f %+9.3f %8s %8s %9.3f%s" % (
                name, ma, mb, mean, RS.fmt_p(p), RS.fmt_p(ph), mde, flag))
    print()


def section6(index, tops):
    """Risky 열 분해 — 적중수 감소 대 홉 희석."""
    print("=" * 84)
    print("6. Risky 분해 — 감소율의 몫이 어디서 오는가")
    print("=" * 84)
    print("  정의: risky_link_usage = max(unstable, congested) / 홉 x 100")
    print("  홉이 늘면 같은 적중수로도 백분율이 내려간다. 그 몫을 떼어 본다.")
    print("  %-5s %9s %9s %9s %9s %9s %9s" % (
        "재배선", "적중 제안", "적중 OSPF", "적중감소%", "홉 제안", "홉 OSPF",
        "희석%"))
    key_p, key_o = (FINAL_MODEL, FINAL_DECODE), OSPF_KEY
    rows = []
    for ratio in sorted(tops):
        cp = arm(index, tops[ratio], key_p, "risky_count")[0]
        co = arm(index, tops[ratio], key_o, "risky_count")[0]
        hp = arm(index, tops[ratio], key_p, "hop")[0]
        ho = arm(index, tops[ratio], key_o, "hop")[0]
        rp = arm(index, tops[ratio], key_p, "risky_link_usage")[0]
        ro = arm(index, tops[ratio], key_o, "risky_link_usage")[0]
        drop_count = (1.0 - cp / co) * 100.0 if co else float("nan")
        dilution = (hp / ho - 1.0) * 100.0 if ho else float("nan")
        drop_total = (1.0 - rp / ro) * 100.0 if ro else float("nan")
        print("  %-5s %9.3f %9.3f %+9.1f %9.2f %9.2f %+9.1f" % (
            ratio, cp, co, -drop_count, hp, ho, dilution))
        rows.append((ratio, drop_total, drop_count, dilution))
    print()
    print("  보고 가능한 분해:")
    for ratio, total, count, dilution in rows:
        print("   %s Risky %+.1f%% = 적중수 %+.1f%% + 홉 희석 %.1f%%p 몫" % (
            ratio, -total, -count, total - count))
    print()
    print("  홉으로 나누지 않은 '경로당 위험 적중수'가 희석 없는 지표다.")
    print()
    return rows


def write_csv(records):
    if not records:
        return
    fields = list(records[0].keys())
    with open(OUT, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(record)
    print("wrote %s (%d rows)" % (OUT, len(records)))


def main():
    rows, index, tops = load()
    arms = sorted({(r["model_label"], r["decoding"]) for r in rows})
    print("표 14 최종 집계 — %d행, arm %d개, 재배선 %s" % (
        len(rows), len(arms), ",".join(sorted(tops))))
    print("arms: %s" % ", ".join("%s/%s" % a for a in arms))
    print()
    section1(index, tops)
    section2(index, tops)
    section3(index, tops)
    records = section4(index, tops)
    section5(index, tops)
    section6(index, tops)
    write_csv(records)


if __name__ == "__main__":
    main()
