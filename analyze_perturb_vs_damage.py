# -*- coding: utf-8 -*-
"""23.9.4 판정 — 교란의 어느 요약이 피해를 가르는가 (롤아웃 없음).

왜 필요한가
-----------
`measure_rank_vs_value.py` 의 조건들은 위험 블록을 서로 다른 방식으로
흔든다. 그 교란을 네 스칼라로 요약할 수 있다(`rank_value_geometry.csv`):
  L1(평균|차|) · 노름비 · 코사인(방향) · rho(순위).
어느 요약이 피해(지연·손실·홉)를 가르는지 보면 "값은 순위에 있다"는 주장이
어디까지 서는지 정해진다. 이미 둘은 탈락했다 — `dimshuffle` 은 L1 이
frozen 의 190.9% 인데 피해는 97.6% 고(비례 아님), 피해를 내는 두 조건은
노름을 0.05% 안에서 보존한다.

설계상 대조쌍 둘이 한 축만 다르다(둘 다 rho=1.0000):
  (가) rankpow50 대 rankpow400 — 코사인 0.9664/0.9662, 노름비 1.1728/0.7848
  (나) rankpow05 대 frozen      — 코사인 0.7662/0.7636, rho 1.0000/0.2495

한계를 먼저 적는다
-----------------
- 조건 수가 적다(7~8). 조건 수준 상관은 **기술 통계**이고 검정이 아니다.
- 네 요약은 서로 독립이 아니다. 단조 사상은 L1 과 코사인을 같이 움직인다.
- '피해 있음'은 지연 p_holm<0.05 로 정의한다. 그 p 는 모델 수준이고
  23.9.5 의 중복 문제가 그대로 있다 — 중복축약 열도 같이 낸다.
- 분리력(간극)은 조건 7~8개로 계산한 것이라 새 조건이 들어오면 바뀐다.

출력: review_results/perturb_vs_damage.csv
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import review_stats as RS                            # noqa: E402
import reanalyze_cluster_n as RC                     # noqa: E402

OUT_DIR = RC.OUT_DIR
METRICS = RC.METRICS
BASE = "intact"
MAIN_CSV = "rank_vs_value.csv"
# 부분 실행(RV_TAG) 파일을 **이름으로 하드코딩하지 않는다**. 하드코딩하면
# 뒤에 만든 태그(axis 등)가 집계에서 조용히 빠진다 — reanalyze_cluster_n.py
# 에서 같은 함정을 한 번 겪었다. 실제로 있는 파일을 전부 읽는다.
GEO_CSV = "rank_value_geometry.csv"
# 사전 선언한 대조쌍: (이름, 조건A, 조건B, 고정축, 다른축)
PAIRS = (("가", "risk_rankpow50", "risk_rankpow400", "코사인", "노름비"),
         ("나", "risk_rankpow05", "risk_frozen", "코사인", "rho"))


def spearman(xs, ys):
    """조건 수준 n 이 7~8 이라 단순 구현. 동점은 평균순위.
    review_stats 에는 순위상관이 없고, measure_rank_vs_value 를 끌어오면
    torch 와 원본 network_env 까지 로드돼서 여기 둔다."""
    def rank(v):
        idx = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(idx):
            j = i
            while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    a, b = rank(list(xs)), rank(list(ys))
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((x - mb) ** 2 for x in b) ** 0.5
    return num / (da * db) if da * db else float("nan")


def read_geometry():
    """조건 -> 네 예측자. 없으면 빈 칸."""
    path = os.path.join(OUT_DIR, GEO_CSV)
    geo = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            c = row["condition"]
            if c.startswith("_") or c == "true_vector_ties":
                continue
            try:
                geo[c] = {
                    "l1": float(row["mean_abs_diff"]),
                    "normdev": abs(float(row["norm_ratio_vs_true"]) - 1.0),
                    "cosdev": 1.0 - float(row["cosine"]),
                    "rhodev": 1.0 - float(row["mean_rho"]),
                }
            except (KeyError, ValueError):
                continue
    return geo


def extra_csvs():
    """review_results/ 에 있는 rank_vs_value_*.csv 전부 (본 파일 제외)."""
    pat = os.path.join(OUT_DIR, "rank_vs_value_*.csv")
    out = []
    for p in sorted(glob.glob(pat)):
        base = os.path.basename(p)
        if base == MAIN_CSV or base.startswith("_"):
            continue
        out.append(base)
    return out


def merge_damage():
    """본 실행과 부분 실행들을 합친다. intact 일치를 전제로 검사한다."""
    data = RC.read_metrics(MAIN_CSV)
    note = []
    files = extra_csvs()
    if not files:
        note.append("부분 실행 파일 없음 (rank_vs_value_*.csv)")
        return data, note
    note.append("부분 실행 파일 %d개: %s" % (len(files), ", ".join(files)))
    worst = 0.0
    for fn in files:
        extra = RC.read_metrics(fn)
        common = sorted(set(data) & set(extra))
        added = set()
        for name in common:
            for met in METRICS:
                worst = max(worst, abs(data[name][BASE][met]
                                       - extra[name][BASE][met]))
            for cond, vals in extra[name].items():
                if cond != BASE:
                    data[name][cond] = vals
                    added.add(cond)
        note.append("  %s: 모델 %d개 공통 · 조건 %s"
                    % (fn, len(common), ",".join(sorted(added)) or "(없음)"))
    note.append("모든 파일의 intact 최대 불일치 %.6g "
                "(0 이어야 합친 게 옳다)" % worst)
    return data, note


def damage_table(data, conds):
    """조건 x 지표 -> (n=10 차이, p_holm, 축약 차이, p_holm)."""
    models = sorted(data.keys())
    rows, p1s, p5s = [], [], []
    for cond in conds:
        for met in METRICS:
            per = {}
            for nm in models:
                per[nm] = data[nm][cond][met] - data[nm][BASE][met]
            vals = [per[n] for n in models]
            m1, _s1, t1, _d1, pv1 = RC.one_sample(vals)
            dup = RC.collapse(per, lambda n: RC.DUP_GROUP[n])
            m5, _s5, t5, _d5, pv5 = RC.one_sample(dup)
            rows.append([cond, met, m1, t1, pv1, m5, t5, pv5, len(dup)])
            p1s.append(pv1)
            p5s.append(pv5)
    h1 = RS.holm_bonferroni(p1s)
    h5 = RS.holm_bonferroni(p5s)
    for i, r in enumerate(rows):
        r.extend([h1[i], h5[i]])
    return rows


def margin(geo, conds, hurt, key):
    """무해군 최악값과 피해군 최선값 사이 간극. 양수면 단일 문턱 분리."""
    ok = [geo[c][key] for c in conds if c in geo and c not in hurt]
    bad = [geo[c][key] for c in conds if c in geo and c in hurt]
    if not ok or not bad:
        return None, None, None
    return min(bad) - max(ok), max(ok), min(bad)


def main():
    geo = read_geometry()
    data, note = merge_damage()
    RC.duplication_report(MAIN_CSV, BASE)
    for nm in data:
        RC.DUP_GROUP.setdefault(nm, -1)
    conds = [c for c in RC.conds_in(MAIN_CSV, BASE)]
    for nm in data:
        for c in data[nm]:
            if c != BASE and c not in conds:
                conds.append(c)
    bar = "=" * 86
    print("")
    print(bar)
    print("[0] 입력")
    print("-" * 86)
    for s in note:
        print("  " + s)
    print("  기하학 조건 %d개 · 피해 조건 %d개 · 모델 %d개"
          % (len(geo), len(conds), len(data)))
    miss = [c for c in conds if c not in geo]
    if miss:
        print("  기하학 없는 조건(빈 칸으로 둔다): %s" % ", ".join(miss))

    rows = damage_table(data, conds)
    print("")
    print("[1] 조건별 피해 (intact 대응차 · Holm 족 %d)" % len(rows))
    print("-" * 86)
    print("%-18s %-6s %9s %8s | %9s %8s %4s"
          % ("조건", "지표", "차이", "p_holm", "축약차이", "p_holm", "n"))
    for r in rows:
        print("%-18s %-6s %9.4f %8s | %9.4f %8s %4d"
              % (r[0], r[1], r[2], RS.fmt_p(r[9]), r[5],
                 RS.fmt_p(r[10]), r[8]))
    # 피해 기준을 **세 벌** 만든다. 원래 이 스크립트는 '지연 p_holm<0.05'
    # 하나로 피해를 정의했는데, 23.9.6.10 에서 지연은 홉 단축에 **가려져**
    # 보정 전 n=5 를 못 넘고(0.0820), 손실이 홉당평균·합 **양쪽 괄호**에서
    # 서는 지표로 판정됐다. 그래서 기준을 바꾸지 않고 셋 다 찍어서
    # [3] 판정이 기준에 의존하는지 드러낸다.

    def hurt_set(met, idx):
        return set(r[0] for r in rows if r[1] == met and r[idx] < 0.05)

    CRIT = (("지연 n=10", hurt_set("delay", 9)),
            ("손실 n=10", hurt_set("loss", 9)),
            ("손실 n=5(최보수)", hurt_set("loss", 10)))
    hurt = CRIT[0][1]
    print("")
    for lab, hs in CRIT:
        print("  '피해'(%s p_holm<0.05, 족 %d): %s"
              % (lab, len(rows), ", ".join(sorted(hs)) if hs else "없음"))
    print("  (기준에 따라 소속이 바뀌는 조건: %s)"
          % (", ".join(sorted(
              set.union(*[h for _, h in CRIT])
              - set.intersection(*[h for _, h in CRIT]))) or "없음"))

    dl = dict((r[0], r[2]) for r in rows if r[1] == "delay")
    ls = dict((r[0], r[2]) for r in rows if r[1] == "loss")
    print("")
    print("[2] 교란 요약 네 개와 피해 (조건 수준 · 기술 통계)")
    print("-" * 86)
    print("%-18s %9s %9s %9s %9s %10s %8s"
          % ("조건", "L1", "|노름비-1|", "1-코사인", "1-rho",
             "지연차이", "피해"))
    for c in conds:
        g = geo.get(c)
        if g is None:
            print("%-18s %9s %9s %9s %9s %10.4f %8s"
                  % (c, "-", "-", "-", "-", dl.get(c, float("nan")),
                     "O" if c in hurt else "."))
            continue
        print("%-18s %9.5f %9.4f %9.4f %9.4f %10.4f %8s"
              % (c, g["l1"], g["normdev"], g["cosdev"], g["rhodev"],
                 dl.get(c, float("nan")), "O" if c in hurt else "."))
    both = [c for c in conds if c in geo]
    if len(both) >= 3:
        print("")
        for key, lab in (("l1", "L1"), ("normdev", "|노름비-1|"),
                         ("cosdev", "1-코사인"), ("rhodev", "1-rho")):
            xs = [geo[c][key] for c in both]
            # 지연과 손실 **둘 다** 찍는다. 지연은 홉에 가려진 지표이므로
            # (23.9.6.10) 지연 상관만 보고하면 한쪽에 치우친다.
            print("  %-10s 대 지연차이 rho = %+.4f · 손실차이 rho = %+.4f"
                  "  (조건 n=%d, 검정 아님)"
                  % (lab, spearman(xs, [dl[c] for c in both]),
                     spearman(xs, [ls[c] for c in both]), len(both)))

    print("")
    print("[3] 분리력 — 무해군 최악과 피해군 최선 사이 간극")
    print("-" * 86)
    print("  양수면 그 요약 하나로 피해 유무를 가를 수 있다(겹치지 않는다).")
    print("%-12s %10s %10s %10s %s"
          % ("요약", "간극", "무해최악", "피해최선", "판정"))
    print("  간극 0 은 '겹침'이 아니라 **동점**이다 — 같은 코사인을 가진 두")
    print("  조건이 문턱 양쪽에 걸리면 어떤 문턱도 못 가른다(구조적 한계).")
    seps = {}
    for clab, hs in CRIT:
        print("")
        print("  [기준: %s]  피해군 %d개 / 무해군 %d개"
              % (clab, len(hs), len([c for c in conds if c not in hs])))
        print("  %-12s %10s %10s %10s %s"
              % ("요약", "간극", "무해최악", "피해최선", "판정"))
        for key, lab in (("l1", "L1"), ("normdev", "|노름비-1|"),
                         ("cosdev", "1-코사인"), ("rhodev", "1-rho")):
            gap, okw, badb = margin(geo, conds, hs, key)
            seps[(clab, key)] = gap
            if gap is None:
                print("  %-12s %10s %10s %10s %s"
                      % (lab, "-", "-", "-", "계산 불가(한쪽 군이 빈다)"))
                continue
            vd = ("분리" if gap > 0 else
                  ("동점 -> 문턱 불가" if abs(gap) < 1e-9 else
                   "겹침 -> 설명 못 함"))
            print("  %-12s %+10.4f %10.4f %10.4f %s"
                  % (lab, gap, okw, badb, vd))

    print("")
    print("[4] 사전 선언 대조쌍 판정")
    print("-" * 86)
    for tag, ca, cb, fixed, moved in PAIRS:
        if ca not in dl or cb not in dl:
            print("  (%s) %s 대 %s — 아직 측정 없음" % (tag, ca, cb))
            continue
        ha = "피해" if ca in hurt else "무해"
        hb = "피해" if cb in hurt else "무해"
        print("  (%s) %s(%s, 지연 %+.2f) 대 %s(%s, 지연 %+.2f)"
              % (tag, ca, ha, dl[ca], cb, hb, dl[cb]))
        print("      %s 는 맞췄고 %s 만 다르다 -> "
              % (fixed, moved), end="")
        if ha == hb:
            print("%s 축은 피해를 안 가른다" % moved)
        else:
            print("%s 축이 피해를 가른다" % moved)

    path = os.path.join(OUT_DIR, "perturb_vs_damage.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "metric", "diff_n10", "t_n10", "p_holm_n10",
                    "diff_collapsed", "t_collapsed", "p_holm_collapsed",
                    "n_collapsed", "l1", "norm_dev", "cos_dev", "rho_dev",
                    "hurt_delay_n10", "hurt_loss_n10", "hurt_loss_n5",
                    "holm_family"])
        for r in rows:
            g = geo.get(r[0], {})
            w.writerow([r[0], r[1], "%.4f" % r[2], "%.3f" % r[3],
                        "%.6f" % r[9], "%.4f" % r[5], "%.3f" % r[6],
                        "%.6f" % r[10], r[8],
                        "%.6f" % g["l1"] if g else "",
                        "%.6f" % g["normdev"] if g else "",
                        "%.6f" % g["cosdev"] if g else "",
                        "%.6f" % g["rhodev"] if g else "",
                        1 if r[0] in CRIT[0][1] else 0,
                        1 if r[0] in CRIT[1][1] else 0,
                        1 if r[0] in CRIT[2][1] else 0,
                        len(rows)])
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
