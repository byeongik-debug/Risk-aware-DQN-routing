# -*- coding: utf-8 -*-
"""23절의 대응 검정을 '정책 클래스' 단위로 다시 잰다 (환경 재실행 없음).

왜 필요한가
-----------
23절의 모든 Holm 판정은 모델 10개를 **독립 표본**으로 보고 n=10, df=9 로 했다.
그런데 22.4절이 이미 측정해둔 사실은 그 10개가 방문 상태에서 **정책 3개**라는
것이다(클래스 1: 6개, 클래스 2: 3개, 클래스 4: 1개). 실제로 200에피소드 본실행에서
intact 지연이 **350.7908 로 6개 모델이 소수점까지 동일**하다.

같은 정책을 여섯 번 센 것이라면 대응차 10개는 독립 관측 10개가 아니다. 중복은
모델간 sd 를 실제보다 **작게** 만들고(같은 값이 반복되므로) t = m/(sd/sqrt(n)) 의
분모를 줄이고 n 을 늘려 **t 를 양쪽에서 부풀린다**. 따라서 23절의 유의성이
군집화를 견디는지 **재서 확인해야** 한다. 견디면 23절이 강해지고, 안 견디면
23절의 p 값을 전부 약화해 써야 한다.

방법
----
롤아웃을 다시 돌리지 않는다. 이미 저장된 모델별 지표를 읽어
(1) 모델 수준 n=10, df=9 (23절 재현)
(2) 클래스 수준 n=3, df=2 — 클래스 안에서 대응차를 평균해 관측 하나로 축약
둘을 같은 Holm 족 안에서 나란히 낸다. 클래스 소속은 22.4절의 greedy 묶음
(`review_results/policy_classes.csv`) 그대로 쓴다.

읽는 법: 클래스 수준에서도 유의하면 "효과가 세 정책 전부에서 같은 방향"이라는
뜻이고 23절 결론은 유지된다. 클래스 수준에서 떨어지면 23절의 별표는 **같은 정책을
여섯 번 센 것**이 만든 것이므로 23절을 약화해 써야 한다.
"""
import csv
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
import review_stats as RS                            # noqa: E402

OUT_DIR = os.path.join(HERE, "review_results")
METRICS = ("delay", "loss", "hop")

# 22.4절 greedy 묶음 결과. (팔, 런) -> 클래스
CLASS_MAP = {
    ("factor_full", 1): 1, ("factor_full", 2): 1,
    ("factor_full", 3): 1, ("factor_full", 4): 1,
    ("window5", 1): 1, ("window5", 2): 1,
    ("factor_full", 5): 2, ("window5", 3): 2, ("window5", 5): 2,
    ("window5", 4): 4,
}


def model_class(name):
    """'window5_run03' -> 클래스 번호."""
    arm, _, run = name.rpartition("_run")
    return CLASS_MAP[(arm, int(run))]


def load_class_map_from_csv():
    """policy_classes.csv 의 greedy 행으로 CLASS_MAP 을 검증한다."""
    path = os.path.join(OUT_DIR, "policy_classes.csv")
    got = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["battery"] != "greedy":
                continue
            key = (row["arm"], int(row["run_id"]))
            if key in CLASS_MAP:
                got[key] = int(row["class_id"])
    if got != CLASS_MAP:
        raise SystemExit("CLASS_MAP 이 policy_classes.csv 와 다르다: %r" % (got,))
    return len(got)


def read_metrics(fname):
    """{모델: {조건: {지표: 값}}} 로 읽는다."""
    out = {}
    fp = os.path.join(OUT_DIR, fname)
    with open(fp, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            m = out.setdefault(row["model"], {})
            m[row["condition"]] = dict((k, float(row[k])) for k in METRICS)
    return out


def one_sample(pairs):
    """대응차 리스트 -> (평균, sd, t, df, p). 23절 세 스크립트와 같은 규약."""
    n = len(pairs)
    m, sd = RS.mean_std(pairs)
    if sd == 0.0:
        return m, sd, float("inf") if m else 0.0, n - 1, 0.0 if m else 1.0
    t = m / (sd / n ** 0.5)
    return m, sd, t, n - 1, RS.t_two_sided_p(t, n - 1)


# intact 롤아웃이 글자 단위로 같은 모델 묶음. duplication_report 가 채운다.
DUP_GROUP = {}


def collapse(per_model, keyfn):
    """모델별 대응차 -> 묶음별 평균 (묶음 키 순)."""
    buckets = {}
    for name, val in per_model.items():
        buckets.setdefault(keyfn(name), []).append(val)
    return [sum(v) / len(v) for _, v in sorted(buckets.items())]


def analyze(fname, baseline, conds, label):
    data = read_metrics(fname)
    models = sorted(data.keys())
    rows = []
    pm_p, pd_p, pc_p = [], [], []
    for cond in conds:
        for met in METRICS:
            per_model = {}
            for name in models:
                per_model[name] = (data[name][cond][met]
                                   - data[name][baseline][met])
            vals = [per_model[n] for n in models]
            m1, sd1, t1, _d1, p1 = one_sample(vals)
            dup = collapse(per_model, lambda n: DUP_GROUP[n])
            m5, sd5, t5, _d5, p5 = one_sample(dup)
            cls = collapse(per_model, model_class)
            m2, sd2, t2, _d2, p2 = one_sample(cls)
            rows.append([cond, met, m1, sd1, t1, p1, m5, sd5, t5, p5,
                         m2, sd2, t2, p2, len(set(round(v, 6) for v in vals))])
            pm_p.append(p1)
            pd_p.append(p5)
            pc_p.append(p2)
    h1 = RS.holm_bonferroni(pm_p)
    h5 = RS.holm_bonferroni(pd_p)
    h2 = RS.holm_bonferroni(pc_p)
    for i, r in enumerate(rows):
        r.append(h1[i])
        r.append(h5[i])
        r.append(h2[i])
    print("")
    print("=" * 94)
    print("[%s] %s  (기준 %s · Holm 족 %d · 세 수준 각각 따로 보정)"
          % (label, fname, baseline, len(rows)))
    print("%-16s %-6s %9s %9s %8s | %9s %8s | %9s %8s | %s"
          % ("조건", "지표", "차이", "t(n=10)", "p_holm", "t(n=5)", "p_holm",
             "t(n=3)", "p_holm", "다른차이"))
    for r in rows:
        (cond, met, m1, _s1, t1, _p1, _m5, _s5, t5, _p5,
         _m2, _s2, t2, _p2, nuniq, ph1, ph5, ph2) = r
        print("%-16s %-6s %9.4f %9.3f %8s | %9.3f %8s | %9.3f %8s | %d/10"
              % (cond, met, m1, t1, RS.fmt_p(ph1), t5, RS.fmt_p(ph5),
                 t2, RS.fmt_p(ph2), nuniq))
    return rows


def conds_in(fname, baseline):
    """CSV 에 실제로 있는 조건(기준 제외, 파일 등장 순)."""
    data = read_metrics(fname)
    seen = []
    for name in sorted(data.keys()):
        for c in data[name]:
            if c != baseline and c not in seen:
                seen.append(c)
    return tuple(seen)


def duplication_report(fname, baseline):
    """intact 지표가 몇 개 모델에서 글자 단위로 같은가."""
    data = read_metrics(fname)
    groups = {}
    for name, conds in data.items():
        key = tuple(round(conds[baseline][m], 9) for m in METRICS)
        groups.setdefault(key, []).append(name)
    print("")
    print("[중복 진단] %s 의 %s 지표가 같은 모델 묶음" % (fname, baseline))
    for gid, (_k, names) in enumerate(
            sorted(groups.items(), key=lambda kv: -len(kv[1]))):
        for nm in names:
            DUP_GROUP[nm] = gid
    for key, names in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print("  지연 %9.4f · 손실 %.6f · 홉 %.4f  ->  %d개  %s"
              % (key[0], key[1], key[2], len(names), ", ".join(sorted(names))))
    return groups


def main():
    n_chk = load_class_map_from_csv()
    print("[0] 전제")
    print("  22.4절 greedy 클래스 소속을 policy_classes.csv 로 재확인: %d/10 일치" % n_chk)
    print("  클래스 1 = 6개 모델 · 클래스 2 = 3개 · 클래스 4 = 1개")
    print("  t 임계값: df=9 -> %.3f · df=4 -> %.3f · df=2 -> %.3f"
          % (RS.t_critical(9), RS.t_critical(4), RS.t_critical(2)))
    print("  세 수준을 나란히 낸다:")
    print("   n=10 : 23절이 쓴 수준. 모델 하나 = 관측 하나.")
    print("   n=5  : intact 롤아웃이 글자 단위로 같은 모델을 하나로 묶은 수준.")
    print("          중복이 실측된 것만 묶으므로 근거가 가장 단단하다.")
    print("   n=3  : 22.4절 greedy 정책 클래스 수준. 가장 보수적인 하한.")
    print("          단 클래스 2의 세 모델은 intact 지연이 서로 다르다")
    print("          (373.92/369.51/380.99) - 250상태 탐침에서만 같았다는 뜻이므로")
    print("          이 수준은 진짜 중복보다 과하게 묶는다.")

    duplication_report("state_occlusion_rollout.csv", "intact")

    rows = []
    rows += [["occlusion"] + r for r in analyze(
        "state_occlusion_rollout.csv", "intact",
        ("risk_zero", "risk_shuffle", "risk_mean",
         "curr_zero", "visited_shuffle", "dest_zero"), "1")]
    rows += [["frozen_quality"] + r for r in analyze(
        "risk_frozen_quality.csv", "intact",
        ("risk_frozen", "risk_mean", "risk_shuffle"), "2")]
    # 조건을 하드코딩하면 뒤에 추가한 조건(rankpow*)이 집계에서 조용히
    # 빠진다. CSV 에 실제로 있는 조건을 읽어 쓴다. RV_CONDS 부분 실행
    # 결과(rank_vs_value_pow2.csv)도 있으면 같은 기준으로 같이 분석한다.
    # 태그 파일 이름을 하드코딩하면 뒤에 만든 태그가 조용히 빠진다.
    # 실제로 있는 rank_vs_value*.csv 를 전부 읽는다(`_` 로 시작하는
    # 백업/로그는 뺀다).
    _files = ["rank_vs_value.csv"]
    for _p in sorted(glob.glob(os.path.join(OUT_DIR, "rank_vs_value_*.csv"))):
        _b = os.path.basename(_p)
        if not _b.startswith("_"):
            _files.append(_b)
    for i, fn in enumerate(_files):
        if not os.path.exists(os.path.join(OUT_DIR, fn)):
            continue
        conds = conds_in(fn, "intact")
        if not conds:
            continue
        rows += [[fn[:-4]] + r for r in analyze(
            fn, "intact", conds, "3" + (chr(ord("a") + i) if i else ""))]

    print("")
    print("=" * 94)
    print("[4] 판정 — 어느 주장이 가장 보수적인 수준에서도 서는가")
    allsig, mid, only10 = [], [], []
    for r in rows:
        tag = "%-14s %-16s %-6s" % (r[0], r[1], r[2])
        p1, p5, p2 = r[16], r[17], r[18]
        trip = (tag, p1, p5, p2)
        if p1 < 0.05 and p5 < 0.05 and p2 < 0.05:
            allsig.append(trip)
        elif p1 < 0.05 and p5 < 0.05:
            mid.append(trip)
        elif p1 < 0.05:
            only10.append(trip)

    def dump(items, none_msg):
        if not items:
            print("    %s" % none_msg)
        for tag, p1, p5, p2 in items:
            print("    %s  %s / %s / %s"
                  % (tag, RS.fmt_p(p1), RS.fmt_p(p5), RS.fmt_p(p2)))

    print("")
    print("  (가) 세 수준 전부 유의 — 군집화와 무관하게 선다:")
    dump(allsig, "없음")
    print("")
    print("  (나) n=10·n=5 유의, n=3 에서 떨어짐 — 실측 중복은 견디고"
          " 클래스 축약은 못 견딤:")
    dump(mid, "없음")
    print("")
    print("  (다) n=10 에서만 유의 — 중복 모델이 만든 별표다. 쓰면 안 된다:")
    dump(only10, "없음 — n=10 의 모든 별표가 실측 중복 축약을 견딘다")

    out = os.path.join(OUT_DIR, "cluster_n_reanalysis.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "condition", "metric",
                    "diff_n10", "sd_n10", "t_n10", "p_n10",
                    "diff_n5", "sd_n5", "t_n5", "p_n5",
                    "diff_n3", "sd_n3", "t_n3", "p_n3",
                    "n_unique_of_10",
                    "p_holm_n10", "p_holm_n5", "p_holm_n3"])
        for r in rows:
            w.writerow(r)
    print("")
    print("저장: %s" % out)


if __name__ == "__main__":
    main()
