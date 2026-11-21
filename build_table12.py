# -*- coding: utf-8 -*-
"""Table 12 (path setup time) from the two measured runs.

Both source files already exist; this only reshapes them and computes the
ratios, so that the ratio that goes into the paper is not one I quoted from
memory.

The one thing this script is built to make unmissable: Risk-DQN's cost is
per-decision x hops (it chooses at every hop, while OSPF and QoS-Dijkstra
compute a path once), and the per-decision cost moves by ~3.4x with the torch
thread count while OSPF barely moves. So the headline "N times slower" is not a
property of the method alone -- it is a property of the method *and* the thread
setting, which reviewer note 28 requires us to state.

Usage:
    python -X utf8 build_table12.py
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
SCEN = ["normal", "congestion", "unstable", "dynamic"]
MAIN = ["OSPF", "QoS-Dijkstra", "Risk-DQN"]


def load(path, thread_col):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["scenario"], row["method"])] = row
    return out


def main():
    d12 = load(os.path.join(RES, "path_setup_time.csv"), None)
    d1 = load(os.path.join(RES, "path_setup_time_threads1.csv"), "torch_threads")

    rows = []
    for sc in SCEN:
        for m in MAIN:
            a, b = d12[(sc, m)], d1[(sc, m)]
            rows.append({
                "scenario": sc,
                "method": m,
                "n_paths": int(float(a["n_paths"])),
                # The table reports the 1-thread run, so its decision count has to
                # come from that run too.  This used to read a["decisions_mean"],
                # the 12-thread run's count: dynamic averaged 7.7 decisions there
                # and 7.6 here, so the Dynamic row printed 7.7 decisions beside a
                # 1-thread 2.664 ms and 0.350 ms/decision -- 2.664/7.7 = 0.346, a
                # row that disagreed with itself.  The 12-thread count is kept as
                # its own column instead of being dropped.
                "decisions": float(b["decisions_mean"]),
                "decisions_t12": float(a["decisions_mean"]),
                "per_decision_ms_t12": float(a["per_decision_ms_mean"]),
                "per_decision_ms_t1": float(b["per_decision_ms_mean"]),
                "per_path_ms_t12": float(a["per_path_ms_mean"]),
                "per_path_ms_std_t12": float(a["per_path_ms_std"]),
                "per_path_ms_t1": float(b["per_path_ms_mean"]),
                "per_path_ms_std_t1": float(b["per_path_ms_std"]),
                # 4.7 (3) quotes "forward is 53-55% of a decision".  That share is a
                # ratio of two measured columns, so it is computed here rather than
                # by hand; decision_breakdown_* reproduces it independently (52.8%).
                # The internal-consistency check that caught the mixed pairing: if
                # the 12-thread decision count belongs in this row, then dividing
                # this row's own per-path time by it has to give this row's own
                # per-decision time.  It does not (0.346 vs 0.350).
                "per_decision_if_mixed_t1": (
                    float(b["per_path_ms_mean"]) / float(a["decisions_mean"])
                    if m == "Risk-DQN" else ""),
                "forward_pct_of_decision_t1": (
                    float(d1[(sc, "Risk-DQN forward only")]["per_decision_ms_mean"])
                    / float(b["per_decision_ms_mean"]) * 100.0
                    if m == "Risk-DQN" else ""),
            })

    for r in rows:
        o12 = [x for x in rows if x["scenario"] == r["scenario"]
               and x["method"] == "OSPF"][0]
        r["ratio_vs_ospf_t12"] = r["per_path_ms_t12"] / o12["per_path_ms_t12"]
        r["ratio_vs_ospf_t1"] = r["per_path_ms_t1"] / o12["per_path_ms_t1"]
        r["thread_inflation"] = r["per_path_ms_t12"] / r["per_path_ms_t1"]

    out = os.path.join(RES, "table12_path_setup.csv")
    cols = ["scenario", "method", "n_paths", "decisions",
            "per_path_ms_t1", "per_path_ms_std_t1", "ratio_vs_ospf_t1",
            "per_path_ms_t12", "per_path_ms_std_t12", "ratio_vs_ospf_t12",
            "per_decision_ms_t1", "per_decision_ms_t12", "thread_inflation",
            "forward_pct_of_decision_t1", "per_decision_if_mixed_t1"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(dict((c, r[c]) for c in cols))

    # 4.7 (2) quotes a regression of per-path time on decision count.  It is a
    # function of this table's own four rows, so it is computed here instead of
    # being quoted by hand -- and both vectors are written out, because the
    # published fit (0.513 + 0.281, R2 0.993) turned out to be the one you get by
    # pairing 1-thread times with the 12-thread decision count.
    dqn = [r for r in rows if r["method"] == "Risk-DQN"]
    reg = []
    for label, xkey in (("t1_decisions", "decisions"),
                        ("mixed_t12_decisions", "decisions_t12")):
        xs = [r[xkey] for r in dqn]
        ys = [r["per_path_ms_t1"] for r in dqn]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        slope = sxy / sxx
        icept = my - slope * mx
        syy = sum((y - my) ** 2 for y in ys)
        ssr = sum((y - (icept + slope * x)) ** 2 for x, y in zip(xs, ys))
        reg.append({"x_source": label, "n_points": n,
                    "intercept_ms": icept, "slope_ms_per_decision": slope,
                    "r2": 1.0 - ssr / syy if syy else float("nan"),
                    "x_min": min(xs), "x_max": max(xs)})
    rout = os.path.join(RES, "table12_regression.csv")
    with open(rout, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(reg[0].keys()))
        w.writeheader()
        for r in reg:
            w.writerow(r)

    print("표 12 — 경로 설정 시간 (단일 스레드 기준, n=30 경로/시나리오)\n")
    hdr = "%-11s | %-13s | %6s | %8s | %7s | %8s" % (
        "시나리오", "방법", "결정수", "ms/경로", "배수", "ms/결정")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print("%-11s | %-13s | %6.1f | %5.3f±%.3f | %6.1fx | %8.3f" % (
            r["scenario"], r["method"], r["decisions"],
            r["per_path_ms_t1"], r["per_path_ms_std_t1"],
            r["ratio_vs_ospf_t1"], r["per_decision_ms_t1"]))

    print("\n스레드 수에 따른 변동 (같은 코드, 같은 모델)")
    print("%-11s | %-13s | %9s | %9s | %7s" % (
        "시나리오", "방법", "1스레드", "12스레드", "팽창"))
    for r in rows:
        print("%-11s | %-13s | %7.3f ms | %7.3f ms | %5.2fx" % (
            r["scenario"], r["method"], r["per_path_ms_t1"],
            r["per_path_ms_t12"], r["thread_inflation"]))

    dqn = [r for r in rows if r["method"] == "Risk-DQN"]
    print("\n요약")
    print("  Risk-DQN 대 OSPF:  1스레드 %.1f~%.1f배,  12스레드 %.1f~%.1f배"
          % (min(r["ratio_vs_ospf_t1"] for r in dqn),
             max(r["ratio_vs_ospf_t1"] for r in dqn),
             min(r["ratio_vs_ospf_t12"] for r in dqn),
             max(r["ratio_vs_ospf_t12"] for r in dqn)))
    print("  → 같은 모델·같은 코드인데 보고 배수가 %.1f배 차이 난다."
          % (max(r["ratio_vs_ospf_t12"] for r in dqn)
             / min(r["ratio_vs_ospf_t1"] for r in dqn)))
    print("  절대값: Risk-DQN %.2f~%.2f ms/경로 (1스레드), 홉마다 %.3f~%.3f ms"
          % (min(r["per_path_ms_t1"] for r in dqn),
             max(r["per_path_ms_t1"] for r in dqn),
             min(r["per_decision_ms_t1"] for r in dqn),
             max(r["per_decision_ms_t1"] for r in dqn)))
    print("\n4.7 (2) 회귀 — 경로당 = 절편 + 기울기 x 결정 횟수")
    for r in reg:
        print("  %-20s 절편 %7.4f  기울기 %7.4f  R2 %.4f  (x %.1f~%.1f)" % (
            r["x_source"], r["intercept_ms"], r["slope_ms_per_decision"],
            r["r2"], r["x_min"], r["x_max"]))
    print("  → 발표된 0.513 + 0.281 (R2 0.993) 은 mixed 쪽이다. 절편 자체는"
          " decision_breakdown_* 에서 철회됐다(p 0.75, 복제 간 부호 반전).")
    print("\nwrote %s" % out)
    print("wrote %s" % rout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
