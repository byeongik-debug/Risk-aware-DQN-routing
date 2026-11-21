# -*- coding: utf-8 -*-
"""`policy_path_distance_pairs.csv` 를 집계해 초안이 인용할 칸을 만든다.

초안에 숫자를 넣으려면 그 값이 집계 CSV 의 **칸**으로 존재해야 한다
(`audit_draft_numbers.py` 규약 — 문서에 적는 것은 출처가 아니다). 31.10 의
[2] 블록과 팔별 응집 표는 쌍 CSV 에서 평균·중앙·최소·최대를 다시 계산한 값이라
칸이 없었다. 그래서 여기서 집계해 저장한다. 손으로 옮기지 않는다.
"""
from __future__ import print_function

import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "review_results")
SRC = os.path.join(RD, "policy_path_distance_pairs.csv")
OUT = os.path.join(RD, "policy_path_distance_summary.csv")

COND = ["cond_d_edit", "cond_d_jacc_edge", "cond_d_hop", "cond_shared_prefix"]
UNCOND = ["d_edit", "d_jacc_edge", "d_hop", "shared_prefix", "diff_pct",
          "same_pct"]


def stats(vs):
    vs = sorted(vs)
    n = len(vs)
    return dict(n=n, mean=sum(vs) / n, median=vs[n // 2],
                vmin=vs[0], vmax=vs[-1])


def main():
    rows = list(csv.DictReader(io.open(SRC, encoding="utf-8", newline="")))
    out = []

    for k in COND:
        vs = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        st = stats(vs)
        out.append(dict(scope="불일치 에피소드만", group="전체 쌍", metric=k, **st))
    for k in UNCOND:
        vs = [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        st = stats(vs)
        out.append(dict(scope="전체 에피소드", group="전체 쌍", metric=k, **st))

    arms = sorted(set(r["arm_a"] for r in rows))
    for arm in arms:
        win = [r for r in rows if r["arm_a"] == arm and r["arm_b"] == arm]
        bet = [r for r in rows if (r["arm_a"] == arm) != (r["arm_b"] == arm)]
        for grp, sub in (("군내", win), ("군간", bet)):
            if not sub:
                continue
            for k in ("same_pct", "d_edit", "d_jacc_edge", "shared_prefix"):
                vs = [float(r[k]) for r in sub if r[k] not in ("", "nan")]
                if vs:
                    out.append(dict(scope="팔별 응집", group=grp + " " + arm,
                                    metric=k, **stats(vs)))

    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["scope", "group", "metric", "n",
                                           "mean", "median", "vmin", "vmax"])
        w.writeheader()
        for r in out:
            for k in ("mean", "median", "vmin", "vmax"):
                r[k] = "%.6f" % r[k]
            w.writerow(r)
    print("저장: %s (%d행)" % (OUT, len(out)))
    for r in out[:4]:
        print("  %-18s %-12s 평균 %s  중앙 %s  (n=%s)"
              % (r["scope"], r["metric"], r["mean"], r["median"], r["n"]))


if __name__ == "__main__":
    main()
