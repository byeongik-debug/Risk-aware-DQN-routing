# -*- coding: utf-8 -*-
u"""표 10 `w/o IC` 행이 왜 **정확히** 0 인가 — 그 근거 수치를 산출물로 남긴다.

왜 다시 재는가. 초안(`DRAFT_4_6_2_table10.md` L201)과 기록
(`FINDINGS_table10_f1.md` 3절)은 혼잡 지표가 중복인 근거로 세 숫자를 쓴다 —
혼잡 표시 링크가 t>=1 전체의 **4.81%**, 그중 **76.62%** 가 이용률 상한에
포화, **83.11%** 가 유형 위험 0.8. 그런데 이 셋을 쓴 스크립트가 없다.
그때 즉석으로 계산해 문서에 옮겨 적었고 CSV 가 남지 않았다. 감사
(32절)에서 `DOC_ONLY` 경보로 뜬다 — 내 문서에 적힌 것은 출처가 아니다.

여기서는 새로 측정하지 않는다. `build_table10_f1.py` 가 남긴 원시 로그
캐시(`_table10_rawlog.pkl`, 그 표를 만든 바로 그 한 번의 환경 통과)를 읽어
같은 정의로 다시 집계한다. 캐시가 없으면 `collect()` 를 불러 다시 만든다.

정의(문서에 적힌 문장 그대로):
  - 분모는 t>=1 의 (스냅숏 x 링크) 칸. t=0 은 reset 이라 danger 유형이 아직
    안 붙는다(그래서 문서도 "t>=1 에서" 라고 적었다).
  - 포화: `utilization >= 1.0`. 환경은 `np.clip(util, 0, 1.5)` 지만 위험식은
    `min(util, 1.0)` 으로 자른다 — 1.0 이상은 전부 같은 값이 된다.
  - 유형 위험 0.8 = `congested`.
원본 폴더는 읽기만 한다. 출력은 review_results/ 로만.
"""
import csv
import io
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "review_results")
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_table10_f1 as B   # noqa: E402

SAT = 1.0          # 위험식이 자르는 상한
TRISK_CONG = 0.8   # TYPE_RISK["congested"]


def load():
    if os.path.exists(B.CACHE):
        with open(B.CACHE, "rb") as fh:
            eps = pickle.load(fh)
        print(u"캐시 사용: %s (에피소드 %d개)" % (B.CACHE, len(eps)))
        return eps
    print(u"캐시 없음 — collect() 로 다시 만든다")
    return B.collect(B.SEEDS)


def main():
    eps = load()

    cong, util, trisk, lab = [], [], [], []
    n_snap = 0
    for e in eps:
        nt = e["util"].shape[0]
        n_snap += nt
        # t>=1 만. 라벨은 링크당 하나라 시간축으로 펼친다.
        cong.append(e["cong"][1:].ravel())
        util.append(e["util"][1:].ravel())
        trisk.append(e["trisk"][1:].ravel())
        lab.append(np.tile(e["label"], (nt - 1, 1)).ravel())

    cong = np.concatenate(cong)
    util = np.concatenate(util)
    trisk = np.concatenate(trisk)
    lab = np.concatenate(lab)
    n = cong.size
    m = cong > 0.5

    rows = [
        dict(name="snapshots_total", value="%d" % n_snap,
             note=u"스냅숏 수(t=0 포함) = 시드 10 x 에피소드 30 x (1+30)"),
        dict(name="cells_t_ge_1", value="%d" % n,
             note=u"t>=1 의 (스냅숏 x 링크) 칸 수 = 분모"),
        dict(name="congested_cells", value="%d" % int(m.sum()),
             note=u"is_congested = 1 인 칸"),
        dict(name="congested_pct", value="%.4f" % (100.0 * m.mean()),
             note=u"혼잡 표시 비율 (초안: 4.81%)"),
        dict(name="cong_label1_pct",
             value="%.4f" % (100.0 * float((lab[m] == 1).mean())),
             note=u"혼잡 칸 중 danger_label=1 비율 (초안: 100.00%)"),
        dict(name="cong_util_saturated_pct",
             value="%.4f" % (100.0 * float((util[m] >= SAT).mean())),
             note=u"혼잡 칸 중 이용률 >= 1.0 비율 (초안: 76.62%)"),
        dict(name="cong_trisk08_pct",
             value="%.4f" % (100.0 * float(
                 (np.abs(trisk[m] - TRISK_CONG) < 1e-12).mean())),
             note=u"혼잡 칸 중 유형 위험 0.8 비율 (초안: 83.11%)"),
        dict(name="cong_either_pct",
             value="%.4f" % (100.0 * float(
                 ((util[m] >= SAT)
                  | (np.abs(trisk[m] - TRISK_CONG) < 1e-12)).mean())),
             note=u"혼잡 칸 중 둘 중 하나라도 해당"),
        dict(name="cong_both_pct",
             value="%.4f" % (100.0 * float(
                 ((util[m] >= SAT)
                  & (np.abs(trisk[m] - TRISK_CONG) < 1e-12)).mean())),
             note=u"혼잡 칸 중 둘 다 해당"),
        dict(name="noncong_util_saturated_pct",
             value="%.4f" % (100.0 * float((util[~m] >= SAT).mean())),
             note=u"대조: 혼잡 아닌 칸의 포화 비율"),
    ]

    out = os.path.join(RES, "ic_redundancy.csv")
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["name", "value", "note"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    for r in rows:
        print(u"  %-28s %12s  %s" % (r["name"], r["value"], r["note"]))
    print(u"\n저장: %s (%d행)" % (out, len(rows)))


if __name__ == "__main__":
    sys.exit(main())
