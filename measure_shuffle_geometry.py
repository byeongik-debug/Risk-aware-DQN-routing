# -*- coding: utf-8 -*-
"""**부분 섞기** 족의 기하를 잰다 — 롤아웃 전에 k 를 고르기 위해.

왜 필요한가
-----------
23.9.6.15 에서 `risk_frozen`(피해 +165.5) 과 `risk_rankpow05n`(+94.7) 이
`cos_dev` 0.2364 대 0.2338, `norm_dev` 0.0005 대 0.0000 으로 **두 축이
맞춰진 채** 1.75배 갈린다. 둘을 가르는 것은 `rho_dev`(0.7505 대 0.0000)
이고 `l1` 은 **반대 방향**(0.1463 < 0.2685)이다. 그러면 "코사인·노름을
맞춰도 순위 교란이 따로 피해를 만든다" 가 가설이 된다.

그런데 `frozen` 은 **다른 모델(공여)의 값**을 넣으므로 주변분포까지
같이 바뀐다 — 교란이 남아 있다. `measure_rank_vs_value.py` 의 새
`SHUFK` 족(50칸 중 k칸만 섞기)은 값 다중집합을 **정확히** 보존하므로
노름비가 1.0000 이고 주변분포가 동일하다. 코사인과 rho 만 k 로 함께
움직인다. 따라서 **코사인이 `pow05n` 과 같아지는 k** 를 쓰면
"코사인·노름·주변분포가 전부 같고 순위만 다른" 쌍이 생긴다.

이 스크립트는 **정책 롤아웃을 하지 않는다**. 기하는 상태의 위험 블록과
치환만의 함수이므로, 90분짜리 롤아웃을 던지기 전에 k 를 고를 수 있다.

전제 맞추기: `measure_rank_value_geometry.py` 와 **같은 배터리·같은
추출 수**(RVG_DRAW, 기본 2000)를 쓴다. 그래야 여기서 나온 cos_dev 를
`rank_value_geometry.csv` 의 기존 조건 값과 나란히 읽을 수 있다.

한계
----
- 이 기하는 배터리 상태(greedy 롤아웃으로 모은 상태) 위에서 잰 평균이다.
  실제 피해 측정은 조건마다 **다른 경로**를 걸으므로 그때 보는 상태 분포는
  다르다. 그래서 여기 값은 **k 선택용**이고, 최종 기하는 기존 파이프라인
  (`measure_rank_value_geometry.py`)으로 다시 재야 한다.
- 고정점이 생길 수 있다(k칸 안의 균등 치환이므로). "k칸이 전부 움직인다"
  가 아니라 "k칸 안에서 재배치된다" 다. 실제로 움직인 칸 비율을 같이 찍는다.

출력: review_results/shuffle_geometry.csv
환경변수: RVG_DRAW(기본 2000)
"""
import csv
import io
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import occlude_state_blocks as OC  # noqa: E402

sys.path.append(OC.HERE)
import measure_rank_vs_value as RV  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

LO, HI = RV.LO, RV.HI
N_DRAW = int(os.environ.get("RVG_DRAW", "2000"))
# 섞기 족 + 비교 기준점 셋
SHUF = tuple(sorted(RV.SHUFK, key=lambda c: RV.SHUFK[c]))
REFC = ("risk_dimshuffle", "risk_rankpow05n", "risk_rankpow05",
        "risk_frozen")
CONDS = SHUF + REFC
TARGET = "risk_rankpow05n"


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()
    net = OC.load_net(os.path.join(OC.ORIG_MODELS, "window5_run01.pth"),
                      device)
    pool = [s for s, _ in OC.build_battery("greedy", device, net)]
    bar = "=" * 92
    print(bar)
    print("부분 섞기 기하 — 롤아웃 없이 k 를 고른다")
    print("  배터리 상태 %d · 추출 %d · 블록 칸 %d"
          % (len(pool), N_DRAW, HI - LO))
    print(bar)

    rng = random.Random(RV.EVAL_SEED)
    keys = ("l1", "cos", "rho", "rt", "moved")
    acc = dict((c, dict((k, []) for k in keys)) for c in CONDS)
    for _ in range(N_DRAW):
        state = pool[rng.randrange(len(pool))]
        donor = pool[rng.randrange(len(pool))]
        true = np.asarray(state[LO:HI], dtype=np.float64)
        tnorm = float(np.sqrt((true ** 2).sum()))
        perm = np.random.RandomState(
            rng.randrange(1 << 30)).permutation(HI - LO)
        for c in CONDS:
            fed = np.asarray(RV.make_fed(state, c, donor, perm)[LO:HI],
                             dtype=np.float64)
            d = fed - true
            acc[c]["l1"].append(float(np.abs(d).mean()))
            acc[c]["rho"].append(RV.spearman_np(fed, true))
            fnorm = float(np.sqrt((fed ** 2).sum()))
            acc[c]["rt"].append(fnorm / tnorm if tnorm else float("nan"))
            den = fnorm * tnorm
            acc[c]["cos"].append(float(np.dot(fed, true) / den)
                                 if den > 0 else float("nan"))
            acc[c]["moved"].append(100.0 * float((np.abs(d) > 1e-9).mean()))

    def m(c, k):
        return float(np.mean(acc[c][k]))

    print("")
    print("[1] 족 전체 — 노름비가 1.0000 인지가 설계의 전제다")
    print("-" * 92)
    print("%-20s %5s %9s %9s %9s %9s %9s %8s"
          % ("조건", "k", "cos", "cos_dev", "rho", "rho_dev",
             "노름비", "움직임%"))
    for c in CONDS:
        kk = RV.SHUFK.get(c, "" if c != "risk_dimshuffle" else HI - LO)
        print("%-20s %5s %9.4f %9.4f %9.4f %9.4f %9.4f %7.1f%%"
              % (c, kk, m(c, "cos"), 1.0 - m(c, "cos"), m(c, "rho"),
                 1.0 - m(c, "rho"), m(c, "rt"), m(c, "moved")))

    bad = [c for c in SHUF if abs(m(c, "rt") - 1.0) > 1e-6]
    print("")
    if bad:
        print("  **경고** 노름비가 1 이 아닌 섞기 조건: %s" % ", ".join(bad))
        print("  값 다중집합 보존이 깨졌다 — 설계 전제가 무너진다.")
    else:
        print("  섞기 족 전부 노름비 1.000000 (값 다중집합 정확히 보존)")
        print("  → `norm_dev` 가 **정확히 0** 이라 노름 축이 고정된다.")

    print("")
    print("[2] k 고르기 — 목표는 %s 의 cos_dev 와 같아지는 k" % TARGET)
    print("-" * 92)
    tgt = 1.0 - m(TARGET, "cos")
    print("  목표 cos_dev = %.4f (%s, rho_dev %.4f)"
          % (tgt, TARGET, 1.0 - m(TARGET, "rho")))
    best, bd = None, None
    for c in SHUF:
        dd = abs((1.0 - m(c, "cos")) - tgt)
        print("    %-16s k=%2d cos_dev %.4f (차 %.4f) · rho_dev %.4f"
              % (c, RV.SHUFK[c], 1.0 - m(c, "cos"), dd, 1.0 - m(c, "rho")))
        if bd is None or dd < bd:
            best, bd = c, dd
    print("")
    print("  → 가장 가까운 조건 **%s** (k=%d, cos_dev 차 %.4f)"
          % (best, RV.SHUFK[best], bd))
    print("    그 조건의 rho_dev %.4f 대 %s 의 %.4f"
          % (1.0 - m(best, "rho"), TARGET, 1.0 - m(TARGET, "rho")))
    print("    l1 %.6f 대 %.6f · 노름비 %.4f 대 %.4f"
          % (m(best, "l1"), m(TARGET, "l1"),
             m(best, "rt"), m(TARGET, "rt")))
    if bd > 0.02:
        print("  **주의** 차가 0.02 를 넘는다. k 격자를 더 촘촘히 해야")
        print("  '코사인을 맞췄다' 고 말할 수 있다.")

    print("")
    print("[3] 이 족이 기존 조건과 **다른 궤적**인지")
    print("-" * 92)
    print("  pow 족은 단조변환이라 rho_dev=0 이고 코사인만 움직인다.")
    print("  섞기 족은 둘이 **같이** 움직인다. 그래서 둘을 겹쳐 놓으면")
    print("  같은 코사인에서 rho_dev 만 다른 쌍이 생긴다.")
    print("%-20s %9s %9s %12s"
          % ("조건", "cos_dev", "rho_dev", "rho_dev/cos_dev"))
    for c in CONDS:
        cd = 1.0 - m(c, "cos")
        rd = 1.0 - m(c, "rho")
        print("%-20s %9.4f %9.4f %12s"
              % (c, cd, rd, "%.3f" % (rd / cd) if cd > 1e-6 else "-"))

    path = os.path.join(OC.OUT_DIR, "shuffle_geometry.csv")
    with io.open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["condition", "k", "cosine", "cos_dev", "mean_rho",
                    "rho_dev", "norm_ratio", "mean_abs_diff", "moved_pct",
                    "n_draw"])
        for c in CONDS:
            w.writerow([c, RV.SHUFK.get(c, HI - LO
                                        if c == "risk_dimshuffle" else ""),
                        "%.6f" % m(c, "cos"), "%.6f" % (1.0 - m(c, "cos")),
                        "%.6f" % m(c, "rho"), "%.6f" % (1.0 - m(c, "rho")),
                        "%.6f" % m(c, "rt"), "%.6f" % m(c, "l1"),
                        "%.3f" % m(c, "moved"), N_DRAW])
    print("")
    print("저장: %s" % path)
    print(bar)


if __name__ == "__main__":
    main()
