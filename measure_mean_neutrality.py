# -*- coding: utf-8 -*-
"""23.8절이 쓰려는 설명 — '평균 벡터는 중립, 공여 벡터는 자신있게 틀림' — 을 잰다.

risk_frozen(+165.51 ms)이 risk_mean(+51.11 ms)보다 3.2배 나쁜 것을 "평균은
모든 엣지를 고르게 중간으로 말하고(중립), 공여는 특정 엣지에 대해 확신을 갖고
틀리게 말한다"로 설명하려 한다. 그런데 그 '중립'은 **설계상의 기대**일 뿐 아직
측정하지 않았다. 측정하지 않은 메커니즘을 단정하지 않기 위해 여기서 잰다.

재는 것 (롤아웃 없음, 배치 묶음만):
  (1) 벡터 내 분산: 위험 50칸의 칸간 표준편차. 평균 벡터가 공여들보다 작으면
      '평탄 = 중립'이 수치로 확인된다.
  (2) 극단 주장 수: |x - 블록평균| > 2*칸간sd 인 칸의 개수. 공여는 많고 평균
      벡터는 0이어야 한다.
  (3) 오정보의 크기: 공여 벡터와 같은 묶음의 다른 상태 사이 칸별 차이의 RMS.
      평균 벡터 대비 같은 양을 재서 '얼마나 다른 말을 하는가'를 비교한다.
  (4) 순위 교란: 공여의 위험 순위와 참 상태의 위험 순위 사이 스피어만 rho.
      평균 벡터에 대해서도 같은 것을 잰다.

**실측 결과로 (1)(2)(3) 의 기대가 다 꺾였고 (4) 가 답이었다 (23.8절).**
평균 벡터는 평탄하지 않고(칸간 sd 0.201 대 공여 0.281 = 1.40배뿐), 공여는 극단
주장을 하지 않으며(2sd 초과 칸 중위 0개), 참 상태와의 차이도 1.22배뿐이다.
가른 것은 **순위 충실도**다 - 평균 rho 0.4607 대 공여 rho 0.2695. 즉 평균 벡터는
'중립'이 아니라 **구조 성분만 남긴 더 좋은 추정**이고(잡음 평균화:
sqrt(0.2695)=0.519 ~= 0.4607), 공여는 **순서를 틀리게** 말한다.

출력: review_results/mean_neutrality.csv
"""
import csv
import os

import occlude_state_blocks as OC               # noqa: E402

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

LO, HI = 0, 50


def spearman(a, b):
    """scipy 없이. 동순위는 평균순위로."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    ra, rb = rank(a), rank(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def main():
    device = torch.device("cpu")
    os.makedirs(OC.OUT_DIR, exist_ok=True)
    OC.check_env()

    ref = OC.load_net(os.path.join(OC.ORIG_MODELS, "window5_run01.pth"), device)
    battery = OC.build_battery("greedy", device, ref)
    blocks = np.stack([s[LO:HI] for s, _ in battery])
    meanv = blocks.mean(axis=0)
    dimsd = blocks.std(axis=0, ddof=1)          # 칸별 (상태 간) 표준편차
    print("공여 상태 %d개 · 위험 50칸" % len(blocks))
    print("")

    # (1) 벡터 내 분산 — 평탄한가
    within_mean = float(meanv.std(ddof=1))
    within_don = [float(b.std(ddof=1)) for b in blocks]
    wm = float(np.median(within_don))
    print("[1] 벡터 내 칸간 표준편차 (클수록 '특정 엣지를 콕 집어 말한다')")
    print("    평균 벡터          %.5f" % within_mean)
    print("    공여 벡터 중위     %.5f  (최소 %.5f, 최대 %.5f)"
          % (wm, min(within_don), max(within_don)))
    print("    비 (공여/평균)     %.2f배" % (wm / within_mean)
          if within_mean > 0 else "    비: 평균 벡터가 완전 평탄")
    print("")

    # (2) 극단 주장 수
    thr = 2.0 * dimsd
    ext_mean = int(np.sum(np.abs(meanv - meanv) > thr))
    ext_don = [int(np.sum(np.abs(b - meanv) > thr)) for b in blocks]
    print("[2] 극단 주장 칸 수 (|x - 블록평균| > 2*칸별sd, 50칸 중)")
    print("    평균 벡터          %d개  (정의상 0)" % ext_mean)
    print("    공여 벡터 중위     %.1f개 (최대 %d개)"
          % (float(np.median(ext_don)), max(ext_don)))
    print("")

    # (3) 오정보의 크기 — 참 상태와 얼마나 다른 말을 하는가
    rng = np.random.RandomState(20261114)
    idx = rng.randint(0, len(blocks), size=(2000, 2))
    rms_don = [float(np.sqrt(np.mean((blocks[i] - blocks[j]) ** 2)))
               for i, j in idx if i != j]
    rms_mean = [float(np.sqrt(np.mean((meanv - blocks[j]) ** 2)))
                for _, j in idx]
    print("[3] 참 상태와의 칸별 차이 RMS (클수록 '다른 말을 한다')")
    print("    공여 대 참 상태    %.5f +- %.5f"
          % (np.mean(rms_don), np.std(rms_don)))
    print("    평균 대 참 상태    %.5f +- %.5f"
          % (np.mean(rms_mean), np.std(rms_mean)))
    print("    비 (공여/평균)     %.2f배"
          % (np.mean(rms_don) / np.mean(rms_mean)))
    print("")

    # (4) 순위 교란
    rho_don = [spearman(list(blocks[i]), list(blocks[j]))
               for i, j in idx[:400] if i != j]
    rho_mean = [spearman(list(meanv), list(blocks[j])) for _, j in idx[:400]]
    print("[4] 위험 순위의 스피어만 rho (1=같은 순서, 0=무관)")
    print("    공여 대 참 상태    %.4f" % float(np.nanmean(rho_don)))
    print("    평균 대 참 상태    %.4f" % float(np.nanmean(rho_mean)))
    print("")

    path = os.path.join(OC.OUT_DIR, "mean_neutrality.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "mean_vector", "donor_median", "ratio"])
        w.writerow(["within_vector_sd", "%.6f" % within_mean,
                    "%.6f" % wm, "%.3f" % (wm / within_mean)])
        w.writerow(["extreme_dims", 0, "%.1f" % float(np.median(ext_don)), ""])
        w.writerow(["rms_vs_true", "%.6f" % float(np.mean(rms_mean)),
                    "%.6f" % float(np.mean(rms_don)),
                    "%.3f" % (np.mean(rms_don) / np.mean(rms_mean))])
        w.writerow(["spearman_vs_true", "%.4f" % float(np.nanmean(rho_mean)),
                    "%.4f" % float(np.nanmean(rho_don)), ""])
    print("저장: %s" % path)


if __name__ == "__main__":
    main()
