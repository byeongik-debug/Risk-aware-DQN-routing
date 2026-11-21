# -*- coding: utf-8 -*-
"""상태 벡터 네 블록의 크기와 변동 — 가림 실험의 해석 가드.

`occlude_state_blocks.py` 가 위험 블록을 0으로 만들면 지표가 크게 나빠진다.
그런데 그게 **정보 손실** 때문인지 **크기 충격** 때문인지는 따로 재야 한다.
위험 50칸이 상태벡터 노름의 절반을 넘게 차지하면 0으로 만드는 것은 입력의
대부분을 지우는 것이고, 그 비교는 one-hot 25칸(노름 1.0)을 지우는 것과 공정하지
않다. 그래서 블록별 평균 L2, 칸별 표준편차, 상수인 칸 수를 센다.

부수 효과로 19절의 "`dest_onehot` 25칸은 상수"가 독립 확인된다 (살아있는 칸 0개).

출력: review_results/state_block_scale.csv
"""
import csv
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import train_factor_ablation as T                # noqa: E402
from network_env import NetworkEnv               # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

OUT_DIR = os.path.join(HERE, "review_results")
REFERENCE = os.path.join(ORIG, "models", "window5_run01.pth")
BLOCKS = (("risk", 0, 50), ("curr", 50, 75),
          ("dest", 75, 100), ("visited", 100, 125))
BATTERY_N = int(os.environ.get("BATTERY_N", "250"))
SEED = 20261114


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    state = torch.load(REFERENCE, map_location="cpu")
    net = T.QNetwork(125, 25)
    net.load_state_dict(state, strict=True)
    net.eval()

    def argmax(s, valid):
        with torch.no_grad():
            q = net(torch.FloatTensor(s).unsqueeze(0))[0].numpy()
        return max(valid, key=lambda a: q[a])

    random.seed(SEED)
    np.random.seed(SEED)
    rows = []
    while len(rows) < BATTERY_N:
        env = NetworkEnv(use_risk=True, scenario="dynamic", risk_window=5)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        s = env.get_state()
        done = False
        while not done and len(rows) < BATTERY_N:
            valid = env.get_valid_actions()
            if not valid:
                break
            rows.append(np.asarray(s, dtype=np.float32))
            s, _, done, _ = env.step(argmax(s, valid))
    X = np.stack(rows)
    total_sq = float(np.linalg.norm(X, axis=1).mean()) ** 2

    out = []
    for name, lo, hi in BLOCKS:
        B = X[:, lo:hi]
        sd = B.std(axis=0)
        l2 = float(np.linalg.norm(B, axis=1).mean())
        out.append({"block": name, "dims": hi - lo,
                    "mean_l2": round(l2, 6),
                    "share_of_sq_norm_pct": round(100.0 * l2 ** 2 / total_sq, 4),
                    "mean_value": round(float(B.mean()), 6),
                    "mean_dim_sd": round(float(sd.mean()), 6),
                    "median_dim_sd": round(float(np.median(sd)), 6),
                    "max_dim_sd": round(float(sd.max()), 6),
                    "live_dims": int((sd > 1e-9).sum()),
                    "min_value": round(float(B.min()), 6),
                    "max_value": round(float(B.max()), 6)})

    path = os.path.join(OUT_DIR, "state_block_scale.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)

    print("배치 상태 %d개, 상태벡터 평균 L2 %.4f" % (len(X), total_sq ** 0.5))
    print("%-9s %5s %9s %10s %9s %11s %11s" %
          ("블록", "차원", "평균L2", "노름지분%", "평균값",
           "칸별sd중위", "살아있는칸"))
    for r in out:
        print("%-9s %5d %9.4f %10.2f %9.4f %11.5f %7d/%d" %
              (r["block"], r["dims"], r["mean_l2"],
               r["share_of_sq_norm_pct"], r["mean_value"],
               r["median_dim_sd"], r["live_dims"], r["dims"]))
    print("")
    print("해석: 위험 블록이 노름 지분 %.1f%% 이므로 risk_zero 는 정보 제거가 아니라"
          % out[0]["share_of_sq_norm_pct"])
    print("크기 충격을 포함한다. 공정한 절제는 크기를 유지하는 risk_mean 이다.")
    print("19절 확인: dest 살아있는 칸 %d개 (상수여야 0)" % out[2]["live_dims"])
    print("쓴 파일: %s" % path)


if __name__ == "__main__":
    main()
