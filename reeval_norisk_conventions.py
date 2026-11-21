# -*- coding: utf-8 -*-
"""표 11 세 행 중 **NoRisk-DQN 의 종단간 손실**을 처음 측정한다.

왜 필요한가. 표 11 의 Loss 열은 `network_env.py:454` 의 `total_loss / hop_count`
= **홉당 평균**이다. 같은 함수가 지연은 **합**으로 돌려준다. 제안 기법의 경로가
NoRisk 보다 0.68 홉 길므로 홉당 평균은 제안에 구조적으로 유리하다
(`FINDINGS_loss_metric.md`). 4.6.3 절에 쓰려던 문장은 "손실 7.8% 를 지연 5.2% 로
산다" 는 **교환 관계**인데, 그 교환이 성립하는지는 손실을 종단간으로 봐야 정해진다.
제안 기법 행은 `reeval_loss_conventions.py` 가 이미 세 규약으로 쟀고
(`review_results/loss_conventions_exact.csv`), **NoRisk 행만 없었다.**

규약 셋은 `reeval_loss_conventions.loss_conventions` 를 그대로 import 해서 쓴다
(계산이 글자 그대로 같아야 비교가 성립한다):

  published   mean(p_i)               표 11 이 적는 값 — 홉당 평균
  path_sum    sum(p_i)                지연이 이미 쓰고 있는 양
  compounded  1 - prod(1 - p_i)       경로 어딘가에서 떨어질 확률

측정 규약은 원본 `eval_norisk.py:39-40` 과 같다 — 시드 블록마다
`np.random.seed(seed)` · `random.seed(seed)` 한 번, 200 에피소드, 워밍업 10 스텝,
마스킹된 argmax. 두 팔이 같은 시드 다섯 개를 쓰므로 블록 수준에서 공통난수다.

**발표값과 비트 일치는 기대하지 않는다.** 표 11 의 NoRisk 행을 만든
`train_norisk_dqn.py` 는 평가 앞에 시드를 박지 않고 **학습이 남긴 난수 상태**를
물려받는다(7월식). 그러므로 이 스크립트가 보는 것은 "같은 모델 · 같은 환경 ·
같은 지표 정의에서 규약을 바꾸면 비교가 뒤집히는가" 이고, 발표값과의 차는
26.10.6 의 규약 산포 안인지로만 읽는다.

팔:
  Proposed   `risk_only_model_final.pth` (동결 1개) × 시드 5개
             — 표 11 Proposed 행의 ± 정의와 같다
  NoRisk     `models/norisk_run01..05.pth` (독립 학습 5개) × 시드 5개
             — 표 11 NoRisk 행의 ± 정의는 모델 5개이므로, 모델별로 시드 5개를
               평균해 모델값 5개를 만들어 검정한다

출력: `review_results/norisk_conventions.csv` (행별 raw, arm × model × seed)
원본 `SDN_DQN_new_env/` 는 읽기만 한다.
"""

import argparse
import csv
import io
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RD = os.path.join(HERE, "review_results")
OUT = os.path.join(RD, "norisk_conventions.csv")

sys.path.insert(0, HERE)
sys.path.insert(0, ORIG)

import review_stats as RS                       # noqa: E402
from reeval_loss_conventions import (           # noqa: E402
    CONVENTIONS, EVAL_EPISODES, SEEDS, WARMUP, loss_conventions, seed_mean,
)
from network_env import NetworkEnv              # noqa: E402
from topology import DESTINATION                # noqa: E402

RISK_MODEL = os.path.join(ORIG, "risk_only_model_final.pth")
NORISK_MODELS = [os.path.join(ORIG, "models", "norisk_run%02d.pth" % i)
                 for i in range(1, 6)]
RISK_WINDOW = 5
SCEN = "dynamic"
METRICS = ("hop", "delay") + CONVENTIONS

# 표 11 이 인쇄한 값 (출처: results/norisk_reference.csv, exp1_reference.csv)
PUB = {
    "NoRisk": {"delay": 332.85556682983463, "published": 0.09063384250192018,
               "hop": 7.0},
    "Proposed": {"delay": 350.124972, "published": 0.083550, "hop": 7.677},
}


def build_net(state_size, action_size, layernorm):
    import torch.nn as nn

    layers = [nn.Linear(state_size, 512)]
    if layernorm:
        layers.append(nn.LayerNorm(512))
    layers += [nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
               nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size)]
    return nn.Sequential(*layers)


def load_model(path, use_risk):
    """체크포인트의 실제 차원으로 망을 세운다 — 하드코딩하지 않는다."""
    import torch
    import torch.nn as nn

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit("예상 못 한 체크포인트 형식: %s" % path)
    # 키 접두어가 `net.` 인지 확인하고, 첫 층에서 상태 차원을 읽는다
    first = [k for k in sd if k.endswith("0.weight")]
    if not first:
        raise SystemExit("첫 층을 못 찾음: %s" % path)
    w = sd[first[0]]
    state_size, action_size = int(w.shape[1]), None
    last = [k for k in sd if k.endswith("weight")][-1]
    action_size = int(sd[last].shape[0])
    layernorm = any("LayerNorm" in k or ".1.weight" in k and
                    sd[k].dim() == 1 for k in sd)
    # LayerNorm 유무는 키 번호로 판정: net.1.weight 가 1차원이면 LayerNorm
    layernorm = ("net.1.weight" in sd and sd["net.1.weight"].dim() == 1)

    class QNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = build_net(state_size, action_size, layernorm)

        def forward(self, x):
            return self.net(x)

    model = QNetwork()
    model.load_state_dict(sd)
    model.eval()
    return model, state_size, action_size, layernorm


def rollout(model, use_risk, seed, episodes):
    """eval_norisk.py / eval_baseline.py 와 같은 규약으로 한 시드 블록."""
    import torch

    np.random.seed(seed)
    random.seed(seed)
    out = []
    for _ in range(episodes):
        if use_risk:
            env = NetworkEnv(use_risk=True, scenario=SCEN,
                             risk_window=RISK_WINDOW)
        else:
            env = NetworkEnv(use_risk=False, scenario=SCEN)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            with torch.no_grad():
                q = model(torch.FloatTensor(state).unsqueeze(0))[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node == DESTINATION:
            out.append(loss_conventions(env, env.visited))
    return seed_mean(out), len(out) / float(episodes) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

    arms = []
    m, ss, acts, ln = load_model(RISK_MODEL, True)
    print("Proposed  %s  state=%d action=%d LayerNorm=%s"
          % (os.path.basename(RISK_MODEL), ss, acts, ln))
    arms.append(("Proposed", os.path.basename(RISK_MODEL), m, True))
    for p in NORISK_MODELS:
        m, ss, acts, ln = load_model(p, False)
        print("NoRisk    %s  state=%d action=%d LayerNorm=%s"
              % (os.path.basename(p), ss, acts, ln))
        arms.append(("NoRisk", os.path.basename(p), m, False))
    print()

    rows = []
    t0 = time.time()
    for arm, name, model, use_risk in arms:
        for sd in SEEDS:
            r, reach = rollout(model, use_risk, sd, args.episodes)
            row = {"arm": arm, "model": name, "seed": sd,
                   "episodes": args.episodes, "reach_pct": "%.4f" % reach}
            for k in METRICS:
                row[k] = "%.10f" % r[k]
            rows.append(row)
            print("  %-9s %-22s seed=%d  delay=%9.4f  pub=%.6f  "
                  "sum=%.6f  comp=%.6f  hop=%.4f  reach=%.1f%%  (%.1f min)"
                  % (arm, name, sd, r["delay"], r["published"], r["path_sum"],
                     r["compounded"], r["hop"], reach, (time.time() - t0) / 60))

    os.makedirs(RD, exist_ok=True)
    with io.open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n기록: %s (%d행)" % (OUT, len(rows)))

    # ---- 집계 ----
    def vals(arm, metric, by_model):
        if by_model:
            names = sorted({r["model"] for r in rows if r["arm"] == arm})
            return [float(np.mean([float(r[metric]) for r in rows
                                   if r["arm"] == arm and r["model"] == n]))
                    for n in names]
        return [float(r[metric]) for r in rows if r["arm"] == arm]

    print()
    print("=" * 78)
    print("A. 발표값 재현 점검 (7월식 대 이 하네스 — 비트 일치는 기대 안 함)")
    print("=" * 78)
    for arm in ("Proposed", "NoRisk"):
        by_model = (arm == "NoRisk")
        for k in ("delay", "published", "hop"):
            xs = vals(arm, k, by_model)
            mu, s = RS.mean_std(xs)
            pub = PUB[arm][k]
            dev = (mu - pub) / pub * 100.0 if pub else float("nan")
            print("  %-9s %-10s 측정 %12.6f ± %-10.6f  발표 %12.6f  "
                  "편차 %+7.3f%%  (n=%d, %s)"
                  % (arm, k, mu, s, pub, dev, len(xs),
                     "모델 5개" if by_model else "시드 5개"))

    print()
    print("=" * 78)
    print("B. 규약을 바꾸면 손실 비교가 어떻게 되는가 (dynamic)")
    print("=" * 78)
    hdr = "%-12s %14s %14s %12s %12s %10s"
    print(hdr % ("규약", "제안", "NoRisk", "차", "제안 우위%", "p_holm"))
    print("-" * 78)
    tests = []
    for conv in CONVENTIONS:
        xp = vals("Proposed", conv, False)
        xn = vals("NoRisk", conv, True)
        diff, t, df, p = RS.welch(xp, xn)
        mp, sp = RS.mean_std(xp)
        mn, sn = RS.mean_std(xn)
        adv = -(mp - mn) / mn * 100.0      # 음수면 제안이 낮다 -> 양수 우위
        tests.append((conv, mp, sp, mn, sn, diff, p, adv))
    ps = [x[6] for x in tests]
    holm = RS.holm_bonferroni(ps)
    for (conv, mp, sp, mn, sn, diff, p, adv), hp in zip(tests, holm):
        ph = hp[1] if isinstance(hp, (list, tuple)) else hp
        print(hdr % (conv, "%.6f" % mp, "%.6f" % mn, "%+.6f" % diff,
                     "%+.2f%%" % adv, RS.fmt_p(ph)))
    print("-" * 78)
    xp = vals("Proposed", "delay", False)
    xn = vals("NoRisk", "delay", True)
    d, t, df, p = RS.welch(xp, xn)
    mp, _ = RS.mean_std(xp)
    mn, _ = RS.mean_std(xn)
    print("  지연: 제안 %.4f  NoRisk %.4f  차 %+.4f ms  "
          "(제안이 %+.2f%%)  p=%s"
          % (mp, mn, d, (mp - mn) / mn * 100.0, RS.fmt_p(p)))
    hp_ = vals("Proposed", "hop", False)
    hn_ = vals("NoRisk", "hop", True)
    print("  홉:   제안 %.4f  NoRisk %.4f  차 %+.4f 홉"
          % (np.mean(hp_), np.mean(hn_), np.mean(hp_) - np.mean(hn_)))
    print()
    print("  읽는 법: `published` 는 홉당 평균이라 긴 경로에 유리하다.")
    print("  `path_sum` 은 지연과 같은 양(합)이고 `compounded` 는 확률이다.")
    print("  셋이 같은 방향이면 교환 관계 문장이 선다. 부호가 갈리면")
    print("  '손실 N% 를 지연 M% 로 산다' 는 문장은 **쓸 수 없다**.")


if __name__ == "__main__":
    main()
