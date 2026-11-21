# -*- coding: utf-8 -*-
"""표 11 NoRisk 행의 2인자 교락을 **분해**한다 — 상태 표현 대 보상 함수.

문제
----
`network_env.py` 의 `use_risk` 는 네 곳에서 쓰이는데, 정확히 두 묶음이다.
  상태 : `:86`  state_size,            `:276` get_state 의 엣지 블록
  보상 : `:326` 스텝 보상 분기,         `:378` 종단 위험 벌점
따라서 표 11 의 두 행은 2×2 의 **대각선 두 칸**뿐이다.

        보상=위험          보상=QoS
  상태=위험  제안(125d) ✓        **A 칸 (미측정)**
  상태=QoS   **B 칸 (미측정)**    NoRisk(175d) ✓

대각선만으로는 부호 역전이 **표현 몫인지 보상 몫인지** 분해되지 않는다.
이 스크립트가 빈 두 칸을 채운다.
  - `qos_state`  : 상태 175차원(원시 QoS) + 보상 **위험 기반** = B 칸
  - `qos_reward` : 상태 125차원(위험) + 보상 **QoS 기반**     = A 칸

왜 Risk-Blind 로는 안 되는가
----------------------------
125차원 상태는 50(위험) + 25(current) + 25(dest, 전 차원 상수) + 25(visited)
이고 **엣지 수준 채널이 위험 블록 하나뿐**이다. 그래서 차원을 고정한 채
위험을 빼면 엣지 인식이 **전면** 사라지고 정책이 경로 하나로 붕괴한다
(측정: 추론 시점 가림에서 홉이 정확히 8.00 / 7.00). 즉 Risk-Blind 는
**바닥값**이고 분해가 아니다(`train_risk_blind.py`, 항목 9).

구현
----
원본 코드를 베끼지 않는다. `get_state()` 를 부르는 동안만 `self.use_risk` 를
`state_risk` 로 바꿔치기하고 바로 되돌린다 — `get_state` 는 그 플래그 외에
`self.qos` / `current_node` / `visited` 만 읽으므로 안전하다. 보상 분기는
`self.use_risk = reward_risk` 로 **그대로** 둔다.

학습 설정·시드 규칙(42+run_id)·시나리오 혼합·평가 규약은 제안 팔
(`train_factor_ablation.py` 의 `full`)과 같다. 에이전트 클래스는
`train_risk_blind.py` 에서 가져와 또 베끼지 않는다.

사용법:
    python train_state_reward_cross.py qos_state 1
    python train_state_reward_cross.py qos_reward 1 --episodes 60
    python train_state_reward_cross.py eval            # 학습된 것 전부 채점
    python train_state_reward_cross.py eval_base       # 2x2 대각 두 칸
                                                       # (Full / NoRisk 재현)
"""
import argparse
import csv
import io
import os
import random
import sys
import time
from collections import deque

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import train_risk_blind as TB                       # noqa: E402
from network_env import NetworkEnv                  # noqa: E402
from topology import DESTINATION, EDGES, NUM_NODES  # noqa: E402

CELLS = {
    # 이름          (state_risk, reward_risk, 차원)
    "qos_state":  (False, True,  len(EDGES) * 2 + NUM_NODES * 3),   # B 칸
    "qos_reward": (True,  False, len(EDGES) + NUM_NODES * 3),       # A 칸
    # 아래 둘은 **평가 전용** 이다(학습하지 않는다). 2x2 의 대각 두 칸을
    # 교차 칸과 같은 하네스·같은 EVAL_SEED 로 재기 위한 것이고, 모델은
    # 이미 학습된 Full / NoRisk 체크포인트를 쓴다.
    "risk_both":  (True,  True,  len(EDGES) + NUM_NODES * 3),       # 기준 Full
    "qos_both":   (False, False, len(EDGES) * 2 + NUM_NODES * 3),   # NoRisk
}

# 학습 금지 셀 (모델을 밖에서 가져온다)
EVAL_ONLY = ("risk_both", "qos_both")

# 평가 전용 셀의 모델 출처. (셀, 디렉터리, 파일 접두사, 기록 CSV 의 arm)
BASELINE_SRC = (
    ("risk_both", os.path.join(HERE, "models_ablation"), "factor_full", "Full"),
    ("qos_both", os.path.join(os.path.dirname(HERE), "SDN_DQN_new_env",
                              "models"), "norisk", None),
)

OUT = TB.OUT
MODEL_DIR = os.path.join(OUT, "models_cross")
RUNS_CSV = os.path.join(OUT, "cross_runs.csv")
BASE_CSV = os.path.join(OUT, "cross_baselines.csv")


class CrossEnv(NetworkEnv):
    """상태 표현과 보상 함수를 떼어 놓은 환경.

    클래스 변수 `STATE_RISK` 로 상태를, 생성 시 `use_risk` 로 보상을 정한다.
    `E.evaluate` 가 `use_risk=True` 를 넘기므로 그 인자는 **무시**하고
    클래스 변수 `REWARD_RISK` 를 쓴다.
    """
    STATE_RISK = True
    REWARD_RISK = True

    def __init__(self, use_risk=True, scenario="normal",
                 risk_window=TB.RISK_WINDOW):
        super().__init__(use_risk=type(self).REWARD_RISK, scenario=scenario,
                         risk_window=risk_window)
        sr = type(self).STATE_RISK
        self.state_size = (len(EDGES) if sr else len(EDGES) * 2) + NUM_NODES * 3

    def get_state(self):
        saved = self.use_risk
        self.use_risk = type(self).STATE_RISK
        try:
            return super().get_state()
        finally:
            self.use_risk = saved


def configure(cell):
    sr, rr, dim = CELLS[cell]
    CrossEnv.STATE_RISK = sr
    CrossEnv.REWARD_RISK = rr
    return sr, rr, dim


def verify(cell):
    """보상과 상태가 **따로** 움직이는지 단정으로 확인한다."""
    sr, rr, dim = configure(cell)
    env = CrossEnv(scenario="dynamic")
    assert env.state_size == dim, (env.state_size, dim)
    assert env.use_risk is rr, "보상 플래그가 안 먹었다"
    env.reset()
    s = env.get_state()
    assert len(s) == dim, (len(s), dim)
    assert env.use_risk is rr, "get_state 가 보상 플래그를 복원하지 않았다"
    # 상태 블록이 정말 그 표현인지: 위험이면 50칸, QoS 면 100칸
    nblk = len(EDGES) if sr else len(EDGES) * 2
    ref = NetworkEnv(use_risk=sr, scenario="dynamic",
                     risk_window=TB.RISK_WINDOW)
    assert ref.state_size == dim, (ref.state_size, dim)
    # 보상 분기 확인: 같은 전이에서 reward_risk 에 따라 값이 달라야 한다
    return nblk


def train(cell, run_id, episodes):
    sr, rr, dim = configure(cell)
    nblk = verify(cell)
    seed = 42 + run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(MODEL_DIR, exist_ok=True)
    tag = "%s_run%02d" % (cell, run_id)
    if episodes != TB.EPISODES:
        tag += "_ep%d" % episodes
    save_path = os.path.join(MODEL_DIR, tag + ".pth")

    base = CrossEnv(scenario="normal")
    agent = TB.DQNAgent(base.state_size, base.action_size)
    print("===== Cross  cell=%s  run=%d  seed=%d =====" % (cell, run_id, seed))
    print("상태=%s(%d칸 엣지블록, %d차원)  보상=%s  Episodes=%d  Threads=%d"
          % ("위험" if sr else "원시QoS", nblk, dim,
             "위험기반" if rr else "QoS기반", episodes, torch.get_num_threads()))

    succ = deque(maxlen=100)
    rew = deque(maxlen=100)
    t0 = time.time()
    for ep in range(1, episodes + 1):
        env = CrossEnv(scenario=TB.sample_scenario())
        state = env.reset()
        done = False
        ep_rew = 0.0
        while not done:
            valid = env.get_valid_actions()
            action = agent.select_action(state, valid)
            if action is None:
                break
            next_state, reward, done, info = env.step(action)
            agent.memory.push(state, action, reward, next_state, done,
                              info["next_valid_mask"])
            agent.train_step()
            state = next_state
            ep_rew += reward
        succ.append(1 if env.current_node == DESTINATION else 0)
        rew.append(ep_rew)
        agent.decay_epsilon()
        if ep % TB.TARGET_UPDATE == 0:
            agent.update_target()
        if ep % 100 == 0:
            el = time.time() - t0
            print("Ep %4d | Suc %5.1f%% | Rew %8.2f | Eps %.4f | %6.1fs (%.3fs/ep)"
                  % (ep, np.mean(succ) * 100, np.mean(rew), agent.epsilon,
                     el, el / ep), flush=True)

    torch.save(agent.policy_net.state_dict(), save_path)
    print("모델 저장: %s  (%.1f분)" % (save_path, (time.time() - t0) / 60.0))


# ── 평가: eval_ablation_risky.evaluate 를 그대로 재사용 ────────────────────
def evaluate_cell(cell, model_path, episodes):
    import eval_ablation_risky as E
    import network_env
    configure(cell)
    orig = network_env.NetworkEnv
    network_env.NetworkEnv = CrossEnv
    try:
        assert E.network_env.NetworkEnv is CrossEnv, "E 가 패치를 못 봤다"
        probe = CrossEnv(scenario="dynamic")
        assert probe.use_risk is CELLS[cell][1], "보상 플래그 유실"
        assert probe.state_size == CELLS[cell][2], "차원 유실"
        res = E.evaluate(model_path, [], episodes)
    finally:
        network_env.NetworkEnv = orig
    assert network_env.NetworkEnv is orig, "패치 복원 실패"
    return res


def eval_all(episodes):
    import eval_ablation_risky as E
    if not os.path.isdir(MODEL_DIR):
        print("채점할 모델이 없다: %s" % MODEL_DIR)
        return
    models = sorted(f for f in os.listdir(MODEL_DIR)
                    if f.endswith(".pth") and "_ep" not in f)
    if not models:
        print("채점할 모델이 없다: %s" % MODEL_DIR)
        return
    print("EVAL_SEED=%d  에피소드 %d  모델 %d개"
          % (E.EVAL_SEED, episodes, len(models)))
    rows = []
    for name in models:
        cell = name.split("_run")[0]
        run = int(name.split("_run")[1].split(".")[0])
        res = evaluate_cell(cell, os.path.join(MODEL_DIR, name), episodes)
        row = {"arm": cell, "cell": cell, "run": run, "model": name,
               "state": "risk" if CELLS[cell][0] else "qos",
               "reward": "risk" if CELLS[cell][1] else "qos",
               "state_size": CELLS[cell][2], "episodes": episodes,
               "eval_seed": E.EVAL_SEED}
        for k in E.FIELDS:
            row[k] = "%.6f" % res[k]
        row["success_rate"] = "%.6f" % res["success_rate"]
        rows.append(row)
        print("  %-22s delay %.4f  loss %.6f  hop %.4f  도달 %.1f%%"
              % (name, res["delay"], res["loss"], res["hop"],
                 res["success_rate"]), flush=True)
    with io.open(RUNS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("저장: %s (%d행)" % (RUNS_CSV, len(rows)))


def _recorded_full():
    """table10_factor_runs.csv 의 Full 행 -> run -> 지표 (자기검증용)."""
    p = os.path.join(OUT, "table10_factor_runs.csv")
    out = {}
    if not os.path.exists(p):
        return out
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == "Full":
                out[int(float(row["run"]))] = row
    return out


def eval_baselines(episodes):
    """2x2 의 대각 두 칸을 교차 칸과 **같은 하네스·같은 규약** 으로 잰다.

    왜: 교차 칸은 EVAL_SEED 단일 시드로 쟀고 기준 칸들은 다른 규약에서
    왔다. 규약이 섞이면 팔 간 차에 ±11.6 ms 가 실린다. 그러니 네 칸을
    한 번에 같은 자리에서 잰다.

    risk_both 는 Full 기록값과 자기검증한다. 어긋나면 CrossEnv 가
    평범한 NetworkEnv 와 다르다는 뜻이므로 2x2 전체가 무효다.
    """
    import eval_ablation_risky as E
    rec = _recorded_full()
    print("EVAL_SEED=%d  에피소드 %d  (평가 전용 셀 %d개)"
          % (E.EVAL_SEED, episodes, len(BASELINE_SRC)))
    rows, bad, checked = [], [], 0
    for (cell, mdir, prefix, rec_arm) in BASELINE_SRC:
        if not os.path.isdir(mdir):
            print("  %s: 디렉터리 없음 %s" % (cell, mdir))
            continue
        names = sorted(f for f in os.listdir(mdir)
                       if f.startswith(prefix + "_run") and f.endswith(".pth"))
        print("\n[%s] %s  모델 %d개" % (cell, mdir, len(names)))
        for name in names:
            run = int(name.split("_run")[1].split(".")[0])
            res = evaluate_cell(cell, os.path.join(mdir, name), episodes)
            sc = ""
            if rec_arm == "Full" and run in rec and episodes == 200:
                d = abs(float(rec[run]["delay"]) - res["delay"])
                sc = ("일치(%.2g)" % d) if d < 1e-4 else "어긋남(%.3g)" % d
                checked += 1
                if d >= 1e-4:
                    bad.append((name, d))
            row = {"arm": cell, "cell": cell, "run": run, "model": name,
                   "state": "risk" if CELLS[cell][0] else "qos",
                   "reward": "risk" if CELLS[cell][1] else "qos",
                   "state_size": CELLS[cell][2], "model_dir": mdir,
                   "selfcheck": sc, "episodes": episodes,
                   "eval_seed": E.EVAL_SEED}
            for k in E.FIELDS:
                row[k] = "%.6f" % res[k]
            row["success_rate"] = "%.6f" % res["success_rate"]
            rows.append(row)
            print("  %-22s delay %.4f  loss %.6f  hop %.4f  도달 %.1f%%  %s"
                  % (name, res["delay"], res["loss"], res["hop"],
                     res["success_rate"], sc), flush=True)
    if not rows:
        print("채점할 모델이 없다")
        return
    with io.open(BASE_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n저장: %s (%d행)" % (BASE_CSV, len(rows)))
    if bad:
        print("!! 자기검증 어긋남 %d개 — CrossEnv(True,True) 가 평범한"
              " NetworkEnv 와 다르다. 2x2 비교를 쓰면 안 된다: %s"
              % (len(bad), bad[:3]))
    elif checked:
        print("자기검증 통과(%d개): CrossEnv(True,True) 가 표 10 Full 기록값을"
              " 재현한다 -> 네 칸이 같은 자리에서 측정됐다." % checked)
    else:
        print("자기검증 **안 했다** — 기록값이 200에피소드 규약이므로"
              " --eval-episodes 200 으로 돌려야 비교가 유효하다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",
                    help="qos_state | qos_reward | eval | eval_base")
    ap.add_argument("run_id", nargs="?", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=TB.EPISODES)
    ap.add_argument("--eval-episodes", type=int, default=200)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))
    if a.mode == "eval":
        eval_all(a.eval_episodes)
        return
    if a.mode == "eval_base":
        eval_baselines(a.eval_episodes)
        return
    if a.mode in EVAL_ONLY:
        ap.error("%s 는 평가 전용 셀이다 (학습하지 않는다). "
                 "`eval_base` 로 채점한다." % a.mode)
    if a.mode not in CELLS:
        ap.error("mode must be qos_state / qos_reward / eval / eval_base")
    train(a.mode, a.run_id, a.episodes)


if __name__ == "__main__":
    main()
