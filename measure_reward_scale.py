# -*- coding: utf-8 -*-
u"""두 보상(위험 기반 · QoS 기반)의 **규모 차이**를 같은 전이에서 잰다.

왜 다시 재는가. 2x2 교차(`train_state_reward_cross.py`)의 한계로 초안
(`DRAFT_4_6_3_table11.md` L391)과 기록(`FINDINGS_table10_f1.md` 26.x)은
"같은 전이에서 위험 기반 -20.68 / -29.53 / -29.00 / -29.73, QoS 기반
-1.78 / -13.32 / -10.90 / -14.35" 이라고 적는다. 그런데 그 여덟 값을 쓴
스크립트가 없다 — 표본 네 개가 **어느 전이인지 기록이 없고** CSV 도 없다.
감사 32절에서 `DOC_ONLY` 경보로 뜬다. 표본 네 개를 되찾으려 추측하는 대신,
**규약을 선언하고 전이 전체에서 다시 잰다**. 주장("두 보상은 규모가 달라
보상 인자 비교에 크기가 섞인다")은 분포로 받쳐야 더 강하다.

어떻게 재는가 — 원본 보상식을 **베끼지 않는다**. `step()` 의 보상 분기는
(그 시점 QoS, 거리 변화, 재방문, 도착) 의 함수라 같은 전이에서 두 분기를
모두 평가할 수 있다. 그래서 매 전이마다
  1) stdlib `random` 상태를 저장하고 환경을 깊은 복사한 뒤
  2) 사본의 `use_risk` 만 뒤집고 **같은 난수 상태 · 같은 행동**으로 step
해 두 보상을 얻는다. `update_dynamic_qos()` 가 사본과 원본에서 같은 수를
뽑으므로 두 보상은 **같은 QoS 스냅숏**에서 나온 값이다(검증: 사본과 원본의
`info["delay"] / info["loss"]` 가 비트 단위로 같아야 한다).

규약: scenario=dynamic, 시드 900~929(에피소드 30개), 행동은 유효 행동 중
균등 무작위이되 **환경 난수열과 분리된 RNG**로 뽑는다(환경 스트림을
건드리면 두 사본이 갈라진다).

원본 폴더는 읽기만 한다. 출력은 review_results/ 로만.
"""
import copy
import csv
import io
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, os.pardir, "SDN_DQN_new_env"))
RES = os.path.join(HERE, "review_results")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from network_env import NetworkEnv   # noqa: E402

SCENARIO = "dynamic"
SEEDS = list(range(900, 930))


def rollout(seed, rows):
    env = NetworkEnv(use_risk=True, scenario=SCENARIO)
    random.seed(seed)
    np.random.seed(seed)
    env.reset()
    pick = random.Random(seed ^ 0x5EED)      # 환경 스트림과 분리
    t = 0
    while True:
        valid = env.get_valid_actions()
        if not valid:
            break
        a = pick.choice(sorted(valid))

        st = random.getstate()
        clone = copy.deepcopy(env)
        clone.use_risk = False

        _, r_risk, done, info = env.step(a)
        random.setstate(st)
        _, r_qos, done_c, info_c = clone.step(a)

        # 같은 전이인지 확인 — QoS 스냅숏이 비트 단위로 같아야 한다.
        assert done == done_c, (seed, t, done, done_c)
        for k in ("delay", "loss"):
            if k in info and k in info_c:
                assert info[k] == info_c[k], (seed, t, k, info[k], info_c[k])

        rows.append(dict(seed=seed, step=t, action=a,
                         reason=info.get("reason", ""),
                         delay="%.6f" % info.get("delay", float("nan")),
                         loss="%.6f" % info.get("loss", float("nan")),
                         reward_risk="%.6f" % r_risk,
                         reward_qos="%.6f" % r_qos,
                         ratio=("%.4f" % (r_risk / r_qos))
                         if r_qos else ""))
        t += 1
        if done:
            break


def greedy_rollout(model_path, seed, rows, policy):
    u"""학습된 정책(ε=0)의 전이에서도 같은 측정을 한다.

    무작위 정책은 ε-greedy 학습의 **초반**을 대표한다. 수렴 뒤에도 규모 차이가
    남는지 보려면 학습된 정책의 전이에서도 재야 한다. 평가 규약은
    `eval_ablation_risky.evaluate()` 와 같다(워밍업 10스텝 · 마스킹 argmax).
    """
    import torch                                   # noqa: E402
    import train_factor_ablation as T              # noqa: E402

    base = NetworkEnv(use_risk=True, scenario=SCENARIO,
                      risk_window=T.RISK_WINDOW)
    agent = T.DQNAgent(base.state_size, base.action_size)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device, weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = NetworkEnv(use_risk=True, scenario=SCENARIO,
                     risk_window=T.RISK_WINDOW)
    env.reset()
    for _ in range(10):
        env.update_dynamic_qos()
    state = env.get_state()
    t, done = 0, False
    while not done:
        valid = env.get_valid_actions()
        if not valid:
            break
        st_t = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
        with torch.no_grad():
            q = agent.policy_net(st_t)[0]
        masked = torch.full_like(q, -1e9)
        for a in valid:
            masked[a] = q[a]
        a = int(torch.argmax(masked).item())

        st = random.getstate()
        clone = copy.deepcopy(env)
        clone.use_risk = False

        state, r_risk, done, info = env.step(a)
        random.setstate(st)
        _, r_qos, done_c, info_c = clone.step(a)
        assert done == done_c, (seed, t)
        for k in ("delay", "loss"):
            if k in info and k in info_c:
                assert info[k] == info_c[k], (seed, t, k)

        rows.append(dict(policy=policy, seed=seed, step=t, action=a,
                         reason=info.get("reason", ""),
                         delay="%.6f" % info.get("delay", float("nan")),
                         loss="%.6f" % info.get("loss", float("nan")),
                         reward_risk="%.6f" % r_risk,
                         reward_qos="%.6f" % r_qos,
                         ratio=("%.4f" % (r_risk / r_qos)) if r_qos else ""))
        t += 1


def main():
    rows = []
    for s in SEEDS:
        rollout(s, rows)
    for r in rows:
        r.setdefault("policy", "random")

    # 학습된 정책 — K=5(제안 설정) 모델 5개. 원본은 읽기만 한다.
    grows = []
    mdir = os.path.join(SRC, "models")
    for run in range(1, 6):
        mp = os.path.join(mdir, "window5_run%02d.pth" % run)
        if not os.path.exists(mp):
            print(u"  (없음, 건너뜀) %s" % mp)
            continue
        for s in SEEDS:
            greedy_rollout(mp, s, grows, "greedy_w5_run%02d" % run)
    if grows:
        gk = np.array([float(r["reward_risk"]) for r in grows])
        gq = np.array([float(r["reward_qos"]) for r in grows])
        print(u"학습 정책 전이 %d개  위험 [%.2f, %.2f] 평균 %.3f | "
              u"QoS [%.2f, %.2f] 평균 %.3f | 배율 %.2f"
              % (len(grows), gk.min(), gk.max(), gk.mean(),
                 gq.min(), gq.max(), gq.mean(),
                 np.abs(gk).mean() / np.abs(gq).mean()))
        rows.extend(grows)

    def pick_rows(pol, cont):
        return [r for r in rows
                if (r["policy"] == "random" if pol == "random"
                    else r["policy"] != "random")
                and (r["reason"] == "continue" if cont else True)]

    groups = []
    for pol in ("random", "greedy"):
        for cont in (False, True):
            sub = pick_rows(pol, cont)
            if not sub:
                continue
            groups.append((pol + ("_continue" if cont else ""), sub))

    for nm, sub in groups:
        a = np.array([float(r["reward_risk"]) for r in sub])
        b = np.array([float(r["reward_qos"]) for r in sub])
        print(u"  %-16s n=%4d  위험 [%7.2f, %6.2f] 평균 %8.3f | "
              u"QoS [%7.2f, %6.2f] 평균 %8.3f | |위험|/|QoS| %.2f배"
              % (nm, len(sub), a.min(), a.max(), a.mean(),
                 b.min(), b.max(), b.mean(),
                 np.abs(a).mean() / np.abs(b).mean()))

    out = os.path.join(RES, "reward_scale.csv")
    cols = ["policy", "seed", "step", "action", "reason", "delay", "loss",
            "reward_risk", "reward_qos", "ratio"]
    with io.open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    sm = os.path.join(RES, "reward_scale_summary.csv")

    def q(a, p):
        return float(np.percentile(a, p))

    srows = []
    for nm, sub in groups:
        for arm, col in (("risk", "reward_risk"), ("qos", "reward_qos")):
            a = np.array([float(r[col]) for r in sub])
            srows.append(dict(
                policy=nm, arm=arm, n=len(a), mean="%.6f" % a.mean(),
                sd="%.6f" % a.std(ddof=1), min="%.6f" % a.min(),
                p05="%.6f" % q(a, 5), median="%.6f" % np.median(a),
                p95="%.6f" % q(a, 95), max="%.6f" % a.max(),
                mean_abs="%.6f" % np.abs(a).mean()))
    with io.open(sm, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
        w.writeheader()
        for r in srows:
            w.writerow(r)
    print(u"\n저장: %s (%d행)\n      %s (%d행)"
          % (out, len(rows), sm, len(srows)))


if __name__ == "__main__":
    sys.exit(main())
