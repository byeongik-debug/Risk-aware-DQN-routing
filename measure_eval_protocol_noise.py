# -*- coding: utf-8 -*-
"""평가 규약이 **모델을 고정한 채로** 만드는 산포를 잰다.

왜 재는가
---------
7월에 `train_factor_ablation.py` 가 학습 직후 스스로 쓴
`SDN_DQN_new_env/results/factor_*_dynamic_run*_result.csv` 25개가
표 10 계열 숫자의 출처다. 그 평가 루프는 200 에피소드 앞에 시드를 **박지
않는다** — 8000 에피소드 학습이 남긴 난수 상태에서 그대로 이어 돈다
(`train_factor_ablation.py:216-243`). 그래서 모델마다 **다른 200 에피소드**를
겪는다.

11월 하네스(`eval_ablation_risky.evaluate:164-169`)는 반대로 에피소드마다
`EVAL_SEED + i` 를 박아 모든 모델이 **같은 200 에피소드**를 겪게 한다
(공통난수). 지표 계산은 두 루프가 글자 그대로 같다.

그 결과 11월에 지연이 `354.294012` 하나로 붕괴한 모델 여섯
(`-Cong-r03·-Cong-r05·-dDelay-r01·-dDelay-r04·-dLoss-r02·-dLoss-r03`)이
7월 CSV 에서는 346.73 ~ 355.82 로 흩어져 있다. 이 여섯은 200 에피소드
전 구간에서 경로가 **100.0000% 일치**함을 따로 측정한 모델들이다
(`path_identity_modal_arms.csv`).

그러니 그 9.1 ms 산포는 '정책 차이가 0인데도 7월 규약이 만든 폭'으로
읽고 싶어진다. 하지만 그 읽기는 **경로 동일을 11월 표본 하나에서만**
확인했다는 가정에 기댄다. 다른 표본에서 갈라질 가능성이 남는다.

그래서 가정을 없애고 직접 잰다: **모델을 하나로 고정**하고 7월 방식으로
R회 평가한다. 정책 차이는 정의상 정확히 0 이므로, 남는 산포는 전부 규약
몫이다.

무엇이 갈리는가
---------------
  (1) 7월식 산포가 표 10 의 팔 간 차이만큼 크면, 그 차이는 읽을 수 없다.
  (2) 7월식 산포가 작으면 7월 CSV 의 흩어짐은 규약이 아니라 **정책 차이**
      이고, 그러면 11월 하네스가 모델을 구분하지 못한 것(= 붕괴가 측정
      산물)이라는 반대 결론이 선다.
  (3) `-Util` 은 표 10 에서 유일하게 살아남은 효과(+44.34 ms)다. 그 간격이
      규약 산포보다 큰지 같은 자리에서 확인한다.

11월식도 시드 기반을 바꿔 R회 재서 두 규약의 잡음을 같은 축에 놓는다.

출력: `review_results/eval_protocol_noise.csv`
원본(`SDN_DQN_new_env/`)은 읽기만 한다.
"""

import argparse
import csv
import glob
import io
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RESULT_DIR = os.path.join(HERE, "review_results")
OUT_CSV = os.path.join(RESULT_DIR, "eval_protocol_noise.csv")

sys.path.insert(0, HERE)
import eval_ablation_risky as E          # noqa: E402  (HERE 를 떼고 ORIG 를 꽂는다)

NE = E.network_env
RC = E.risk_calculator
T = E.T
torch = E.torch
np = E.np
RS = E.RS

# (라벨, 파일명, 0으로 만든 인자, 창)
TARGETS = [
    ("-Cong-r03", "factor_cong_run03.pth", ["congestion"], 5),
    ("-dLoss-r03", "factor_loss_inst_run03.pth", ["loss_inst"], 5),
    ("-Util-r03", "factor_util_run03.pth", ["utilization"], 5),
]

FIELDS = ("delay", "loss", "hop", "risky_pct")


def rollout(model_path, zeroed, episodes, seed, protocol, risk_window=5):
    """7월/11월 평가 루프. 지표 계산은 두 원본과 글자 그대로 같다.

    protocol="july" : 200 에피소드 블록 앞에 시드를 한 번만 박는다.
    protocol="nov"  : 에피소드마다 seed + i 를 박는다(공통난수).
    """
    RC.ZEROED_FACTORS = list(zeroed)
    base = NE.NetworkEnv(use_risk=True, scenario="dynamic",
                         risk_window=risk_window)
    agent = T.DQNAgent(base.state_size, base.action_size)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device, weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    if protocol == "july":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    acc = dict((k, []) for k in FIELDS)
    reached = 0
    for i in range(episodes):
        if protocol == "nov":
            random.seed(seed + i)
            np.random.seed(seed + i)
            torch.manual_seed(seed + i)

        env = NE.NetworkEnv(use_risk=True, scenario="dynamic",
                            risk_window=risk_window)
        env.reset()
        for _ in range(10):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            st = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                q = agent.policy_net(st)[0]
            masked = torch.full_like(q, -1e9)
            for a in valid:
                masked[a] = q[a]
            state, _, done, _ = env.step(int(torch.argmax(masked).item()))
        if env.current_node != NE.DESTINATION:
            continue
        reached += 1
        m = env.calculate_path_metrics(env.visited)
        hop = max(int(m["hop_count"]), 1)
        risky = max(int(m["unstable_count"]), int(m["congested_count"]))
        acc["delay"].append(float(m["total_delay"]))
        acc["loss"].append(float(m["total_loss"]))
        acc["hop"].append(float(m["hop_count"]))
        acc["risky_pct"].append(risky / hop * 100.0)

    out = dict((k, (float(np.mean(v)) if v else 0.0)) for k, v in acc.items())
    out["reached"] = reached
    out["reach_pct"] = reached / float(episodes) * 100.0
    return out


def july_modal_spread():
    """경로 동일로 측정된 최빈 구성원들의 **7월 CSV** 값 산포."""
    members = os.path.join(RESULT_DIR, "path_identity_modal_arms.csv")
    if not os.path.exists(members):
        return None
    keep = set()
    with io.open(members, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["label"].startswith("*"):
                continue          # 대조군은 구성원이 아니다
            keep.add(row["model"])

    k2f = {"util": "-Util", "cong": "-Cong", "delay_inst": "-dDelay",
           "loss_inst": "-dLoss", "link_type": "-Type"}
    vals = []
    pat = os.path.join(ORIG, "results",
                       "factor_*_dynamic_run*_result.csv")
    for p in sorted(glob.glob(pat)):
        m = re.match(r"factor_(.+)_dynamic_run(\d+)_result\.csv",
                     os.path.basename(p))
        if not m:
            continue
        key, run = m.group(1), int(m.group(2))
        model = "factor_%s_run%02d.pth" % (key, run)
        if model not in keep:
            continue
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            row = list(csv.DictReader(fh))[0]
        vals.append(("%s-r%02d" % (k2f.get(key, key), run),
                     float(row["delay"])))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=T.EVAL_EPISODES)
    ap.add_argument("--protocols", default="july,nov")
    ap.add_argument("--seed0", type=int, default=1000)
    a = ap.parse_args()

    protocols = [p for p in a.protocols.split(",") if p]
    os.makedirs(RESULT_DIR, exist_ok=True)

    rows = []
    store = {}
    for proto in protocols:
        for label, fname, zeroed, win in TARGETS:
            path = os.path.join(ORIG, "models", fname)
            if not os.path.exists(path):
                print("  [없음] %s" % path)
                continue
            vals = []
            for d in range(a.draws):
                seed = a.seed0 + d * 7919          # 겹치지 않는 간격
                r = rollout(path, zeroed, a.episodes, seed, proto, win)
                vals.append(r["delay"])
                rows.append({
                    "protocol": proto, "label": label, "model": fname,
                    "zeroed": ",".join(zeroed), "draw": d + 1, "seed": seed,
                    "episodes": a.episodes, "reached": r["reached"],
                    "reach_pct": "%.4f" % r["reach_pct"],
                    "delay": "%.6f" % r["delay"],
                    "loss": "%.8f" % r["loss"],
                    "hop": "%.6f" % r["hop"],
                    "risky_pct": "%.6f" % r["risky_pct"],
                })
                print("  %-5s %-11s draw%02d seed=%-7d delay=%10.4f "
                      "reach=%5.1f%%"
                      % (proto, label, d + 1, seed, r["delay"],
                         r["reach_pct"]))
            store[(proto, label)] = vals

    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("\n기록: %s (%d행)" % (OUT_CSV, len(rows)))

    print("\n" + "=" * 78)
    print("A. 모델 고정 — 규약이 혼자 만드는 산포 (정책 차이 = 0)")
    print("=" * 78)
    hdr = "%-6s %-11s %4s %11s %9s %11s %11s %9s"
    print(hdr % ("규약", "모델", "R", "평균", "표준편차", "최소", "최대",
                 "범위"))
    print("-" * 78)
    for (proto, label), v in sorted(store.items()):
        mu, sd = RS.mean_std(v)
        print(hdr % (proto, label, len(v), "%.4f" % mu, "%.4f" % sd,
                     "%.4f" % min(v), "%.4f" % max(v),
                     "%.4f" % (max(v) - min(v))))

    sp = july_modal_spread()
    if sp:
        d = [x[1] for x in sp]
        mu, sd = RS.mean_std(d)
        print("\n" + "=" * 78)
        print("B. 경로 100% 동일로 측정된 최빈 구성원들의 7월 CSV 값")
        print("=" * 78)
        for name, val in sorted(sp, key=lambda x: x[1]):
            print("  %-12s %10.4f" % (name, val))
        print("  ---- n=%d  평균 %.4f  표준편차 %.4f  범위 %.4f"
              % (len(d), mu, sd, max(d) - min(d)))
        print("  (11월 하네스에서는 이 모델 전부 354.294012 — 경로까지 동일)")

    print("\n" + "=" * 78)
    print("C. 읽는 법")
    print("=" * 78)
    jm = store.get(("july", "-Cong-r03"))
    jl = store.get(("july", "-dLoss-r03"))
    ju = store.get(("july", "-Util-r03"))

    if jm and jl:
        same = sum(1 for x, y in zip(jm, jl) if abs(x - y) < 1e-9)
        print("  경로 동일 두 모델이 **같은 시드**에서 같은 값: %d/%d draw"
              % (same, min(len(jm), len(jl))))
        print("    -> 이 draw 들은 EVAL_SEED+i 와 다른 표본이다. 전부 같으면")
        print("       정책 동일이 **새 표본에서도** 재확인된 것이다.")
        print("       (한 draw 안에서 두 모델은 같은 표본을 겪으므로 이 칸은")
        print("        규약 잡음이 아니라 정책 동일성을 본다.)")

    if jm:
        mu_m, sd_m = RS.mean_std(jm)
        rng = max(jm) - min(jm)
        print("\n  [예측 검정] 모델 고정 산포 대 7월 6개 모델 산포")
        print("    모델 고정(7월식, R=%d, %d에피소드): 표준편차 %.4f, 범위 %.4f"
              % (len(jm), a.episodes, sd_m, rng))
        if sp:
            d = [x[1] for x in sp]
            _, sd_j = RS.mean_std(d)
            ratio = (sd_j / sd_m) if sd_m > 0 else float("inf")
            print("    7월 CSV 6개 모델(경로 동일):        표준편차 %.4f, 범위 %.4f"
                  % (sd_j, max(d) - min(d)))
            print("    비 = %.2f배" % ratio)
            print("    -> 1 에 가까우면 그 9.1 ms 는 **전부 규약 몫**이고,")
            print("       표 10 의 그만한 팔 간 차이는 읽을 수 없다.")
            print("       1 보다 크게 크면 6개 모델의 7월 차이는 실제 정책")
            print("       차이이고, 그러면 11월 하네스의 붕괴가 **측정 산물**")
            print("       이라는 반대 결론이 선다. 둘 중 하나만 쓸 것.")

    if jm and ju:
        gap = abs(float(np.mean(ju)) - float(np.mean(jm)))
        rng = max(max(jm) - min(jm), max(ju) - min(ju))
        print("\n  -Util 과 최빈 구성원의 평균 간격: %.4f ms" % gap)
        print("    모델 고정 산포의 최대 범위:      %.4f ms" % rng)
        print("    간격/범위 = %.1f배" % (gap / rng if rng > 0 else 0.0))
        print("    -> 1 보다 훨씬 크면 표 10 의 유일한 생존 효과는 규약")
        print("       산포로 설명되지 않는다.")

    if jm:
        print()
        print("  족 D '재현 바닥' 10.1081 ms 는 이 산포와 "
              "견주지 **않는다**.")
        print("    Full(360.8420) 과 K=5(370.9502) 는 둘 다 "
              "11월 하네스의 같은 200 에피소드에서")
        print("    나왔으니 **쌍** 비교다. 쌍 비교에 남는 표본 "
              "항은 아래 '쌍 sd' 이고 이 범위가 아니다(26.10.7).")

    print()
    print("  [쌍 대 비쌍] 같은 추첨 안에서 -Util-r03 - -Cong-r03")
    for proto in ("nov", "july"):
        u = store.get((proto, "-Util-r03")) or []
        c = store.get((proto, "-Cong-r03")) or []
        ks = list(range(min(len(u), len(c))))
        if len(ks) < 3:
            continue
        dmu, dsd = RS.mean_std([u[i] - c[i] for i in ks])
        su = RS.mean_std([u[i] for i in ks])[1]
        sc = RS.mean_std([c[i] for i in ks])[1]
        unp = (su * su + sc * sc) ** 0.5
        off = (1.0 - dsd / unp) * 100.0 if unp else 0.0
        print("    %-5s 쌍 sd %6.4f | 비쌍 기대 %6.4f | "
              "상쇄 %+3.0f%%  (차 %+.4f)"
              % (proto, dsd, unp, off, dmu))
    print("    -> 11월식만 상쇄가 보인다. 7월식은 블록 시드가 같아도 한 번 갈라지면")
    print("       난수 흐름이 어긋나 상쇄가 사라진다(26.10.6).")

    print("\n  주의: 여기서 재는 것은 유의성이 아니라 **크기**다. 7월식은")
    print("  모델마다 다른 난수 상태를 물려받으므로 표 10 의 팔 간 차이에는")
    print("  이 산포가 그대로 섞여 있다. 11월식은 공통난수라 섞이지 않는다.")


if __name__ == "__main__":
    main()
