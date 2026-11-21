# -*- coding: utf-8 -*-
"""제안 정책이 상태의 위험 50칸을 실제로 쓰는가 — 추론 시점 가림(occlusion).

동기: 22.7절에서 표 11의 `NoRisk-DQN` 행이 입력 175차원(위험 50칸 자리에 원시 QoS
100칸)이라는 것이 드러났다. 즉 그 행은 위험 절제가 아니라 **입력 표현 교체**이고,
논문에는 "학습된 정책이 위험 점수를 쓰는가"를 재는 측정이 없다. 진짜 절제는 125차원을
유지한 채 위험 50칸만 상수화해서 재학습하는 것인데 1런에 7.64시간이 든다(21절 측정).

그래서 학습 없이 **추론 시점에서** 가린다. 두 부분이다.

(A) 롤아웃 없음: 고정 상태 묶음에서 블록을 가리고 마스킹 argmax 가 바뀌는 비율.
    블록은 위험[0:50] / 현재[50:75] / 목적[75:100] / 방문[100:125] 네 개
    (`network_env.get_state`, EDGES 50개 · NUM_NODES 25개).
    가림 방식 세 가지 - 0으로, 묶음 내 다른 상태의 같은 블록으로(주변분포 유지),
    묶음 평균으로.
    **음성 통제가 설계에 박혀 있다**: 19절에서 `dest_onehot` 25칸은 상수
    (`DESTINATION` 고정)이므로 목적 블록 가림은 0%에 가까워야 한다. 그게 안 나오면
    측정 코드가 틀린 것이다.

(B) 롤아웃: 전방계산 직전에 상태를 같은 방식으로 고쳐서 실제 평가 지표를 낸다.
    조건 - 원본 / 위험0 / 위험섞기 / 목적0(통제).

비교군은 위험 수식과 K 가 같은 `window5`·`factor_full` 10개 모델뿐이다
(factor_* 의 다른 팔은 위험 수식 자체가 달라 블록 의미가 다르다).

출력: review_results/state_occlusion_battery.csv
      review_results/state_occlusion_rollout.csv
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

import risk_calculator                           # noqa: E402
import network_env                               # noqa: E402
import train_factor_ablation as T                # noqa: E402
from network_env import NetworkEnv               # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

OUT_DIR = os.path.join(HERE, "review_results")
ORIG_MODELS = os.path.join(ORIG, "models")
ABL_MODELS = os.path.join(HERE, "models_ablation")

SCENARIO = "dynamic"
WINDOW = 5
WARMUP = 10
SEED = 20261114
BATTERY_N = int(os.environ.get("BATTERY_N", "250"))
STATE_DIM = 125
EPISODES = int(os.environ.get("EPISODES", "50"))
EVAL_SEED = int(os.environ.get("EVAL_SEED", "20261114"))

BLOCKS = (("risk", 0, 50), ("curr", 50, 75),
          ("dest", 75, 100), ("visited", 100, 125))
MODES = ("zero", "shuffle", "mean")

ARMS = (("window5", ORIG_MODELS, "window5_run%02d.pth"),
        ("factor_full", ABL_MODELS, "factor_full_run%02d.pth"))
RUNS = (1, 2, 3, 4, 5)

# 조건 일곱 개. risk_mean 이 진짜 절제에 가장 가깝다(정보 0, 크기는 그럴듯함).
# curr_zero·visited_shuffle 는 "필수 블록을 가리면 얼마나 망가지는가"의 척도이고,
# dest_zero 는 음성 통제다(19절에서 상수로 밝혀진 블록).
ROLLOUT_CONDS = (("intact", None, None),
                 ("risk_zero", "risk", "zero"),
                 ("risk_shuffle", "risk", "shuffle"),
                 ("risk_mean", "risk", "mean"),
                 ("curr_zero", "curr", "zero"),
                 ("visited_shuffle", "visited", "shuffle"),
                 ("dest_zero", "dest", "zero"))


def load_net(path, device):
    state = torch.load(path, map_location=device)
    if int(state["net.0.weight"].shape[1]) != STATE_DIM:
        raise SystemExit("%s 입력 차원이 %d 가 아니다" % (path, STATE_DIM))
    net = T.QNetwork(STATE_DIM, state["net.7.weight"].shape[0]).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def qvals(net, state, device):
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return net(st)[0].cpu().numpy()


def masked_argmax_from_q(q, valid):
    best, best_v = valid[0], q[valid[0]]
    for a in valid[1:]:
        if q[a] > best_v:
            best, best_v = a, q[a]
    return int(best)


def occlude(state, lo, hi, mode, donor, blockmean):
    """상태 사본의 [lo:hi] 를 가린다. donor 는 섞기용 공여 상태."""
    out = np.array(state, dtype=np.float32, copy=True)
    if mode == "zero":
        out[lo:hi] = 0.0
    elif mode == "shuffle":
        out[lo:hi] = np.asarray(donor[lo:hi], dtype=np.float32)
    elif mode == "mean":
        out[lo:hi] = np.asarray(blockmean[lo:hi], dtype=np.float32)
    else:
        raise ValueError(mode)
    return out


def build_battery(kind, device, ref_net, target=BATTERY_N, seed=SEED):
    """(state, valid) 쌍. cluster_policy_classes.py 와 같은 절차·같은 시드."""
    random.seed(seed)
    np.random.seed(seed)
    pairs = []
    guard = 0
    while len(pairs) < target and guard < 10000:
        guard += 1
        env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done and len(pairs) < target:
            valid = env.get_valid_actions()
            if not valid:
                break
            pairs.append((np.asarray(state, dtype=np.float32),
                          tuple(sorted(valid))))
            if kind == "random":
                action = random.choice(valid)
            else:
                action = masked_argmax_from_q(qvals(ref_net, state, device),
                                              list(valid))
            state, _, done, _ = env.step(action)
    return pairs


def check_env():
    if list(getattr(risk_calculator, "ZEROED_FACTORS", [])) != []:
        raise SystemExit("ZEROED_FACTORS 가 비어 있지 않다")
    if getattr(risk_calculator, "CUSTOM_WEIGHTS", None) is not None:
        raise SystemExit("CUSTOM_WEIGHTS 가 설정돼 있다")
    if network_env.RISK_WINDOW != WINDOW:
        raise SystemExit("RISK_WINDOW != %d" % WINDOW)
    for mod in (network_env, risk_calculator, T):
        if not os.path.abspath(mod.__file__).startswith(ORIG):
            raise SystemExit("%s 가 원본에서 오지 않았다" % mod.__name__)


def battery_occlusion(nets, battery, device, seed=SEED):
    """(A) 묶음에서 블록을 가리고 argmax 가 바뀌는 비율. 롤아웃 없음."""
    rng = random.Random(seed)
    n = len(battery)
    blockmean = np.mean(np.stack([s for s, _ in battery]), axis=0)
    donors = [rng.randrange(n) for _ in range(n)]
    donors = [d if d != i else (d + 1) % n for i, d in enumerate(donors)]

    base_act, base_q = {}, {}
    for name, net in nets:
        acts, qs = [], []
        for s, v in battery:
            q = qvals(net, s, device)
            qs.append(q)
            acts.append(masked_argmax_from_q(q, list(v)))
        base_act[name], base_q[name] = acts, qs

    rows = []
    for blk, lo, hi in BLOCKS:
        for mode in MODES:
            for name, net in nets:
                changed, dq = 0, []
                for i, (s, v) in enumerate(battery):
                    alt = occlude(s, lo, hi, mode, battery[donors[i]][0],
                                  blockmean)
                    q = qvals(net, alt, device)
                    a = masked_argmax_from_q(q, list(v))
                    if a != base_act[name][i]:
                        changed += 1
                    b = base_act[name][i]
                    dq.append(abs(float(q[b] - base_q[name][i][b])))
                rows.append({"block": blk, "mode": mode, "model": name,
                             "n_states": len(battery),
                             "changed": changed,
                             "change_pct": changed / len(battery) * 100.0,
                             "mean_abs_dq": float(np.mean(dq))})
    return rows, base_act


def risk_dim_sensitivity(nets, battery, device, base_act):
    """위험 50칸을 한 칸씩 0으로 — 정책이 몇 개 엣지를 읽는가."""
    out = []
    for name, net in nets:
        per_dim = []
        for d in range(50):
            changed = 0
            for i, (s, v) in enumerate(battery):
                alt = np.array(s, copy=True)
                alt[d] = 0.0
                a = masked_argmax_from_q(qvals(net, alt, device), list(v))
                if a != base_act[name][i]:
                    changed += 1
            per_dim.append(changed)
        live = sum(1 for c in per_dim if c > 0)
        out.append({"model": name, "live_dims": live,
                    "max_changed": max(per_dim),
                    "total_changed": sum(per_dim),
                    "per_dim": per_dim})
    return out


def run_eval_occluded(net, device, blk, mode, donor_pool, blockmean,
                      episodes=EPISODES, seed=EVAL_SEED):
    """(B) 전방계산 직전에 상태를 가리고 실제 평가 지표를 낸다.

    행동 선택은 RNG 를 쓰지 않는 인라인 argmax 이고, 섞기용 공여 추출은 전용
    RNG 인스턴스를 쓴다. 그래서 네 조건이 **같은 평가 에피소드 열**을 본다
    (22.8절의 평가기 교란을 이 측정에 섞지 않기 위한 것).
    """
    from network_env import DESTINATION
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    drng = random.Random(seed + 1)

    lo = hi = None
    if blk is not None:
        for nm, a, b in BLOCKS:
            if nm == blk:
                lo, hi = a, b
    delays, losses, hops = [], [], []
    steps = 0
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            if lo is None:
                fed = state
            else:
                donor = donor_pool[drng.randrange(len(donor_pool))]
                fed = occlude(state, lo, hi, mode, donor, blockmean)
            action = masked_argmax_from_q(qvals(net, fed, device), list(valid))
            steps += 1
            state, _, done, _ = env.step(action)
        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
    return {"delay": float(np.mean(delays)) if delays else float("nan"),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "hop": float(np.mean(hops)) if hops else float("nan"),
            "success_rate": len(delays) / episodes * 100.0,
            "n_success": len(delays), "steps": steps}


def mean_std(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, var ** 0.5


def main():
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    check_env()

    models = []
    for arm, d, pat in ARMS:
        for r in RUNS:
            p = os.path.join(d, pat % r)
            if os.path.exists(p):
                models.append(("%s_run%02d" % (arm, r), arm, r, p))
    if len(models) != 10:
        raise SystemExit("엄격 비교군 10개가 아니다: %d" % len(models))
    nets = [(nm, load_net(p, device)) for nm, _, _, p in models]
    print("모델 %d개 적재 (125차원 엄격 비교군)" % len(nets))

    ref = dict(nets)["window5_run01"]
    battery = build_battery("greedy", device, ref)
    print("배치 상태 묶음 %d개 (greedy, window5_run01 기준)" % len(battery))
    blockmean = np.mean(np.stack([s for s, _ in battery]), axis=0)
    donor_pool = [s for s, _ in battery]

    brows, base_act = battery_occlusion(nets, battery, device)
    with open(os.path.join(OUT_DIR, "state_occlusion_battery.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["block", "mode", "model",
                                          "n_states", "changed",
                                          "change_pct", "mean_abs_dq"])
        w.writeheader()
        for r in brows:
            w.writerow(r)

    dims = risk_dim_sensitivity(nets, battery, device, base_act)

    rrows = []
    for nm, net in nets:
        for cond, blk, mode in ROLLOUT_CONDS:
            s = run_eval_occluded(net, device, blk, mode, donor_pool,
                                  blockmean)
            s.update({"model": nm, "condition": cond,
                      "episodes": EPISODES, "eval_seed": EVAL_SEED})
            rrows.append(s)
            print("  %-16s %-13s delay=%.4f hop=%.3f succ=%.1f%%"
                  % (nm, cond, s["delay"], s["hop"], s["success_rate"]))
    with open(os.path.join(OUT_DIR, "state_occlusion_rollout.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "condition", "episodes",
                                          "eval_seed", "delay", "loss",
                                          "hop", "success_rate",
                                          "n_success", "steps"])
        w.writeheader()
        for r in rrows:
            w.writerow(r)

    report(brows, dims, rrows)


def report(brows, dims, rrows):
    import review_stats as RS
    print("")
    print("=" * 74)
    print("[1] 블록 가림 — 배치 상태에서 행동이 바뀌는 비율 (모델 10개 평균)")
    print("-" * 74)
    print("%-9s %-9s %10s %10s %12s" % ("블록", "가림", "평균%", "표준편차",
                                        "평균|dQ|"))
    for blk, _, _ in BLOCKS:
        for mode in MODES:
            sel = [r for r in brows if r["block"] == blk and r["mode"] == mode]
            m, sd = mean_std([r["change_pct"] for r in sel])
            dq, _ = mean_std([r["mean_abs_dq"] for r in sel])
            print("%-9s %-9s %10.2f %10.2f %12.5f" % (blk, mode, m, sd, dq))
    print("")
    print("음성 통제: dest 블록은 19절에서 상수로 밝혀졌다 → 0%에 가까워야 한다.")

    print("")
    print("[2] 위험 50칸 — 한 칸씩 0으로 (정책이 읽는 엣지 수)")
    print("-" * 74)
    print("%-16s %10s %12s %12s" % ("모델", "살아있는칸", "최대변화수",
                                    "총변화수"))
    for d in dims:
        print("%-16s %10d %12d %12d" % (d["model"], d["live_dims"],
                                        d["max_changed"], d["total_changed"]))
    lv, lsd = mean_std([float(d["live_dims"]) for d in dims])
    print("평균 살아있는 칸 %.1f / 50 (sd %.1f)" % (lv, lsd))

    print("")
    print("[3] 롤아웃 — 조건별 지표 (모델 10개, %d 에피소드)" % EPISODES)
    print("-" * 74)
    print("%-16s %14s %12s %10s %10s"
          % ("조건", "지연", "손실", "홉", "성공%"))
    agg = {}
    for cond, _, _ in ROLLOUT_CONDS:
        sel = [r for r in rrows if r["condition"] == cond]
        agg[cond] = sel
        dm, dsd = mean_std([r["delay"] for r in sel])
        lm, _ = mean_std([r["loss"] for r in sel])
        hm, _ = mean_std([r["hop"] for r in sel])
        sm, _ = mean_std([r["success_rate"] for r in sel])
        print("%-16s %7.2f±%-6.2f %12.5f %10.3f %10.2f"
              % (cond, dm, dsd, lm, hm, sm))

    print("")
    print("[4] 원본 대비 짝지은 검정 (같은 모델, 같은 에피소드 열)")
    print("-" * 74)
    print("%-16s %-13s %10s %9s %10s %10s"
          % ("조건", "지표", "차이", "t", "p", "p_holm"))
    tests, labels = [], []
    for cond, _, _ in ROLLOUT_CONDS[1:]:
        for metric in ("delay", "loss", "hop", "success_rate"):
            pairs = []
            for r in agg[cond]:
                base = [b for b in agg["intact"] if b["model"] == r["model"]]
                if base and r[metric] == r[metric]:
                    pairs.append(r[metric] - base[0][metric])
            if len(pairs) < 2:
                continue
            m, sd = mean_std(pairs)
            se = sd / len(pairs) ** 0.5
            t = m / se if se > 0 else 0.0
            p = RS.t_two_sided_p(t, len(pairs) - 1) if se > 0 else 1.0
            tests.append((cond, metric, m, t, p))
            labels.append("%s/%s" % (cond, metric))
    holm = RS.holm_bonferroni([x[4] for x in tests])
    for (cond, metric, m, t, p), ph in zip(tests, holm):
        print("%-16s %-13s %10.4f %9.3f %10s %10s"
              % (cond, metric, m, t, RS.fmt_p(p), RS.fmt_p(ph)))
    print("")
    print("쓴 파일: review_results/state_occlusion_battery.csv,")
    print("         review_results/state_occlusion_rollout.csv")


if __name__ == "__main__":
    sys.path.append(HERE)
    main()
