# -*- coding: utf-8 -*-
"""표 10 Full 기준선의 9.6 ms 차이는 학습 탓인가 평가기 탓인가.

발표된 Full 368.80 = window5_run01~05 (train_window_dqn.py 가 학습하고 평가)
동일조건 재학습 Full 359.16 = factor_full_run01~05 (train_factor_ablation.py 규약)

두 평가기는 epsilon=0 에서 같은 행동을 고르지만 RNG 소비량이 다르다.
  - window 쪽(train_window_dqn.py:272)은 select_action() 을 부르고 그 안에
    random.random() 이 있다 -> 스텝마다 난수 1개 소비
  - factor 쪽(train_factor_ablation.py:230-236)은 select_action 을 우회하고
    인라인 masked argmax 를 쓴다 -> 소비 0개
network_env 는 stdlib random 만 쓰므로(np.random 호출 0곳) 평가 에피소드 열이 갈린다.

그래서 2x2 로 잰다: 팔(window5 / factor_full) x 평가기(select / inline).
같은 모델을 두 평가기로 재면 순수한 평가기 효과가, 같은 평가기로 두 팔을 재면
순수한 학습 효과가 나온다. 원본 디렉터리는 읽기만 한다.

출력: review_results/eval_protocol_compare.csv
"""
import csv
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))

# review_response/risk_calculator.py 사본이 원본을 가리면 안 된다 (ZEROED_FACTORS 훅 없음)
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import numpy as np                              # noqa: E402
import torch                                    # noqa: E402

import risk_calculator                           # noqa: E402
import network_env                               # noqa: E402
import train_factor_ablation as T                # noqa: E402
from network_env import NetworkEnv               # noqa: E402
from topology import DESTINATION                 # noqa: E402

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

OUT_DIR = os.path.join(HERE, "review_results")
ARMS = (
    ("window5", os.path.join(ORIG, "models"), "window5_run%02d.pth"),
    ("factor_full", os.path.join(HERE, "models_ablation"), "factor_full_run%02d.pth"),
)
RUNS = (1, 2, 3, 4, 5)
SCENARIO = "dynamic"
WINDOW = 5
WARMUP = 10          # 두 평가기 공통
EPISODES = int(os.environ.get("EPISODES", "200"))
EVAL_SEEDS = [int(v) for v in os.environ.get("EVAL_SEEDS", "20261114,20261115").split(",")]
EVAL_SEED = EVAL_SEEDS[0]
METRICS = ("delay", "loss", "hop")
PUBLISHED = {"window5": 368.7961, "factor_full": 359.1636}


def check_env():
    """원본에서 임포트됐는지, Full 조건인지 확인한다."""
    for mod in (risk_calculator, network_env, T):
        got = os.path.abspath(mod.__file__)
        if not got.startswith(ORIG):
            raise SystemExit("원본이 아닌 모듈: " + got)
    if list(getattr(risk_calculator, "ZEROED_FACTORS", [])) != []:
        raise SystemExit("ZEROED_FACTORS 가 비어 있지 않다 - Full 조건이 아니다")
    if getattr(risk_calculator, "CUSTOM_WEIGHTS", None) is not None:
        raise SystemExit("CUSTOM_WEIGHTS 가 설정돼 있다 - Full 조건이 아니다")
    if network_env.RISK_WINDOW != WINDOW:
        raise SystemExit("RISK_WINDOW != %d" % WINDOW)


def load_net(path, device):
    """체크포인트를 원본 QNetwork 에 strict 로 올린다 (구조 불일치는 여기서 터진다)."""
    state = torch.load(path, map_location=device)
    in_dim = state["net.0.weight"].shape[1]
    out_dim = state["net.7.weight"].shape[0]
    net = T.QNetwork(in_dim, out_dim).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, in_dim, out_dim


def pick_inline(net, state, valid, device):
    """train_factor_ablation.py:230-236 의 평가 경로. RNG 를 쓰지 않는다."""
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(st)[0]
    masked = torch.full_like(q, -1e9)
    for a in valid:
        masked[a] = q[a]
    return int(torch.argmax(masked).item())


def pick_select(net, state, valid, device):
    """train_window_dqn.py 의 select_action() epsilon=0 경로를 그대로 재현한다.

    random.random() 은 결과를 바꾸지 않지만 난수를 하나 먹는다 - 그게 차이다.
    고른 행동이 인라인 경로와 같은지 매 스텝 확인한다(정책이 아니라 RNG 만 달라야 한다).
    """
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(st)[0]
        _ = torch.max(q).item()
    drawn = random.random()
    if drawn < 0.0:                      # 항상 거짓 (epsilon = 0.0)
        return random.choice(valid)
    masked = torch.full_like(q, -1e9)
    for a in valid:
        masked[a] = q[a]
    return int(torch.argmax(masked).item())


PICKERS = {"select": pick_select, "inline": pick_inline}


def run_eval(net, picker_name, device, episodes=EPISODES, seed=EVAL_SEED):
    """두 평가기의 공통 골격. 시나리오/워밍업/성공조건/지표추출은 두 원본과 동일하다."""
    picker = PICKERS[picker_name]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    delays, losses, hops = [], [], []
    sigs = []
    steps = 0
    for _ in range(episodes):
        env = NetworkEnv(use_risk=True, scenario=SCENARIO, risk_window=WINDOW)
        env.reset()
        for _ in range(WARMUP):
            env.update_dynamic_qos()
        # 워밍업 직후 환경 지문: 평가 에피소드 열이 갈리는지 보기 위한 것
        sigs.append(round(sum(q["utilization"] for q in env.qos.values()), 9))
        state = env.get_state()
        done = False
        while not done:
            valid = env.get_valid_actions()
            if not valid:
                break
            action = picker(net, state, valid, device)
            steps += 1
            state, _, done, _ = env.step(action)
        if env.current_node == DESTINATION:
            m = env.calculate_path_metrics(env.visited)
            delays.append(m["total_delay"])
            losses.append(m["total_loss"])
            hops.append(m["hop_count"])
    return {
        "delay": float(np.mean(delays)) if delays else float("nan"),
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "hop": float(np.mean(hops)) if hops else float("nan"),
        "success_rate": len(delays) / episodes * 100.0,
        "n_success": len(delays),
        "steps": steps,
        "sigs": sigs,
    }


def mean_std(xs):
    n = len(xs)
    mean = sum(xs) / n
    if n < 2:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    return mean, math.sqrt(var)


def check_policy_equivalence(net, device, episodes=3, seed=EVAL_SEED):
    """두 평가기가 같은 상태에서 같은 행동을 고르는지 확인한다.

    고르는 행동이 다르면 아래 비교는 'RNG 소비 차이'가 아니라 '정책 차이'를
    재는 것이 되므로, 이 확인이 통과해야 결과를 해석할 수 있다.
    """
    random.seed(seed)
    np.random.seed(seed)
    mismatch = 0
    checked = 0
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
            a_inline = pick_inline(net, state, valid, device)
            a_select = pick_select(net, state, valid, device)
            checked += 1
            if a_inline != a_select:
                mismatch += 1
            state, _, done, _ = env.step(a_inline)
    return checked, mismatch


def main():
    check_env()
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)

    probe_net, in_dim, out_dim = load_net(
        os.path.join(ORIG, "models", "window5_run01.pth"), device)
    print("체크포인트 입출력 차원: %d -> %d" % (in_dim, out_dim))
    checked, mismatch = check_policy_equivalence(probe_net, device)
    print("정책 동일성 확인: %d 스텝 중 불일치 %d" % (checked, mismatch))
    if mismatch != 0:
        raise SystemExit("두 평가기가 다른 행동을 고른다 - 비교 전제가 깨졌다")

    rows = []
    sig_store = {}
    t0 = time.time()
    for arm, model_dir, pattern in ARMS:
        for run in RUNS:
            path = os.path.join(model_dir, pattern % run)
            net, _, _ = load_net(path, device)
            for proto in ("select", "inline"):
                for seed in EVAL_SEEDS:
                    res = run_eval(net, proto, device, seed=seed)
                    sig_store[(arm, run, proto, seed)] = res.pop("sigs")
                    res.update({"arm": arm, "run_id": run, "protocol": proto,
                                "eval_seed": seed, "model": os.path.basename(path)})
                    rows.append(res)
                    print("  %-12s run%02d %-6s seed=%d delay=%8.3f loss=%.5f "
                          "hop=%.3f succ=%5.1f%% (%.0fs)"
                          % (arm, run, proto, seed, res["delay"], res["loss"],
                             res["hop"], res["success_rate"], time.time() - t0))

    cols = ["arm", "run_id", "model", "protocol", "eval_seed", "delay", "loss", "hop",
            "success_rate", "n_success", "steps"]
    out = os.path.join(OUT_DIR, "eval_protocol_compare.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print("저장: " + out)
    return rows, sig_store


def report(rows, sig_store):
    """평가기 효과와 학습 효과를 분리해서 보고한다."""
    import review_stats as RS

    def grab(arm, proto, metric, seed=None):
        seed = EVAL_SEED if seed is None else seed
        return [r[metric] for r in sorted(rows, key=lambda x: x["run_id"])
                if r["arm"] == arm and r["protocol"] == proto
                and r["eval_seed"] == seed]

    print("")
    print("[1] 평가 에피소드 열이 실제로 갈리는가 (워밍업 직후 환경 지문)")
    arm, run = ARMS[0][0], RUNS[0]
    a = sig_store[(arm, run, "select", EVAL_SEED)]
    b = sig_store[(arm, run, "inline", EVAL_SEED)]
    first_same = (a[0] == b[0])
    n_diff = sum(1 for x, y in zip(a, b) if x != y)
    print("    1화 동일? %s  (스텝 전에는 RNG 소비가 같으므로 동일해야 한다)" % first_same)
    print("    %d개 에피소드 중 지문이 다른 것: %d개" % (len(a), n_diff))
    first_diff = next((i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print("    처음 갈리는 에피소드: %s" % first_diff)

    print("")
    print("[2] 평가기 효과 — 같은 모델을 두 평가기로 (대응, n=%d)" % (len(ARMS) * len(RUNS)))
    print("    %-7s %12s %12s %10s" % ("metric", "select", "inline", "paired d"))
    for metric in METRICS:
        sel, inl, diffs = [], [], []
        for a_name, _, _ in ARMS:
            s_vals, i_vals = grab(a_name, "select", metric), grab(a_name, "inline", metric)
            sel += s_vals
            inl += i_vals
            diffs += [s - i for s, i in zip(s_vals, i_vals)]
        d_mean, d_sd = mean_std(diffs)
        se = d_sd / math.sqrt(len(diffs)) if d_sd > 0 else float("nan")
        t = d_mean / se if se == se and se > 0 else float("nan")
        p = RS.t_two_sided_p(t, len(diffs) - 1) if t == t else float("nan")
        print("    %-7s %12.5f %12.5f %10.5f  (t=%.3f, p=%s)"
              % (metric, mean_std(sel)[0], mean_std(inl)[0], d_mean, t, RS.fmt_p(p)))

    print("")
    print("[3] 학습 효과 — 평가기를 고정하고 두 팔을 (Welch, n=5 대 5)")
    for proto in ("select", "inline"):
        print("    평가기 %s" % proto)
        for metric in METRICS:
            xs = grab("window5", proto, metric)
            ys = grab("factor_full", proto, metric)
            diff, t, df, p = RS.welch(xs, ys)
            mde = RS.min_detectable_effect(xs, ys)
            print("      %-7s window5=%9.4f±%.4f  factor_full=%9.4f±%.4f  "
                  "diff=%+8.4f  p=%-8s MDE=%.4f"
                  % (metric, mean_std(xs)[0], mean_std(xs)[1],
                     mean_std(ys)[0], mean_std(ys)[1], diff, RS.fmt_p(p), mde))

    print("")
    print("[4] 발표된 값과의 대조 (dynamic delay)")
    pub_gap = PUBLISHED["window5"] - PUBLISHED["factor_full"]
    print("    발표/재학습 격차: %.4f - %.4f = %+.4f ms" %
          (PUBLISHED["window5"], PUBLISHED["factor_full"], pub_gap))
    for proto in ("select", "inline"):
        xs = grab("window5", proto, "delay")
        ys = grab("factor_full", proto, "delay")
        print("    평가기 %-6s 고정시 팔 격차: %+8.4f ms (발표 격차의 %.0f%%)"
              % (proto, mean_std(xs)[0] - mean_std(ys)[0],
                 (mean_std(xs)[0] - mean_std(ys)[0]) / pub_gap * 100.0))
    for arm, _, _ in ARMS:
        sel = mean_std(grab(arm, "select", "delay"))[0]
        inl = mean_std(grab(arm, "inline", "delay"))[0]
        print("    팔 %-12s 고정시 평가기 격차: %+8.4f ms (발표 격차의 %.0f%%)"
              % (arm, sel - inl, (sel - inl) / pub_gap * 100.0))

    print("")
    print("[5] 재학습 Full 의 run05 단독 영향 (leave-one-out)")
    for proto in ("select", "inline"):
        ys = grab("factor_full", proto, "delay")
        full = mean_std(ys)[0]
        for drop in range(len(ys)):
            kept = [v for i, v in enumerate(ys) if i != drop]
            if drop == 4:
                print("    %-6s 전체 %.3f -> run05 제외 %.3f (%+.3f ms)"
                      % (proto, full, mean_std(kept)[0], mean_std(kept)[0] - full))

    print("")
    print("[6] 평가 시드 효과 — 같은 모델, 같은 평가기, 시드만 다르게")
    for proto in ("select", "inline"):
        for metric in METRICS:
            spread = []
            for a_name, _, _ in ARMS:
                per_seed = [grab(a_name, proto, metric, sd) for sd in EVAL_SEEDS]
                for vals in zip(*per_seed):
                    spread.append(max(vals) - min(vals))
            m, sd = mean_std(spread)
            print("    %-6s %-7s 모델별 시드간 범위 평균 %.5f (최대 %.5f)"
                  % (proto, metric, m, max(spread)))

    print("")
    print("[7] 체크포인트 10개가 만드는 서로 다른 행동 부류의 수")
    for proto in ("select", "inline"):
        for sd in EVAL_SEEDS:
            classes = {}
            for r in rows:
                if r["protocol"] != proto or r["eval_seed"] != sd:
                    continue
                key = (round(r["delay"], 6), round(r["loss"], 8),
                       round(r["hop"], 6), r["steps"])
                classes.setdefault(key, []).append(
                    "%s_r%02d" % (r["arm"][:3], r["run_id"]))
            print("    %-6s seed=%d: 모델 %d개 -> 서로 다른 결과 %d개"
                  % (proto, sd, sum(len(v) for v in classes.values()), len(classes)))
            for key in sorted(classes, key=lambda k: -len(classes[k])):
                members = classes[key]
                if len(members) > 1:
                    print("        delay=%9.4f hop=%.3f  <- %s"
                          % (key[0], key[2], ", ".join(sorted(members))))


if __name__ == "__main__":
    sys.path.insert(0, HERE)      # review_stats 는 review_response 쪽에 있다
    report(*main())
