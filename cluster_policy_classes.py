# -*- coding: utf-8 -*-
"""표 10·표 12 의 체크포인트들이 실제로 몇 개의 서로 다른 정책인가.

compare_eval_protocols.py 에서 서로 다른 체크포인트 10개가 평가 지표를 소수점까지
똑같이 내는 것을 봤다 (6개가 한 부류, 두 학습 스크립트를 가로지른다). 그게 20
에피소드의 우연인지 정책 자체가 같은 것인지 가린다.

방법: 환경 롤아웃을 다시 돌리지 않는다. 고정된 상태 묶음(battery)을 한 번 만들고
각 체크포인트의 마스킹 argmax 행동 벡터를 뽑아 동일한 벡터끼리 묶는다.

묶음은 두 종류다. 하나는 무작위 유효행동 보행(정책과 무관), 하나는 기준 모델
(window5_run01)의 그리디 경로(어느 한 정책에 유리한 분포). 둘을 따로 집계해서
분포 편향이 결론을 만드는지 보이게 한다.

주의: 상태 벡터의 앞 50칸은 위험값이고 위험 수식은 팔마다 다르다 (factor_*는 인자를
0으로, weight_*는 가중치를 바꾼다). 그래서 공통 묶음으로 재는 것은 '배치 상태에서의
행동'이 아니라 '같은 입력에 대한 정책 함수의 동일성'이다. 수식과 K가 같은
window5 / factor_full 10개만 엄격 비교군이고 나머지는 참고로 표기한다.

출력: review_results/policy_classes.csv
"""
import collections
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
BATTERY_N = 250          # 묶음 종류별 상태 수
SCENARIO = "dynamic"
WINDOW = 5
WARMUP = 10
SEED = 20261114
PERMS = int(os.environ.get("PERMS", "2000"))
REFERENCE = os.path.join(ORIG_MODELS, "window5_run01.pth")

# 엄격 비교군: 위험 수식과 K 가 같아서 배치 상태까지 동일한 팔
STRICT = ("window5", "factor_full")


def model_inventory():
    """(arm, run_id, path) 목록. 원본과 review_response 양쪽을 모은다."""
    items = []
    for d in (ORIG_MODELS, ABL_MODELS):
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".pth"):
                continue
            stem = name[:-4]
            if "_run" not in stem:
                continue
            arm, run = stem.rsplit("_run", 1)
            if not run.isdigit():
                continue
            items.append((arm, int(run), os.path.join(d, name)))
    return items


STATE_DIM = 125          # 제안 모델의 상태 차원 (위험 50 + 현재 25 + 목적 25 + 방문 25)


def load_net(path, device):
    state = torch.load(path, map_location=device)
    net = T.QNetwork(state["net.0.weight"].shape[1],
                     state["net.7.weight"].shape[0]).to(device)
    net.load_state_dict(state, strict=True)
    net.eval()
    return net


def input_dim(path):
    return int(torch.load(path, map_location="cpu")["net.0.weight"].shape[1])


def masked_argmax(net, state, valid, device):
    st = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        q = net(st)[0]
    masked = torch.full_like(q, -1e9)
    for a in valid:
        masked[a] = q[a]
    return int(torch.argmax(masked).item())


def build_battery(kind, device, ref_net=None, target=BATTERY_N, seed=SEED):
    """(state, valid) 쌍을 모은다. kind: 'random' 또는 'greedy'."""
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
            pairs.append((list(state), tuple(sorted(valid))))
            if kind == "random":
                action = random.choice(valid)
            else:
                action = masked_argmax(ref_net, state, valid, device)
            state, _, done, _ = env.step(action)
    return pairs


def action_vector(net, battery, device):
    return tuple(masked_argmax(net, s, list(v), device) for s, v in battery)


def main():
    device = torch.device("cpu")
    os.makedirs(OUT_DIR, exist_ok=True)
    if list(getattr(risk_calculator, "ZEROED_FACTORS", [])) != []:
        raise SystemExit("ZEROED_FACTORS 가 비어 있지 않다 - 묶음이 Full 조건이 아니다")
    if network_env.RISK_WINDOW != WINDOW:
        raise SystemExit("RISK_WINDOW != %d" % WINDOW)

    ref_net = load_net(REFERENCE, device)
    batteries = {
        "random": build_battery("random", device),
        "greedy": build_battery("greedy", device, ref_net=ref_net),
    }
    for kind, b in batteries.items():
        states = set(tuple(s) for s, _ in b)
        print("묶음 %-7s 상태 %d개 (서로 다른 상태 %d개)" % (kind, len(b), len(states)))
        if len(states) < len(b) * 0.9:
            print("  경고: 중복 상태가 10%% 넘는다 - 분해능이 낮다")

    items = model_inventory()
    print("체크포인트 %d개" % len(items))

    # norisk 는 위험점수 50칸 대신 원시 QoS 100칸을 받아 상태가 175차원이다.
    # 공통 묶음으로 잴 수 없으므로 제외하고 그 사실을 보고한다.
    skipped = collections.Counter()
    kept = []
    for arm, run, path in items:
        d = input_dim(path)
        if d != STATE_DIM:
            skipped[(arm, d)] += 1
        else:
            kept.append((arm, run, path))
    for (arm, d), cnt in sorted(skipped.items()):
        print("  제외 %-18s %d개 (상태 %d차원 != %d)" % (arm, cnt, d, STATE_DIM))
    items = kept
    print("비교 대상 %d개" % len(items))

    vectors = {}
    for arm, run, path in items:
        net = load_net(path, device)
        for kind, b in batteries.items():
            vectors[(kind, arm, run)] = action_vector(net, b, device)

    rows = []
    for kind in batteries:
        classes = {}
        for (k, arm, run), vec in vectors.items():
            if k != kind:
                continue
            classes.setdefault(vec, []).append((arm, run))
        ranked = sorted(classes.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1])))
        for idx, (vec, members) in enumerate(ranked, start=1):
            for arm, run in sorted(members):
                rows.append({"battery": kind, "class_id": idx,
                             "class_size": len(members), "arm": arm, "run_id": run})

    out = os.path.join(OUT_DIR, "policy_classes.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["battery", "class_id", "class_size",
                                           "arm", "run_id"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("저장: " + out)
    return batteries, vectors, items


def report(batteries, vectors, items):
    arms = sorted(set(a for a, _, _ in items))

    for kind in sorted(batteries):
        n = len(batteries[kind])
        classes = collections.defaultdict(list)
        for (k, arm, run), vec in vectors.items():
            if k == kind:
                classes[vec].append((arm, run))
        print("")
        print("=== 묶음 %s (상태 %d개) ===" % (kind, n))
        print("체크포인트 %d개 -> 서로 다른 정책 %d개" %
              (sum(len(v) for v in classes.values()), len(classes)))
        ranked = sorted(classes.items(), key=lambda kv: -len(kv[1]))
        for idx, (vec, members) in enumerate(ranked, start=1):
            if len(members) < 2:
                continue
            tags = ", ".join("%s_r%02d" % (a, r) for a, r in sorted(members))
            print("  부류 %d (%d개): %s" % (idx, len(members), tags))
        singles = sum(1 for _, m in ranked if len(m) == 1)
        print("  단독 부류: %d개" % singles)

        print("  -- 엄격 비교군 (%s) --" % ", ".join(STRICT))
        strict_classes = collections.defaultdict(list)
        for vec, members in classes.items():
            keep = [(a, r) for a, r in members if a in STRICT]
            if keep:
                strict_classes[vec].extend(keep)
        print("  모델 %d개 -> 서로 다른 정책 %d개" %
              (sum(len(v) for v in strict_classes.values()), len(strict_classes)))
        for vec, members in sorted(strict_classes.items(), key=lambda kv: -len(kv[1])):
            tags = ", ".join("%s_r%02d" % (a, r) for a, r in sorted(members))
            print("    %d개: %s" % (len(members), tags))

        print("  -- 팔 안쪽 일치율 (같은 팔 런끼리 평균 행동일치) --")
        for arm in arms:
            vs = [vectors[(kind, a, r)] for a, r, _ in
                  [(x, y, z) for x, y, z in items if x == arm]]
            if len(vs) < 2:
                continue
            agree = []
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    same = sum(1 for x, y in zip(vs[i], vs[j]) if x == y)
                    agree.append(same / float(n) * 100.0)
            print("    %-18s %5.1f%% (최소 %5.1f%%, 쌍 %d개)"
                  % (arm, sum(agree) / len(agree), min(agree), len(agree)))

        print("  -- window5 대 다른 팔 (팔간 평균 행동일치) --")
        base = [vectors[(kind, "window5", r)] for _, r, _ in
                [(x, y, z) for x, y, z in items if x == "window5"]]
        for arm in arms:
            if arm == "window5":
                continue
            other = [vectors[(kind, arm, r)] for _, r, _ in
                     [(x, y, z) for x, y, z in items if x == arm]]
            agree = []
            for u in base:
                for v in other:
                    agree.append(sum(1 for x, y in zip(u, v) if x == y) / float(n) * 100.0)
            print("    %-18s %5.1f%% (최대 %5.1f%%)"
                  % (arm, sum(agree) / len(agree), max(agree)))

    print("")
    print("[8] 팔 라벨이 정책을 가리는가 — 순열검정 (B=%d)" % PERMS)
    for kind in sorted(batteries):
        n = len(batteries[kind])
        labels = [a for a, _, _ in items]
        vecs = [vectors[(kind, a, r)] for a, r, _ in items]
        m = len(vecs)
        agree = [[0.0] * m for _ in range(m)]
        for i in range(m):
            for j in range(i + 1, m):
                v = sum(1 for x, y in zip(vecs[i], vecs[j]) if x == y) / float(n)
                agree[i][j] = agree[j][i] = v * 100.0

        def stat(labs):
            win, cross = [], []
            for i in range(m):
                for j in range(i + 1, m):
                    (win if labs[i] == labs[j] else cross).append(agree[i][j])
            return (sum(win) / len(win)) - (sum(cross) / len(cross))

        obs = stat(labels)
        rng = random.Random(SEED)
        hits = 0
        for _ in range(PERMS):
            sh = labels[:]
            rng.shuffle(sh)
            if stat(sh) >= obs:
                hits += 1
        print("    묶음 %-7s 같은팔 - 다른팔 = %+.3f%%p, p = %.4f"
              % (kind, obs, (hits + 1) / float(PERMS + 1)))

        per_arm = []
        for arm in arms:
            idx = [i for i, a in enumerate(labels) if a == arm]
            if len(idx) < 2:
                continue
            inset = set(idx)
            win = [agree[i][j] for i in idx for j in idx if i < j]
            cross = [agree[i][j] for i in idx for j in range(m)
                     if j not in inset]
            d = sum(win) / len(win) - sum(cross) / len(cross)
            rng2 = random.Random(SEED + 1)
            k = len(idx)
            null = []
            for _ in range(PERMS):
                pick = rng2.sample(range(m), k)
                ps = set(pick)
                w2 = [agree[i][j] for i in pick for j in pick if i < j]
                c2 = [agree[i][j] for i in pick for j in range(m) if j not in ps]
                null.append(sum(w2) / len(w2) - sum(c2) / len(c2))
            hits2 = sum(1 for v in null if v >= d)
            null.sort()
            thresh = null[int(0.95 * (len(null) - 1))]
            per_arm.append((arm, len(idx), d, (hits2 + 1) / float(PERMS + 1), thresh))
        import review_stats as RS
        adj = RS.holm_bonferroni([x[3] for x in per_arm])
        for (arm, k, d, praw, thresh), ph in zip(per_arm, adj):
            mark = "유의" if ph < 0.05 else ""
            print("      %-18s n=%2d  내부우위 %+7.3f%%p  p=%.4f  p_holm=%.4f %s"
                  % (arm, k, d, praw, ph, mark))
        print("      -- 검출 한계: 보정 없이 단독 검정했을 때의 95% 문턱 --")
        for (arm, k, d, praw, thresh) in per_arm:
            room = "검출 가능" if d >= thresh else "문턱 미만"
            print("      %-18s n=%2d  관측 %+7.3f%%p  문턱 %+6.3f%%p  %s"
                  % (arm, k, d, thresh, room))


if __name__ == "__main__":
    sys.path.append(HERE)      # review_stats 는 review_response 쪽에 있다
    report(*main())
