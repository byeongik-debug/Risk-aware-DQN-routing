# -*- coding: utf-8 -*-
"""독립 학습한 모델들이 **같은 경로**를 고르는지 직접 확인한다.

왜 필요한가
-----------
`table10_factor_runs.csv` 에서 Full 의 모델 5개 중 4개가 지연·손실·홉·
Risky% 네 지표 전부 소수 4자리까지 같았다(354.2940 / 0.084779 / 7.7100 /
21.8929). 200 에피소드 평균이 네 지표 다 같으면 경로 선택이 같다고 **추론**
할 수는 있지만, 그건 추론이다. 서로 다른 경로 집합이 우연히 네 평균을 다
맞출 확률이 낮다는 것뿐이고, 측정한 게 아니다.

그래서 경로를 직접 비교한다. `env.visited` 가 그 에피소드의 노드 열이므로
에피소드별 경로를 전부 기록해 모델 쌍마다 몇 %가 같은지 센다.

같은 값이 **학습 스크립트를 가로질러** 나오는 것도 확인한다(K=10·K=5·
-Cong 도 354.2940 을 낸다). 다른 스크립트가 만든 모델이 같은 경로를 고른다면
"이 환경에서 기본 가중치로 학습하면 몇 개 해 중 하나로 수렴한다" 는 뜻이고,
표 10 의 ± 는 성능의 산포가 아니라 **그 추첨의 산포**다.

평가 코드를 베끼는 문제
-----------------------
`E.evaluate` / `X.evaluate_arm` 은 경로를 버리므로 롤아웃 루프를 다시 써야
한다. 베끼면 하네스가 갈라질 위험이 있으니 **자기 검증**을 넣었다: 같은
루프에서 네 지표를 다시 계산해 이미 기록된 모델별 CSV 값과 맞춰 본다.
어긋나면 내 루프가 원래 평가와 달라진 것이므로 그 자리에서 드러나고,
경로 비교 결과도 믿을 수 없다고 판정한다.
"""
import argparse
import csv
import io
import os
import random
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

import eval_ablation_risky as E              # noqa: E402
import eval_table10_extra as X               # noqa: E402

T = E.T
risk_calculator = E.risk_calculator
network_env = E.network_env
RESULT_DIR = E.RESULT_DIR
EVAL_SEED = E.EVAL_SEED
ORIG = E.ORIG

OUT_PAIRS = os.path.join(RESULT_DIR, "path_identity_pairs.csv")
OUT_ARMS = os.path.join(RESULT_DIR, "path_identity_arms.csv")


# (라벨, 모델경로, risk_window, 가중치키, 절제목록, 모델별CSV의 arm/run)
def build_targets(runs):
    tg = []
    for r in range(1, runs + 1):
        tg.append(("Full-run%02d" % r,
                   os.path.join(HERE, "models_ablation",
                                "factor_full_run%02d.pth" % r),
                   5, None, [], ("Full", r)))
    # 같은 값(354.2940)을 낸 다른 스크립트/다른 설정의 모델들
    tg.append(("-Cong-run03",
               os.path.join(ORIG, "models", "factor_cong_run03.pth"),
               5, None, ["congestion"], ("-Cong", 3)))
    tg.append(("K5-run01",
               os.path.join(ORIG, "models", "window5_run01.pth"),
               5, None, [], ("K=5", 1)))
    tg.append(("K10-run01",
               os.path.join(ORIG, "models", "window10_run01.pth"),
               10, None, [], ("K=10", 1)))
    # 대조: 최빈 결과를 한 번도 내지 않은 팔, 그리고 값이 달랐던 Full 모델
    tg.append(("-Util-run01*",
               os.path.join(ORIG, "models", "factor_util_run01.pth"),
               5, None, ["utilization"], ("-Util", 1)))
    tg.append(("Full-run05*",
               os.path.join(HERE, "models_ablation",
                            "factor_full_run05.pth"),
               5, None, [], ("Full", 5)))
    return tg


def rollout_paths(model_path, risk_window, wkey, zeroed, episodes,
                  detail=False):
    """E.evaluate 와 같은 절차. 다른 점은 경로를 기록하는 것뿐."""
    risk_calculator.ZEROED_FACTORS = list(zeroed)
    risk_calculator.CUSTOM_WEIGHTS = (None if wkey is None
                                      else X.WEIGHTS[wkey])
    base = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                  risk_window=risk_window)
    agent = T.DQNAgent(base.state_size, base.action_size)
    agent.policy_net.load_state_dict(
        torch.load(model_path, map_location=agent.device,
                   weights_only=True))
    agent.policy_net.eval()
    agent.epsilon = 0.0

    paths, met = [], {"delay": [], "loss": [], "hop": [], "risky_pct": []}
    for i in range(episodes):
        random.seed(EVAL_SEED + i)
        np.random.seed(EVAL_SEED + i)
        torch.manual_seed(EVAL_SEED + i)
        env = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
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
        if env.current_node != network_env.DESTINATION:
            paths.append(None)          # 미도달도 자리를 지켜 짝을 맞춘다
            continue
        paths.append(tuple(env.visited))
        m = env.calculate_path_metrics(env.visited)
        hop = max(int(m["hop_count"]), 1)
        risky = max(int(m["unstable_count"]), int(m["congested_count"]))
        met["delay"].append(float(m["total_delay"]))
        met["loss"].append(float(m["total_loss"]))
        met["hop"].append(float(m["hop_count"]))
        met["risky_pct"].append(risky / hop * 100.0)
    risk_calculator.ZEROED_FACTORS = []
    risk_calculator.CUSTOM_WEIGHTS = None
    means = dict((k, float(np.mean(v)) if v else float("nan"))
                 for k, v in met.items())
    if detail:
        # 에피소드별 리스트까지 돌려준다. 위험 엣지 '수' 는 따로 모으지
        # 않았지만 risky_pct * hop / 100 으로 정확히 되살아난다.
        return paths, means, met
    return paths, means


def load_meta():
    """모델별 CSV 두 개에서 (arm, run) -> (경로, risk_window, wkey, 절제목록).

    절제·디렉터리·윈도우·가중치를 **CSV 에 적힌 대로** 되살린다. 손으로
    다시 쓰면 그 자리에서 조건이 어긋난다.
    """
    meta = {}
    pf = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
    if os.path.exists(pf):
        with io.open(pf, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                mdir = os.path.basename(row["model_dir"].rstrip("/\\"))
                root = HERE if mdir == "models_ablation" else ORIG
                z = [x for x in row["zeroed"].split(",") if x]
                meta[(row["arm"], int(float(row["run"])))] = (
                    os.path.join(root, mdir, row["model"]), 5, None, z)
    pe = os.path.join(RESULT_DIR, "table10_extra_runs.csv")
    if os.path.exists(pe):
        with io.open(pe, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                wk = row["weights"]
                wkey = None if wk in ("", "default") else wk
                meta[(row["arm"], int(float(row["run"])))] = (
                    os.path.join(ORIG, "models", row["model"]),
                    int(float(row["risk_window"])), wkey, [])
    return meta


def build_targets_membership(outcome_id, contrasts=True):
    """outcome_collapse 의 한 결과값에 속한 모델 **전부**를 대상으로 만든다.

    26.7절의 헤드라인("23개가 같은 정책")은 그중 7개만 경로로 재고 나머지
    16개는 네 지표가 같다는 **추론**이었다. 그 추론을 없앤다.
    """
    mp = os.path.join(RESULT_DIR, "outcome_membership.csv")
    if not os.path.exists(mp):
        print("!! %s 가 없다. analyze_outcome_collapse.py 를 먼저 돌려라."
              % mp)
        return []
    meta = load_meta()
    tg, missing = [], []
    with io.open(mp, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    want = [r for r in rows if int(float(r["outcome_id"])) == outcome_id]
    for r in want:
        key = (r["arm"], int(float(r["run"])))
        m = meta.get(key)
        if m is None:
            missing.append(key)
            continue
        path, rw, wkey, z = m
        tg.append(("%s-r%02d" % (key[0], key[1]), path, rw, wkey, z, key))
    if contrasts:
        # 대조: 같은 결과값에 들어가지 않은 모델 둘(멀리·가까이)
        others = [r for r in rows if int(float(r["outcome_id"])) != outcome_id]
        others.sort(key=lambda r: abs(float(r["delay"])
                                      - float(want[0]["delay"]))
                    if want else 0.0)
        for r in (others[:1] + others[-1:]):
            key = (r["arm"], int(float(r["run"])))
            m = meta.get(key)
            if m is None:
                continue
            path, rw, wkey, z = m
            tg.append(("*%s-r%02d" % (key[0], key[1]), path, rw, wkey, z,
                       key))
    if missing:
        print("!! 메타데이터를 못 찾은 모델: %s" % missing)
    return tg


def build_targets_all():
    """표 10 모수 **전부**(두 CSV 의 모든 arm/run)를 대상으로 만든다.

    왜: 최빈 모드(--membership)는 **최빈 결과값에 속한 모델만** 골라 경로를
    쟀다. 그 집합이 같은 경로를 쓴다는 결과는 선택 기준과 순환이 있다(네
    지표가 같은 모델을 골랐으니 경로도 같을 가능성이 높다). 그리고 69개
    모수 중 25개만 쟀으므로 나머지 44개의 '정책 동일' 은 평균 일치에서
    온 **추론**이다. 전수로 재면 둘 다 없어진다.

    `-Type`·`W-Equal`·`W-Temporal` 은 최빈 집합에 아예 없었으므로 이 모드가
    그 팔들의 **첫 경로 측정**이다.
    """
    meta = load_meta()
    tg = []
    for (arm, run) in sorted(meta.keys()):
        path, rw, wkey, z = meta[(arm, run)]
        tg.append(("%s-r%02d" % (arm, run), path, rw, wkey, z, (arm, run)))
    return tg


def load_recorded():
    """모델별 CSV 두 개에서 (arm, run) -> 지표."""
    out = {}
    for name in ("table10_factor_runs.csv", "table10_extra_runs.csv"):
        p = os.path.join(RESULT_DIR, name)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                out[(row["arm"], int(float(row["run"])))] = row
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=T.EVAL_EPISODES)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--membership", type=int, default=0,
                    help="outcome_membership.csv 의 결과값 id (예: 1)")
    ap.add_argument("--all", action="store_true",
                    help="표 10 모수 전부(두 CSV 의 모든 arm/run)를 잰다")
    args = ap.parse_args()

    global OUT_ARMS, OUT_PAIRS
    if args.all:
        OUT_ARMS = os.path.join(RESULT_DIR, "path_identity_all_arms.csv")
        OUT_PAIRS = os.path.join(RESULT_DIR, "path_identity_all_pairs.csv")
    elif args.membership:
        OUT_ARMS = os.path.join(RESULT_DIR,
                                "path_identity_modal_arms.csv")
        OUT_PAIRS = os.path.join(RESULT_DIR,
                                 "path_identity_modal_pairs.csv")
    rec = load_recorded()
    print("=" * 96)
    print("경로 동일성 직접 확인 - 네 지표가 같으면 경로도 같은가")
    print("EVAL_SEED=%d, 에피소드=%d" % (EVAL_SEED, args.episodes))
    print("=" * 96)

    got, arm_rows = [], []
    bad_check = 0
    if args.all:
        targets = build_targets_all()
    elif args.membership:
        targets = build_targets_membership(args.membership)
    else:
        targets = build_targets(args.runs)
    if not targets:
        return 1
    if args.all:
        tag = u"표 10 모수 전수"
    elif args.membership:
        tag = u"결과값 id=%d 구성원 + 대조" % args.membership
    else:
        tag = u"고정 목록"
    print("대상 모델 %d개 (%s)" % (len(targets), tag))
    for label, path, rw, wkey, zeroed, key in targets:
        if not os.path.exists(path):
            print("[skip] %-12s 모델 없음: %s" % (label, path))
            continue
        paths, means = rollout_paths(path, rw, wkey, zeroed, args.episodes)
        got.append((label, paths, key))
        uniq = len(set(p for p in paths if p is not None))
        miss = sum(1 for p in paths if p is None)
        # 자기 검증: 기록된 모델별 값과 맞는가
        r = rec.get(key)
        verdict, worst = "기록없음", float("nan")
        if r is not None and args.episodes == T.EVAL_EPISODES:
            worst = max(abs(means[k] - float(r[k]))
                        for k in ("delay", "loss", "hop", "risky_pct"))
            verdict = "일치" if worst < 1e-4 else "어긋남"
            if worst >= 1e-4:
                bad_check += 1
        elif r is not None:
            verdict = "비교안함(에피소드수 다름)"
        print("  %-12s 서로 다른 경로 %3d/%3d (미도달 %d)  지연 %9.4f  "
              "자기검증 %s(%.2g)"
              % (label, uniq, len(paths), miss, means["delay"], verdict,
                 worst))
        arm_rows.append(dict(label=label, model=os.path.basename(path),
                             risk_window=rw, zeroed=",".join(zeroed),
                             episodes=args.episodes, distinct_paths=uniq,
                             unreached=miss, selfcheck=verdict,
                             selfcheck_maxdiff="%.6g" % worst,
                             **dict((k, "%.6f" % v)
                                    for k, v in means.items())))

    if bad_check:
        print("")
        print("!! %d 개 모델에서 자기 검증이 어긋났다. 내 롤아웃 루프가 원래"
              % bad_check)
        print("   평가와 다르다는 뜻이므로 아래 경로 비교를 믿으면 안 된다.")
    elif got:
        print("")
        print("-> 자기 검증 통과. 이 루프는 기록된 평가와 같은 값을 낸다.")

    print("")
    print("[쌍별 경로 일치율] 에피소드마다 노드 열이 완전히 같은 비율")
    print("-" * 96)
    wide = len(got) <= 11
    if not wide:
        print("대상이 %d개라 행렬은 생략하고 요약만 찍는다 (CSV 에 전부"
              " 있다)." % len(got))
    hdr = "%-13s" % ""
    for label, _p, _k in got:
        hdr += "%12s" % label[:12]
    if wide:
        print(hdr)
    pair_rows = []
    for la, pa, _ka in got:
        line = "%-13s" % la[:13]
        for lb, pb, _kb in got:
            n = min(len(pa), len(pb))
            same = sum(1 for i in range(n)
                       if pa[i] is not None and pa[i] == pb[i])
            frac = 100.0 * same / n if n else float("nan")
            line += "%11.1f%%" % frac
            if la < lb:
                pair_rows.append(dict(a=la, b=lb, episodes=n, same=same,
                                      same_pct="%.4f" % frac))
        if wide:
            print(line)
    if not wide:
        star = [r for r in pair_rows
                if r["a"].startswith("*") or r["b"].startswith("*")]
        core = [r for r in pair_rows if r not in star]
        full = [r for r in core if float(r["same_pct"]) >= 99.9999]
        print("  구성원 쌍 %d개 중 100%% 일치 **%d개**"
              % (len(core), len(full)))
        if core:
            lo = min(core, key=lambda r: float(r["same_pct"]))
            print("  구성원 쌍 최저 일치율: %s%% (%s 대 %s)"
                  % (lo["same_pct"], lo["a"], lo["b"]))
        if star:
            print("  대조(*) 쌍 %d개: 일치율 %s"
                  % (len(star),
                     ", ".join(sorted(set(r["same_pct"] for r in star)))))

    for path, rows, flds in (
            (OUT_ARMS, arm_rows,
             ["label", "model", "risk_window", "zeroed", "episodes",
              "distinct_paths", "unreached", "delay", "loss", "hop",
              "risky_pct", "selfcheck", "selfcheck_maxdiff"]),
            (OUT_PAIRS, pair_rows, ["a", "b", "episodes", "same",
                                    "same_pct"])):
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print("-> %s (%d 행)" % (path, len(rows)))

    print("")
    print("읽는 법: Full-run01~04 끼리 100% 면 '독립 학습한 모델 네 개가")
    print("같은 경로를 고른다' 가 측정된 것이다. 다른 스크립트 산물(K5·K10·")
    print("-Cong)까지 100% 면 수렴 지점이 학습 설정을 가로질러 같다는 뜻이고,")
    print("Full-run05* 는 값이 달랐던 모델이라 낮아야 한다(대조).")
    return 1 if bad_check else 0


if __name__ == "__main__":
    sys.exit(main())
