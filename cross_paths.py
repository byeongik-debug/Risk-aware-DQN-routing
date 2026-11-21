# -*- coding: utf-8 -*-
"""2x2(상태 표현 x 보상 함수) 네 칸의 **경로**를 직접 비교한다.

왜 지연 평균만으로는 모자라나
-----------------------------
`cross_runs.csv` 에서 보상을 QoS 로 바꾼 칸은 다섯 모델의 홉이 전부
7.0000 이고 지연이 335~341 ms 로 모여 있다. 반면 Full 칸은 5런 중 4개가
354.2940 으로 같고 하나가 387.0342 다. 평균 차이(약 -23 ms)가 **정책이
바뀐 것**인지, 아니면 수렴 추첨의 산포인지는 평균으로 못 가린다.

경로는 다르다. 에피소드별 노드 열은 부동소수 잡음·torch 스레드 수에
둔감하고, 두 모델이 200 에피소드 전부 같은 경로를 고르면 '같은 정책'이
성립한다. 그래서 칸 안(잡음 바닥)과 칸 사이를 같은 자로 잰다.

분모를 먼저 잰다
----------------
"A 칸과 B 칸이 x% 일치" 는 **칸 안에서 런끼리 몇 % 일치하는지** 를 모르면
해석할 수 없다. 칸내 일치율이 이미 낮으면 칸 사이의 불일치는 인자의
효과가 아니라 학습 시드의 효과다. 그래서 칸내를 먼저 찍는다.

하네스
------
평가 루프는 `check_path_identity.rollout_paths` 를 **그대로** 쓴다(베끼지
않는다). 상태/보상 조합은 `train_state_reward_cross.evaluate_cell` 과 같은
방식으로 `network_env.NetworkEnv` 를 `CrossEnv` 로 갈아끼워 만든다.
자기검증: 같은 루프가 낸 네 지표를 `cross_runs.csv` /
`cross_baselines.csv` 의 기록값과 맞춘다. 어긋나면 경로 비교도 무효다.

쓰는 법
-------
    python cross_paths.py --episodes 200
"""
import argparse
import csv
import io
import itertools
import os

import check_path_identity as C
import network_env
import train_state_reward_cross as TC

RESULT_DIR = C.RESULT_DIR
OUT_ARMS = os.path.join(RESULT_DIR, "cross_paths_arms.csv")
OUT_PAIRS = os.path.join(RESULT_DIR, "cross_paths_pairs.csv")

CELL_ORDER = ["risk_both", "qos_reward", "qos_state", "qos_both"]
CELL_LABEL = {"risk_both": "상태위험/보상위험(Full)",
              "qos_reward": "상태위험/보상QoS",
              "qos_state": "상태QoS/보상위험",
              "qos_both": "상태QoS/보상QoS(NoRisk)"}


def load_targets():
    """두 CSV 에서 (셀, run) -> (모델경로, 기록행). 기록값이 자기검증 기준."""
    tg = {}
    for name in ("cross_runs.csv", "cross_baselines.csv"):
        p = os.path.join(RESULT_DIR, name)
        if not os.path.exists(p):
            continue
        with io.open(p, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                cell = row["cell"]
                mdir = row.get("model_dir") or TC.MODEL_DIR
                tg[(cell, int(float(row["run"])))] = (
                    os.path.join(mdir, row["model"]), row)
    return tg


def rollout_in_cell(cell, model_path, episodes):
    """CrossEnv 로 갈아끼운 상태에서 rollout_paths 를 돌린다."""
    TC.configure(cell)
    orig = network_env.NetworkEnv
    network_env.NetworkEnv = TC.CrossEnv
    try:
        assert C.network_env.NetworkEnv is TC.CrossEnv, "패치를 못 봤다"
        probe = TC.CrossEnv(scenario="dynamic")
        assert probe.use_risk is TC.CELLS[cell][1], "보상 플래그 유실"
        assert probe.state_size == TC.CELLS[cell][2], "차원 유실"
        return C.rollout_paths(model_path, TC.TB.RISK_WINDOW, None, [],
                               episodes, detail=True)
    finally:
        network_env.NetworkEnv = orig


def risky_per_episode(met):
    """에피소드별 위험 엣지 '수' 를 되살린다.

    `rollout_paths` 는 수를 따로 모으지 않지만
    `risky_pct = risky / max(hop,1) * 100` 이므로 hop 과 함께 쓰면 정확히
    복원된다. 정수로 떨어지지 않으면 복원이 틀린 것이므로 단정으로 막는다.
    """
    out = []
    for pct, hop in zip(met["risky_pct"], met["hop"]):
        h = max(int(hop), 1)
        v = pct * h / 100.0
        iv = int(round(v))
        assert abs(v - iv) < 1e-6, "복원 실패 %r" % (v,)
        out.append(iv)
    return out


def agree(pa, pb):
    assert len(pa) == len(pb), "에피소드 수가 다르다"
    return 100.0 * sum(1 for x, y in zip(pa, pb) if x == y) / len(pa)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()

    tg = load_targets()
    keys = [k for k in sorted(tg.keys()) if k[0] in CELL_ORDER]
    if not keys:
        print("대상이 없다 — `eval` 과 `eval_base` 를 먼저 돌릴 것.")
        return
    print("대상 모델 %d개 / 에피소드 %d" % (len(keys), a.episodes))

    paths, risky, arows, bad, miss = {}, {}, [], [], []
    for (cell, run) in keys:
        mp, rec = tg[(cell, run)]
        if not os.path.exists(mp):
            print("  %s r%02d: 모델 없음 %s" % (cell, run, mp))
            continue
        pp, mm, met = rollout_in_cell(cell, mp, a.episodes)
        # 미도달이 있으면 met 리스트가 paths 보다 짧아져 모델 간 수열이
        # 조용히 어긋난다. 가정하지 않고 막는다.
        nmiss = sum(1 for x in pp if x is None)
        if nmiss or len(met["hop"]) != len(pp):
            print("  %s r%02d: 미도달 %d개(met %d/paths %d) — 이 모델은"
                  " 수열 비교에서 뺀다"
                  % (cell, run, nmiss, len(met["hop"]), len(pp)))
            miss.append((cell, run, nmiss))
            continue
        paths[(cell, run)] = pp
        risky[(cell, run)] = risky_per_episode(met)
        nd = len(set(map(tuple, pp)))
        sc = ""
        if a.episodes == 200 and rec.get("delay"):
            d = abs(float(rec["delay"]) - mm["delay"])
            sc = ("일치(%.2g)" % d) if d < 1e-4 else "어긋남(%.3g)" % d
            if d >= 1e-4:
                bad.append(((cell, run), d))
        print("  %-26s r%02d  서로 다른 경로 %3d/%d  지연 %.4f  %s"
              % (CELL_LABEL[cell], run, nd, a.episodes, mm["delay"], sc),
              flush=True)
        arows.append(dict(cell=cell, label=CELL_LABEL[cell], run=run,
                          model=os.path.basename(mp), episodes=a.episodes,
                          n_distinct_paths=nd,
                          delay="%.6f" % mm["delay"],
                          loss="%.6f" % mm["loss"],
                          hop="%.4f" % mm["hop"],
                          risky_pct="%.4f" % mm["risky_pct"],
                          risky_cnt_mean="%.6f"
                          % (sum(risky[(cell, run)])
                             / float(len(risky[(cell, run)]))),
                          risky_seq_id="",
                          selfcheck=sc))

    if miss:
        print("\n!! 미도달이 있는 모델 %d개를 뺐다 — 수열 비교의"
              " 분모가 모델마다 달라지기 때문이다: %s"
              % (len(miss), miss))
    if bad:
        print("\n!! 자기검증 어긋남 %d개 — 경로 비교를 쓰면 안 된다: %s"
              % (len(bad), bad[:3]))

    # ── 쌍별 일치율 ──────────────────────────────────────────────────────
    prows = []
    for (ka, kb) in itertools.combinations(sorted(paths.keys()), 2):
        prows.append(dict(a_cell=ka[0], a_run=ka[1], b_cell=kb[0],
                          b_run=kb[1],
                          same_cell=int(ka[0] == kb[0]),
                          same_run=int(ka[1] == kb[1]),
                          episodes=a.episodes,
                          agree_pct="%.4f" % agree(paths[ka], paths[kb])))

    within = {}
    cross = {}
    for r in prows:
        v = float(r["agree_pct"])
        if r["same_cell"]:
            within.setdefault(r["a_cell"], []).append(v)
        else:
            cross.setdefault(tuple(sorted((r["a_cell"], r["b_cell"]))),
                             []).append(v)

    print("\n" + "-" * 78)
    print("칸 안의 일치율 = 잡음 바닥 (이걸 모르면 칸 사이 차이를 못 읽는다)")
    print("-" * 78)
    for c in CELL_ORDER:
        v = within.get(c)
        if not v:
            continue
        print("  %-26s 쌍 %2d  평균 %6.2f%%  최소 %6.2f%%  최대 %6.2f%%"
              % (CELL_LABEL[c], len(v), sum(v) / len(v), min(v), max(v)))

    print("\n" + "-" * 78)
    print("칸 사이의 일치율")
    print("-" * 78)
    for k in sorted(cross.keys()):
        v = cross[k]
        wa = within.get(k[0], [0.0])
        wb = within.get(k[1], [0.0])
        print("  %-24s x %-24s 쌍 %2d  평균 %6.2f%%  (칸내 %5.1f / %5.1f)"
              % (CELL_LABEL[k[0]], CELL_LABEL[k[1]], len(v),
                 sum(v) / len(v), sum(wa) / len(wa), sum(wb) / len(wb)))

    # ── 위험 엣지 수가 정책에 따라 변하나 ───────────────────────────────
    print("\n" + "-" * 78)
    print("경로 위의 위험 엣지 수 — 평균만 같은가, 에피소드마다 같은가")
    print("-" * 78)
    seqs = {}
    for k in sorted(risky.keys()):
        sig = tuple(risky[k])
        seqs.setdefault(sig, []).append(k)
    for r in arows:
        sig = tuple(risky[(r["cell"], r["run"])])
        r["risky_seq_id"] = "S%d" % (sorted(seqs.keys()).index(sig) + 1)
    print("  모델 %d개 -> 서로 다른 '에피소드별 위험 엣지 수 수열' %d개"
          % (len(risky), len(seqs)))
    for i, sig in enumerate(sorted(seqs.keys())):
        who = seqs[sig]
        cells_in = sorted(set(k[0] for k in who))
        print("   S%d  모델 %2d개  평균 %.6f  칸 %s"
              % (i + 1, len(who), sum(sig) / float(len(sig)),
                 ", ".join(CELL_LABEL[c] for c in cells_in)))
    if len(seqs) == 1:
        print("\n  -> 스무 모델이 **에피소드마다** 같은 수를 쓴다. 경로를"
              " 바꿔도 위험 엣지 수는 안 바뀐다(이 평가 집합에서).")
        print("     그러면 Risky% 차이는 전부 `수/홉` 의 홉 몫이다 —"
              " 위험 회피의 증거로 못 쓴다.")
    else:
        print("\n  -> 수열이 갈린다. 평균이 같아 보였던 것은 상쇄다."
              " 희석이라고 단정하면 안 된다.")

    for path, rows in ((OUT_ARMS, arows), (OUT_PAIRS, prows)):
        if not rows:
            continue
        with io.open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print("\n-> %s (%d행)" % (path, len(rows)))

    print("\n주의: 칸 사이 일치율이 낮아도, 그 칸의 **칸내** 일치율이 낮으면"
          " 원인은 인자가 아니라 학습 시드다. 두 수를 같이 읽을 것.")


if __name__ == "__main__":
    main()
