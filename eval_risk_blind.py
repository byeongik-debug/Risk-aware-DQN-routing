# -*- coding: utf-8 -*-
"""Risk-Blind 팔을 **표 10 Full 팔과 같은 하네스**로 채점한다.

왜 이 파일인가
--------------
비교 기준은 `review_response/models_ablation/factor_full_run01..05.pth`
(= `train_factor_ablation.py` 의 `full`, 제안 팔)이고 그 값은 이미
`review_results/table10_factor_runs.csv` 에 **고정 시드 규약**
(`EVAL_SEED=20261114 + 에피소드 색인`)으로 커밋돼 있다. Risk-Blind 를
같은 규약으로 재야 쌍체 비교가 선다.

평가 코드를 **베끼지 않는다** — `eval_ablation_risky.evaluate` 를 그대로
부르고, 그 모듈이 보는 `network_env.NetworkEnv` 심볼만 상수 래퍼로
바꿔치기한다. 전에 `RISK_WINDOW` 몽키패치가 조용히 죽은 적이 있으므로
**패치가 먹었는지 단정으로 확인**한 뒤에만 채점한다.

Risk-Blind 는 위험 인자를 절제하지 않는다(`zeroed=[]`) — 위험도 계산과
보상은 제안 팔과 **완전히 같고**, 상태의 위험 50칸만 상수다.

사용법:
    python eval_risk_blind.py                 # models_risk_blind/*.pth 전부
    python eval_risk_blind.py --episodes 200
"""
import argparse
import csv
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import eval_ablation_risky as E                        # noqa: E402
import network_env                                     # noqa: E402
import train_risk_blind as TB                          # noqa: E402

RS = E.RS
RESULT_DIR = E.RESULT_DIR
FIELDS = E.FIELDS
RUNS_CSV = os.path.join(RESULT_DIR, "risk_blind_runs.csv")
CMP_CSV = os.path.join(RESULT_DIR, "risk_blind_vs_full.csv")
FULL_RUNS = os.path.join(RESULT_DIR, "table10_factor_runs.csv")


def blind_env_class(const):
    """NetworkEnv 를 상속해 상태 앞 50칸만 상수로 덮는 클래스를 만든다."""
    class _Blind(network_env.NetworkEnv):
        CONST = const

        def get_state(self):
            s = super().get_state()
            s[:TB.N_EDGES] = _Blind.CONST
            return s
    return _Blind


def assert_patch_live(cls, const):
    """패치가 실제로 먹는지 — 상태 앞 50칸이 상수와 같아야 하고,
    원본 클래스는 그래야 **다른** 값을 내야 한다."""
    env = cls(use_risk=True, scenario="dynamic", risk_window=E.T.RISK_WINDOW)
    env.reset()
    for _ in range(10):
        env.update_dynamic_qos()
    got = env.get_state()[:TB.N_EDGES]
    assert np.allclose(got, const, atol=1e-6), "상수 블록이 안 먹었다"
    raw = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                 risk_window=E.T.RISK_WINDOW)
    raw.reset()
    for _ in range(10):
        raw.update_dynamic_qos()
    rawblk = raw.get_state()[:TB.N_EDGES]
    assert not np.allclose(rawblk, const, atol=1e-6), \
        "원본도 상수를 낸다 — 통제가 통제가 아니다"
    return float(np.abs(rawblk - const).mean())


def eval_blind(model_path, const, episodes):
    """E.evaluate 를 그대로 쓰되 그 모듈이 보는 환경 클래스만 바꿔치기."""
    cls = blind_env_class(const)
    gap = assert_patch_live(cls, const)
    orig = network_env.NetworkEnv
    network_env.NetworkEnv = cls
    try:
        assert E.network_env.NetworkEnv is cls, "E 모듈이 패치를 못 봤다"
        res = E.evaluate(model_path, [], episodes)
    finally:
        network_env.NetworkEnv = orig
    assert network_env.NetworkEnv is orig, "패치 복원 실패"
    res["_raw_vs_const_absmean"] = gap
    return res


def load_full():
    """커밋된 Full 팔 모델별 값 (같은 EVAL_SEED 규약)."""
    rows = []
    with io.open(FULL_RUNS, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "Full":
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()

    const_base = TB.load_const()
    mdir = TB.MODEL_DIR
    models = sorted(f for f in os.listdir(mdir) if f.endswith(".pth")) \
        if os.path.isdir(mdir) else []
    models = [m for m in models if "_ep" not in m]          # 탐침본 제외
    if not models:
        print("채점할 Risk-Blind 모델이 없다: %s" % mdir)
        return

    E.probe_hook([])        # full 과 같은 조건임을 확인 (위험도 안 움직여야)
    print("EVAL_SEED=%d  에피소드 %d  모델 %d개"
          % (E.EVAL_SEED, args.episodes, len(models)))

    rows = []
    for name in models:
        variant = name.split("_run")[0]
        run = int(name.split("_run")[1].split(".")[0])
        const = TB.const_for(variant, const_base)
        res = eval_blind(os.path.join(mdir, name), const, args.episodes)
        row = {"arm": "Blind-%s" % variant, "variant": variant, "run": run,
               "model": name,
               "const_sd": "%.6f" % float(np.std(const)),
               "raw_vs_const": "%.6f" % res.pop("_raw_vs_const_absmean")}
        for k in FIELDS:
            row[k] = "%.6f" % res[k]
        row["success_rate"] = "%.6f" % res["success_rate"]
        rows.append(row)
        print("  %-18s delay %.4f  loss %.6f  hop %.4f  도달 %.1f%%"
              % (name, res["delay"], res["loss"], res["hop"],
                 res["success_rate"]), flush=True)

    with io.open(RUNS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("저장: %s (%d행)" % (RUNS_CSV, len(rows)))

    # ── Full 팔과 비교 ────────────────────────────────────────────────────
    full = load_full()
    if not full:
        print("Full 팔 행을 못 찾았다 — 비교 생략")
        return
    out = []
    for metric in ("delay", "loss", "hop", "risk", "loss_total",
                   "risky_pct", "success_rate"):
        fy = [float(r[metric]) for r in full]
        for variant in sorted(set(r["variant"] for r in rows)):
            bx = [float(r[metric]) for r in rows if r["variant"] == variant]
            if len(bx) < 2 or len(fy) < 2:
                d = (np.mean(bx) - np.mean(fy)) if bx else float("nan")
                out.append({"metric": metric, "variant": variant,
                            "n_blind": len(bx), "n_full": len(fy),
                            "blind_mean": "%.6f" % np.mean(bx),
                            "blind_sd": "", "full_mean": "%.6f" % np.mean(fy),
                            "full_sd": "", "diff": "%.6f" % d,
                            "pct": "", "t": "", "df": "", "p": "",
                            "distinct_blind": len(set(bx))})
                continue
            bm, bs = RS.mean_std(bx)
            fm, fs = RS.mean_std(fy)
            diff, t, df, p = RS.welch(bx, fy)
            out.append({
                "metric": metric, "variant": variant,
                "n_blind": len(bx), "n_full": len(fy),
                "blind_mean": "%.6f" % bm, "blind_sd": "%.6f" % bs,
                "full_mean": "%.6f" % fm, "full_sd": "%.6f" % fs,
                "diff": "%.6f" % diff,
                "pct": "%.4f" % (100.0 * diff / fm) if fm else "",
                "t": "%.4f" % t, "df": "%.3f" % df, "p": "%.6g" % p,
                "distinct_blind": len(set(bx)),
            })
    ps = [float(r["p"]) for r in out if r["p"]]
    if ps:
        holm = RS.holm_bonferroni(ps)
        it = iter(holm)
        for r in out:
            r["p_holm"] = "%.6g" % next(it) if r["p"] else ""
    with io.open(CMP_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print("저장: %s (%d행, Holm 족 %d)" % (CMP_CSV, len(out), len(ps)))
    print("\n%-12s %-14s %12s %12s %10s %9s %9s %4s"
          % ("지표", "팔", "Blind", "Full", "차", "p", "p_holm", "값종"))
    for r in out:
        print("%-12s %-14s %12s %12s %10s %9s %9s %4s"
              % (r["metric"], r["variant"], r["blind_mean"], r["full_mean"],
                 r["diff"], r["p"], r.get("p_holm", ""), r["distinct_blind"]))


if __name__ == "__main__":
    main()
