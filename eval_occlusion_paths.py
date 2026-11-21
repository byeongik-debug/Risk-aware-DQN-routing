# -*- coding: utf-8 -*-
"""가림(occlusion) 측정을 체크포인트 5런으로 올리고 **경로 동일성을 직접** 센다.

왜 다시 재나
--------------
주6(표 11 초안)에 넣은 가림 표는 Full **run01 하나**에서 나왔고, "정책이 고정
경로 하나로 붕괴한다" 는 근거가 **홉 평균이 정확히 정수(8.000 / 7.000)** 라는
간접 증거뿐이었다. 정수 평균은 단일 경로의 필요조건이지 충분조건이 아니다
(길이가 같은 서로 다른 경로를 섞어도 정수가 된다). 그리고 run01 의 우연일
수도 있다. 그래서
  (1) Full 5런 전부에 대해 재고,
  (2) 에피소드마다 **실제 경로**를 받아 서로 다른 경로가 몇 개인지 센다.

경로 수는 지연과 달리 스레드 수·부동소수에 흔들리지 않으므로, n=5 의 해상도
한계(Welch MDE 약 30 ms)에 걸리지 않는 증거다.

측정법
------
평가 코드는 베끼지 않는다. `network_env.NetworkEnv` 를 기록용 서브클래스로
몽키패치하고 `eval_ablation_risky.evaluate` 를 **그대로** 부른다.
  - `get_state()` : CONST 가 있으면 앞 50칸을 덮는다 (없으면 통제군)
  - `reset()`     : 에피소드 슬롯을 하나 연다
  - `calculate_path_metrics()` : 그 슬롯에 경로를 적는다 (한 에피소드에
    두 번 불려도 덮어쓰므로 슬롯당 하나)
둘 다 super() 로 위임만 하므로 평가 결과를 바꾸지 않는다 — 통제군이 커밋값
354.294012 를 재현하는지로 확인한다.

사용법:
    python eval_occlusion_paths.py            # 5런 x (통제/scalar/edgemean)
    python eval_occlusion_paths.py --episodes 20
"""
import argparse
import csv
import io
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 순서 중요: eval_ablation_risky 가 sys.path 를 **원본** 쪽으로 돌려놓고
# 임포트된 모듈이 원본인지 단정으로 검사한다. 먼저 넣지 않으면
# review_response 사본이 캐시되어 그 단정이 터진다.
import eval_ablation_risky as E                      # noqa: E402
import network_env                                   # noqa: E402
import train_risk_blind as TB                        # noqa: E402
from topology import EDGES                           # noqa: E402

N_EDGES = len(EDGES)
OUT = TB.OUT
MODEL_DIR = os.path.join(HERE, "models_ablation")
RUNS_CSV = os.path.join(OUT, "occlusion_paths_runs.csv")
PATHS_CSV = os.path.join(OUT, "occlusion_paths_detail.csv")
COMMITTED = os.path.join(OUT, "table10_factor_runs.csv")


def recorder_class(const):
    """경로를 기록하는 서브클래스. const=None 이면 가리지 않는다."""
    class Rec(network_env.NetworkEnv):
        CONST = const
        PATHS = []

        def reset(self):
            type(self).PATHS.append(None)
            return super().reset()

        def get_state(self):
            s = super().get_state()
            if type(self).CONST is not None:
                s[:N_EDGES] = type(self).CONST
            return s

        def calculate_path_metrics(self, path):
            if type(self).PATHS:
                type(self).PATHS[-1] = tuple(path)
            return super().calculate_path_metrics(path)
    return Rec


def assert_hook(cls, const):
    """패치가 정말 먹었는지 / 원본은 그러지 않는지 단정으로 확인한다."""
    env = cls(use_risk=True, scenario="dynamic", risk_window=TB.RISK_WINDOW)
    env.reset()
    for _ in range(10):
        env.update_dynamic_qos()
    s = env.get_state()
    assert len(s) == N_EDGES + 75, len(s)
    raw = network_env.NetworkEnv(use_risk=True, scenario="dynamic",
                                 risk_window=TB.RISK_WINDOW)
    raw.reset()
    for _ in range(10):
        raw.update_dynamic_qos()
    r = raw.get_state()
    if const is None:
        return 0.0
    assert np.allclose(s[:N_EDGES], const), "가림이 안 먹었다"
    assert not np.allclose(r[:N_EDGES], const), \
        "원본도 상수를 낸다 — 통제가 통제가 아니다"
    return float(np.mean(np.abs(r[:N_EDGES] - const)))


def run_one(model_path, const, episodes):

    cls = recorder_class(const)
    gap = assert_hook(cls, const)
    cls.PATHS = []
    orig = network_env.NetworkEnv
    network_env.NetworkEnv = cls
    try:
        assert E.network_env.NetworkEnv is cls, "E 모듈이 패치를 못 봤다"
        res = E.evaluate(model_path, [], episodes)
    finally:
        network_env.NetworkEnv = orig
    assert network_env.NetworkEnv is orig, "패치 복원 실패"
    paths = [p for p in cls.PATHS if p]
    return res, paths, gap


def committed_full():
    """표 10 에 커밋된 Full 5런의 지연 — 통제군이 이걸 재현해야 한다."""
    out = {}
    if not os.path.exists(COMMITTED):
        return out
    with io.open(COMMITTED, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("key") == "full" and not row.get("zeroed"):
                out[int(row["run"])] = float(row["delay"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

    vec = TB.load_const()
    consts = [("none", None),
              ("scalar", TB.const_for("scalar", vec)),
              ("edgemean", TB.const_for("edgemean", vec))]
    ref = committed_full()
    print("평가 에피소드 %d  스레드 %d  커밋 Full %d런"
          % (a.episodes, torch.get_num_threads(), len(ref)))

    rows, detail = [], []
    for run in range(1, 6):
        mp = os.path.join(MODEL_DIR, "factor_full_run%02d.pth" % run)
        if not os.path.exists(mp):
            print("없음: %s" % mp)
            continue
        for name, const in consts:
            res, paths, gap = run_one(mp, const, a.episodes)
            cnt = Counter(paths)
            top, topn = (cnt.most_common(1)[0] if cnt else ((), 0))
            row = {"run": run, "blind": name,
                   "const_sd": "%.6f" % (0.0 if const is None
                                         else float(np.std(const))),
                   "raw_gap": "%.6f" % gap,
                   "episodes_reached": len(paths),
                   "n_distinct_paths": len(cnt),
                   "top_path_share": "%.4f" % (topn / len(paths)
                                               if paths else float("nan")),
                   "hop_sd": "%.6f" % (float(np.std([len(p) - 1
                                                     for p in paths]))
                                       if paths else float("nan"))}
            for k in ("delay", "loss", "hop", "risk", "risky_pct"):
                row[k] = "%.6f" % res[k]
            row["success_rate"] = "%.4f" % res["success_rate"]
            if name == "none" and run in ref:
                row["committed_delay"] = "%.6f" % ref[run]
                row["dev_ms"] = "%+.6f" % (res["delay"] - ref[run])
            else:
                row["committed_delay"] = ""
                row["dev_ms"] = ""
            rows.append(row)
            print("  run%02d %-9s delay %9.4f  홉 %.4f(sd %.3f)  "
                  "경로 %2d개  최빈 %5.1f%%  도달 %5.1f%%%s"
                  % (run, name, res["delay"], res["hop"],
                     float(np.std([len(p) - 1 for p in paths])) if paths
                     else float("nan"),
                     len(cnt), 100.0 * topn / len(paths) if paths else 0.0,
                     res["success_rate"],
                     ("  편차 %+.6f" % (res["delay"] - ref[run]))
                     if (name == "none" and run in ref) else ""), flush=True)
            for p, c in cnt.most_common():
                detail.append({"run": run, "blind": name,
                               "episodes": c, "hops": len(p) - 1,
                               "path": "-".join(str(x) for x in p)})

    if not rows:
        print("측정한 것이 없다")
        return
    with io.open(RUNS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with io.open(PATHS_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "blind", "episodes",
                                           "hops", "path"])
        w.writeheader()
        w.writerows(detail)
    print("저장: %s (%d행), %s (%d행)"
          % (RUNS_CSV, len(rows), PATHS_CSV, len(detail)))

    # ── 요약: 가림 대가와 경로 붕괴 ──────────────────────────────────
    import review_stats as RS
    by = {}
    for r in rows:
        by.setdefault(r["blind"], []).append(r)
    print("\n팔별 평균 (n=%d런)" % len(set(r["run"] for r in rows)))
    for name, _ in consts:
        g = by.get(name, [])
        if not g:
            continue
        d = np.array([float(x["delay"]) for x in g])
        np_ = np.array([int(x["n_distinct_paths"]) for x in g])
        print("  %-9s 지연 %.4f ± %.4f   서로 다른 경로 %s  (평균 %.1f개)"
              % (name, d.mean(), d.std(ddof=1) if len(d) > 1 else 0.0,
                 [int(v) for v in np_], np_.mean()))
    if "scalar" in by and "edgemean" in by:
        s = np.array([float(x["delay"]) for x in by["scalar"]])
        e = np.array([float(x["delay"]) for x in by["edgemean"]])
        _d, t, _df, p = RS.welch(list(s), list(e))
        print("\nscalar 대 edgemean (정적 서열의 값): 차 %+.4f ms  "
              "Welch t=%.4f p=%.4f  [쌍체 아님 — run_id 짝짓기 금지]"
              % (s.mean() - e.mean(), t, p))
        if "none" in by:
            n = np.array([float(x["delay"]) for x in by["none"]])
            for nm, arr in (("scalar", s), ("edgemean", e)):
                _d2, t2, _df2, p2 = RS.welch(list(arr), list(n))
                print("  %-9s 대 통제: %+.4f ms (%+.2f%%)  p=%.4f"
                      % (nm, arr.mean() - n.mean(),
                         100.0 * (arr.mean() - n.mean()) / n.mean(), p2))


if __name__ == "__main__":
    main()
