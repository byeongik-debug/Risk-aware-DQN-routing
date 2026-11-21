# -*- coding: utf-8 -*-
"""`-Util` 을 **1스레드로** 재학습해 족 A 의 스레드 혼입을 직접 제거한다.

왜 필요한가
-----------
26.9.7 절: 족 A(절제)의 기준 `Full` 은 `models_ablation` = 내 스크립트 =
1스레드 고정이고, 절제 팔 다섯(`-Cong`·`-Type`·`-Util`·`-dDelay`·`-dLoss`)은
`SDN_DQN_new_env/models` = 원본 = 비고정(이 기계 12)이다. 그래서 논문이
유일하게 의지하는 절제 효과(`-Util` +44.34 ms = 재현바닥 4.39배)가
**스레드 수와 혼입**되어 있다.

26.9.7 에서 그 혼입이 결과를 설명하지 못한다고 적었지만 근거는 **간접**이다
(비고정인 `-Cong` 이 기준에서 0.05 ms 거리, 두 체제 모델이 같은 최빈값에
들어간다). 이 스크립트는 직접 잰다: `-Util` 을 `Full` 과 **같은 1스레드**로
다시 학습해 +44 ms 가 남는지 본다.

  남으면   -> 혼입을 없앤 상태에서도 효과가 있다 (주장이 단단해진다)
  사라지면 -> 표 10 의 유일한 생존 효과가 스레드 산물이다 (주장 철회)

원본을 건드리지 않는 방법
-------------------------
`measure_seed_determinism.run_one` 에서 검증된 방식을 그대로 쓴다.
`train_factor_ablation.train()` 은 `models/...` 와 `results/...` 를 **상대
경로**로 쓰므로 작업 디렉터리를 스크래치로 옮기면 원본 트리에 아무것도
쓰지 않는다. 산출 `.pth` 는 `review_response/models_thread1/` 로 복사한다.

평가 하네스
-----------
`train()` 내부 평가를 **쓰지 않는다**. 그걸 쓰면 σref 에서 겪은 '다른
하네스를 한 열에 섞는' 오류를 반복한다(327.87 대 360.8). 표 10 의
`table10_factor_runs.csv` 를 만든 `eval_ablation_risky.evaluate` 를 그대로
불러 **같은 하네스**로 잰다.

사용법
------
    python retrain_util_thread1.py --runs 5 --parallel   # 학습 + 평가
    python retrain_util_thread1.py --one --run-id 1      # 하위 프로세스
    python retrain_util_thread1.py --eval-only           # 학습된 것만 평가
    python retrain_util_thread1.py --one --run-id 9 --episodes 50  # 시간 재기
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RESULT_DIR = os.path.join(HERE, "review_results")
MODEL_OUT = os.path.join(HERE, "models_thread1")
SCRATCH = os.environ.get(
    "RETRAIN_DIR", os.path.join(HERE, "_retrain_tmp"))
OUT_CSV = os.path.join(RESULT_DIR, "util_thread1_runs.csv")
FACTOR = "util"


def train_one(run_id, episodes):
    """1스레드로 한 번 학습하고 .pth 를 MODEL_OUT 으로 복사한다."""
    import torch
    workdir = os.path.join(SCRATCH, "run%02d" % run_id)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)
    torch.set_num_threads(1)

    sys.path.insert(0, ORIG)
    import train_factor_ablation as T      # noqa: E402

    if episodes:
        T.EPISODES = int(episodes)
        T.EVAL_EPISODES = 5                # 내부 평가는 쓰지 않는다
    t0 = time.time()
    print("[run%02d] threads=%d episodes=%d cwd=%s"
          % (run_id, torch.get_num_threads(), T.EPISODES, os.getcwd()))
    T.train(FACTOR, run_id)
    dt = time.time() - t0

    src = os.path.join(workdir, "models",
                       "factor_%s_run%02d.pth" % (FACTOR, run_id))
    if not os.path.exists(src):
        print("!! %s 가 없다" % src)
        return 1
    if not os.path.isdir(MODEL_OUT):
        os.makedirs(MODEL_OUT)
    shutil.copy2(src, os.path.join(MODEL_OUT, os.path.basename(src)))
    print("[run%02d] 학습 %.1f초 (%.4f초/에피소드) -> %s"
          % (run_id, dt, dt / max(1, T.EPISODES), MODEL_OUT))
    return 0


def evaluate_all(episodes):
    """표 10 과 **같은** 하네스로 잰다."""
    import csv
    sys.path.insert(0, HERE)
    import eval_ablation_risky as E        # noqa: E402
    T = E.T
    zeroed = T.FACTOR_MAP[FACTOR]
    ep = episodes or T.EVAL_EPISODES

    if not os.path.isdir(MODEL_OUT):
        print("!! %s 가 없다" % MODEL_OUT)
        return 1
    paths = sorted(os.path.join(MODEL_OUT, f)
                   for f in os.listdir(MODEL_OUT)
                   if f.startswith("factor_%s_run" % FACTOR)
                   and f.endswith(".pth"))
    if not paths:
        print("!! %s 에 모델이 없다" % MODEL_OUT)
        return 1
    b, g = E.probe_hook(zeroed)
    print("")
    print("=" * 92)
    print("평가 - eval_ablation_risky.evaluate 그대로, EVAL_SEED=%d, "
          "에피소드/모델=%d" % (E.EVAL_SEED, ep))
    print("zeroed=%s, 탐침 %.6f -> %.6f" % (",".join(zeroed), b, g))
    print("=" * 92)
    rows, delays = [], []
    for i, p in enumerate(paths, 1):
        r = E.evaluate(p, zeroed, ep)
        delays.append(r["delay"])
        rows.append(dict(arm="-Util(1thread)", key=FACTOR, run=i,
                         model=os.path.basename(p), threads=1,
                         model_dir=MODEL_OUT, zeroed=",".join(zeroed),
                         episodes=ep, delay="%.6f" % r["delay"],
                         loss="%.8f" % r["loss"], hop="%.6f" % r["hop"],
                         risky_pct="%.6f" % r["risky_pct"]))
        print("  run%02d  지연 %10.4f  손실 %.6f  홉 %.3f  Risky%% %.4f"
              % (i, r["delay"], r["loss"], r["hop"], r["risky_pct"]))
    flds = ["arm", "key", "run", "model", "threads", "model_dir", "zeroed",
            "episodes", "delay", "loss", "hop", "risky_pct"]
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("")
    print("-> %s (%d 행)" % (OUT_CSV, len(rows)))
    compare(delays)
    return 0


def _mean(v):
    return sum(v) / len(v)


def _sd(v):
    if len(v) < 2:
        return float("nan")
    m = _mean(v)
    return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5


def compare(delays):
    """기록된 Full(고정1) · -Util(비고정12) 과 견준다."""
    import csv
    rec = {}
    p = os.path.join(RESULT_DIR, "table10_factor_runs.csv")
    with io.open(p, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            rec.setdefault(row["arm"], []).append(float(row["delay"]))
    pe = os.path.join(RESULT_DIR, "table10_extra_runs.csv")
    if os.path.exists(pe):
        with io.open(pe, encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                rec.setdefault(row["arm"], []).append(float(row["delay"]))
    full = rec["Full"]
    util12 = rec["-Util"]
    new = list(delays)
    floor = abs(_mean(full) - _mean(rec.get("K=5", full)))

    print("")
    print("=" * 92)
    print("판정 - 스레드를 맞춘 절제 비교")
    print("=" * 92)
    print("%-20s %-10s %3s %10s %9s" % ("", "스레드", "n", "평균지연", "sd"))
    print("-" * 60)
    for lab, th, v in (("Full (기준)", "고정1", full),
                       ("-Util (기록)", "비고정12", util12),
                       ("-Util (새로)", "고정1", new)):
        print("%-20s %-10s %3d %10.4f %9.4f"
              % (lab, th, len(v), _mean(v), _sd(v)))
    d_old = _mean(util12) - _mean(full)
    d_new = _mean(new) - _mean(full)
    print("")
    print("기록 혼입 비교   -Util(12) - Full(1) = %+8.4f ms (바닥 %+.2f배)"
          % (d_old, d_old / floor if floor else float("nan")))
    print("스레드 맞춘 비교 -Util(1)  - Full(1) = %+8.4f ms (바닥 %+.2f배)"
          % (d_new, d_new / floor if floor else float("nan")))
    print("차이의 차이                         = %+8.4f ms" % (d_new - d_old))
    try:
        sys.path.insert(0, HERE)
        import review_stats as RS2          # noqa: E402
        diff, t, df, pv = RS2.welch(new, full)
        mde = RS2.min_detectable_effect(new, full)
        print("Welch 생 p (새 -Util 대 Full) = %.4f "
              "(차 %+.4f, t=%.3f, df=%.2f)" % (pv, diff, t, df))
        print("이 표본에서 검출 가능한 최소 효과 = %.4f ms "
              "(생 p 기준, Holm 전)" % mde)
    except Exception as e:                  # noqa: BLE001
        print("(Welch 못 구했다: %s)" % e)
    print("")
    print("읽는 법: `차이의 차이` 가 0 근처면 스레드 혼입이 효과를 만들지")
    print("않았다는 **직접** 증거다. d_new 가 크게 줄면 표 10 의 유일한")
    print("생존 효과가 스레드 산물이라는 뜻이고 그 주장은 철회해야 한다.")
    print("주의: n=5 로는 Holm 유의를 못 만든다(26.8.1). 여기서 보는 것은")
    print("유의성이 아니라 **효과 크기가 남는지**다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="하위 프로세스 모드")
    ap.add_argument("--run-id", type=int, default=1)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=0,
                    help="0 이면 원본 기본값(8000). 시간 재기용으로만 줄인다")
    ap.add_argument("--eval-episodes", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--parallel", action="store_true",
                    help="런을 동시에 띄운다 (원래 학습 방식과 같다)")
    args = ap.parse_args()

    if args.one:
        return train_one(args.run_id, args.episodes)
    if args.eval_only:
        return evaluate_all(args.eval_episodes)

    for d in (SCRATCH, RESULT_DIR, MODEL_OUT):
        if not os.path.isdir(d):
            os.makedirs(d)
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TORCH_THREADS"] = "1"
    env["RETRAIN_DIR"] = SCRATCH
    print("=" * 92)
    print("-Util 1스레드 재학습 - 원본은 읽기만, 쓰기는 %s 안" % SCRATCH)
    print("런 %d개, parallel=%s" % (args.runs, args.parallel))
    print("=" * 92)
    procs = []
    for r in range(1, args.runs + 1):
        cmd = [sys.executable, "-u", "-X", "utf8",
               os.path.abspath(__file__), "--one", "--run-id", str(r),
               "--episodes", str(args.episodes)]
        if args.parallel:
            procs.append((r, subprocess.Popen(
                cmd, env=env, cwd=HERE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)))
        else:
            print(">> run%02d" % r)
            q = subprocess.run(cmd, env=env, cwd=HERE,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
            for ln in q.stdout.decode("utf-8", "replace").strip(
                    ).split("\n")[-3:]:
                print("   | %s" % ln)
            if q.returncode != 0:
                print("!! run%02d 실패 (exit %d)" % (r, q.returncode))
                return 1
    for r, q in procs:
        out, _ = q.communicate()
        print(">> run%02d (exit %d)" % (r, q.returncode))
        for ln in out.decode("utf-8", "replace").strip().split("\n")[-3:]:
            print("   | %s" % ln)
        if q.returncode != 0:
            print("!! run%02d 실패" % r)
            return 1
    return evaluate_all(args.eval_episodes)


if __name__ == "__main__":
    sys.exit(main())
