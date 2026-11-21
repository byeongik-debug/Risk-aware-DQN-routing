# -*- coding: utf-8 -*-
"""같은 시드로 학습하면 같은 모델이 나오는가 — **측정한다**.

왜 필요한가
-----------
`factor_full_run0N`(내 주입 드라이버가 학습) 과 `window5_run0N`(원본
`train_window_dqn.py` 가 학습) 은 설정이 완전히 같고 시드 규약도 같은데
(`seed = 42 + run_id`) 가중치가 다르다(`max|diff|` 0.82~1.47). 그래서
"Full 대 K=5" 가 재현 바닥 10.1 ms 를 재는 셈이 됐다.

26.6절에서 나는 원인을 **단정하지 않았다**. 내 드라이버는
`train_ablation_temporal.py:41` 에서 `torch.set_num_threads(1)` 을 박아
두었고 원본 학습들은 그러지 않았으므로 스레드 수가 후보지만, 그건 가설이다.
이 스크립트가 그 가설을 직접 친다.

묻는 것 둘
----------
  (ㄱ) 같은 시드 · 같은 스레드 수로 두 번 학습하면 **비트 단위로** 같은가?
       아니라면 스레드와 무관한 비결정성이 있다는 뜻이고, 가설은 거기서
       끝난다(스레드를 맞춰도 재현이 안 되므로).
  (ㄴ) 스레드 수만 1 → 4 로 바꾸면 갈라지는가? 갈라지면 스레드 수가
       비결정성의 **한 경로**임이 확인된다.

원본을 건드리지 않는 방법
-------------------------
`train_factor_ablation.train()` 은 `os.makedirs("models")` 를 **상대
경로**로 부르고 `save_path` 도 상대다. 그래서 작업 디렉터리를 스크래치로
옮기면 원본 트리에 아무것도 쓰지 않는다. 원본 파일은 읽기만 한다.

학습 길이
---------
`EPISODES` 를 8000 → `--episodes`(기본 300) 로 줄인다. 비트 단위 재현성은
처음부터 깨지거나 안 깨지므로 짧은 학습으로도 (ㄱ)·(ㄴ) 의 답이 나온다.
다만 **8000 에피소드 학습의 차이가 전부 이 경로에서 왔다고는 말할 수
없다** — 두 모델군은 학습 시점·torch 버전까지 다를 수 있다. 이 측정은
'스레드 수가 비결정성을 만드는가' 까지만 답한다.

사용법
------
    python measure_seed_determinism.py            # 전체 (하위 프로세스 3개)
    python measure_seed_determinism.py --one --threads 1 --tag a   # 내부용
"""
import argparse
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))
RESULT_DIR = os.path.join(HERE, "review_results")
SCRATCH = os.environ.get(
    "DETERMINISM_DIR",
    os.path.join(HERE, "_determinism_tmp"))
OUT_CSV = os.path.join(RESULT_DIR, "seed_determinism.csv")


def run_one(threads, tag, episodes, run_id):
    """한 번 학습하고 state_dict 를 스크래치에 저장한다 (하위 프로세스용)."""
    import torch
    workdir = os.path.join(SCRATCH, tag)
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    os.chdir(workdir)                      # 원본 트리 보호: 상대 경로 쓰기
    torch.set_num_threads(int(threads))

    sys.path.insert(0, ORIG)
    import train_factor_ablation as T      # noqa: E402

    T.EPISODES = int(episodes)
    T.EVAL_EPISODES = 5                    # 평가는 이 측정에 쓰지 않는다
    print("[%s] threads=%s(실제 %d) episodes=%d run_id=%d cwd=%s"
          % (tag, threads, torch.get_num_threads(), T.EPISODES, run_id,
             os.getcwd()))
    T.train("full", run_id)

    src = os.path.join(workdir, "models",
                       "factor_full_run%02d.pth" % run_id)
    if not os.path.exists(src):
        cand = []
        for root, _d, files in os.walk(workdir):
            for f in files:
                if f.endswith(".pth"):
                    cand.append(os.path.join(root, f))
        if not cand:
            print("!! .pth 를 못 찾았다")
            return 1
        src = cand[0]
    sd = torch.load(src, map_location="cpu", weights_only=True)
    torch.save(sd, os.path.join(SCRATCH, "%s.pth" % tag))
    meta = dict(tag=tag, threads=int(threads),
                torch_threads=torch.get_num_threads(),
                episodes=int(episodes), run_id=run_id,
                torch_version=torch.__version__, model=src)
    with io.open(os.path.join(SCRATCH, "%s.json" % tag), "w",
                 encoding="utf-8") as fh:
        fh.write(json.dumps(meta, ensure_ascii=False))
    print("[%s] 저장 완료" % tag)
    return 0


def compare(tags):
    import torch
    sds, metas = {}, {}
    for t in tags:
        p = os.path.join(SCRATCH, "%s.pth" % t)
        if not os.path.exists(p):
            print("!! %s 가 없다 (학습이 실패했다)" % p)
            return None
        sds[t] = torch.load(p, map_location="cpu", weights_only=True)
        with io.open(os.path.join(SCRATCH, "%s.json" % t),
                     encoding="utf-8") as fh:
            metas[t] = json.loads(fh.read())
    rows = []
    print("")
    print("=" * 92)
    print("쌍별 가중치 비교 (같은 시드 42+run_id, 같은 설정)")
    print("=" * 92)
    print("%-14s %-14s %14s %12s %10s" % ("A", "B", "max|diff|",
                                          "다른 텐서", "판정"))
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            ka, kb = set(sds[a]), set(sds[b])
            if ka != kb:
                print("  키가 다르다: %s vs %s" % (sorted(ka), sorted(kb)))
                continue
            worst, nd, which = 0.0, 0, ""
            for k in sorted(ka):
                d = float((sds[a][k].float()
                           - sds[b][k].float()).abs().max())
                if d > 0:
                    nd += 1
                if d > worst:
                    worst, which = d, k
            same = worst == 0.0
            print("%-14s %-14s %14.6g %6d/%-5d %10s"
                  % (a, b, worst, nd, len(ka),
                     "동일" if same else "다름"))
            rows.append(dict(
                a=a, b=b, threads_a=metas[a]["threads"],
                threads_b=metas[b]["threads"],
                episodes=metas[a]["episodes"],
                max_abs_diff="%.6g" % worst, n_tensors_diff=nd,
                n_tensors=len(ka), worst_tensor=which,
                identical=int(same),
                torch_version=metas[a]["torch_version"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one", action="store_true", help="하위 프로세스 모드")
    ap.add_argument("--threads", default="1")
    ap.add_argument("--tag", default="a")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--run-id", type=int, default=1)
    args = ap.parse_args()

    if args.one:
        return run_one(args.threads, args.tag, args.episodes, args.run_id)

    if not os.path.isdir(SCRATCH):
        os.makedirs(SCRATCH)
    if not os.path.isdir(RESULT_DIR):
        os.makedirs(RESULT_DIR)
    plan = [("t1_a", 1), ("t1_b", 1), ("t4_a", 4)]
    print("=" * 92)
    print("시드 결정성 측정 - 원본은 읽기만, 쓰기는 %s 안에서만" % SCRATCH)
    print("계획: %s (각각 별 프로세스, episodes=%d)"
          % (", ".join("%s@%d스레드" % p for p in plan), args.episodes))
    print("=" * 92)
    env = dict(os.environ)
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    env["PYTHONIOENCODING"] = "utf-8"
    env["DETERMINISM_DIR"] = SCRATCH
    for tag, th in plan:
        env["OMP_NUM_THREADS"] = str(th)
        env["MKL_NUM_THREADS"] = str(th)
        cmd = [sys.executable, "-u", "-X", "utf8", os.path.abspath(__file__),
               "--one", "--threads", str(th), "--tag", tag,
               "--episodes", str(args.episodes),
               "--run-id", str(args.run_id)]
        print("")
        print(">> %s (%d 스레드)" % (tag, th))
        p = subprocess.run(cmd, env=env, cwd=HERE,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
        txt = p.stdout.decode("utf-8", "replace")
        for line in txt.strip().split("\n")[-4:]:
            print("   | %s" % line)
        if p.returncode != 0:
            print("!! %s 가 실패했다 (exit %d)" % (tag, p.returncode))
            return 1

    rows = compare([t for t, _ in plan])
    if not rows:
        return 1
    import csv
    flds = ["a", "b", "threads_a", "threads_b", "episodes",
            "max_abs_diff", "n_tensors_diff", "n_tensors", "worst_tensor",
            "identical", "torch_version"]
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=flds, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print("")
    print("-> %s" % OUT_CSV)

    same_th = [r for r in rows if r["threads_a"] == r["threads_b"]]
    diff_th = [r for r in rows if r["threads_a"] != r["threads_b"]]
    print("")
    print("판정")
    print("-" * 92)
    if same_th and all(r["identical"] for r in same_th):
        print("(ㄱ) 같은 스레드 수에서는 **비트 단위로 재현된다**.")
        if diff_th and not any(r["identical"] for r in diff_th):
            print("(ㄴ) 스레드 수만 바꾸면 **갈라진다** -> 스레드 수가")
            print("     비결정성의 한 경로임이 확인됐다. 그래도 8000")
            print("     에피소드 두 모델군의 차이가 전부 이 경로에서 왔다고는")
            print("     말할 수 없다(학습 시점·torch 버전도 다를 수 있다).")
        elif diff_th:
            print("(ㄴ) 스레드 수를 바꿔도 **같다** -> 스레드 가설은 **기각**.")
            print("     factor_full 과 window5 의 차이는 다른 데서 왔다.")
    elif same_th:
        print("(ㄱ) 같은 시드·같은 스레드인데 **재현되지 않는다**.")
        print("     스레드를 맞춰도 재현이 안 되므로 스레드 가설은 설명력이")
        print("     없다. 이 환경의 학습 자체가 비결정적이다 -> 표 10 의 ±")
        print("     는 '설정 차이' 가 아니라 이 비결정성을 담고 있다.")
    print("")
    print("한계: episodes=%d 짧은 학습이다. 비트 단위 재현성은 처음부터"
          % args.episodes)
    print("깨지거나 안 깨지므로 (ㄱ)·(ㄴ) 의 답에는 충분하지만, 8000")
    print("에피소드 모델군 사이의 **차이 크기**를 설명하지는 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
