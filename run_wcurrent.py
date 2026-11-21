# -*- coding: utf-8 -*-
"""표 10 의 W-Current 행을 **실제로 학습해서** 차용이 정당한지 재다.

왜 필요한가
-----------
표 10 가중치 비교군의 원천은 `table10_extra_runs.csv` 이고 거기에는
`W-Equal`(5런)·`W-Temporal`(5런) 만 있다. **W-Current 행에는 측정된 팔이
없다** (`models/weight_current_*.pth` 도 0개). 그 행의 숫자는 기본 가중치로
학습한 다른 팔(Full)에서 가져온 것이다.

차용이 정당한지는 "W-Current 조건"과 "Full 조건"이 같은지에 달렸다. 읽기만
해서 확인한 것 (원본은 수정하지 않았다):

  * `train_weight_compare.WEIGHT_CONFIGS["current"]`
    == `risk_calculator.DEFAULT_WEIGHTS`  (다섯 값이 같은 리터럴)
    그리고 `risk_calculator.py:89-96` 은 `w["util"]` 처럼 **키로 직접** 꺼내
    더하므로 dict 순서는 무관하다 → 위험값이 비트까지 같아야 한다.
  * `FACTOR_MAP["full"] == []` == weight_compare 가 넣는 `ZEROED_FACTORS = []`
  * 하이퍼파라미터 전부 동일: EPISODES 8000, GAMMA 0.99, LR 3e-4,
    BATCH 64, MEMORY 40000, EPS 1.0->0.05 decay 0.9994, TARGET_UPDATE 50,
    RISK_WINDOW 5, 시나리오 0.25x4, EVAL_EPISODES 200
  * `QNetwork` 정의가 두 스크립트에서 바이트까지 동일
  * `DQNAgent` 차이는 변수명과 RNG 를 쓰지 않는 `current_max_q` 기록 한 줄뿐
  * 시나리오 추첨식이 달라 보이지만
    `random.choices(SCENARIOS, weights=[0.25]*4)[0]` 과
    `random.choices(list(PROBS.keys()), weights=list(PROBS.values()), k=1)[0]`
    은 같은 시드에서 8000회 수열이 동일하고 RNG 소비량도 같다(실측).
  * 두 스크립트가 `seed = 42 + run_id` 를 쓴다 → 같은 run_id = 같은 시드

여기까지는 **구조 비교**다. 구조가 같다고 결과가 같다고 단정하지 않기 위해
실제로 돌려서 `factor_full_run{NN}.pth` 와 맞춰 본다.

판정 도구 둘
------------
(ㄱ) 비트 비교: 같으면 결정적이다. **다르면 애매하다** — Full 모델을 학습할
     때의 torch 스레드 수를 모르고, 스레드 수는 부동소수 축약 순서를 바꾼다
     (이전 측정에서 같은 시드가 스레드 수 때문에 갈린 적이 있다).
(ㄴ) 200에피소드 경로 동일성: 스레드 수·부동소수 잡음에 둔감하다. 경로가
     전부 같으면 **정책이 같다** 가 성립하므로 이쪽이 판정이다.

쓰는 법
-------
    python run_wcurrent.py 1        # run01 학습 (8000ep, 약 35분)

원본 `train_weight_compare.train()` 을 **그대로** 호출한다(몽키패치 없음).
`train()` 안의 저장 경로 `models/...` 와 `RESULT_DIR="results"` 는 상대경로
이므로, cwd 를 `review_results/wcurrent/` 로 옮겨 원본 트리에 아무것도
쓰지 않게 한다. `network_env`·`risk_calculator`·`topology` 에 상대경로 파일
적재가 없음을 grep 으로 확인했기 때문에 cwd 를 옮겨도 동작이 바뀌지 않는다.
"""
import os
import sys

SDN = (r"c:/Users/kye/.vscode/thesis_papers/ROUTING TEST/SDN_DQN_new_env")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "review_results", "wcurrent")


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_wcurrent.py <run_id>")
        return 1
    run_id = int(sys.argv[1])

    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, SDN)
    os.chdir(OUT)

    import torch
    import risk_calculator
    import train_weight_compare as T

    # ── 전제 검사: 두 조건이 정말 같은 가중치인가 ─────────────────────────
    cur = T.WEIGHT_CONFIGS["current"]
    dflt = risk_calculator.DEFAULT_WEIGHTS
    print("W-Current 가중치 : %s" % cur)
    print("기본    가중치 : %s" % dflt)
    same = (sorted(cur.items()) == sorted(dflt.items()))
    print("두 가중치 동일? %s" % same)
    if not same:
        print("!! 같지 않다 — 차용 논증의 전제가 깨졌다. 중단한다.")
        return 2
    print("torch 스레드 %d  (기록용: 스레드 수가 부동소수 축약 순서를 바꾼다)"
          % torch.get_num_threads())
    print("cwd = %s  (원본 트리에는 쓰지 않는다)" % os.getcwd())
    print("-" * 72)

    T.train("current", run_id)
    print("\n저장 위치: %s" % os.path.join(
        OUT, "models", "weight_current_run%02d.pth" % run_id))
    return 0


if __name__ == "__main__":
    sys.exit(main())
