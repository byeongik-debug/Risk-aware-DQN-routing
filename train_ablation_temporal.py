# -*- coding: utf-8 -*-
"""표 10의 빠진 행을 학습한다 — 원본 디렉터리를 건드리지 않는 주입 드라이버.

SDN_DQN_new_env/train_factor_ablation.py 를 모듈로 불러 FACTOR_MAP에 조합을
주입한다. 기존 절제 모델 5회(factor_util/cong/delay_inst/loss_inst/link_type)와
같은 환경·같은 risk_calculator·같은 시드 규칙(42+run_id)을 쓰므로 한 표에 올릴 수 있다.

주의: review_response/risk_calculator.py 에는 ZEROED_FACTORS 훅이 없다.
그쪽이 임포트되면 절제가 조용히 무효화되므로, 자기 디렉터리를 sys.path에서
제거하고 원본 디렉터리를 맨 앞에 넣은 뒤 __file__ 로 검증한다.

사용법:
    python train_ablation_temporal.py <variant> <run_id>
    variant: temporal        (w/o VD, VL  — 두 변동성 항 동시 제거, 재정규화 없음)
             temporal_renorm (w/o VD, VL  — 원고 서술대로 나머지 가중치 재정규화)
             full            (기준 행 — 아무것도 절제하지 않는다. 원본 results/ 에
                              per-run CSV도 모델도 없어서 짝 검정·홉·Risky% 비교
                              대상이 없었다. 절제 행과 같은 코드·같은 시드 규약으로
                              다시 뽑는다. 절제가 걸리지 않았음을 아래 검증 4가 확인)

    run_id 는 1..5. 시드는 원본 규약 그대로 42+run_id 이며, 런마다 따로 호출한다.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.path.abspath(os.path.join(HERE, "..", "SDN_DQN_new_env"))

# 자기 디렉터리를 치우고 원본을 맨 앞에 — 복사본이 원본을 가리지 않게
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
sys.path.insert(0, ORIG)

import risk_calculator                      # noqa: E402
import network_env                          # noqa: E402
import train_factor_ablation as T           # noqa: E402
import torch                               # noqa: E402

# 1스레드가 이 크기의 망에서는 더 빠르고(0.63 vs 0.88 s/ep), 5회를 병렬로
# 돌릴 수 있게 해 준다. 측정값이 아니라 학습이므로 스레드 수는 결과를
# 바꾸지 않는다(경로 설정 시간 측정에서만 보고 대상이다).
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))

# 검증 1: 원본에서 임포트됐는가
for mod in (risk_calculator, network_env, T):
    got = os.path.abspath(mod.__file__)
    assert got.startswith(ORIG), "원본이 아닌 사본이 임포트됐다: %s" % got

# 검증 2: 절제 훅이 존재하는가
assert hasattr(risk_calculator, "ZEROED_FACTORS"), "ZEROED_FACTORS 훅이 없다"
assert hasattr(risk_calculator, "CUSTOM_WEIGHTS"), "CUSTOM_WEIGHTS 훅이 없다"

TEMPORAL = ["delay_inst", "loss_inst"]

# 재정규화판: 두 변동성 항을 빼고 남은 셋의 가중치를 합 1로 비례 조정
_W = risk_calculator.DEFAULT_WEIGHTS
_KEEP = ("util", "cong", "type")
_S = sum(_W[k] for k in _KEEP)
RENORM = {k: (_W[k] / _S if k in _KEEP else 0.0) for k in _W}

T.FACTOR_MAP["temporal"] = TEMPORAL
T.LABEL_MAP["temporal"] = "-VD,VL"
T.FACTOR_MAP["temporal_renorm"] = TEMPORAL
T.LABEL_MAP["temporal_renorm"] = "-VD,VL(renorm)"

T.RESULT_DIR = os.path.join(HERE, "review_results")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    variant, run_id = sys.argv[1], int(sys.argv[2])
    if variant not in ("temporal", "temporal_renorm", "full"):
        print("variant는 temporal / temporal_renorm / full 중 하나")
        return 1

    # full은 표 10의 기준 행이다. 원본 results/ 에 factor_full_* 가 per-run CSV도
    # 모델도 없어서(find 0건) 짝 검정·홉·Risky% 비교 대상이 없다. 절제 행과 같은
    # 코드·같은 시드 규약으로 다시 뽑는다. 절제가 아니므로 아무것도 0으로 만들지
    # 않는 것이 맞고, 그것을 아래에서 역으로 확인한다.

    if variant == "temporal_renorm":
        risk_calculator.CUSTOM_WEIGHTS = dict(RENORM)
        print("CUSTOM_WEIGHTS = %s  (합 %.4f)"
              % ({k: round(v, 4) for k, v in RENORM.items()},
                 sum(RENORM.values())))

    # 검증 3: 절제가 실제로 위험도를 바꾸는가 — 학습 전에 확인
    probe = dict(utilization=0.7, is_congested=True,
                 delay_history=[10.0, 40.0, 25.0, 60.0, 15.0],
                 loss_history=[0.01, 0.06, 0.02, 0.09, 0.03],
                 link_type="unstable")
    risk_calculator.ZEROED_FACTORS = []
    before = risk_calculator.calculate_link_risk(**probe)
    risk_calculator.ZEROED_FACTORS = TEMPORAL
    after = risk_calculator.calculate_link_risk(**probe)
    print("절제 검증: 변동성 큰 링크 위험도 %.4f -> %.4f (차 %+.4f)"
          % (before, after, after - before))
    assert abs(after - before) > 1e-9, "절제가 위험도를 바꾸지 않았다 — 훅이 안 걸렸다"
    risk_calculator.ZEROED_FACTORS = []   # train()이 다시 설정한다

    # 검증 4: full은 절제가 아니어야 한다 — 위 탐침이 훅이 산다는 걸 보였으니,
    # 이번엔 반대로 아무것도 안 걸리는지 본다. 걸려 있으면 기준 행이 오염된다.
    if variant == "full":
        assert T.FACTOR_MAP[variant] == [], T.FACTOR_MAP[variant]
        assert risk_calculator.CUSTOM_WEIGHTS is None
        risk_calculator.ZEROED_FACTORS = T.FACTOR_MAP[variant]
        base = risk_calculator.calculate_link_risk(**probe)
        print("기준 검증: 절제 없음 위험도 %.4f (탐침 원본 %.4f)" % (base, before))
        assert abs(base - before) < 1e-12, "full인데 위험도가 움직였다"

    os.makedirs(os.path.join(HERE, "models_ablation"), exist_ok=True)
    os.chdir(HERE)
    T.train(variant, run_id)

    # 모델을 원본 models/ 가 아니라 review_response 쪽으로 옮겨 둔다
    name = "factor_%s_run%02d.pth" % (variant, run_id)
    src = os.path.join(HERE, "models", name)
    dst = os.path.join(HERE, "models_ablation", name)
    if os.path.exists(src):
        os.replace(src, dst)
        print("모델 이동: %s" % dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
