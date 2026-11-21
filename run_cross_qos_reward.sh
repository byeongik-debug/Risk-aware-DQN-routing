#!/usr/bin/env bash
# 2x2 의 A 칸(qos_reward: 상태=위험 125d, 보상=QoS 기반) 5런을 1스레드로 병렬 학습.
#
# 왜 스크립트로 띄우나: 앞서 포그라운드 Bash 호출 안에서 `nohup ... &` 로 띄웠다가
# 그 호출이 반환될 때 프로세스 그룹이 함께 종료되어 5런 전부 ep 900 에서 끊겼다
# (트레이스백 없음 = 크래시가 아니라 종료). 이 스크립트를 background 태스크로
# 돌려 `wait` 가 끝까지 붙잡게 한다.
#
# 스레드 1 고정: 학습 스레드 수가 측정값을 26.13 ms 움직인 적이 있다.
set -u
cd "$(dirname "$0")"
mkdir -p review_results/logs_cross
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=""
for run in 1 2 3 4 5; do
  log="review_results/logs_cross/qos_reward_run0${run}.log"
  python -u -X utf8 train_state_reward_cross.py qos_reward "$run" > "$log" 2>&1 &
  pids="$pids $!"
  echo "launched qos_reward run$run -> $log (pid $!)"
done

fail=0
for p in $pids; do
  wait "$p" || fail=$((fail+1))
done
echo "=== 전부 종료. 실패 $fail 개 ==="
ls -la review_results/models_cross/ 2>/dev/null
