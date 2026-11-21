#!/usr/bin/env bash
# Risk-Blind 10런(scalar 5 + edgemean 5)을 각 1스레드로 병렬 학습.
# 스레드를 1로 고정하는 이유: 학습 스레드 수가 측정값을 26.13 ms 움직인 적이 있다.
set -u
cd "$(dirname "$0")"
mkdir -p review_results/logs_risk_blind
export KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 TORCH_THREADS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=""
for variant in scalar edgemean; do
  for run in 1 2 3 4 5; do
    log="review_results/logs_risk_blind/${variant}_run0${run}.log"
    python -u -X utf8 train_risk_blind.py "$variant" "$run" > "$log" 2>&1 &
    pids="$pids $!"
    echo "launched $variant run$run -> $log (pid $!)"
  done
done

fail=0
for p in $pids; do
  wait "$p" || fail=$((fail+1))
done
echo "=== 전부 종료. 실패 $fail 개 ==="
ls -la review_results/models_risk_blind/ 2>/dev/null
