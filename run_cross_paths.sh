#!/bin/sh
cd "c:/Users/kye/.vscode/thesis_papers/ROUTING TEST/review_response"
KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 TORCH_THREADS=1 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u -X utf8 cross_paths.py --episodes 200 \
  > review_results/logs_cross/cross_paths200.log 2>&1 &
wait
echo "exit=$?"
tail -30 review_results/logs_cross/cross_paths200.log
