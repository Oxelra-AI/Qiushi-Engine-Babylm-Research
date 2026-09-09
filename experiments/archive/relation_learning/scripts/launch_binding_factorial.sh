#!/usr/bin/env bash
set -euo pipefail
OUT=experiments/archive/relation_learning/data/binding_factorial
CACHE=experiments/archive/relation_learning/data/binding_factorial/hf_cache
mkdir -p "$OUT" "$CACHE"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Run the three arms in sequence or concurrently as resources permit. The
# python script configures a writable per-job HF_MODULES_CACHE before importing
# transformers from CACHE_BASE.
run_arm() {
  local arm=$1
  local gpu=$2
  local logbase=$OUT/${arm}_gpu${gpu}
  CACHE_BASE=$CACHE/${arm} CUDA_VISIBLE_DEVICES=$gpu \
    python -B experiments/archive/relation_learning/scripts/binding_factorial_train.py \
      --arm "$arm" \
      --device cuda:0 \
      --epochs 20 \
      --batch-size 16 \
      --eval-batch-size 64 \
      --eval-every 5 \
      --lr 3e-5 \
      --weight-decay 0.01 \
      --seed 73073 \
      --out-root "$OUT" \
      > "${logbase}.stdout.log" 2> "${logbase}.stderr.log"
}
run_arm answer_clean 0 &
pid_answer=$!
run_arm uniform_wwm 1 &
pid_wwm=$!
wait $pid_answer
wait $pid_wwm
run_arm answer_corrupt_update_state 0
python -B experiments/archive/relation_learning/scripts/summarize_binding_factorial.py --out-root "$OUT"
