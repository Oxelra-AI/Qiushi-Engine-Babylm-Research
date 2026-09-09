#!/usr/bin/env bash
set -euo pipefail
# research: Launch chck_82M binding candidate with frozen-slow + fresh-private architecture.
#
# This reuses the research balanced paired binding trainer's proven data/masking/eval
# pipeline, but replaces the model with research's frozen architecture.
#
# Architecture: frozen chck_82M slow path + fresh 995,584-parameter private adapter
# Data: balanced recombination pairs with answer-only credit
# Evaluation: margin-based paired-null gating + cheap7
#
# Usage:
#   GPU=0 bash launch_binding_candidate.sh
#   GPU=0 EPOCHS=40 SCALE=0.75 bash launch_binding_candidate.sh

GPU=${GPU:-0}
EPOCHS=${EPOCHS:-30}
SCALE=${SCALE:-0.75}
LR=${LR:-5e-4}
PAIR_BS=${PAIR_BS:-8}
EVAL_EVERY=${EVAL_EVERY:-5}
OUT=experiments/archive/relation_learning/data/chck82_binding_candidate
CACHE=$OUT/hf_cache
MIN_FREE_MB=${MIN_FREE_MB:-25000}

mkdir -p "$OUT" "$CACHE"

wait_gpu_free() {
  if ! command -v nvidia-smi &>/dev/null; then return; fi
  while true; do
    local free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')
    if [[ -n "$free" ]] && (( free >= MIN_FREE_MB )); then break; fi
    echo "[gpu_wait] gpu=$GPU free_mb=$free min_free_mb=$MIN_FREE_MB" >&2
    sleep 10
  done
}

echo "[research] Starting chck_82M binding candidate: scale=$SCALE epochs=$EPOCHS lr=$LR gpu=$GPU"

wait_gpu_free

# Export cache variables BEFORE python import
export TRANSFORMERS_CACHE="$CACHE"
export HF_HOME="$CACHE"
export CACHE_BASE="$CACHE"

CUDA_VISIBLE_DEVICES="$GPU" python -B experiments/archive/relation_learning/scripts/chck82_binding_candidate.py \
  --private-adapter-scale "$SCALE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --pair-batch-size "$PAIR_BS" \
  --eval-every "$EVAL_EVERY" \
  --gpu 0

echo "[research] Binding candidate complete. Evaluating cheap7..."

# Run cheap7 evaluation on the produced checkpoint
CKPT="$OUT/scale_${SCALE}/checkpoint"
if [ -d "$CKPT" ]; then
  wait_gpu_free
  python -B experiments/archive/frontier_consolidation/scripts/frozen82_tail_eval_one.py \
    --checkpoint "$CKPT" \
    --parent experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M \
    --target binding_candidate_cheap7 \
    --gpu "$GPU" \
    --out-dir "$OUT/scale_${SCALE}/cheap7" \
    --columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading \
    --force 2>&1 || echo "[research] cheap7 eval returned nonzero; check logs"
fi

echo "[research] All done."
