#!/bin/bash
# research: evaluate the fixed-data WWM->token masking recipe isolation run.
#
# Scientific purpose
# ------------------
# Evaluate the completed recipe-isolation run: same clean-Qwen data and
# protected 8x480/baseline16k/AdamW recipe as COMPACT_EXPERIENCE research, but masking switches
# from whole-word to token-level after switch_frac=0.7.  The resulting trajectory
# separates masking-recipe effects from data effects before any combined route.

set -uo pipefail

RUN="experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022"
MODEL_ROOT="${RUN}/hf_model"
METRICS="${RUN}/scientific_metrics.json"
OUT_ROOT="experiments/archive/representation_and_objectives/data/wwm_to_token_fullzeroshot_reading"
EVAL_SCRIPT="experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py"

if [ ! -f "${METRICS}" ]; then
  echo "ERROR: training metrics not found yet: ${METRICS}"
  exit 1
fi
if [ ! -d "${MODEL_ROOT}/chck_100M" ]; then
  echo "ERROR: chck_100M not found under ${MODEL_ROOT}"
  exit 1
fi

python - <<'PY'
import json, pathlib, sys
m=json.loads(pathlib.Path('experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/scientific_metrics.json').read_text())
ck=[c.get('name') for c in m.get('saved_checkpoints', [])]
need=[f'chck_{i}M' for i in range(10,101,10)]
print(json.dumps({
  'word_exposure':m.get('word_exposure'),
  'actual_training_steps':m.get('actual_training_steps'),
  'loss_first':m.get('loss_first'),
  'loss_last':m.get('loss_last'),
  'checkpoints_available_10M_spacing':[x for x in need if x in ck],
  'missing_10M_spacing':[x for x in need if x not in ck],
  'masking_curriculum':m.get('masking_curriculum'),
  'switch_frac':m.get('switch_frac'),
}, indent=2))
if m.get('word_exposure') != 100000000:
  sys.exit('word_exposure is not 100M')
if any(x not in ck for x in need):
  sys.exit('missing a 10M-spaced checkpoint')
PY

GPU_ID="${GPU:-0}"
CUDA_VISIBLE_DEVICES=${GPU_ID} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -B "${EVAL_SCRIPT}" \
  --target qwen_wwm_to_token \
  --model_root "${MODEL_ROOT}" \
  --gpu "${GPU_ID}" \
  --out_root "${OUT_ROOT}" \
  --checkpoints chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M
