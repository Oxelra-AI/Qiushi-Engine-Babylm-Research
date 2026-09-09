#!/bin/bash
# research: minimal downstream conversion for the fixed-data WWM->token schedule.
#
# Scientific purpose
# ------------------
# This evaluates only the post-switch checkpoints of the already-trained clean-Qwen
# WWM->token run.  research CPU measurement showed that this run is training-trace
# identical to the clean-Qwen fixed-WWM reference through the 70M-word WWM prefix,
# and first differs at research / 70,062,953 words when masking changes to token
# level.  Therefore the lowest-cost downstream conversion is chck_80M, chck_90M,
# and chck_100M, compared to the existing COMPACT_EXPERIENCE fixed-WWM trajectory summary.
#
# Use only when a physical GPU is genuinely isolated and using it will not delay
# the repaired FineWeb seqsafe96 task, its selected full-eval converter, or frontier_consolidation.
# This script does not wait for a GPU: it verifies the requested GPU is free enough
# and exits if not, to avoid racing with managed work.
#
# Optional environment variables:
#   GPU=0|1        physical GPU index to use (default 0)
#   MEM_FREE_MB=65000
#   UTIL_MAX=25
#   FORCE=1       pass --force to the evaluator
#   DRY_RUN=1     perform all cheap checks and print the command without eval

set -uo pipefail

SESSION="experiments/archive/representation_and_objectives"
WORKSPACE="${SESSION}/workspace"
RUN="${WORKSPACE}/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022"
MODEL_ROOT="${RUN}/hf_model"
METRICS="${RUN}/scientific_metrics.json"
MEASUREMENT="${WORKSPACE}/data/wwm_to_token_training_measurement/wwm_to_token_training_measurement.json"
OUT_ROOT="${WORKSPACE}/data/wwm_to_token_postswitch_fullzeroshot_reading"
EVAL_SCRIPT="experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py"
SUMMARY_SCRIPT="${WORKSPACE}/training/scripts/summarize_wwm_to_token_recipe_delta.py"
TARGET="qwen_wwm_to_token_postswitch"
CHECKPOINTS=(chck_80M chck_90M chck_100M)
GPU_ID="${GPU:-0}"
MEM_FREE_MB="${MEM_FREE_MB:-65000}"
UTIL_MAX="${UTIL_MAX:-25}"
LOG="${OUT_ROOT}/postswitch_eval.log"
FORCE_ARG=()
if [ "${FORCE:-0}" = "1" ]; then
  FORCE_ARG=(--force)
fi

mkdir -p "${OUT_ROOT}"
{
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) research WWM-to-token post-switch minimal downstream eval"
  echo "run=${RUN}"
  echo "measurement=${MEASUREMENT}"
  echo "target=${TARGET} checkpoints=${CHECKPOINTS[*]} gpu=${GPU_ID} mem_free_mb=${MEM_FREE_MB} util_max=${UTIL_MAX} force=${FORCE:-0} dry_run=${DRY_RUN:-0}"
} >> "${LOG}"

if [ ! -f "${METRICS}" ]; then
  echo "ERROR: missing completed training metrics: ${METRICS}" | tee -a "${LOG}" >&2
  exit 2
fi
if [ ! -f "${MEASUREMENT}" ]; then
  echo "ERROR: missing research CPU measurement: ${MEASUREMENT}" | tee -a "${LOG}" >&2
  exit 3
fi
for ck in "${CHECKPOINTS[@]}"; do
  if [ ! -d "${MODEL_ROOT}/${ck}" ]; then
    echo "ERROR: missing checkpoint ${MODEL_ROOT}/${ck}" | tee -a "${LOG}" >&2
    exit 4
  fi
done

python -B - <<'PY'
import json, pathlib, sys
m = json.loads(pathlib.Path('experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/scientific_metrics.json').read_text())
meas = json.loads(pathlib.Path('experiments/archive/representation_and_objectives/data/wwm_to_token_training_measurement/wwm_to_token_training_measurement.json').read_text())
need = ['chck_80M','chck_90M','chck_100M']
ck = [x.get('name') for x in m.get('saved_checkpoints', [])]
assert m.get('word_exposure') == 100000000, m.get('word_exposure')
assert m.get('masking_curriculum') == 'wwm_to_token', m.get('masking_curriculum')
assert meas.get('same_prefix_steps_excluding_elapsed') == 1761, meas.get('same_prefix_steps_excluding_elapsed')
assert meas.get('first_token_training_row', {}).get('step') == 1762, meas.get('first_token_training_row')
missing = [x for x in need if x not in ck]
if missing:
    raise SystemExit(f'missing checkpoints {missing}')
print(json.dumps({
  'event':'preflight_ok',
  'word_exposure':m.get('word_exposure'),
  'same_prefix_steps_excluding_elapsed':meas.get('same_prefix_steps_excluding_elapsed'),
  'first_token_step':meas.get('first_token_training_row', {}).get('step'),
  'checkpoints':need,
}, indent=2))
PY
PRE_RC=$?
if [ "${PRE_RC}" -ne 0 ]; then
  echo "ERROR: preflight failed rc=${PRE_RC}" | tee -a "${LOG}" >&2
  exit "${PRE_RC}"
fi

GPU_LINE=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits | awk -F',' -v target="${GPU_ID}" '
  {idx=$1; used=$2; free=$3; util=$4; gsub(/^[ \t]+|[ \t]+$/, "", idx); gsub(/^[ \t]+|[ \t]+$/, "", used); gsub(/^[ \t]+|[ \t]+$/, "", free); gsub(/^[ \t]+|[ \t]+$/, "", util); if (idx == target) {print idx","used","free","util; exit}}
')
if [ -z "${GPU_LINE}" ]; then
  echo "ERROR: requested GPU ${GPU_ID} not reported by nvidia-smi" | tee -a "${LOG}" >&2
  exit 5
fi
IFS=',' read -r _GPU _USED _FREE _UTIL <<< "${GPU_LINE}"
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) gpu_status index=${_GPU} used=${_USED} free=${_FREE} util=${_UTIL}" | tee -a "${LOG}"
if [ "${_FREE}" -lt "${MEM_FREE_MB}" ] || [ "${_UTIL}" -gt "${UTIL_MAX}" ]; then
  echo "ERROR: GPU ${GPU_ID} is not isolated enough for this optional eval (free=${_FREE}, util=${_UTIL}). Not launching." | tee -a "${LOG}" >&2
  exit 6
fi

CMD=(python -B "${EVAL_SCRIPT}" --target "${TARGET}" --model_root "${MODEL_ROOT}" --gpu "${GPU_ID}" --out_root "${OUT_ROOT}" --checkpoints "${CHECKPOINTS[@]}" "${FORCE_ARG[@]}")
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) eval_cmd=${CMD[*]}" | tee -a "${LOG}"
if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: not running GPU evaluation" | tee -a "${LOG}"
  exit 0
fi

TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${CMD[@]}" > "${OUT_ROOT}/postswitch_eval.stdout.log" 2> "${OUT_ROOT}/postswitch_eval.stderr.log"
RC=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) post-switch no-AoA eval rc=${RC}" | tee -a "${LOG}"
if [ "${RC}" -ne 0 ]; then
  exit "${RC}"
fi

python -B "${SUMMARY_SCRIPT}" --eval-root "${OUT_ROOT}" --target "${TARGET}" > "${OUT_ROOT}/postswitch_summary.stdout.log" 2> "${OUT_ROOT}/postswitch_summary.stderr.log"
RC=$?
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) post-switch summary rc=${RC}" | tee -a "${LOG}"
exit "${RC}"
