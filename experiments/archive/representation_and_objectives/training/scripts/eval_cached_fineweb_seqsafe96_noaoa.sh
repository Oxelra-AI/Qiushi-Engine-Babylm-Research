#!/bin/bash
# research: no-AoA official-compatible trajectory evaluation for the cached FineWeb seqsafe96 contrast.
# Run only after launch_cached_fineweb_seqsafe96_contrast.sh has produced both endpoints.
# Optional env: GPU_TREAT=0 GPU_CTRL=1 OUT_ROOT=... bash ...

set -uo pipefail

EVAL_SCRIPT="experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py"
SUMMARY_SCRIPT="experiments/archive/representation_and_objectives/training/scripts/summarize_fineweb_seqsafe96_noaoa_delta.py"
OUT_ROOT="${OUT_ROOT:-experiments/archive/representation_and_objectives/data/cached_fineweb_seqsafe96_noaoa_eval}"
RUNS_DIR="experiments/archive/representation_and_objectives/training/runs"
TREAT_RUN="${RUNS_DIR}/cleanqwen_fineweb_seqsafe96_8x480_16k_wwm_seed43022"
CTRL_RUN="${RUNS_DIR}/cleanqwen_official_seqsafe96_control_8x480_16k_wwm_seed43022"
TREAT_MODEL="${TREAT_RUN}/hf_model"
CTRL_MODEL="${CTRL_RUN}/hf_model"
CHECKPOINTS="chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M"
GPU_TREAT_ID="${GPU_TREAT:-0}"
GPU_CTRL_ID="${GPU_CTRL:-1}"

if [ "${GPU_TREAT_ID}" = "${GPU_CTRL_ID}" ]; then
    echo "ERROR: treatment/control GPUs must differ"
    exit 1
fi

for ROOT in "${TREAT_MODEL}" "${CTRL_MODEL}"; do
    for CK in ${CHECKPOINTS}; do
        if [ ! -d "${ROOT}/${CK}" ]; then
            echo "ERROR: missing checkpoint ${ROOT}/${CK}"
            exit 1
        fi
    done
done
mkdir -p "${OUT_ROOT}"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) research seqsafe96 FineWeb no-AoA eval launch"

CUDA_VISIBLE_DEVICES=${GPU_TREAT_ID} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -B "${EVAL_SCRIPT}" \
    --target fineweb_seqsafe96_treatment \
    --model_root "${TREAT_MODEL}" \
    --gpu ${GPU_TREAT_ID} \
    --out_root "${OUT_ROOT}" \
    --checkpoints ${CHECKPOINTS} \
    > "${OUT_ROOT}/fineweb_seqsafe96_treatment_eval.stdout.log" 2>&1 &
PID_TREAT=$!

CUDA_VISIBLE_DEVICES=${GPU_CTRL_ID} TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -B "${EVAL_SCRIPT}" \
    --target fineweb_seqsafe96_control \
    --model_root "${CTRL_MODEL}" \
    --gpu ${GPU_CTRL_ID} \
    --out_root "${OUT_ROOT}" \
    --checkpoints ${CHECKPOINTS} \
    > "${OUT_ROOT}/fineweb_seqsafe96_control_eval.stdout.log" 2>&1 &
PID_CTRL=$!

wait ${PID_TREAT}
RC_TREAT=$?
wait ${PID_CTRL}
RC_CTRL=$?

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) research seqsafe96 FineWeb eval return codes: treatment=${RC_TREAT} control=${RC_CTRL}"
if [ ${RC_TREAT} -ne 0 ] || [ ${RC_CTRL} -ne 0 ]; then
    echo "ERROR: at least one evaluation failed. Inspect ${OUT_ROOT}/*.stdout.log"
    exit $(( RC_TREAT + RC_CTRL ))
fi

python -B "${SUMMARY_SCRIPT}" \
    --out-root "${OUT_ROOT}" \
    --treatment-target fineweb_seqsafe96_treatment \
    --control-target fineweb_seqsafe96_control
