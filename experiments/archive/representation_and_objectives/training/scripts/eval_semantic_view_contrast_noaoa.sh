#!/bin/bash
# research: no-AoA official-compatible zero-shot + Reading evaluation for the
# semantic-view packet-local contrast.
#
# Run this only after launch_semantic_view_packet_local_training.sh has
# produced both 100M training endpoints.  It evaluates the treatment and the
# packet-local source-only control on the same 10M-spaced checkpoint trajectory
# using the COMPACT_EXPERIENCE evaluator, then summarizes matched deltas.

set -uo pipefail

EVAL_SCRIPT="experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py"
SUMMARY_SCRIPT="experiments/archive/representation_and_objectives/training/scripts/summarize_semantic_view_noaoa_delta.py"
OUT_ROOT="experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval"
RUNS_DIR="experiments/archive/representation_and_objectives/training/runs"
TREAT_RUN="${RUNS_DIR}/semantic_view_treatment_8x480_16k_wwm_seed43022"
PACKET_RUN="${RUNS_DIR}/original_packet_local_8x480_16k_wwm_seed43022"
TREAT_MODEL="${TREAT_RUN}/hf_model"
PACKET_MODEL="${PACKET_RUN}/hf_model"
CHECKPOINTS="chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M"

for ROOT in "${TREAT_MODEL}" "${PACKET_MODEL}"; do
  for CK in ${CHECKPOINTS}; do
    if [ ! -d "${ROOT}/${CK}" ]; then
      echo "ERROR: missing checkpoint ${ROOT}/${CK}"
      exit 1
    fi
  done
done

mkdir -p "${OUT_ROOT}"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) research semantic-view no-AoA eval launch"

CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -B "${EVAL_SCRIPT}" \
  --target semantic_view_treatment \
  --model_root "${TREAT_MODEL}" \
  --gpu 0 \
  --out_root "${OUT_ROOT}" \
  --checkpoints ${CHECKPOINTS} \
  > "${OUT_ROOT}/semantic_view_treatment_eval.stdout.log" 2>&1 &
PID_TREAT=$!

CUDA_VISIBLE_DEVICES=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -B "${EVAL_SCRIPT}" \
  --target original_packet_local \
  --model_root "${PACKET_MODEL}" \
  --gpu 1 \
  --out_root "${OUT_ROOT}" \
  --checkpoints ${CHECKPOINTS} \
  > "${OUT_ROOT}/original_packet_local_eval.stdout.log" 2>&1 &
PID_PACKET=$!

echo "  treatment eval pid=${PID_TREAT} GPU0"
echo "  packet-local eval pid=${PID_PACKET} GPU1"

wait ${PID_TREAT}
RC_TREAT=$?
wait ${PID_PACKET}
RC_PACKET=$?

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) research semantic-view eval return codes: treatment=${RC_TREAT} packet_local=${RC_PACKET}"
if [ ${RC_TREAT} -ne 0 ] || [ ${RC_PACKET} -ne 0 ]; then
  echo "ERROR: at least one evaluation failed. Inspect ${OUT_ROOT}/*.stdout.log"
  exit $(( RC_TREAT + RC_PACKET ))
fi

python -B "${SUMMARY_SCRIPT}" \
  --out-root "${OUT_ROOT}" \
  --treatment-target semantic_view_treatment \
  --control-target original_packet_local
