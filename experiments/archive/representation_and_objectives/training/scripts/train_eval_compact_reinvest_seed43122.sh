#!/bin/bash
# research: independent-seed training + fast no-AoA screen for compact_view_reinvest.
#
# Expensive-work admission: the seed43022 compact_view_reinvest endpoint is the
# strongest SOTA-facing candidate from the fast screen. Full evaluations of
# compact_view_core and reinvest are already underway. The remaining
# load-bearing uncertainty is seed stability of the
# strongest endpoint. This task trains exactly one new seed of the already chosen
# reinvest corpus and runs the lowest-cost official-compatible task-family screen;
# it does not add a new route, tokenizer, architecture, or full-eval burden.
set -u

TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

ARM="cleanqwen_fineweb_compact_view_reinvest"
TARGET_LABEL="compact_view_reinvest_seed43122"
EXTRA_INIT_SEED=43122
TRAIN_RNG_SEED=43123
A02_LAUNCHER="experiments/archive/frontier_consolidation/scripts/train_density_arm_seed.py"
RUN_DIR="experiments/archive/representation_and_objectives/training/runs/repl_compact_view_reinvest_seed43122"
OUT_ROOT="experiments/archive/representation_and_objectives/data/compact_reinvest_seed43122_fast"
FAST_EVAL="experiments/archive/representation_and_objectives/training/scripts/fast_eval_density_seed.py"
LOG="${OUT_ROOT}/reinvest_seed43122_controller.log"
mkdir -p "${OUT_ROOT}"

echo "$(TS) research compact_view_reinvest seed43122 controller start" | tee -a "${LOG}"

# Frozen-data preflight before GPU work.
PREFLIGHT_JSON="${OUT_ROOT}/preflight.json"
echo "$(TS) preflight dry-run with hash check arm=${ARM}" | tee -a "${LOG}"
PYTHONDONTWRITEBYTECODE=1 python -B "${A02_LAUNCHER}" \
  --arm "${ARM}" --gpu 0 \
  --extra-init-seed "${EXTRA_INIT_SEED}" --train-rng-seed "${TRAIN_RNG_SEED}" \
  --run-dir "${RUN_DIR}" --check-hash --dry-run > "${PREFLIGHT_JSON}" 2>>"${LOG}"
PRC=$?
if [ "${PRC}" -ne 0 ]; then
  echo "$(TS) preflight failed rc=${PRC}; abort" | tee -a "${LOG}"
  exit 20
fi
if ! grep -q '"hash_ok": true' "${PREFLIGHT_JSON}"; then
  echo "$(TS) preflight hash mismatch; abort" | tee -a "${LOG}"
  exit 21
fi

echo "$(TS) preflight OK" | tee -a "${LOG}"

pick_gpu() {
  python -B - <<'PY'
import subprocess, json
out = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu","--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
best=None
for line in out.strip().splitlines():
    parts=[x.strip() for x in line.split(',')]
    if len(parts) != 3:
        continue
    idx, free, util = int(parts[0]), int(parts[1]), int(parts[2])
    # Training requires a fully isolated H100; do not share with full evaluation.
    if free >= 70000 and util <= 15:
        if best is None or free > best[1]:
            best=(idx, free, util)
print(json.dumps(best if best else []))
PY
}

WAITED=0
MAX_WAIT=${MAX_WAIT_GPU_SEC:-14400}
INTERVAL=120
GPU=""
while [ "${WAITED}" -lt "${MAX_WAIT}" ]; do
  S1=$(pick_gpu); sleep 3
  S2=$(pick_gpu); sleep 3
  S3=$(pick_gpu)
  G1=$(echo "${S1}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  G2=$(echo "${S2}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  G3=$(echo "${S3}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  if [ "${G1}" != "-1" ] && [ "${G1}" = "${G2}" ] && [ "${G2}" = "${G3}" ]; then
    GPU="${G1}"
    echo "$(TS) isolated GPU ${GPU} samples=${S1} ${S2} ${S3}" | tee -a "${LOG}"
    break
  fi
  echo "$(TS) no isolated GPU after ${WAITED}s samples=${S1} ${S2} ${S3}" | tee -a "${LOG}"
  sleep "${INTERVAL}"
  WAITED=$((WAITED+INTERVAL))
done
if [ -z "${GPU}" ]; then
  echo "$(TS) no isolated GPU within ${MAX_WAIT}s; exit without training" | tee -a "${LOG}"
  exit 30
fi

echo "$(TS) launch training ${TARGET_LABEL} on GPU ${GPU}" | tee -a "${LOG}"
PYTHONDONTWRITEBYTECODE=1 python -B "${A02_LAUNCHER}" \
  --arm "${ARM}" --gpu "${GPU}" \
  --extra-init-seed "${EXTRA_INIT_SEED}" --train-rng-seed "${TRAIN_RNG_SEED}" \
  --run-dir "${RUN_DIR}" --check-hash >> "${LOG}" 2>&1
TRC=$?
echo "$(TS) training rc=${TRC}" | tee -a "${LOG}"
if [ "${TRC}" -ne 0 ]; then
  exit "${TRC}"
fi
if [ ! -f "${RUN_DIR}/hf_model/chck_100M/model.safetensors" ]; then
  echo "$(TS) trained endpoint missing; abort fast eval" | tee -a "${LOG}"
  exit 31
fi

echo "$(TS) launch fast no-AoA screen ${TARGET_LABEL} on GPU ${GPU}" | tee -a "${LOG}"
PYTHONDONTWRITEBYTECODE=1 python -B "${FAST_EVAL}" \
  --run-dir "${RUN_DIR}" --target-label "${TARGET_LABEL}" \
  --endpoint chck_100M --gpu "${GPU}" --out-root "${OUT_ROOT}" >> "${LOG}" 2>&1
ERC=$?
echo "$(TS) fast eval rc=${ERC}" | tee -a "${LOG}"
exit "${ERC}"
