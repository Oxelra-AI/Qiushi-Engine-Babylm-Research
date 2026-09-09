#!/bin/bash
# Independent-seed replication of the compact_view_core density candidate
# at a DIFFERENT seed (extra_init_seed=43122, train_rng_seed=43123), then fast
# no-AoA official-compatible screen and comparison against seed 43022.
#
# Scientific purpose: compact_view_core (seed 43022) was the strongest
# BabyLM Strict-Small candidate in the screen: fast seven-column mean 43.849, projected Overall
# ~41.917 with clean SuperGLUE and AoA=0, which would pass the visible 41.8 leader.
# A single-seed fast screen cannot support a SOTA claim. This replication checks
# reproducibility at an independent seed. The interpretation rule is: if seed 43122 reproduces
# equal7_full_entity within ~0.5 of 43.928 and keeps Supplement/EWoK/Entity/COMPS
# gains over compact_repeat_core, the result supports seed reproducibility and
# prioritizing full evaluation and submission preparation; if it collapses toward the ~41.7 repeat
# baseline, the fast-screen surface is seed-fragile and does not justify that commitment.
#
# GPU discipline: this waits for a GENUINELY isolated GPU (>=70GB free AND <=15% util
# across three samples) to avoid contention with full evaluation and trajectory
# tasks. It exits cleanly if no GPU frees within the wait budget.
set -u

TS() { date -u +%Y-%m-%dT%H:%M:%SZ; }

REPO_ROOT="."
A02_LAUNCHER="experiments/archive/frontier_consolidation/scripts/train_density_arm_seed.py"
A02_FAST_EVAL="experiments/archive/frontier_consolidation/scripts/fast_eval_density_arms.py"
ARM="cleanqwen_fineweb_compact_view_core_neutral"
EXTRA_INIT_SEED=43122
TRAIN_RNG_SEED=43123
RUN_DIR="experiments/archive/representation_and_objectives/training/runs/repl_compact_view_core_seed43122"
OUT_ROOT="experiments/archive/representation_and_objectives/data/compact_view_core_replication"
LOG="${OUT_ROOT}/controller.log"

mkdir -p "${OUT_ROOT}"
echo "$(TS) compact_view_core replication controller start" | tee -a "${LOG}"

# ---- Preflight: verify frozen reference data hash before spending any GPU time ----
echo "$(TS) preflight dry-run with hash check" | tee -a "${LOG}"
PREFLIGHT_JSON="${OUT_ROOT}/preflight.json"
PYTHONDONTWRITEBYTECODE=1 python -B "${A02_LAUNCHER}" \
  --arm "${ARM}" --gpu 0 \
  --extra-init-seed "${EXTRA_INIT_SEED}" --train-rng-seed "${TRAIN_RNG_SEED}" \
  --run-dir "${RUN_DIR}" \
  --check-hash --dry-run > "${PREFLIGHT_JSON}" 2>>"${LOG}"
PRC=$?
if [ "${PRC}" -ne 0 ]; then
  echo "$(TS) preflight FAILED rc=${PRC}; not launching" | tee -a "${LOG}"
  exit 20
fi
if ! grep -q '"hash_ok": true' "${PREFLIGHT_JSON}"; then
  echo "$(TS) preflight hash mismatch; not launching" | tee -a "${LOG}"
  exit 21
fi
echo "$(TS) preflight OK, frozen data hash matches the seed43022 corpus" | tee -a "${LOG}"

# ---- Wait for the seed43022 full official-compatible result before replication ----
# The cheaper decisive full evaluation of seed43022 takes priority over replication.
# If that result fails far below the useful range, replication is no longer the
# immediate bottleneck and this controller exits before training. If the result is
# SOTA or near-SOTA, independent seed replication becomes the next load-bearing
# evidence.
A02_FULL_TARGET="experiments/archive/frontier_consolidation/data/density_full_eval/per_target/compact_view_core.json"
WAIT_A02=0
MAX_WAIT_A02=7200
while [ "${WAIT_A02}" -lt "${MAX_WAIT_A02}" ]; do
  A02_PARSE=$(python -B - "${A02_FULL_TARGET}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    print(json.dumps({"status":"missing"})); raise SystemExit(0)
try:
    data = json.loads(p.read_text(encoding='utf-8'))
    oo = data.get('official_overall') or {}
    overall = oo.get('Overall') or oo.get('overall') or oo.get('provisional_overall')
    complete = bool(oo.get('complete_for_provisional_overall') or oo.get('submit_ready_overall') or overall is not None)
    print(json.dumps({"status":"present", "overall": overall, "complete": complete, "finished_utc": data.get('finished_utc')}))
except Exception as exc:
    print(json.dumps({"status":"parse_error", "error": repr(exc)}))
PY
)
  echo "$(TS) seed43022 full-eval check after ${WAIT_A02}s: ${A02_PARSE}" | tee -a "${LOG}"
  A02_STATUS=$(echo "${A02_PARSE}" | python -B -c "import sys,json; print(json.load(sys.stdin).get('status'))")
  A02_COMPLETE=$(echo "${A02_PARSE}" | python -B -c "import sys,json; print(json.load(sys.stdin).get('complete', False))")
  if [ "${A02_STATUS}" = "present" ] && [ "${A02_COMPLETE}" = "True" ]; then
    A02_OVERALL=$(echo "${A02_PARSE}" | python -B -c "import sys,json; v=json.load(sys.stdin).get('overall'); print('nan' if v is None else v)")
    PROCEED=$(python -B - <<PY
v=float('${A02_OVERALL}')
print('yes' if v >= 41.6 else 'no')
PY
)
    if [ "${PROCEED}" = "yes" ]; then
      echo "$(TS) seed43022 compact_view_core full-eval is near/SOTA-relevant (Overall=${A02_OVERALL}); proceeding to GPU wait for independent-seed replication" | tee -a "${LOG}"
      break
    else
      echo "$(TS) seed43022 compact_view_core full-eval Overall=${A02_OVERALL} is below immediate replication threshold 41.6; exiting without training" | tee -a "${LOG}"
      exit 34
    fi
  fi
  sleep 120
  WAIT_A02=$((WAIT_A02+120))
done
if [ "${WAIT_A02}" -ge "${MAX_WAIT_A02}" ]; then
  echo "$(TS) seed43022 full-eval result did not finalize within ${MAX_WAIT_A02}s; exiting without training" | tee -a "${LOG}"
  exit 35
fi

# ---- Wait for a genuinely isolated GPU (do not oversubscribe) ----
pick_gpu() {
  python -B - <<'PY'
import subprocess, json
out = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free,utilization.gpu",
                      "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
best=None
for line in out.strip().splitlines():
    idx, free, util = [x.strip() for x in line.split(",")]
    idx=int(idx); free=int(free); util=int(util)
    if free>=70000 and util<=15:
        if best is None or free>best[1]:
            best=(idx, free, util)
print(json.dumps(best if best else []))
PY
}

WAITED=0
MAX_WAIT=5400   # up to 90 min wait for a GPU to free
INTERVAL=120
GPU=""
while [ "${WAITED}" -lt "${MAX_WAIT}" ]; do
  # require three consecutive stable isolated samples
  S1=$(pick_gpu); sleep 3
  S2=$(pick_gpu); sleep 3
  S3=$(pick_gpu)
  G1=$(echo "${S1}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  G2=$(echo "${S2}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  G3=$(echo "${S3}" | python -B -c "import sys,json; v=json.load(sys.stdin); print(v[0] if v else -1)")
  if [ "${G1}" != "-1" ] && [ "${G1}" = "${G2}" ] && [ "${G2}" = "${G3}" ]; then
    GPU="${G1}"
    echo "$(TS) isolated GPU ${GPU} (samples ${S1} ${S2} ${S3})" | tee -a "${LOG}"
    break
  fi
  echo "$(TS) no isolated GPU yet after ${WAITED}s (s1=${S1} s2=${S2} s3=${S3})" | tee -a "${LOG}"
  sleep "${INTERVAL}"
  WAITED=$((WAITED+INTERVAL))
done

if [ -z "${GPU}" ]; then
  echo "$(TS) no isolated GPU within ${MAX_WAIT}s; exiting without training" | tee -a "${LOG}"
  exit 30
fi

# ---- Train replication ----
echo "$(TS) launching replication train on GPU ${GPU}" | tee -a "${LOG}"
PYTHONDONTWRITEBYTECODE=1 python -B "${A02_LAUNCHER}" \
  --arm "${ARM}" --gpu "${GPU}" \
  --extra-init-seed "${EXTRA_INIT_SEED}" --train-rng-seed "${TRAIN_RNG_SEED}" \
  --run-dir "${RUN_DIR}" \
  --check-hash >> "${LOG}" 2>&1
TRC=$?
echo "$(TS) replication train rc=${TRC}" | tee -a "${LOG}"
if [ "${TRC}" -ne 0 ]; then
  echo "$(TS) training FAILED; not evaluating" | tee -a "${LOG}"
  exit 31
fi

if [ ! -d "${RUN_DIR}/hf_model/chck_100M" ]; then
  echo "$(TS) chck_100M missing after training; abort eval" | tee -a "${LOG}"
  exit 32
fi

# ---- Fast no-AoA screen on the replication endpoint ----
echo "$(TS) fast no-AoA screen on replication chck_100M (GPU ${GPU})" | tee -a "${LOG}"
PYTHONDONTWRITEBYTECODE=1 python -B \
  "experiments/archive/representation_and_objectives/training/scripts/fast_eval_replication.py" \
  --run-dir "${RUN_DIR}" \
  --endpoint chck_100M \
  --gpu "${GPU}" \
  --out-root "${OUT_ROOT}" >> "${LOG}" 2>&1
ERC=$?
echo "$(TS) fast eval rc=${ERC}" | tee -a "${LOG}"
exit "${ERC}"
