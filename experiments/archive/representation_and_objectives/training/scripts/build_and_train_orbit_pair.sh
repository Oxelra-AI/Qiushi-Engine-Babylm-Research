#!/usr/bin/env bash
# research: build both deterministic token-matched entity-orbit corpora, then train
# the two matched Strict-Small arms in parallel on GPU0 and GPU1.
#
# Scientific design
#   reference : data/external/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022 (already trained)
#   per_row   : identity persistence across documents destroyed
#   stable    : identical detector, alias bank, substitution count and token geometry,
#               but one corpus-stable alias per lexical key, so identity persists
#
# per_row minus stable isolates removal of cross-document identity persistence from
# the lexical substitution itself. Both arms share the reference tokenizer, model,
# seeds, batch/schedule, row order, per-row word counts, and 100M exposure; each
# changed row also preserves the complete tokenizer word-start vector, so WWM group
# structure and RNG-selected mask positions are matched.
set -euo pipefail

CORPUS_DIR="experiments/archive/representation_and_objectives/data/deterministic_token_matched_entity_orbit"
SCRIPT="experiments/archive/representation_and_objectives/scripts/deterministic_token_matched_entity_orbit.py"
TOKENIZER="experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"
TRAINER="experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py"

for MODE in per_row stable; do
  echo "[research] building corpus mode=${MODE}"
  python3 -B "${SCRIPT}" --output_dir "${CORPUS_DIR}" --assignment_mode "${MODE}" --progress_every 20000
done

python3 - "${CORPUS_DIR}" <<'PY'
import json, sys, hashlib
from pathlib import Path
d = Path(sys.argv[1])
ref100 = Path("experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl")
refseq = []
for line in ref100.open(encoding="utf-8"):
    if line.strip():
        o = json.loads(line)
        refseq.append((o.get("example_id"), o["words"]))
print(json.dumps({"event": "reference_seq", "rows": len(refseq), "words": sum(w for _, w in refseq)}))
for mode in ["per_row", "stable"]:
    p = d / f"det_tokenmatched_orbit_{mode}_100M.jsonl"
    rows = words = 0
    ok = True
    for i, line in enumerate(p.open(encoding="utf-8")):
        if not line.strip():
            continue
        o = json.loads(line)
        rows += 1
        words += o["words"]
        if o["words"] != len(o["text"].split()):
            raise SystemExit(f"{mode}: word field mismatch row {rows}")
        if (o.get("example_id"), o["words"]) != refseq[i]:
            ok = False
            raise SystemExit(f"{mode}: order mismatch at row {rows}")
    assert rows == len(refseq) and words == 100_000_000, (mode, rows, words)
    print(json.dumps({"event": "corpus_verified", "mode": mode, "rows": rows, "words": words, "order_matches_reference": ok}))
PY

declare -A GPU=( [per_row]=0 [stable]=1 )
PIDS=()
for MODE in per_row stable; do
  RUN_DIR="experiments/archive/representation_and_objectives/training/runs/orbit_${MODE}_16k_seed43022"
  mkdir -p "${RUN_DIR}"
  echo "[research] training mode=${MODE} on GPU${GPU[$MODE]} -> ${RUN_DIR}"
  CUDA_VISIBLE_DEVICES=${GPU[$MODE]} python3 -B "${TRAINER}" \
    --example_jsonl "${CORPUS_DIR}/det_tokenmatched_orbit_${MODE}_100M.jsonl" \
    --example_jsonl_label "det_tokenmatched_orbit_${MODE}" \
    --example_jsonl_meta "${CORPUS_DIR}/det_tokenmatched_orbit_${MODE}_manifest.json" \
    --output_dir "${RUN_DIR}" \
    --tokenizer_path "${TOKENIZER}" \
    --tokenizer_label baseline16k \
    --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
    --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
    --batch_size 256 --seq_length 256 --max_seq_length 256 \
    --learning_rate 0.001 --warmup_fraction 0.06 --weight_decay 0.01 \
    --masking_curriculum wwm_fixed --mask_prob_start 0.15 --mask_prob_end 0.15 \
    --checkpoint_words 20000000 --max_word_exposure 100000000 \
    --num_workers 0 --log_every 200 --dynamics_trace_every 500 \
    > "${RUN_DIR}/train_stdout.log" 2> "${RUN_DIR}/train_stderr.log" &
  PIDS+=($!)
done

FAIL=0
for P in "${PIDS[@]}"; do
  wait "${P}" || FAIL=1
done

for MODE in per_row stable; do
  RUN_DIR="experiments/archive/representation_and_objectives/training/runs/orbit_${MODE}_16k_seed43022"
  if [ -s "${RUN_DIR}/scientific_metrics.json" ]; then
    python3 -c "
import json,sys
m=json.load(open('${RUN_DIR}/scientific_metrics.json'))
print(json.dumps({'event':'train_done','mode':'${MODE}','steps':m.get('actual_training_steps'),'word_exposure':m.get('word_exposure'),'loss_first':m.get('loss_first'),'loss_last':m.get('loss_last'),'ckpts':[c['name'] for c in m.get('saved_checkpoints',[])]}))
"
  else
    echo "{\"event\":\"train_missing_metrics\",\"mode\":\"${MODE}\"}"
    FAIL=1
  fi
done
exit ${FAIL}
