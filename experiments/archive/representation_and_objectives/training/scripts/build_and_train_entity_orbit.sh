#!/usr/bin/env bash
# research: build the high-precision entity-orbit corpus, then train the matched
# Strict-Small DeBERTa arm with the exact historical compact_view_reinvest recipe.
#
# Matching contract against the reference run
#   experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022
# tokenizer legal/baseline16k, DeBERTa-v2 8x480, seed 43 / init 43022 / train rng 43023,
# batch 256, seq 256, LR 1e-3, warmup 0.06, wwm_fixed 0.15, 100M word exposure,
# row order and per-row word counts identical, so the update schedule and batch words match.
# The only difference is the surface identity of repeated proper-name mentions.
set -euo pipefail

CORPUS_DIR="experiments/archive/representation_and_objectives/data/entity_orbit_alias_corpus_v2"
TRAIN100="${CORPUS_DIR}/entity_orbit_alias_compact_reinvest_100M.jsonl"
MANIFEST="${CORPUS_DIR}/entity_orbit_alias_manifest.json"
RUN_DIR="experiments/archive/representation_and_objectives/training/runs/entity_orbit_alias_16k_seed43022"
TOKENIZER="experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model"

echo "[research] building full entity-orbit corpus"
python3 -B experiments/archive/representation_and_objectives/scripts/language_entity_orbit_corpus_v2.py \
  --output_dir "${CORPUS_DIR}" --progress_every 10000

test -s "${TRAIN100}"
test -s "${MANIFEST}"
python3 - "$TRAIN100" <<'PY'
import json, sys
p = sys.argv[1]
rows = words = 0
for line in open(p, encoding="utf-8"):
    if not line.strip():
        continue
    o = json.loads(line)
    rows += 1
    words += o["words"]
    if o["words"] != len(o["text"].split()):
        raise SystemExit(f"word mismatch row {rows}")
assert words == 100_000_000, words
print(json.dumps({"event": "corpus_verified", "rows": rows, "words": words}))
PY

echo "[research] training matched arm on GPU1"
mkdir -p "${RUN_DIR}"
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py \
  --example_jsonl "${TRAIN100}" \
  --example_jsonl_label entity_orbit_alias_compact_reinvest \
  --example_jsonl_meta "${MANIFEST}" \
  --output_dir "${RUN_DIR}" \
  --tokenizer_path "${TOKENIZER}" \
  --tokenizer_label baseline16k \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --warmup_fraction 0.06 --weight_decay 0.01 \
  --masking_curriculum wwm_fixed --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --checkpoint_words 20000000 --max_word_exposure 100000000 \
  --num_workers 0 --log_every 100 --dynamics_trace_every 200

test -s "${RUN_DIR}/scientific_metrics.json"
python3 -c "
import json
m=json.load(open('${RUN_DIR}/scientific_metrics.json'))
print(json.dumps({'event':'train_done','steps':m['actual_training_steps'],'word_exposure':m['word_exposure'],'loss_first':m['loss_first'],'loss_last':m['loss_last'],'ckpts':[c['name'] for c in m['saved_checkpoints']]}))
"
