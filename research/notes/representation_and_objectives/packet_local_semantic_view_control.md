# packet local semantic view control — packet-local control for semantic-view experiment

## Why this correction matters

semantic view contrast materialization made a matched semantic-view experiment, but the strategist pointed out a remaining scientific confound. The `semantic_view_treatment` prefix keeps each row as a coherent source packet: original SimpleWiki source + accepted simplification/paraphrase view(s). The semantic view contrast materialization stream control, `original_stream_matched`, uses the same selected SimpleWiki sources and the same row-length sequence, but chunks a shuffled/cycled source stream. That can splice unrelated articles inside one training row. A treatment gain could then come from coherent single-topic packing, not from semantic-view transformation.

packet local semantic view control repaired the design by adding a first-class **packet-local balanced source-only control**:

- `original_packet_local_10M.jsonl` / `original_packet_local_100M.jsonl`
- Each control row uses only that row's own original SimpleWiki source words.
- It starts with the exact original source, matching the treatment prefix, then uses cyclic offsets only for extra repetitions needed to reach the exact treatment row length. This avoids the first naive cyclic variant's nuisance of starting some rows mid-sentence.
- It preserves source identity, topic/article boundary, row length, repetition, official filler, init/RNG, tokenizer, architecture, optimizer, and training schedule, while removing generated wording.

This arm is now the decisive transformation comparison. The stream arm remains useful to quantify packing effects, but it is not sufficient as the only control.

## Code changes

Updated materializer:

- `experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py`
- Added `balanced_packet_local_original_to_length(source_text, target_words, offset)`.
- `build_semantic_rows(...)` now emits:
  - `semantic_view_treatment`
  - `original_packet_local` (new causal arm)
  - `original_stream_matched` (packing-effect arm)
  - `original_packet_exact_matched` (old prefix-repeated forensic artifact)
- Full materialization now writes packet-local 10M and optional 100M files plus `packet_local_rows_meta.jsonl`.

Updated auditor:

- `experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py`
- Now validates packet-local row-length identity, identical filler after semantic prefix, source-key/source-article identity against semantic packet rows, semantic-prefix token-length profile, full-pool token-length profile, and 100M training-file word exposure.

New launch script:

- `experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_packet_local_training.sh`
- Trains `semantic_view_treatment_100M.jsonl` vs `original_packet_local_100M.jsonl` first on two H100s under protected DeBERTa-v2 8x480 / baseline16k / WWM fixed / AdamW recipe.
- Requires the full contrast audit to pass packet-local identity and word-exposure checks before launch.
- Uses 1M checkpoint spacing so a promising result remains AoA/submission-compatible without retraining.
- The stream arm is intentionally left for a later third run if treatment-vs-packet-local suggests a transformation signal or if packing-effect separation remains scientifically necessary.

Also wrote the post-completion evaluator launcher for the fixed-data masking recipe probe:

- `experiments/archive/representation_and_objectives/training/scripts/eval_wwm_to_token_recipe_isolation.sh`
- After fixed-data WWM-to-token training completes, it validates 100M exposure and 10M-spaced checkpoints and then runs the COMPACT_EXPERIENCE no-AoA full zero-shot + Reading checkpoint trajectory.

Syntax checks passed:

```bash
bash -n experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_packet_local_training.sh
bash -n experiments/archive/representation_and_objectives/training/scripts/eval_wwm_to_token_recipe_isolation.sh
python -B -c "import ast, pathlib; [ast.parse(pathlib.Path(p).read_text()) for p in ['experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py','experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py']]; print('syntax_ok')"
```

## Smoke validation on generation slice quality generation slice (balanced packet-local v2)

Materialized slice smoke with the new packet-local arm:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py \
  --prompts experiments/archive/representation_and_objectives/training/data/generation_slice/slice_prompts.jsonl \
  --outputs experiments/archive/representation_and_objectives/training/runs/generation_slice_qwen/outputs.jsonl \
  --out-dir experiments/archive/representation_and_objectives/training/data/semantic_view/slice_packet_local_smoke_v2 \
  --note research/notes/representation_and_objectives/semantic_view_packet_local_smoke_v2.md \
  --total-words 100000 --passes 2 --max-semantic-words 25000 --write-training
```

Result:

- accepted views: 149 / 256 generated simpara prompts in the slice
- accepted by type: simplification 68, paraphrase 81
- semantic rows: 149
- semantic words: 9,398
- filler words: 90,602
- row-length sequence identical across treatment, stream, packet-local, packet-exact arms
- metadata: `experiments/archive/representation_and_objectives/training/data/semantic_view/slice_packet_local_smoke_v2/semantic_view_materialization_metadata.json`

Audited slice smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py \
  --corpus-dir experiments/archive/representation_and_objectives/training/data/semantic_view/slice_packet_local_smoke_v2 \
  --total-words 100000 --passes 2
```

Audit result:

- row counts: treatment 716, stream control 716, packet-local control 716, packet-exact control 716
- filler identical after prefix for stream control: true
- filler identical after prefix for packet-local control: true
- packet-local source identity and row fields identical to semantic packet rows: true
- full-pool token delta stream minus treatment: mean -0.3254 tokens, over-256 row delta 0, excess-token delta 0
- full-pool token delta packet-local minus treatment: mean -0.1522 tokens, over-256 row delta 0, excess-token delta 0
- all four 2-pass 200k-word smoke training files exist and have expected exposure
- audit JSON: `experiments/archive/representation_and_objectives/training/data/semantic_view/slice_packet_local_smoke_v2/semantic_view_contrast_audit.json`

Semantic-prefix token check (computed directly on first 149 rows, excluding identical official filler):

```json
{
  "treatment": {"prefix_rows": 149, "mean_tokens": 108.9866, "max_tokens": 242, "rows_over_256": 0, "p95": 190},
  "packet_local": {"prefix_rows": 149, "mean_tokens": 108.255, "max_tokens": 236, "rows_over_256": 0, "p95": 189},
  "stream": {"prefix_rows": 149, "mean_tokens": 107.4228, "max_tokens": 224, "rows_over_256": 0, "p95": 187}
}
```

The full-pool audit reports over-256 rows because identical official filler contains old 160-word official rows that can exceed 256 tokens under baseline16k; this is shared across arms and already part of the inherited protected recipe. The semantic-view prefix itself has no seq256 truncation risk on the slice.

## Direct evidence for the stream-control packing confound

On the smoke semantic-prefix rows, `stream_matched_packet_rows_meta.jsonl` shows that stream control rows usually mix sources:

```json
{
  "stream_prefix_rows": 149,
  "rows_with_multiple_source_keys": 140,
  "fraction_multi_key": 0.9396,
  "mean_source_keys_per_row": 2.8725,
  "max_source_keys_per_row": 5,
  "rows_with_multiple_articles": 140,
  "fraction_multi_article": 0.9396,
  "mean_articles_per_row": 2.8725,
  "max_articles_per_row": 5
}
```

Text inspection illustrates the same issue. Treatment row 400002 stays on one O'Dell medical/death event plus a simplified view; packet-local row 400002 uses only balanced repeated words from the same O'Dell source and starts with the exact source; stream row 400002 combines Baillon's crake, BestGore/Luka Magnotta, an Indian archer, WHTZ radio, and consonant material inside one length-matched row. Therefore the stream arm is not a clean semantic-view transformation control by itself.

## Fallback factual-source finding

A quick local fallback inspection found that the exact public FineWeb simplification-pairs training file is not present locally in INITIAL_MODEL_STUDIES: `experiments/archive/initial_model_studies/staging/fineweb_simplification_pairs` contains a README plus incomplete download locks (`FineWeb_simplification_pairs.train.lock`) but no train file. The INITIAL_MODEL_STUDIES FineWeb-Edu cache contains the dataset README/cache and older materialized 3M samples, not a complete local copy of the leader's 9,999,969-word simplification-pair corpus. If simpara semantic-view is weak, reopening a broader factual source will require verifying current FineWeb-Edu reachability or using the older INITIAL_MODEL_STUDIES streaming materializer under strict <=10M accounting, not assuming the leader data exists locally.

## Full-run sequence after pending jobs return

Background jobs still running during packet local semantic view control:

- `s3_t31_tool1`: full source-conservative Qwen simplification/paraphrase generation to `experiments/archive/representation_and_objectives/training/runs/gen_simpara_full_qwen/outputs.jsonl`
- `s3_t32_tool1`: fixed clean-Qwen WWM-to-token recipe-isolation training to `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022`

After the semantic-view data construction completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py \
  --out-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast \
  --note research/notes/representation_and_objectives/semantic_view_contrast_materialization.md \
  --total-words 10000000 --passes 10 --max-semantic-words 2500000 --write-training

PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py \
  --corpus-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast \
  --total-words 10000000 --passes 10
```

Then launch treatment vs packet-local, not treatment vs stream first:

```bash
bash experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_packet_local_training.sh
```

After the fixed-data WWM-to-token training completes:

```bash
bash experiments/archive/representation_and_objectives/training/scripts/eval_wwm_to_token_recipe_isolation.sh
```

Scientific reading of semantic-view training results:

1. `semantic_view_treatment` > `original_packet_local`: evidence for semantic-view transformation under coherent packet-local matched repetition.
2. `semantic_view_treatment` ≈ or < `original_packet_local`: closes the simple same-source simpara semantic-view route; do not spend more H100 time adding more paraphrases of the same SimpleWiki facts.
3. Stream arm, if run later, separates packing/coherence effects: if packet-local is strong but stream is weak, coherent source packing/repetition matters; if treatment beats both, generated semantic views are more likely carrying a true learning signal.

This correction keeps the experiment faithful to the BabyLM SOTA goal: the next expensive two-H100 pair should answer a real mechanism question, not conflate transformation with packing.

## No-AoA evaluation readout prepared

To make the semantic-view contrast immediately readable after training completes, packet local semantic view control also prepared:

- `experiments/archive/representation_and_objectives/training/scripts/eval_semantic_view_contrast_noaoa.sh`
- `experiments/archive/representation_and_objectives/training/scripts/summarize_semantic_view_noaoa_delta.py`

The launcher evaluates `semantic_view_treatment` and `original_packet_local` on the COMPACT_EXPERIENCE official-compatible zero-shot + Reading trajectory for checkpoints `chck_10M` through `chck_100M` and writes a matched delta summary. The summary table reports treatment, control, and treatment-minus-control deltas for BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading, and equal7 no-AoA. Syntax check passed:

```bash
bash -n experiments/archive/representation_and_objectives/training/scripts/eval_semantic_view_contrast_noaoa.sh
python -B -c "import ast, pathlib; paths=['experiments/archive/representation_and_objectives/training/scripts/summarize_semantic_view_noaoa_delta.py']; [ast.parse(pathlib.Path(p).read_text()) for p in paths]; print('eval_scripts_syntax_ok')"
# stdout: eval_scripts_syntax_ok
```
