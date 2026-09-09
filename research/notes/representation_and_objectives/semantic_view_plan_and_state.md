# semantic view contrast materialization — semantic-view contrast machinery, coverage reality, and next-run plan

## What semantic view contrast materialization established (Execute, no expensive training launched)

1. **Coverage reality of the pending simpara shard.** `training/scripts/measure_simpara_source_coverage.py` →
   `training/data/semantic_view/simpara_prompt_source_coverage.json`, note `notes/simpara_source_coverage.md`.
   - The 30,000 shard1 prompts are only **14,985 unique source texts** (502,284 source words) across 8,990
     SimpleWiki article labels; every unique source has **both** a simplification and a paraphrase prompt.
   - Using generation slice quality slice length ratios, the maximum semantic-view pool if every generation were accepted is
     ~**1.42M words (14.2%)** of a 10M corpus.
   - Scientific meaning: the simpara shard is **not a broad new factual source**. It is two linguistic views of
     the same SimpleWiki rows. It can only be a *semantic-view learning* mechanism, not the factual/entity/causal
     breadth hypothesized to explain the leader gap.

2. **Matched contrast machinery (validated on the generation slice quality slice, both smoke dirs).**
   `training/scripts/materialize_semantic_view_contrast.py` builds, with exact ≤10M word accounting:
   - `semantic_view_treatment`: each accepted source row concatenated with accepted simplification/paraphrase views.
   - `original_stream_matched` (**primary control**): a shuffled/cycled stream of the *same selected SimpleWiki
     source texts* chunked to the identical treatment row-length sequence — same source articles, same original
     word mass, same row lengths, no Qwen transformation.
   - `original_packet_exact_matched` (**secondary forensic control**): rowwise repeated original to the same length
     (kept for inspection; over-emphasizes source prefixes, not the preferred causal arm).
   - Identical official filler in all arms after the semantic prefix; identical row-length sequence; 10-pass
     repeated exposure training files when `--write-training` is set.
   - View acceptance uses transparent number/entity/overlap/completeness screens (same spirit as COMPACT_EXPERIENCE compact density subtask delta).
     On the 384-slice: 149/256 accepted (68 simplification, 81 paraphrase); rejections dominated by entity-recall
     and number risk, which is the intended conservatism.
   - Audit: `training/scripts/audit_semantic_view_contrast.py` verified exact word totals, identical filler
     after the prefix, and no seq256 truncation risk (token delta control−treatment −0.78 mean tokens, 0 over-256).

## Why this is the right *fast, decision-changing* next experiment
The strategist correction is now operational: any score movement between `semantic_view_treatment` and
`original_stream_matched` is interpretable as **linguistic-view transformation of the same source facts**, not as
mixed transformation + repeated-content allocation, because word exposure, source article set, and row lengths are
matched. This distinguishes the actual mechanism the simpara data can provide.

## Immediate next actions (for the collecting step)
1. **Collect `s3_t31_tool1`** (full simpara generation → `training/runs/gen_simpara_full_qwen/outputs.jsonl`).
   Then run the full materializer with `--write-training`:
   ```
   PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/materialize_semantic_view_contrast.py \
     --out-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast \
     --note research/notes/representation_and_objectives/semantic_view_contrast_materialization.md \
     --total-words 10000000 --passes 10 --max-semantic-words 2500000 --write-training
   ```
   Then audit:
   ```
   PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/representation_and_objectives/training/scripts/audit_semantic_view_contrast.py \
     --corpus-dir experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast \
     --total-words 10000000 --passes 10
   ```
   Expected semantic-packet fraction is well under the 25% cap (the pool max is ~14%), which is scientifically fine —
   this is a matched-view test, not an inflated 25% corpus.

2. **Collect `s3_t32_tool1`** (fixed clean-Qwen 8×480/16k WWM→token recipe isolation training →
   `training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022`). Inspect `scientific_metrics.json`
   (word_exposure=100M, saved chck_1M..chck_100M, loss trajectory) and run the established no-AoA official-compatible
   zero-shot + Reading trajectory (`experiments/archive/compact_experience/scripts/eval_checkpoint_trajectory_fullzeroshot.py`,
   full_eval task paths, strict eval cwd) against the protected clean-Qwen WWM baseline
   (`data/external/qwen_clean_aligned_16k_seed43022`, official Overall 41.3443 with SuperGLUE 70.31).
   This separates the masking-recipe effect from the data effect. COMPACT_EXPERIENCE fast-pair evidence already showed WWM→token
   trades Supplement/Entity/Reading for BLiMP/GlobalPIQA with an unstable sign; the fixed-data 100M run tests whether
   that trade is net-positive on the official surface.

3. **Two-H100 training when data arm is ready.** Train `semantic_view_treatment` and `original_stream_matched`
   under the protected 8×480/16k WWM/AdamW recipe (`masking_curriculum_trainer.py`, LR 0.001, batch 256, seq 256,
   `--extra_init_seed 43022 --train_rng_seed 43023`, `--checkpoint_words 10000000 --max_word_exposure 100000000`),
   shared init/RNG, then run the full official nine-column evaluation (reuse COMPACT_EXPERIENCE `compact reinvest full eval summary/compact density subtask delta` runner pattern
   including SuperGLUE and AoA) and read ΔOverall.

## Decision rule for the data mechanism
- If `semantic_view_treatment` − `original_stream_matched` ΔOverall is clearly positive (≥ +0.25 on matched full
  official eval, without collapsing Supplement/Entity/Reading), semantic-view learning is a real lever; scale it and
  combine with the strongest recipe.
- If the contrast is weak/negative, this closes the "more simpara views of the same SimpleWiki rows" route and the
  data effort should reopen a genuinely broader, verifiable factual source. Local `HuggingFaceFW/fineweb-edu`
  streaming (used in INITIAL_MODEL_STUDIES `fineweb_relation_matched_materialize.py`) was reachable in INITIAL_MODEL_STUDIES; verify current
  reachability before assuming external download is possible, and account words strictly under the ≤10M budget.

## Key artifacts
- Coverage: `training/data/semantic_view/simpara_prompt_source_coverage.json`
- Materializer: `training/scripts/materialize_semantic_view_contrast.py`
- Auditor: `training/scripts/audit_semantic_view_contrast.py`
- Validated slice smoke (repaired control): `training/data/semantic_view/slice_contrast_smoke_v2/`
- Protected baseline coordinate: COMPACT_EXPERIENCE `data/full_eval/full_eval_summary.json`
  (`qwen_clean_aligned` Overall 41.3443; matched official control 38.94; leader 41.8).
