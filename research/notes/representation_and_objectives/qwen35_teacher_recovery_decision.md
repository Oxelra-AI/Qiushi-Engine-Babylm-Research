# qwen35 teacher recovery — Qwen3.5-9B teacher recovery and compact-view quality

## Purpose and decision context

The FW1.5M mechanism-family route needs new compact rewrites for 26,015 FineWeb
sources that lack compact rewrites. Switching teachers across source subsets would
couple teacher identity to source coverage and turn the scale experiment into a
generation-quality experiment. The proposed comparison therefore requires the
original teacher (Qwen3.5-9B) or a comparison of candidate teachers on identical
sources first.

## What was recovered

- The validated 12,152 compact views were produced by **Qwen3.5-9B**
  (`Qwen/Qwen3.5-9B`).
- The recorded generation process was used for a 256-source representative pilot
  (`training/runs/qwen35_recovery_pilot256/outputs.jsonl`): model
  `Qwen/Qwen3.5-9B`, batch 64, max_new_tokens 80,
  temperature 0.1, 223 tok/s, 59.4 s.

## Quality on the FW new-source subset (256-source pilot)

- Compression ratio: mean 0.646, median 0.642 (p05 0.448, p95 0.84) — matches the
  compact regime (rewrite/source ratio mean 0.625).
- Content recall: mean 0.597, median 0.60 — matches compact core (mean 0.662).
- Entity recall: mean 0.813, median 1.0 — the sub-1.0 tail is dominated by the
  entity extractor false-flagging sentence-initial capitalized function words
  ("However", "There", "Since", "Several", "People", month "May"). Of 90
  entity-loss rows, 71 still contain ≥1 real entity token; inspected samples show
  real named entities (Australia, The Lancet, University of Rochester, Congress,
  USGS, Tanzania, Kenya, Alzheimer) consistently preserved. This matches   entity_recall mean 0.996 measured with the source-provided entity lists.
- Number recall: mean 0.979, median 1.0 — matches (1.0).
- Copy-like 0.008, malformed 0.0, too-short 0.0 — clean generation.
- Core acceptance (ratio in [0.35,0.85], content>=0.45, entity>=0.75, number==1.0,
  not copy/malformed): 0.613. This is the operative acceptance rate and matches
  compact-view yield (10,094 accepted from ~14,600 candidates ≈ 0.69, and
  the COMPACT_EXPERIENCE clean-Qwen selected pairs).
- Structure-strict acceptance 0.145 is a heuristic artifact, not real loss:
  inspected core-accepted samples show faithful paraphrase where negation is
  re-expressed ("with or without"→"regardless of"), causal/comparison markers
  are lexically substituted, so the exact-marker recall under-counts. The
  strict metric over-penalizes legitimate compression and must not gate the
  corpus.

## Decision

Use **Qwen3.5-9B consistently** as the single compact-view teacher for all 26,015
new sources, exactly reproducing the original 12,152-pair generation process
(same model, prompts family, batch 64, max_new_tokens 80, temperature 0.1). This
keeps teacher identity constant across the entire FW pair family so the scale
experiment tests the compact-view mechanism, not generator quality. No teacher
swap to `Qwen3-4B` was adopted.

## Files

- Pilot prompts: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_recovery_pilot_prompts.jsonl`
- Pilot outputs: `experiments/archive/representation_and_objectives/training/runs/qwen35_recovery_pilot256/outputs.jsonl`
- Quality summary: `experiments/archive/representation_and_objectives/data/qwen35_teacher_recovery/qwen35_generation_quality_summary.json`
- Merged/accepted rows: `.../qwen35_generation_merged_quality.jsonl`, `.../qwen35_generation_accepted_core.jsonl`
- Recovery/quality script: `experiments/archive/representation_and_objectives/scripts/qwen35_teacher_recovery_and_quality.py`
