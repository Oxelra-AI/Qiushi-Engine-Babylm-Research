# lead provenance guard and contingency routes — Lead provenance guard and contingency routes

## Scope

No result was inferred from the incomplete retraining comparisons:

- Original same-pool-tokenizer reinvest retraining.
- Byte-alphabet-tokenizer reinvest retraining.

It also did not inspect active run directories, launch GPU work, evaluate any model, alter the frozen tokenizer/corpus/training recipe, or prepare final deliverables. The trusted runtime ledger still reports both retrains as running, and the system instruction is not to poll.



## Why this work was useful

earlier analysis refreshed the current live Strict-Small comparison target: 122 rows in the official HF Space table, current best `wwm_curriculum_simplification_40k_strict-small` at Overall `41.8`. The remaining scientific bottleneck is therefore still the delivered legal tokenizer endpoint score vector, not another leaderboard or tokenizer subset record.

However, a critical independent_review verifier point exposed a useful pre-score repair: the largest remaining possible invalidation risk for a delivered endpoint is exact corpus/exposure provenance, not the already-characterized spatial repair route status newline `<unk>` behavior. Before any post-delivery evaluation starts, the run must be bound to the frozen reinvest 10M pool, its 100M ten-pass training file, and metadata hashes/word counts/row counts.

## independent_review synthesis

independent_review outputs:

- Generator integration: `data/external/independent_review01_generator1_integration.md`
- Verifier integration: `data/external/independent_review01_verifier1_integration.md`
- Branch records:
  - `data/external/independent_review01_sub1_generator.md`
  - `data/external/independent_review01_sub2_generator.md`
  - `data/external/independent_review01_sub3_verifier.md`
  - `data/external/independent_review01_sub4_verifier.md`

Scientific points I am preserving from independent_review:

1. The pending legal endpoints remain the decisive evidence. Do not infer scores from old-tokenizer results, MLM loss, seed behavior, fragmentation, or newline exposure.
2. Both legal tokenizers remain legal endpoint candidates; byte-alphabet is structurally cleaner but not guaranteed to score better because newline byte `Ċ` had zero direct pretraining frequency.
3. Before scoring a delivered endpoint, verify exact tokenizer hash, random initialization/frozen recipe, full checkpoint ladder, `hf_model/chck_100M`, 100M exposure, and exact corpus/provenance.
4. Local pristine Overall is the right experimental coordinate, but a tiny margin over the public rounded 41.8 should be interpreted cautiously because leaderboard values are rounded.
5. Closed or weak routes remain closed: late stopping, checkpoint averaging, broad force filtering, more static newline scans, selecting tokenizers from coverage statistics, unmotivated late LR edits, and larger spatial-lengthening repairs.
6. If neither legal endpoint beats 41.8, the most plausible low-cost successor routes are, in priority order:
   - explicit source/compact paired-view consistency with stop-gradient or aligned-span agreement;
   - information/relation-weighted masking at fixed mask budget;
   - matched-count compactness/source ordering as a temporal allocation test;
   - structural semantic-edge compact views that preserve predicate-argument topology without lengthening prose;
   - generic affordance/state-transition microviews only if they improve GlobalPIQA and physical EWoK without harming stronger columns;
   - corpus-internal early seed/basin selection only if a preregistered held-in signal predicts later broad performance;
   - source-gradient conflict control only after measuring persistent source-group gradient conflict.

These are not authorized 100M launches. They are outcome-conditioned routes requiring cheap matched 10M–30M screens after legal endpoint score vectors exist.

## Strengthened post-delivery safeguards

### Frozen corpus evidence

Read-only checksum/row-count command verified:

```text
92aa4c00b09d201a09da27e7e16677643bc7cda321471e1ec46332600339b3e4  experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/density_cleanqwen_rowholdout_overlay_metadata.json
215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23  experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl
3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691  experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl
64740  cleanqwen_fineweb_compact_view_reinvest_10M.jsonl
647400 cleanqwen_fineweb_compact_view_reinvest_100M.jsonl
```

The metadata itself records `total_words_per_pool=10000000`, `passes=10`, `all_exact_10M=true`, and the compact-reinvest family word totals/row counts matching the frozen reinvest files.

### Patched inspector

Overwrote and strengthened:

- `experiments/archive/frontier_consolidation/scripts/inspect_compliant_retrain.py`

New behavior: the inspector now accepts and checks expected corpus/provenance arguments in addition to endpoint/tokenizer checks:

- expected 100M train file path and SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`;
- expected 10M pool path and SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`;
- expected metadata path and SHA `92aa4c00b09d201a09da27e7e16677643bc7cda321471e1ec46332600339b3e4`;
- expected words/rows: 10M pool `10,000,000` words and `64,740` rows; 100M train file `100,000,000` words and `647,400` rows;
- expected pass count `10`;
- JSONL row/word counting when `--verify-corpus-content` is passed;
- training command path equality for `--example_jsonl`, `pool_10m`, metadata, tokenizer path, max word exposure, tokenizer metadata pool SHA/word/row/path, and frozen command flags; the decisive corpus hashes and JSONL word/row counts are recomputed independently so a valid compliant tokenizer retrain status launch is not rejected merely because its original command did not record a `word_info` field;
- rejection if the command contains pretrained/resume-style flags;
- complete endpoint is false if any required provenance or frozen-recipe field fails.

### Patched post-delivery driver

Patched:

- `experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py`

New behavior:

1. The inspection stage passes all frozen corpus path/hash/word/row/pass arguments above and enables `--verify-corpus-content` before any evaluation column runs.
2. The inspection timeout is 1800 s to allow SHA and JSONL word counting of the 100M file.
3. The driver no longer falls back to a filesystem “latest predictions.json” search for Supplement or EWoK. It now requires the per-target JSON to contain the recorded prediction path and verifies that path lies under the requested isolated target/column output root. This prevents stale or partial predictions from silently entering Supplement slicing or pristine collation after an interrupted evaluation.

### Patched static readiness checker

Patched:

- `experiments/archive/frontier_consolidation/scripts/static_postdelivery_readiness_check.py`

It now requires the dry-run inspect command to contain the frozen reinvest corpus paths, hashes, row/word counts, pass count, and `--verify-corpus-content`.

### Verification after patches

Reran only static/dry-run checks, not active run inspection:

```text
AST_OK inspect_compliant_retrain.py compliant_postdelivery_driver.py static_postdelivery_readiness_check.py
POSTDELIVERY_DRY_RUN bytealphatok_reinvest_seed43022
POSTDELIVERY_DRY_RUN complianttok_reinvest_seed43022
STATIC_POSTDELIVERY_READY
```

Updated artifacts:

- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/bytealphatok_reinvest_seed43022_postdelivery_driver.json`
- `experiments/archive/frontier_consolidation/data/compliant_postdelivery_driver/complianttok_reinvest_seed43022_postdelivery_driver.json`
- `experiments/archive/frontier_consolidation/data/postdelivery_static_readiness/static_postdelivery_readiness.json`
- `research/documents/frontier_consolidation/data/postdelivery_static_readiness/static_postdelivery_readiness.md`

The readiness Markdown now states that the post-delivery inspector requires frozen reinvest corpus provenance before evaluation.

## Current endpoint commands after lead provenance guard and contingency routes patch

### Byte-alphabet endpoint — priority

After byte-alphabet reinvest training completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target bytealphatok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/bytealphatok_reinvest_seed43022 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer_bytealphabet --expected-tokenizer-sha256 b2b317e655a96f2c14eb559ce9fa7573f180904d84819cacfb5f61174cf355cf --gpu 1
```

### spatial repair route status same-pool endpoint — evaluate if delivered and not delaying priority

After the original same-pool reinvest training completes:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compliant_postdelivery_driver.py --arm reinvest --target complianttok_reinvest_seed43022 --run-dir experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2 --expected-tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --expected-tokenizer-sha256 91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9 --gpu 0
```

Both commands now include the stricter corpus/provenance inspection internally; the command line does not need extra corpus arguments.

## Result-dependent next work

1. If a terminal retrain result is delivered, validate the completed result and run the corresponding evaluation driver, byte-alphabet first. Do not bypass the inspector even if `hf_model/chck_100M` exists.
2. If the driver stops at `POSTDELIVERY_STOP_INCOMPLETE_RETRAIN`, treat it as endpoint invalidation/engineering repair evidence, not model-quality evidence; relaunch only the unchanged frozen command if the failure is resource interruption.
3. If a legal endpoint scores above 41.8 by a robust margin, perform independent review of compliance, official-coordinate scoring, provenance, contamination safeguards, and reproducibility.
4. If the margin is tiny, decompose against the live leader vector and inspect whether the apparent advantage is carried by volatile SuperGLUE, newline-affected Supplement rows, or broad column improvements.
5. If no legal endpoint exceeds 41.8 after full official-compatible scoring, use the exact legal score vector to choose among the proposed successor routes. Do not start a new 100M run without a cheap real-training screen that can distinguish continue/change/stop for a specific route.
