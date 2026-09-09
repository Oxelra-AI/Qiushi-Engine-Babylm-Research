# Strict-Small-tokenizer retrain and visibility audit

## State when this note was written

The tokenizer provenance was corrected in an earlier analysis: the inherited `baseline16k` tokenizer came from the Strict-100M setting and is not a compliant Strict-Small representation. The inherited-tokenizer `compact_view_reinvest` seed43022 result (official-coordinate Overall 42.0331347900748) is preserved as mechanism evidence, not as a fully compliant submission endpoint.

Two corrected-tokenizer retrains were in progress:

- Seed43022, GPU0: `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022`
- Seed43122, GPU1: `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122`

Both use the unchanged 100M stream `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl` with SHA-256 `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, same data order seed 43, same DeBERTa-v2 8×480 recipe, and only the tokenizer changed.

At the latest measurement recorded here, both runs had reached 20,840,785 cumulative words, with checkpoints through approximately 20M/21M and losses of approximately 3.75 (seed43022) and 3.61 (seed43122). These are intermediate measurements, not endpoint evidence; the full 100M-word runs and their official-protocol evaluations determine the final comparison.

## Corrected tokenizer identity

Strict-Small tokenizer path:

`experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_retrain/strictsmall_compact_reinvest_16k_tokenizer`

Tokenizer JSON SHA-256:

`4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738`

The chck_1M checkpoint tokenizer in the new seed43022 run already matched this SHA exactly, confirming tokenizer propagation into saved checkpoints.

## Evaluation infrastructure prepared and validated

New scripts:

- Full-eval wrappers:
  - `experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43022.py`
  - `experiments/archive/representation_and_objectives/training/scripts/full_eval_strictsmalltok_seed43122.py`
- Generic official-coordinate replacements:
  - `experiments/archive/representation_and_objectives/training/scripts/official_ewok_reeval_758984f6.py`
  - `experiments/archive/representation_and_objectives/training/scripts/official_aoa_min0.py`
  - `experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py`
- One-command post-training controller:
  - `experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py`

The generic pristine-collation script was validated against the already completed inherited-tokenizer seed43122 endpoint. It reproduced fast full seed43122 reconciliation exactly: Overall `41.24823958912208`, collated SHA-256 `2fb044ef382dfdabc1d77952cc17c8b4febd421c2bd412f7193d94ff087ea3d4`, `collate_returncode=0`, `null_keys=[]`, EWoK total 7,618, AoA row_count_values `[8005]`. Validation output:

`experiments/archive/representation_and_objectives/data/generic_collate_validation_seed43122/pristine_collate_validation_old_seed43122_summary.json`

Dry-runs of the generic EWoK and AoA scripts confirmed they use the corrected official coordinate: EWoK total 7,618 and AoA min_context=0 loads 504 CDI words / 8,005 contexts, while min_context=20 would load the old 328 / 6,560 subset.

## Commands after each retrain finishes

After a run has `chck_100M`, the complete AoA ladder `chck_1M..chck_9M` and `chck_10M..chck_100M`, and checkpoint tokenizer SHA `4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738`, run the evaluator for that seed. Evaluate each seed separately using its own checkpoints and output directory.

Seed43022:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43022 --gpu 0
```

Seed43122:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43122 --gpu 1
```

The controller intentionally runs the inherited full evaluator only for columns needed from it: BLiMP, Supplement, Entity, COMPS, GlobalPIQA_parallel, GlobalPIQA_nonparallel, Reading, and SuperGLUE. It excludes EWoK and AoA there because those are replaced by current-pristine 7,618-row EWoK and official min_context=0 AoA before collation.

Expected final collation outputs:

- seed43022: `experiments/archive/representation_and_objectives/data/strictsmalltok_seed43022_pristine_collate/pristine_collate_strictsmalltok_seed43022_summary.json`
- seed43122: `experiments/archive/representation_and_objectives/data/strictsmalltok_seed43122_pristine_collate/pristine_collate_strictsmalltok_seed43122_summary.json`

## Representation-geometry audit

Input-geometry audit output:

`experiments/archive/representation_and_objectives/data/strictsmall_tokenizer_visibility_audit/strictsmall_tokenizer_visibility_audit.json`

Note:

`research/notes/representation_and_objectives/strictsmall_tokenizer_visibility_audit.md`

Main numbers under the actual trainer interface (`add_special_tokens=False`, `max_length=256`):

- Full 10M pool mean tokens/word: inherited tokenizer 1.475552 → Strict-Small tokenizer 1.463215 (delta −0.012337).
- Full 10M rows over 256 tokens: 15,883 → 15,143 (delta −740).
- Changed compact block mean tokens/word: 1.462849 → 1.436011 (delta −0.026839).
- Changed compact block rows over 256 tokens: 78 → 54 (delta −24).
- Fully visible source+rewrite compact pairs: 12,074/12,155 → 12,098/12,155 (rate 0.993336 → 0.995311; delta +24 pairs).
- Source-visible pairs: 12,142 → 12,147 (delta +5).

Scientific reading: the compliant 10M-trained tokenizer does not hide the compact views more than the inherited tokenizer. It slightly improves token density and pair visibility. Therefore, if final corrected-tokenizer scores fall, the explanation is unlikely to be simple seq256 truncation of compact rewrites; if they survive, the representation repair is not only compliant but also compatible with the compact-view mechanism.

## Do not do yet

- Do not upload or package the inherited-tokenizer 42.0331 endpoint as a compliant Strict-Small submission.
- Do not launch a third seed, adjacency-break run, GlobalPIQA repair, or tokenizer/recipe alteration before the two corrected-tokenizer official-coordinate scores are known.
- Do not use old EWoK/AoA scalar values from the inherited evaluator; final interpretation must use the pristine collation path above.
