# seed43122 official comparison after delivery — seed43122 robustness decision preflight

This is a CPU-only preflight/addendum. The seed43122 full evaluation was still in progress; its partial outputs are not used here.

## Coordinate finding

The seed43122 full wrapper imports the COMPACT_EXPERIENCE full-eval runner and only overrides OUT_ROOT/PER_TARGET/TARGETS. It does not override the runner's WORKSPACE or STRICT path. The inherited runner therefore uses the INITIAL_MODEL_STUDIES strict checkout for zero-shot data and the COMPACT_EXPERIENCE AoA helper. The helper calls `load_eval(word_path, 20, False)`, so its AoA is the old 6,560-row subset, not the official 8,005-row coordinate.

Therefore the inherited full evaluation remains useful for non-EWoK zero-shot, Reading, and SuperGLUE, but seed43122 must use the official EWoK repair and official min_context=0 AoA repair before comparison with seed43022.

## Already established values now made explicit

- Seed43022 official Overall: `42.0331347900748`.
- Seed43122 fast equal7 delta: `-1.355714285714285`.
- Compact reinvest changed-block budget: `423520` words.
- Exact-overlap n-gram occurrences: original source `9`, generated rewrite `0`.
- AoA dry run: min_context=0 `8005` contexts; min_context=20 `6560` contexts.

## Collation prerequisites

Both the full evaluation and the replacement repair evaluation must complete before collation. The first repair attempt failed before scientific execution because its output directory already existed; it produced no evaluation evidence. Collation uses the completed result files:

```bash
python -B experiments/archive/representation_and_objectives/scripts/stage_pristine_collate_seed43122.py --pristine-ewok-predictions <ewok_predictions_path_from_official_ewok_reeval_seed43122.json> --aoa-dir experiments/archive/representation_and_objectives/data/official_aoa_min0_seed43122
```

Use the actual EWoK predictions path from the seed43122 official comparison after delivery EWoK repair JSON. The staged summary should be `experiments/archive/representation_and_objectives/data/pristine_collate_seed43122/pristine_collate_seed43122_summary.json`.

JSON: `experiments/archive/representation_and_objectives/data/seed43122_robustness_decision_preflight/seed43122_robustness_decision_preflight.json`
