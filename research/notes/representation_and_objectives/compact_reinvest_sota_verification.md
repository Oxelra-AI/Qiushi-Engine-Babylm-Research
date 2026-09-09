# compact reinvest refined eval overlap audit compact_view_reinvest SOTA verification

CPU-only verification after the managed full official-compatible evaluation returned successfully. No GPU generation, training, or new evaluation was launched in this script.

## Full score

- Overall: **42.086786**; NLP average: 52.934439; Human-like average: 4.120000.
- Components: BLiMP 66.87, Supplement 63.28, EWoK 53.67, Entity 27.75, COMPS 51.97, SuperGLUE 71.3811, GlobalPIQA 35.62, Reading 8.24, AoA 0.
- Delta vs visible 41.8 leader: Overall +0.286786; components BLiMP -0.330, Supplement +7.270, EWoK -2.400, Entity -0.700, COMPS -1.600, SuperGLUE +1.591, GlobalPIQA -4.050, Reading +2.820, AoA +0.000.
- Delta vs COMPACT_EXPERIENCE clean-Qwen 41.3443: Overall +0.742495; NLP +0.886065; Human-like +0.240000.

## Provenance and official-compatible completeness

- Evaluation result: `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/compact_reinvest_full_eval_summary.json` and `experiments/archive/representation_and_objectives/data/compact_reinvest_full_eval/per_target/compact_view_reinvest.json`.
- Trained model: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`.
- Training file: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`; recomputed sha256 `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`; matches recorded: True.
- 10M corpus rows/words: 64740 rows / 10000000 words; exact 10M: True.
- Required AoA steps present: True; AoA rows 124640 across 19 steps; AoA score 0.0.
- Missing columns: []; unexpected missing/empty artifacts: 0.

## Exact phrase-overlap measurement

- Official eval unique 7-word strings collected: 2197977 from 122 files.
- 10M training corpus exact 7-word overlaps: 1875 unique strings in 1451 rows, 3147 occurrences.
- Sample overlaps are stored in the JSON for inspection; interpret common public phrases by source rather than treating every phrase match as contamination.

## Scientific reading

This is a real endpoint update: compact_view_reinvest exceeds the visible Strict-Small leader by +0.2868 Overall and COMPACT_EXPERIENCE clean-Qwen by +0.7425 under the same official-style arithmetic. The route is not the failed compact-core endpoint: reinvest holds AoA at 0.0, gains +3.48 EWoK, +1.99 Entity, +1.07 SuperGLUE, +0.48 Reading, +0.44 Supplement and +0.19 COMPS over clean-Qwen, while losing -1.00 GlobalPIQA. The largest remaining scientific deficit is practical/affordance reasoning, not the overall endpoint score. Seed43122 remains running and should be read when delivered for robustness and mechanism learning.

Machine-readable JSON: `experiments/archive/representation_and_objectives/data/reinvest_sota_verification/compact_reinvest_sota_verification.json`
