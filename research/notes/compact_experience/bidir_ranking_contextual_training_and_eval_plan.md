# bidir ranking contextual training and eval plan — bidirectional ranking, contextual cap-120 training, and evaluation plan

## Bidirectional trajectory result (AoA-blind)

`data/bidir_trajectory_ranking.json` and `notes/bidir_trajectory_ranking.md` rank the completed bidirectional pair-order checkpoint ladders using only BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading. No AoA/CDI information and no SuperGLUE were used.

Key no-AoA observations:

- Best single bidirectional row: `bidir_seed43022/chck_75M`, equal7 = 43.170000.
  - BLiMP 66.56, Supplement 61.86, EWoK 51.53, Entity 24.97, COMPS 52.10, GlobalPIQA 36.605, Reading 8.565.
- Best clean-Qwen reference row in the same screen: `clean_qwen_seed43022/chck_100M`, equal7 = 43.112857.
  - BLiMP 66.84, Supplement 62.84, EWoK 50.19, Entity 25.76, COMPS 51.78, GlobalPIQA 36.62, Reading 7.76.
- Best dual-seed mean for bidirectional: `chck_90M`, mean equal7 = 43.073214.
- Clean-Qwen dual-seed means remain lower in this no-AoA surface, but the difference is not large enough by itself to close the 41.8 target.

Mechanistic interpretation: reversing half the within-pair order partly repairs the same-window tradeoff profile. The best bidirectional checkpoint improves EWoK, COMPS, and Reading relative to the clean-Qwen 100M reference, while losing Supplement and Entity and slightly lowering BLiMP. This supports the hypothesis that fixed original→rewrite directionality contributes to some representation/position tradeoff, but the gain is narrow and not yet a new main route.

Score geometry: with equal7 = 43.17, beating Overall 41.8 would require `SuperGLUE + AoA_leaderboard > 41.8*9 - 7*43.17 = 74.01`. If AoA remains 0, the needed SuperGLUE is ~74.0, well above the clean-Qwen SuperGLUE 70.31 and visible leader SuperGLUE 69.79. Therefore bidirectional full eval is scientifically useful as a mechanism check, but it should not block the contextual route that has higher chance to alter the broader score geometry.

## Contextual cap-120 status

The contextual one-pair cap-120 experiment was frozen and materialized earlier in this step:

- Metadata: `data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json`.
- Treatment corpus: `data/contextual_one_pair/cap120/training_corpora/qwen_context_onepair_cap120_100M.jsonl`.
- Matched official-only control: `data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_100M.jsonl`.
- Treatment 100M SHA256: `6c90cb644121f4747d5a901206b35d118cb3b6959d856a827f81191ff5903640`.
- Tokenization/truncation audit: `data/contextual_one_pair/cap120/contextual_cap120_tokenization_audit.json` showed pair-side truncation is rare (original side 1 row, rewrite side 12 rows; rewrite visible fraction mean 0.999922).

The matched cap-120 treatment/control pair had started training on both H100s; its completed training and evaluation measurements were pending in this record.

## Prepared next execution assets

These scripts have been written so there is no idle gap when the cap-120 training task finishes:

- `scripts/launch_contextual_cap120_trajectory_screen.sh` — no-AoA trajectory screen for treatment/control across `chck_10M` through `chck_100M`.
- `scripts/rank_contextual_cap120_trajectory.py` — ranks treatment/control and computes the recovery profile `R={BLiMP,EWoK,COMPS,Reading}` and the no-SuperGLUE preservation profile `P={Supplement,Entity}`.
- `scripts/full_eval_candidates.py` — corrected full nine-column wrapper for bidir ranking contextual training and eval plan bidirectional and contextual candidates.
- `scripts/prefill_bidir_eval_from_trajectory.py` — prefill bidirectional zero-shot/Reading columns from trajectory records for later SuperGLUE+AoA only full eval.
- `scripts/prefill_contextual_cap120_eval_from_trajectory.py` — prefill contextual cap-120 zero-shot/Reading columns from trajectory records for later SuperGLUE+AoA only full eval.
- `scripts/launch_full_eval_targets.sh` — parallel full-eval launcher for frozen bidir ranking contextual training and eval plan targets.

## Next interpretation rule

When cap-120 training finishes, immediately trajectory-screen treatment/control. Promote cap-120 treatment checkpoints for full eval if either:

1. the treatment equal7 is competitive with or above clean-Qwen seed43022 100M (43.112857), or
2. the treatment-control contrast shows real recovery in BLiMP/EWoK/COMPS/Reading while preserving Supplement and Entity enough to avoid pure redistribution.

Evaluate the matched control at the same checkpoint or at its own no-AoA optimum if the scientific question is treatment-control mechanism; evaluate treatment best if the question is leaderboard potential. AoA remains only a final measurement inside the frozen full-eval target list.

If cap-120 is promising, the next causal controls should be same-source-unrelated context and contextual original-duplicate, as recorded in `notes/independent_review_contextual_followup_controls.md`.
