# Trajectory-based candidate selection and full-evaluation plan

## Recorded evidence

All four dual-seed grids (clean-Qwen seed43022/43122, devcurr seed43022/43122) cover `chck_10M..chck_100M`. Ranked AoA-blind in `data/trajectory_candidate_ranking.json` / `notes/trajectory_screen_candidate_ranking.md` (52/52 complete rows).

### Key scientific reads (equal7_full_eval, no SuperGLUE/AoA)

- Best clean-Qwen rows: seed43022 `chck_100M` 43.1129, seed43122 `chck_60M` 43.0993, seed43022 `chck_95M` 43.0900. These three are effectively tied within screen noise.
- Best devcurr rows: seed43122 `chck_80M` 42.5371, seed43022 `chck_85M` 42.3386. Devcurr is below clean-Qwen almost everywhere.
- Paired seed means: clean-Qwen `chck_60M` 42.9279, `chck_95M` 42.8607, `chck_100M` 42.7782 are the top three; all devcurr paired means are lower.

**Conclusion:** the AoA-safe source-block developmental first-pass schedule (aoa safety audit and route) is not a SOTA route on the seven-column surface. Its value is now only as a developmental-dynamics/AoA-final measurement, not a late-checkpoint winner.

## Frozen full-eval candidates (AoA-blind selection)

Two candidates were frozen because they match/approach the 100M endpoint on the seven-column screen but were never measured on SuperGLUE/AoA — the two columns that produced clean-Qwen's +2.40 Overall lead:

1. `clean_qwen_seed43022_95M` (trajectory equal7 43.0900)
2. `clean_qwen_seed43122_60M` (trajectory equal7 43.0993)

Their seven zero-shot/Reading columns were prefilled from the trajectory per-checkpoint records via `scripts/prefill_candidate_eval_from_trajectory.py` into `data/full_eval_candidates/per_target/`; the remaining full-evaluation measurements are SuperGLUE + AoA.

Baseline for comparison (corrected AoA units):
- `qwen_clean_aligned` (seed43022 chck_100M) Overall **41.3443** (BLiMP 66.84, Supplement 62.84, EWoK 50.19, Entity 25.76, COMPS 51.78, GlobalPIQA 36.62, SuperGLUE 70.3086, Reading 7.76, AoA 0.0).
- `qwen_clean_aligned_seed43122` (chck_100M) Overall 40.6501.
- Visible leader 41.8.

## Measurements and constructions not yet reported here

SuperGLUE + AoA with corrected scoring were pending for `clean_qwen_seed43022_95M` and `clean_qwen_seed43122_60M` when this note was written.

The proposed follow-up constructions were the bidirectional pair-order corpus (`data/bidirectional_pair_order/`) and the contextual one-pair feasibility audit (`data/contextual_one_pair_audit/`). Their completion is not established by this note.

## Decision rules after full evaluation

1. Read `data/full_eval_candidates/per_target/clean_qwen_seed43022_95M.json` and `clean_qwen_seed43122_60M.json`; compute corrected nine-column Overall from the per-target measurements.
2. If either exceeds 41.3443 and approaches/exceeds 41.8, freeze that exact checkpoint path and preserve all per-target outputs; then run a matched second-seed/other-seed confirmation of the same checkpoint index.
3. If neither exceeds the current best, consider the next AoA-blind candidates (e.g. clean-Qwen `chck_60M` seed43022, `chck_70M` seed43122) or the bidirectional dual-seed mechanism arm.
4. If the contextual audit `located_fraction` is high, build the one-pair contextual materializer as the next mechanism arm to repair Reading/BLiMP/EWoK/COMPS while keeping same-window correspondence.
