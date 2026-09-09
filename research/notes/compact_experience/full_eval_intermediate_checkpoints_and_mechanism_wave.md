# bidir ranking contextual training and eval plan — intermediate checkpoint full eval and mechanism-wave launch

## Intermediate checkpoint full eval result

Full evaluation completed for the two frozen AoA-blind candidates:

- `clean_qwen_seed43022_95M` (`hf_model/chck_95M`)
- `clean_qwen_seed43122_60M` (`hf_model/chck_60M`)

The launcher's printed summary used wrong field names (`overall`/`task_scores` null), but each per-target JSON contains finalized corrected `official_overall`. A fixed summarizer `scripts/score_candidate_overall.py` writes `data/full_eval_candidates/candidate_overall_summary.json`.

### Corrected nine-column scores

`clean_qwen_seed43022_95M`:

- BLiMP 66.79
- Supplement 62.54
- EWoK 50.16
- Entity 25.90
- COMPS 51.81
- GlobalPIQA 36.62
- SuperGLUE 70.16592170868896
- Reading 7.8100000000000005
- AoA raw 0.0 / leaderboard 0.0
- **Overall 41.31065796763211**

`clean_qwen_seed43122_60M`:

- BLiMP 64.58
- Supplement 62.85
- EWoK 50.34
- Entity 25.99
- COMPS 51.20
- GlobalPIQA 39.58
- SuperGLUE 68.9327786907567
- Reading 7.154999999999999
- AoA raw 0.0 / leaderboard 0.0
- **Overall 41.18086429897297**

Current clean-Qwen best remains seed43022 `chck_100M` Overall **41.34429066479573**. The hidden-intermediate-checkpoint hypothesis is therefore closed for the highest-probability nodes: `chck_95M` is essentially tied but lower by -0.0336, and second-seed `chck_60M` is lower by -0.1634. Neither approaches the visible 41.8 leader closely enough to justify more checkpoint mining as the main route.

## Mechanism prep now complete

### Bidirectional pair-order arm

Materialized by `scripts/materialize_bidirectional_pair_order.py`:

- metadata: `data/bidirectional_pair_order/bidirectional_pair_order_metadata.json`
- 10M/100M word totals exact
- exact row identity sequence and 100M pass order preserved
- pair words 1,656,800 (16.568%)
- direction counts: original→rewrite 18,876; rewrite→original 18,718
- direction words: 832,181 vs 824,619
- changed pair rows: 10,524 / 12,236
- sha256_100M: `1341739591673f97e38efc28b45f21bbd52eabe4982c21425a6030ef8cd19c66`

This is a clean structure-only test of whether one-direction pair geometry causes some COMPS/BLiMP/EWoK/Reading cost while preserving same-window generated second-view correspondence.

### Contextual one-pair feasibility

Audit `data/contextual_one_pair_audit/contextual_one_pair_feasibility.json` located all 37,594 selected pairs in their official 160-word rows by sentence splitting (located_fraction=1.0):

- cap 80: mean row words 80.39, mean context words 36.32
- cap 120: mean row words 120.0, mean context words 75.93
- cap 160: mean row words 160.0, mean context words 115.93

This makes a true one-pair official-context arm feasible; it can test whether the current multi-pair packed rows create an artificial semantic-restart genre that harms natural reading/relation columns.

## Running task

The bidirectional pair-order comparison had started on both H100s:

- `training/runs/qwen_bidirectional_pair_order_16k_seed43022`
- `training/runs/qwen_bidirectional_pair_order_16k_seed43122`

Expected output: two 100M word, 100-checkpoint DeBERTa-v2 8×480 WWM baseline16k runs matching the clean qwen compliance and validity recipe except pair-direction structure.

## Next after training

1. Inspect the completed bidirectional pair-order measurements when available.
2. Immediately run the no-AoA trajectory screen on both bidirectional seeds, reusing `scripts/eval_checkpoint_trajectory_fullzeroshot.py` or a new launcher over the same 10M–100M grid.
3. Promote the best bidirectional endpoints for corrected full eval. Compare to clean-Qwen seed-matched checkpoints:
   - if bidirectional improves COMPS/Reading/BLiMP/EWoK without losing Supplement/Entity/SuperGLUE, direction/position bias is a real mechanism cost;
   - if bidirectional loses Entity/SuperGLUE, fixed original→rewrite may be part of the useful denoising cue;
   - if no change, move to one-pair official-context materialization as the next stronger topology repair.
4. Use the contextual feasibility audit to build the one-pair contextual materializer if bidirectional does not produce a clear SOTA-capable lift.

## Efficiency note

Completed measurements for the next mechanism comparison were pending. Intermediate checkpoint mining no longer deserves main-route GPU time unless a later mechanism arm creates a new high plateau; the best route is topology/geometry repair of the validated same-window second-view principle.
