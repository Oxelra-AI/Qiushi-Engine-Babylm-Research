# clean replication and ms direct pending clean preservation replication and exact `(M,S)` direct-comparison status

## Current scientific question

The central Stage III question is no longer whether dense clue suppression can move the model toward contextual evidence use. That has already been established mechanistically by the `(S,S)/(M,S)/(M,M)` family and replicated source/CDI readouts. The current load-bearing question is whether clean eval-mode parent-function preservation turns that movement into a more useful lawful BabyLM Strict-Small frontier candidate than the exact acquisition-only `(M,S)` policy, and whether the clean candidate replicates in the full repaired official coordinate.

## Complete clean seed replication

Clean preservation seed62064 was already complete in earlier analysis:

- Overall: `42.246412332209445`
- delta vs faithful repaired coherent86: `+0.2224443408943415`
- components: BLiMP `68.26`, Supplement `63.28`, EWoK `49.82`, Entity `29.40`, COMPS `52.16`, SuperGLUE `69.047710989885`, GlobalPIQA `40.05`, Reading `8.20`, AoA `0.0`

The completed clean preservation seed62065 SuperGLUE results were inspected and the guarded same-coordinate calculation was repeated. Clean seed62065 is now complete in the same coordinate:

- Overall: `42.23173113265801`
- delta vs faithful repaired coherent86: `+0.20776314134290885`
- delta vs clean seed62064: `-0.014681199551432655`
- components: BLiMP `68.22`, Supplement `63.29`, EWoK `49.72`, Entity `29.45`, COMPS `52.14`, SuperGLUE `69.02058019392214`, GlobalPIQA `40.05`, Reading `8.195`, AoA `0.0`

The updated guarded table is:

- `experiments/archive/functional_learning/data/same_coordinate_all_candidates_after_clean65_superglue/same_coordinate_all_candidates.json`
- `research/documents/functional_learning/data/same_coordinate_all_candidates_after_clean65_superglue/same_coordinate_all_candidates.md`

The replication is scientifically meaningful because seed65 preserves the same trade shape as seed64 under full official-sized evaluation and measured AoA: Entity and GlobalPIQA stay above coherent86, SuperGLUE remains above coherent86, BLiMP/Supplement/EWoK remain below coherent86, and the overall improvement remains positive even under GlobalPIQA-zero sensitivity (`+0.04276314134291089`). This supports a real repeated acquisition-retention trade improvement, not a single-seed accidental score.

## SuperGLUE direct comparison to exact `(M,S)`

Exact acquisition-only `(M,S)` SuperGLUE completed in zero reading and superglue item evidence and was incorporated in clean replication and ms direct pending. The direct SuperGLUE profile is:

- `research/documents/functional_learning/data/superglue_ms_direct_profile/superglue_ms_direct_profile.md`
- `research/documents/functional_learning/data/superglue_all_completed_profile/superglue_all_completed_profile.md`

Key SuperGLUE numbers:

- coherent86: `68.94571192183594`
- dense `(M,M)` seed62064: `68.5318200743064`
- dense `(M,M)` seed62065: `68.80580432073164`
- exact acquisition-only `(M,S)` seed62064: `68.88784158902877`
- clean preservation seed62064: `69.047710989885`
- clean preservation seed62065: `69.02058019392214`

This changes the causal interpretation. Earlier clean-vs-dense `(M,M)` comparisons overstated the preservation-specific SuperGLUE increment because exact `(M,S)` acquisition alone is much stronger than `(M,M)` on SuperGLUE. The direct preservation increment on SuperGLUE is smaller but replicated:

- clean seed62064 minus exact `(M,S)`: `+0.15986940085623758`
- clean seed62065 minus exact `(M,S)`: `+0.13273860489337608`

Task-level structure: clean’s repeatable gain over `(M,S)` comes mainly from MultiRC, MRPC F1, QQP F1, and MNLI; BoolQ/RTE/WSC are equal or slightly seed-dependent. Thus parent-function preservation appears to recover supervised-transfer behavior relative to the acquisition-only policy, but the size is modest and the direct Overall comparison depends on the pending exact `(M,S)` zero-shot/Reading payload.

## Pending exact `(M,S)` zero-shot/Reading

The missing official zero-shot/Reading complement had started and was pending at the time of this note:

- command target: `densemask_sparselabel_seed62064_u0080_zero_reading`
- repaired model: `experiments/archive/functional_learning/data/automodel_repair_candidates/repaired_densemask_sparselabel_seed62064_u0080`
- output root: `experiments/archive/functional_learning/data/repaired_densemask_sparselabel_zero_reading`
- columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA_parallel, GlobalPIQA_nonparallel, Reading

Before it finishes, the direct Overall comparison must remain open. Threshold arithmetic is stored at:

- `research/documents/functional_learning/data/ms_zero_reading_thresholds/ms_zero_reading_thresholds.md`

Because exact `(M,S)` SuperGLUE is `0.15986940085623758` below clean seed62064, if `(M,S)` zero-shot/Reading exactly matched clean seed62064, clean would still lead by `0.017763266761804175` Overall points. To equal clean seed62064 Overall, `(M,S)` must have a seven-component zero/Reading sum `0.15986940085622336` above clean seed62064’s zero/Reading sum. The actual question is empirical: if `(M,S)` improves enough on BLiMP/Supplement/EWoK or Reading relative to clean, it may erase the clean SuperGLUE advantage; if it matches dense/clean trade shape, clean remains better.

## Interpretation to preserve

The current evidence supports clean eval-mode parent-function preservation as a repeated lawful positive practical intervention over coherent86, but it is best understood as improving the acquisition-retention trade rather than universally improving all abilities. The strongest current mechanism statement is restrained:

- dense effective-input clue suppression acquires contextual evidence dependence;
- exact `(M,S)` acquisition alone explains more of the SuperGLUE recovery than dense `(M,M)` controls implied;
- deterministic parent-function preservation adds a repeatable, modest supervised-transfer repair beyond `(M,S)` and strongly bounds ordinary-function/CDI drift in mechanism readouts;
- whether it adds direct Overall value beyond `(M,S)` awaits the running `(M,S)` zero-shot/Reading run.

The direct six-record comparison remained incomplete until the missing zero/Reading payload was validated and included in the guarded same-coordinate table.
