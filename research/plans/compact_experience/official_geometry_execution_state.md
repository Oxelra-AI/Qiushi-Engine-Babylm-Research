# official geometry exposure and steps official-geometry execution state

## Contextual cap-120 summary repair

Repaired `scripts/summarize_contextual_cap120_full_eval.py` to read the actual per-target full-eval schema:

- corrected scores: `official_overall.scores`
- corrected Overall: `official_overall.Overall`
- legacy fallbacks kept for older files
- clean reference now reads `data/full_eval/full_eval_summary.json` under its `targets.qwen_clean_aligned` block

Regenerated:

- `data/contextual_cap120_full_eval_summary.json`
- `notes/contextual_cap120_full_eval_interpretation.md`

Key confirmed measurements:

- `context_cap120_treat_seed43022_80M`: Overall `40.92248881040306`; equal7 `42.66214285714286`; SuperGLUE `69.66739929362754`; AoA `0.0`; gap to visible 41.8 leader `-0.8775111895969374`; gap to clean-Qwen `-0.42180185439266893`.
- `context_cap120_control_seed43022_80M`: Overall `38.77450797674098`; equal7 `42.99285714285714`; SuperGLUE `68.87558637973169`; AoA leaderboard `-20.855014589062833`; no-AoA treatment-control equal7 delta `-0.3307142857142793`.
- Treatment corrected Overall beats the matched control only through the control's negative AoA final measurement; the no-AoA and clean-reference evidence close high-dose contextual Qwen insertion as the immediate frontier route.

## Active official-only geometry training

Launched background task:

- command: `bash experiments/archive/compact_experience/scripts/launch_official_geometry_step_control_training.sh batch256_pair`
- purpose: train official160 G0 and cap120 official-geometry G3 under matched seed/init/tokenizer/WWM recipe, batch256, to separate native row/segmentation geometry from Qwen insertion.
- expected run dirs:
  - `training/runs/official160_b256_16k_seed43022`
  - `training/runs/official_cap120geom_b256_16k_seed43022`
- expected data dirs:
  - `data/official_geometry/official160/`
  - existing cap120 official geometry corpus under `data/contextual_one_pair/cap120/`

The scientific question is whether the strong cap120 official-control BLiMP/equal7 trajectory from contextual cap120 full eval interpretation is a reproducible official-only geometry/segmentation effect, or whether it depended on a hidden one-off interaction in the cap120 training run. This is not a final-result route by itself; it is the load-bearing substrate test before adding any new lower-dose same-window second-view mechanism.

## Script assets checked or repaired in official geometry exposure and steps

- Repaired `scripts/launch_official_geometry_full_eval_targets.sh` final summary block to read `official_overall.Overall` and `official_overall.scores`, preventing another null-score summary.
- Inspected `scripts/launch_official_geometry_trajectory_screen.sh`: it evaluates only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading, with no AoA/SuperGLUE ranking.
- Inspected `scripts/rank_official_geometry_trajectory.py`: it ranks only the same no-AoA seven columns and includes clean-Qwen and cap120 references.
- Inspected `scripts/prefill_official_geometry_eval_from_trajectory.py` and `scripts/full_eval_official_geometry_candidates.py`: frozen targets are prefilled from no-AoA trajectory records; corrected full overall eval evaluator computes final nine-column Overall.
- Wrote `scripts/prepare_official_geometry_full_eval_from_ranking.py`: after a no-AoA ranking exists, freezes each official-geometry family's best checkpoint and optional global top rows into `data/official_geometry_full_eval_targets.json` and `notes/official_geometry_full_eval_targets.md`.
- Wrote `scripts/summarize_official_geometry_full_eval.py`: summarizes official-geometry per-target JSONs using canonical `official_overall.Overall`/`scores`, compares against clean-Qwen and the visible 41.8 leader, and writes `data/official_geometry_full_eval_summary.json` plus `notes/39_official_geometry_full_eval_interpretation.md`.

Compilation of the two new official-geometry exposure/update-count helpers had not completed in this record; their syntax validity remained unverified.

## Immediate next measurement after training completes

1. Confirm both geometry runs have `scientific_metrics.json`, `word_exposure=100000000`, complete checkpoint ladders, and expected step counts.
2. Run no-AoA trajectory screen:

```bash
bash experiments/archive/compact_experience/scripts/launch_official_geometry_trajectory_screen.sh official160_b256 official_cap120geom_b256
python experiments/archive/compact_experience/scripts/rank_official_geometry_trajectory.py
```

3. If `official_cap120geom_b256` clearly exceeds `official160_b256` and clean-Qwen equal7/recovery profile, train one matched-update G3 arm using:

```bash
bash experiments/archive/compact_experience/scripts/launch_official_geometry_step_control_training.sh g3_b286
# or g3_b288 if batch divisibility/step count looks better from the actual metrics
```

with `lr_total_steps=2515` to test whether geometry survives removal of the 2,809-update/LR-time advantage.

4. Freeze full-eval targets only from the no-AoA ranking. Then prefill/run corrected full eval and summarize using the repaired canonical-field scripts. AoA remains final measurement only, not a route selector.

5. Only after a strong official-only geometry is established should the research build a low-dose same-window second-view arm on that geometry, with duplication and shuffled-rewrite controls.
