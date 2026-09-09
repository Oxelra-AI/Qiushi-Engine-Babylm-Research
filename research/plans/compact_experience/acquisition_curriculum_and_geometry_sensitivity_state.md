# official geometry exposure and steps acquisition curriculum and geometry sensitivity preparation

## Scientific interpretation

1. Raw G3-vs-G0 official-geometry comparison at 100M words and batch256 is a whole-recipe comparison, not pure row-topology isolation. G3 has 2,809 optimizer updates while G0 has 2,442, so update count, LR-time, visible tokens, batch statistics, boundary frequency, truncation/padding, and WWM rounding all remain entangled.
2. G3 batch286/288 with `lr_total_steps=2515` is a sensitivity test, not complete isolation. It matches the clean-Qwen update-count coordinate more than the new G0 coordinate and changes batch/noise statistics.
3. Official-only geometry may improve BLiMP/equal7 while harming SuperGLUE or AoA. Clean-Qwen's competitiveness comes partly from SuperGLUE 70.31 and AoA 0.0. Any geometry route must prove it does not repeat the BLiMP-up/AoA-down pattern.
4. A high-value legal path remains corpus-only acquisition pacing on clean-Qwen: reorder early exposure using training-corpus statistics only, then evaluate frozen checkpoints with full nine columns. No official AoA/CDI words, curves, predictions, or scores may be used.

## Active geometry measurement already running

The geometry measurement had started with the following configuration:

```bash
bash experiments/archive/compact_experience/scripts/after_official_geometry_training.sh
```

It should produce:

- `data/official_geometry_audit/official_geometry_exposure_and_steps.json`
- `notes/official_geometry_exposure_and_steps.md`
- `data/official_geometry_trajectory_screen/`
- `data/official_geometry_trajectory_ranking.json`
- `notes/official_geometry_trajectory_ranking.md`
- `data/official_geometry_full_eval_targets.json`
- `notes/official_geometry_full_eval_targets.md`

The no-AoA trajectory screen evaluates only BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading. It does not use SuperGLUE or AoA for ranking.

## Geometry sensitivity assets prepared

New script:

- `scripts/launch_official_geometry_update_sensitivity.sh`

Modes:

- `g3_match_g0`: trains cap120 official geometry with batch295 and `lr_total_steps=2442`, approximating the G0 update count.
- `g0_match_g3`: trains official160 geometry with batch223 and `lr_total_steps=2809`, approximating the G3 update count.
- `both`: runs both in a two-H100 wave.

Registry extensions:

- `scripts/full_eval_official_geometry_candidates.py` now includes `official_cap120geom_b295_lr2442` and `official160_b223_lr2809`.
- `scripts/launch_official_geometry_trajectory_screen.sh` now knows these two target names.
- `scripts/prefill_official_geometry_eval_from_trajectory.py` and `scripts/prepare_official_geometry_full_eval_from_ranking.py` were extended to include these target families if they are later trained.

These sensitivity arms should only be launched if the current G3-vs-G0 batch256 ranking shows a valuable official-only geometry signal. They still do not provide perfect isolation because batch size changes gradient noise and per-update masking statistics.

## Corpus-only acquisition curriculum assets prepared

New materializer:

- `scripts/materialize_acquisition_curriculum_clean_qwen.py`

It builds exact clean-Qwen 10M/100M corpora with no official AoA/CDI use:

- A0 `A0_baseline_order`: exact clean qwen compliance and validity qwen_aligned pool and clean qwen compliance and validity matched shuffled pass-order convention; retraining control.
- A1 `A1_frequency_only`: early passes easy-to-hard by corpus-internal mean log word frequency.
- A2 `A2_freq_entropy_dispersion_spacing`: early passes by a fixed acquisition score using corpus frequency, rare fraction, word length, lexical entropy, source/example dispersion, burstiness, row length, and a spacing heuristic.
- A3 `A3_randomized_label_control`: preserves A2 source/length/generated-bin score distribution but randomizes rank labels inside strata.

Default `early_passes=2`; passes after early window use the clean qwen compliance and validity matched shuffled pass-order convention. All variants preserve the same row multiset and exact 100M word exposure.

Training/evaluation assets:

- `scripts/launch_acquisition_curriculum_training.sh`
- `scripts/launch_acquisition_curriculum_trajectory_screen.sh`
- `scripts/rank_acquisition_curriculum_trajectory.py`
- `scripts/full_eval_acquisition_curriculum_candidates.py`
- `scripts/prepare_acquisition_curriculum_full_eval_from_ranking.py`
- `scripts/prefill_acquisition_curriculum_eval_from_trajectory.py`
- `scripts/launch_acquisition_curriculum_full_eval_targets.sh`
- `scripts/run_acquisition_curriculum_full_eval_from_freeze.sh`
- `scripts/summarize_acquisition_curriculum_full_eval.py`

Compilation had not completed in this record; syntax validity was therefore unverified. The following checks remained pending:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
  experiments/archive/compact_experience/scripts/materialize_acquisition_curriculum_clean_qwen.py \
  experiments/archive/compact_experience/scripts/rank_acquisition_curriculum_trajectory.py \
  experiments/archive/compact_experience/scripts/full_eval_acquisition_curriculum_candidates.py \
  experiments/archive/compact_experience/scripts/prepare_acquisition_curriculum_full_eval_from_ranking.py \
  experiments/archive/compact_experience/scripts/prefill_acquisition_curriculum_eval_from_trajectory.py \
  experiments/archive/compact_experience/scripts/summarize_acquisition_curriculum_full_eval.py
bash -n experiments/archive/compact_experience/scripts/launch_acquisition_curriculum_training.sh
bash -n experiments/archive/compact_experience/scripts/launch_acquisition_curriculum_trajectory_screen.sh
bash -n experiments/archive/compact_experience/scripts/launch_acquisition_curriculum_full_eval_targets.sh
bash -n experiments/archive/compact_experience/scripts/run_acquisition_curriculum_full_eval_from_freeze.sh
```

## Priority after current geometry screen

1. Interpret the completed G3-vs-G0 no-AoA ranking and exposure audit.
2. If G3 is clearly stronger, run one or both geometry sensitivity arms before full nine-column claims.
3. If G3 is weak or its strength looks update-driven, do not spend more GPU on geometry alone; launch acquisition curriculum A0/A1/A2/A3 as the next high-value legal route.
4. If either geometry or acquisition produces a strong no-AoA candidate, freeze targets from no-AoA ranking and run corrected full eval. AoA and SuperGLUE stay final measurements only.
5. Only after a stronger substrate is identified should low-dose same-window second-view experiments be built.
