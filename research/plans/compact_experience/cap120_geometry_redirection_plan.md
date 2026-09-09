# contextual cap120 full eval interpretation cap-120 evidence and geometry redirection plan

## Evidence already established before frozen full eval

Source files:
- `data/contextual_cap120_trajectory_ranking.json`
- `notes/contextual_cap120_trajectory_ranking.md`
- `data/contextual_token_source_audit/cap120_token_source_exposure_audit.json`

No-AoA screen used only BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading. AoA/SuperGLUE were not used for checkpoint selection.

### Ranking

- Clean-Qwen reference best: `clean_qwen_seed43022/chck_100M`, equal7 = 43.112857, R={BLiMP,EWoK,COMPS,Reading}=44.1425, P_no_sglue={Supplement,Entity}=44.30.
- Cap-120 treatment best: `context_cap120_treat_seed43022/chck_80M`, equal7 = 42.662143, R=44.535, P_no_sglue=41.945.
- Cap-120 official length-matched control best: `context_cap120_control_seed43022/chck_80M`, equal7 = 42.992857, R=45.1325, P_no_sglue=41.665.

At late checkpoints the cap-120 treatment loses to its own official control:
- `chck_80M`: treatment - control = -0.330714 equal7; BLiMP -2.13, EWoK +0.76, Supplement +1.63, Entity -1.07, COMPS -0.69, GlobalPIQA -0.485, Reading -0.33.
- `chck_100M`: treatment - control = -0.637143 equal7; BLiMP -1.99, EWoK +1.01, Supplement +1.40, Entity -1.64, COMPS -0.81, GlobalPIQA -2.00, Reading -0.43.

Early treatment gains exist (`chck_20M` +1.022 equal7; `chck_60M` +0.782), but absolute scores remain below the clean frontier. The scientific signature is not a new frontier model: it is a strong official-row-geometry control plus a Qwen insertion that helps Supplement/EWoK but damages BLiMP/COMPS/Entity/GlobalPIQA at scale.

### Token/source exposure audit

Exact trainer rule: `add_special_tokens=False`, right truncation, max seq 256.

Per 10M pool:
- Treatment visible tokens = 14,446,994; control visible tokens = 14,637,292. Treatment has ~190k fewer visible tokens and more over-256 rows (18.14% vs 14.35%).
- Pair visibility is essentially intact: original mean visible fraction 0.999999; rewrite mean 0.999933; pair mean 0.999967; only 9 pair rows have any pair-side truncation. So treatment loss is not caused by pair being hidden.
- Pair rows come mainly from Gutenberg (1.647M words), SimpleWiki (1.299M), OpenSubtitles (0.873M), BNC spoken (0.420M), CHILDES (0.266M). There are 25,486 unique selected examples for 37,594 pairs; 9,010 examples contribute more than one pair, max 7.

### New confound: number of optimizer updates per 100M words

The contextual cap-120 corpora have 71,898 rows/pool and train for 2,809 steps at batch256. clean qwen compliance and validity clean-Qwen and matched official controls trained for 2,515 steps at batch256. Original 160-word official pools have about 62,500 rows/pool and would train for roughly 2,442 steps. Thus cap-120 official control's BLiMP lift may involve not only row geometry but also more parameter-update/LR-time per word. Any geometry route must separate row topology from update count.

## Frozen full eval now running

Command: `PYTHONDONTWRITEBYTECODE=1 bash experiments/archive/compact_experience/scripts/run_contextual_cap120_full_eval_and_summarize.sh`
Expected frozen targets: cap-120 treatment `chck_80M` and control `chck_80M` (possibly plus prep-script selected variants if same checkpoint). AoA enters only inside this corrected full eval as final measurement.

## Prepared scripts in this step

- `scripts/summarize_contextual_cap120_full_eval.py`
- `scripts/run_contextual_cap120_full_eval_and_summarize.sh`
- `scripts/audit_contextual_token_source_exposure.py`
- `scripts/materialize_contextual_shuffled_rewrite_control.py`
- `scripts/launch_contextual_shuffled_rewrite_control_training.sh`
- `scripts/launch_contextual_cap120_seed43122_training.sh`
- `scripts/launch_contextual_cap_variants_training.sh`
- `scripts/materialize_official160_geometry_control.py`
- `scripts/train_single_custom_batch.py`
- `scripts/launch_official_geometry_step_control_training.sh`
- `scripts/launch_official_geometry_trajectory_screen.sh`
- `scripts/rank_official_geometry_trajectory.py`

## Recommended next action after full eval result

If full eval unexpectedly exceeds or approaches 41.8 with SuperGLUE/AoA support, run seed43122 replication and mechanistic controls.

If, as no-AoA evidence suggests, cap-120 treatment remains below clean-Qwen/control, do not spend the next GPU wave on cap80/cap160 or high-dose cap-120 controls. The stronger route is official row geometry:

1. Materialize official160 G0 with `materialize_official160_geometry_control.py`.
2. Train exact G0 vs G3 at batch256 using `launch_official_geometry_step_control_training.sh batch256_pair` to test whether the cap120 official-control BLiMP/equal7 lift is reproducible against a matched new official160 run.
3. If G3 remains strong, train `g3_b286` or `g3_b288` with `lr_total_steps=2515` to test whether its advantage survives approximate update-count/LR-time matching.
4. Screen with `launch_official_geometry_trajectory_screen.sh ...` and rank with `rank_official_geometry_trajectory.py`.
5. Only after identifying the best official-only geometry, add a much lower-dose same-window second-view arm (around 5% rewrite words) with duplication/shuffled controls rather than reusing the 16.568% high-dose cap-120 insertion.
