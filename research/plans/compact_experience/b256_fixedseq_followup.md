# masking curriculum 100M eval follow-up — b256 fixed-seq 100M masking-curriculum screen

## Background task
- Label: `masking curriculum 100M eval b256 fixed-seq 100M masking-curriculum screen on 2xH100`
- Launch script: `experiments/archive/compact_experience/scripts/launch_100M_b256_fixedseq_screen.sh`
- Status in masking curriculum 100M eval: running; wave 1 launched at 2026-08-24 10:47:53 UTC:
  - GPU0: `wwm_fixed_100M_b256_seq256_seed43`
  - GPU1: `wwm_to_token_100M_b256_seq256_seed43`
- Wave 2 starts after wave 1 if both complete:
  - GPU0: `amlm_hard_100M_b256_seq256_seed43`
  - GPU1: `amlm_hard_switch_100M_b256_seq256_seed43`

## Why this screen exists
masking curriculum 100M eval already completed a regime-correct 100M masking-curriculum screen with leader-style sequence schedule `0.0:64,0.5:128,0.8:256` and batch64. That screen showed fixed WWM had the best weighted fast proxy at `chck_100M`, while WWM→token and AMLM arms gained BLiMP/GPIQA but lost Supplement/Reading/Entity enough to lose on the proxy. However, inherited INITIAL_MODEL_STUDIES best coordinate used a stronger baseline recipe: fixed sequence length 256 and batch256. The b256 fixed-seq screen tests whether the negative switch result persists under the inherited baseline regime, instead of over-interpreting a weaker recipe.

## Exact design
Held fixed:
- official Strict-Small 10M pool
- 100M word exposure = 10 epochs
- fixed sequence length 256 (no sequence-length schedule)
- batch size 256
- DeBERTa-v2 8x480, baseline16k tokenizer from earlier analysis local checkpoint
- AdamW LR 1e-3, weight decay 0.01, warmup 0.05
- seed43, no DataLoader workers
- checkpoint every 10M words

Arms:
- `wwm_fixed`: WWM 0.15 all 100M
- `wwm_to_token`: WWM 0.15 through 70M, token masking 0.15 from 70M to 100M
- `amlm_hard`: WWM, mask probability decay 0.30 -> 0.15 plus AMLM hard-token weighting
- `amlm_hard_switch`: AMLM-hard plus WWM→token switch at 70M

## Immediate earlier analysis actions
1. Inspect the completed b256/fixed-sequence measurements when available.
2. Verify all four expected run dirs:
   - `experiments/archive/compact_experience/training/runs/wwm_fixed_100M_b256_seq256_seed43`
   - `experiments/archive/compact_experience/training/runs/wwm_to_token_100M_b256_seq256_seed43`
   - `experiments/archive/compact_experience/training/runs/amlm_hard_100M_b256_seq256_seed43`
   - `experiments/archive/compact_experience/training/runs/amlm_hard_switch_100M_b256_seq256_seed43`
3. For each run, check:
   - `scientific_metrics.json` exists and `word_exposure == 100000000`
   - `training_log.jsonl` exists and has roughly 2442 lines (batch256, 100M / 40960 words per step)
   - `dynamics_traces.jsonl` has 10 checkpoint records
   - `hf_model/chck_100M` exists with `model.safetensors`, `config.json`, `tokenizer.json`, tokenizer configs
   - stdout/stderr logs show return code 0 from the launcher
4. Adapt `experiments/archive/compact_experience/scripts/eval_curriculum_100M_fast.py` to include these `step011_*_b256_seq256` run dirs (or copy it as a earlier analysis evaluator) and evaluate all four `chck_100M` checkpoints on the same fast columns.
5. Compare b256/fixed-seq results to:
   - masking curriculum 100M eval seq-scheduled b64 result: `experiments/archive/compact_experience/data/curriculum_100M_eval/curriculum_100M_eval_summary.json`
   - Inherited INITIAL_MODEL_STUDIES fixed-seq/b256 WWM fast recheck: `experiments/archive/initial_model_studies/data/current_best_direct_checkpoint_recheck.json`
   - INITIAL_MODEL_STUDIES full nine-column coordinate: `experiments/archive/initial_model_studies/data/current_best_internal_coordinate.json`
6. If b256/fixed-seq WWM is close to inherited fast scores, this run calibrates the new trainer against INITIAL_MODEL_STUDIES. If WWM→token or AMLM improves specific columns without Supplement collapse under b256/fixed-seq, evaluate `chck_70M`, `chck_80M`, `chck_100M` trajectory for the promising arm. If fixed WWM remains best, stop the simple prediction-curriculum route and route construction toward a conditional granularity mechanism or a different high-value route.

## masking curriculum 100M eval completed evidence to keep in mind
- `research/notes/compact_experience/masking_curriculum_100M_eval.md`: final 100M seq-scheduled scores.
- `research/notes/compact_experience/masking_curriculum_trajectory_mechanism.md`: trajectory and mechanism interpretation.
- The seq-scheduled screen showed the switch effect begins after 70M and creates a clear trade-off: BLiMP/GPIQA improve but Supplement/Entity/Reading decline. This is the pattern to test under the inherited b256/fixed-seq recipe.
