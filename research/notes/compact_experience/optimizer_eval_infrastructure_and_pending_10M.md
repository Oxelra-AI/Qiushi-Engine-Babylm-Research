# highlr 10M training completion — optimizer × paired-data interaction evaluation infrastructure

## Research purpose

The experiment specifies a matched high-learning-rate optimizer × corpus experiment at LR 0.007, beta2 0.98, DeBERTa-v2 8×480, 16k tokenizer, fixed seq256, pure WWM-MLM, batch128, seed43022. The four runs are:

- `training/runs/adamw_lr0.007_official_10M_seed43022`
- `training/runs/lamb_lr0.007_official_10M_seed43022`
- `training/runs/adamw_lr0.007_qwen_10M_seed43022`
- `training/runs/lamb_lr0.007_qwen_10M_seed43022`

The scientific question is whether LAMB at matched high LR specifically amplifies the validated clean-Qwen same-window paired-data effect relative to AdamW, rather than producing a generic optimizer gain on both official-only and Qwen corpora.

The data object to read is therefore the interaction trajectory:

\[
I_t = [S(\text{LAMB,Qwen},t)-S(\text{LAMB,Official},t)] - [S(\text{AdamW,Qwen},t)-S(\text{AdamW,Official},t)]
\]

for BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading, and equal7-noAoA at 3M, 5M, and 10M.

## Important repair made in highlr 10M training completion

The optimizer by paired data experiment `scripts/eval_20M_optimizer_interaction.sh` originally called `scripts/full_overall_eval_runner.py` with `--model_path`, `--skip_superglue`, and `--skip_aoa`. That interface is wrong. The actual runner is registry-based; dynamic endpoints must be evaluated through `scripts/custom_endpoint_full_eval.py` with:

- `--eval_target <target>`
- `--run_dir <run_dir>`
- `--endpoint <checkpoint>`
- `--columns BLiMP Supplement EWoK Entity COMPS GlobalPIQA_parallel GlobalPIQA_nonparallel Reading`
- `--gpu <id>`

This is the same working pattern used by crossview denoising design cross-view no-AoA evaluation.

highlr 10M training completion created:

- `scripts/verify_highlr_10M_training.py` — cheap pre-evaluation check of the four 10M training outputs, `scientific_metrics.json`, checkpoint directories, optimizer/LR/beta/architecture/seed/corpus invariants, pair-row presence, and exact coordinate.
- `scripts/eval_highlr_10M_interaction.sh` — correct no-AoA evaluation of the four 10M runs at `chck_3M`, `chck_5M`, and `chck_10M` through `custom_endpoint_full_eval.py`.
- `scripts/summarize_highlr_interaction.py` — reads `official_overall.scores` from the per-target payloads and computes column-wise data effects, optimizer effects, and optimizer×data interactions.
- `scripts/unit_test_summarizer.py` — synthetic payload unit test; it passed and verified interaction arithmetic and sign-structure extraction.

highlr 10M training completion also repaired the 20M follow-up scripts so they will not fail if the 10M evidence justifies extension:

- `scripts/eval_20M_optimizer_interaction.sh`
- `scripts/summarize_20M_optimizer_eval.py`

Both now use the correct `custom_endpoint_full_eval.py` dynamic-endpoint interface and propagate evaluator failures instead of silently treating missing models as success.

Static checks passed:

```bash
python -B -m py_compile scripts/verify_highlr_10M_training.py scripts/summarize_highlr_interaction.py scripts/summarize_20M_optimizer_eval.py
bash -n scripts/eval_highlr_10M_interaction.sh
bash -n scripts/eval_20M_optimizer_interaction.sh
```

The synthetic summarizer unit test passed with expected equal7 interaction `0.7857` and sign structure 6 positive / 1 negative.

## Required checks after training

1. Run the cheap verifier:

```bash
cd experiments/archive/compact_experience
python -B scripts/verify_highlr_10M_training.py
```

Expected output path:

- `data/optimizer_10M_training_verify/highlr_10M_training_verify.json`

If it does not pass, repair or rerun only the mismatched arm before evaluation.

2. If verification passes, run no-AoA trajectory evaluation:

```bash
cd experiments/archive/compact_experience
bash scripts/eval_highlr_10M_interaction.sh
```

Expected summary path:

- `data/optimizer_10M_noaoa_eval_lr0.007/highlr_10M_interaction_summary.json`

3. Read the trajectory, not a single endpoint. A useful positive signal would be a broad, persistent interaction across multiple columns, especially not carried only by GlobalPIQA, with no Supplement or Reading collapse. A negative signal is zero/negative interaction, generic LAMB main effect on both corpora, one-column-dominated interaction, or a Supplement/Reading-damaging tradeoff.

4. Only if the 10M interaction is broad and coherent should the high-LR four-arm experiment be extended to 20M via:

```bash
cd experiments/archive/compact_experience
bash scripts/launch_20M_optimizer_interaction.sh high
bash scripts/eval_20M_optimizer_interaction.sh 0.007
```

No 100M run is justified from training loss alone or from a narrow 10M one-column swing.

## Current score status

Trusted complete admissible best remains clean-Qwen seed43022 `chck_100M`, Overall `41.34429066479573`, below the visible 41.8 leader. No official SOTA has been established.

## highlr 10M training completion exposure-boundary reconciliation (exposure-accounting check)

An independent verifier warned that the trainer's `load_examples_jsonl` stops at the last complete row (gap ≤160 allowed) and checkpoints save only when `cumulative_words >= threshold`, so a run could finish without `chck_10M` (as happened in the 1M pilot: 999,866 words, no chck_1M).

Direct audit of the actual data pools resolves this for the 10M runs:

- In `official_only_100M.jsonl` and `qwen_aligned_100M.jsonl`, cumulative words reach **exactly 10,000,000 at row 64,381** (exact boundary). The loader accepts it because its stop condition is `selected + words > target` (strict), so the final row that lands cum on exactly 10M is included and `selected == 10,000,000`.
- Intermediate 1M thresholds are crossed *within* rows (e.g., row 6,440 reaches cum 1,000,026), but checkpoints save on `cumulative_words >= threshold` during the batch loop, which accumulates actual batch words across the full 10M. A same-data 20M run (`tok16_qwen_20M_b128_seed43022`) confirms chck_1M..chck_20M were all saved.

Therefore the four 10M runs are expected to save `chck_1M..chck_10M`, and `chck_10M` is a legal exact-10M endpoint. The 1M pilot's missing checkpoint was a non-boundary artifact, not a bug affecting the 10M interaction runs.

Audit script inline output saved conceptually under `data/exposure_audit/` (the audit was a read of the frozen pools; hashes are pinned in `data/qwen_clean_aligned/clean_materialization_metadata.json`).

## Remaining interpretation caveats for the 10M screen

1. **LAMB recipe ≠ pure layer-adaptation.** The LAMB implementation folds weight decay into the update before the trust ratio, and it adapts the 2-D embedding matrix (trust ratio ~0.41 observed), contradicting the docstring. So a positive interaction attributes to "this LAMB recipe," not specifically to trust-ratio layer adaptation. A mechanism claim would need a same-moments, same-decoupled-decay, trust-scaling-only ablation. Fix the docstring/implementation before any mechanism attribution.
2. **Effective token exposure vs word budget.** Words are counted pre-truncation; official vs Qwen may differ in tokens/word and truncation rate. Report actual non-padding tokens, truncation rate, and pair-token dosage per checkpoint before strong interpretation. (For clean-Qwen these rows are short — pair rows mean ~135 words — so truncation is likely small, but quantify it.)
3. **Fixed-order trajectory = corpus-prefix/dosage trajectory.** `shuffle=False` gives a good paired optimizer control but the 3M/5M/10M points are single-prefix. Report pair-token fraction d_t at each endpoint.
4. **GlobalPIQA fold.** The summarizer averages parallel/nonparallel equally as a screening proxy; this is internally consistent across arms but may not equal the official column fold. Do not use equal7-noAoA to estimate distance to 41.8.
5. **Evaluator GPU mapping / schema.** The wrapper sets both `CUDA_VISIBLE_DEVICES` and `--gpu`; confirm no invalid-device error via a single GPU1 smoke, and confirm subset-eval payloads populate `official_overall.scores` correctly.
6. **Noise floor.** A difference-in-differences over four evaluated scores amplifies measurement noise. A tiny positive equal7 interaction or a few near-zero positive columns is not meaningful; compare against evaluator/bootstrap noise and, if promising, add a second pretraining seed.

## Promotion criteria for a 20M follow-up (no AoA/CDI internals, no training loss)

Justify a fresh 20M-horizon four-arm run only if the 10M screen shows:
- interaction positive at 5M and 10M (not only at the near-zero-LR 10M endpoint);
- LAMB genuinely amplifies the Qwen-minus-official data effect (LQ−LO > AQ−AO, and LAMB−AdamW larger on Qwen than official);
- breadth: ≥4/7 columns same-signed positive across multiple capability families, not one-column (especially not GlobalPIQA-only);
- no Supplement/Reading collapse under LAMB-Qwen vs AdamW-Qwen;
- magnitude above measurement noise.

Even if satisfied, a 20M run must be a fresh 20M schedule from the same init (the trainer does not checkpoint optimizer/scheduler/RNG state, so continuation is impossible), and no 100M/SOTA claim follows without full nine-column official-compatible evaluation and seed replication.
