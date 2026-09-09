# mntp full eval execution state — MNTP negative result and compact tokenizer/data screen state

## Complete execution state MNTP result

The execution state MLM-primary same-corruption token-shift MNTP auxiliary was fully evaluated on the official-style nine-column suite:

- Payload: `data/mlm_mntp_full_eval/per_target/mlm_mntp_aux015_100M.json`
- Analysis: `data/mntp_eval_interpretation.json`
- Overall: `39.72564932846988`
- Scores: BLiMP `66.82`, Supplement `62.07`, EWoK `51.15`, Entity `25.08`, COMPS `51.29`, SuperGLUE `69.76577825313923`, GlobalPIQA `36.15`, Reading `8.62`, AoA leaderboard `-13.414934296910275` (raw `-0.13414934296910275`, p `0.05824680071288274`)

Compared with clean-Qwen seed43022 `chck_100M` Overall `41.34429066479573`, MNTP gives:

- EWoK `+0.96`, Reading `+0.86`
- Supplement `-0.77`, Entity `-0.68`, COMPS `-0.49`, GlobalPIQA `-0.47`, SuperGLUE `-0.54`
- AoA `-13.4149`
- Overall delta `-1.61864`

Interpretation: local gradient compatibility was not enough to preserve official aggregate learning. The exact same-stack 15% norm-calibrated MNTP auxiliary is closed as a SOTA route. The pattern resembles a directional/objective perturbation that can move EWoK/Reading but damages aggregate acquisition balance; do not endpoint-mine AoA.

## Why not revert to the old 12×384/40k/LAMB bundle

Interpretation constraint: a negative MNTP result does not justify the old Phase2 package. That package jointly changed architecture depth/width, tokenizer/vocab, optimizer/LR, sequence schedule, masking schedule, and data state. It does not isolate a clean-Qwen × architecture interaction.

The route should keep the validated same-window clean-Qwen data mechanism and test one factor at a time.

## Tokenizer audit

Script: `scripts/tokenizer_factor_audit.py`
Output: `data/tokenizer_factor_audit.json`

Key audit results comparing baseline 16k to official-only shared 40k on exact clean qwen compliance and validity pools:

- Qwen all mean tokens/word ratio 40k/16k: `0.9683583861611263`
- Official all mean tokens/word ratio 40k/16k: `0.9725340012416565`
- Qwen-pair rows mean tokens/word ratio 40k/16k: `0.9330237416430288`
- Qwen-pair over-256 rate: `0.008662961752206603` → `0.002043151356652501`
- All Qwen over-256 rate: `0.2583060219630015` → `0.2245227629269505`

This gives a real representation/data-interface mechanism: 40k reduces fragmentation most strongly on paired rows, potentially making same-window correspondence easier to represent. It does not by itself prove 40k will help Overall or AoA.

## Execution state of compact screen

The first attempt with batch256 failed in both 40k arms due CUDA OOM during first backward. Logs:

- `training/runs/tok40_official_20M_seed43022/train_stderr.log`
- `training/runs/tok40_qwen_20M_seed43022/train_stderr.log`

The failure is not scientific evidence. It shows 40k vocabulary logits require smaller batch or memory optimization. GPU memory was free afterwards. A batch128 smoke run succeeded:

- Smoke run: `training/runs/tok40_smoke_qwen_b128b`
- 80,121 words, 5 steps, parameter_count `45,826,720`, loss_first `10.708248`, loss_last `10.643279`

Second attempt launched:

- Script: `scripts/launch_tok40_isolation_20M.sh`
- Runs: `training/runs/tok40_official_20M_b128_seed43022` and `training/runs/tok40_qwen_20M_b128_seed43022`
- Design: shared official-only 40k tokenizer, current 8×480/AdamW/fixed-seq256/WWM recipe, 20M exact word exposure, batch128, `lr_total_steps=5030`

This two-arm 40k experiment estimates the clean-Qwen data effect under 40k, not the tokenizer factor by itself. For true tokenization/data interaction, run matched 16k b128 arms and summarize a 2×2.

## Prepared next assets

- 40k no-AoA eval: `scripts/launch_tok40_isolation_noaoa_eval.sh`
- 40k no-AoA summarizer: `scripts/summarize_tok40_isolation_noaoa.py`
- matched 16k b128 20M training launcher: `scripts/launch_tok16_b128_20M_matched.sh`
- matched 16k no-AoA eval: `scripts/launch_tok16_b128_noaoa_eval.sh`
- 2×2 summarizer: `scripts/summarize_tok16_tok40_2x2.py`

## Scientific continuation

If the training comparison completes, first inspect training metrics and run the 40k no-AoA eval. To determine whether tokenization is a true factor or only a 40k-conditioned data effect, launch the matched 16k b128 arms and evaluate them under the same no-AoA screen, then run `scripts/summarize_tok16_tok40_2x2.py`.

A 100M run is justified only if the compact screen shows a target-column-positive interaction or a clear Q40 advantage over Q16 without strong Supplement/Reading damage. Regardless, any final candidate needs complete nine-column evaluation including AoA and SuperGLUE.
