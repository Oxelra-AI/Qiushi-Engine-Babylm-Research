# s2 curriculum verified and true s2 launch — S2 curriculum telemetry verified; true S2 100M launched

## Telemetry verification (compressed mechanism smoke, NOT true S2)

Run: `training/runs/babylm_compressed_curriculum_mechanism_smoke100k`

Confirmed the isolated fork's curriculum implementation is numerically correct on the word-exposure clock:

- `mask_mode_schedule=0.0:wwm,0.5:token`, `curriculum_progress_basis=word`, `curriculum_total_words=100000`.
- Training log shows WWM through memory vs dense untied 1m profile (cumulative 51,200 words, curriculum_frac 0.4864) and token masking from memory stability and next route (cumulative 53,760 words, curriculum_frac 0.512). Switch occurs at frac≈0.5 as intended.
- `mask_mode_step_counts`: wwm 20, token 20. `mask_mode_word_counts`: wwm 51,200, token 48,800 (sum 100,000).
- Length schedule `0.0:64,0.4:128,0.8:256`: `seq_len_step_counts` 64:16, 128:16, 256:8; `unique_training_seq_lengths` [64,128,256]. Length steps 64→128 at frac 0.4, 128→256 at frac 0.8.
- `variant` recorded as `masked_curriculum`.

No-schedule transfer smoke `training/runs/babylm_s1_transfer_noschedule_smoke20k` confirms the fork still reproduces plain S1 12×384/intermediate1280 when no schedule is passed (inert additions).

## Route separation

Two experiments must remain fully distinct in name, interpretation, and downstream decisions:

1. **True S2 (leader-curriculum isolation).** WWM7→Token3 is defined over the ten-epoch cumulative word-exposure clock. On the 100M route the switch must occur at ~70M cumulative words: `--curriculum_progress_basis word --curriculum_total_words 100000000 --mask_mode_schedule 0.0:wwm,0.7:token`, plus length `0.0:64,0.4:128,0.8:256`. This is the protected true-S2 route and must NOT be replaced by any compressed short-run result.
2. **Compressed mechanism probe.** Any 10M (or shorter) run switching at 7M cumulative words tests a *different* early-token-masking intervention. It may inform mechanism but must never be used to eliminate or stand in for true S2.

## True S2 100M launch

Run id: `babylm_true_s2_100M_wordclock_wwm70_len64256`
Full-exposure results were pending when this note was written.

Config: isolated fork only; official corpus; baseline16k; DeBERTa-v2 12×384/intermediate1280; effective batch 256 / micro 128; lr_total_steps 2442; checkpoints every 10M; seq curriculum 64→128→256; mask WWM→token at 70M word-clock. No 40k tokenizer, no LAMB, no new/gated data. Protected trainer untouched.

## Next action

Required checks after training: verify `scientific_metrics.json`: word_exposure 100M, `mask_mode_word_counts` wwm≈70M / token≈30M, length steps present, 2442 optimizer steps, exact 12×384/intermediate1280 config, ten checkpoints. Then evaluate `hf_model/chck_100M` (s1 100m available coordinate scripts) on BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, full local EWoK. Compare true S2 100M against S1 100M (BLiMP 66.84, Supplement 60.31, Entity 20.24, COMPS 52.26, GlobalPIQA mean 37.605, Reading mean 7.25, EWoK 52.02) and protected 8×480 (Overall 40.5269), focusing on whether curriculum preserves GlobalPIQA while improving Entity/EWoK/grammar.
