# official40k oom and accum repair — official40k OOM and accumulation repair

## Failed full run

Original full run:

`training/runs/babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b256/`

The training attempt failed before any optimizer step, at the first backward pass:

- location: `babylm_masked_train_fullcycle.py`, line 666, `loss.backward()`
- error: CUDA OOM, tried to allocate 9.77 GiB
- memory at failure: GPU 0 total 79.18 GiB, 2.88 GiB free, process using 76.28 GiB
- `training_log.jsonl` empty

The expensive data/tokenization phase succeeded:

- 625,000 examples, 100,000,000 words
- official40k full-run tokenization summary: 1.3955 untruncated tokens/word, 1.3726 kept tokens/word
- truncated example fraction at length 256: 0.182256 (lower than baseline16k DeBERTa's 0.292048)
- total tokens lost to truncation: 2,289,650

Interpretation: this is a memory failure from 45.8M-param official40k DeBERTa at micro/effective batch 256, not a scientific negative result about official40k.

## Repair

Cloned trusted trainer:

`training/scripts/babylm_masked_train_fullcycle_accum.py`

Patch adds `--grad_accum_steps`. `--batch_size` is now microbatch size. Optimizer step count becomes `ceil(num_microbatches / grad_accum_steps)`. Logging, LR schedule, checkpoint thresholds, cumulative word exposure, and metrics are tied to optimizer steps and accumulated words.

Smoke run:

`training/runs/babylm_smoke_debertav2_8x480_official40k_accum128x2_80k/`

Command used microbatch 128, `--grad_accum_steps 2`, 80k exposure, 2 optimizer steps. It compiled, trained, saved, and passed training artifact validation. This tests the intended repair: effective batch ≈256 per optimizer step while activation memory is reduced to microbatch 128.

## Full repair run to launch

Run id:

`babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b128_acc2`

Scientific interpretation: this should be compared to baseline16k DeBERTa b256 as the same transformer backbone and intended effective batch/update schedule, but with official40k tokenizer/embedding. Minor non-identity remains because WWM masking is sampled per microbatch rather than one batch-256 tensor; this is acceptable for the rescue and should be recorded.
