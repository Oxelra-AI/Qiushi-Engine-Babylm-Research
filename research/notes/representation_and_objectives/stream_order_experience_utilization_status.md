# mask budget and pad contract — Stream-order experience-utilization status

Depth training remains pending. CPU-only checks prepare an experience-utilization comparison without a data-order confound; no new H100 training or official evaluation was launched.

## Why experience utilization trainer verified needed repair before any expensive launch

The experience utilization trainer verified trainer correctly implemented word-boundary chunking, but it repeated the 10M pool in canonical file order. The matched legal40k and depth baselines train on the materialized 100M stream:

- 10M pool SHA: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- 100M stream SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- Every 64,740-row block of the 100M stream is the same raw-line multiset as the 10M pool and has exactly 10M words.
- But row order is different: epoch0 same-position fraction `1.5446e-05`, adjacent-forward fraction `0.0`, mean absolute displacement `21581.3` rows. All 10 blocks show the same shuffled-order pattern.

Therefore launching experience utilization trainer verified unchanged would test visibility plus a major data-order change relative to existing row256 baselines. The repaired trainer is `experiments/archive/representation_and_objectives/scripts/stream_order_experience_utilization_trainer.py`.

## What the mask budget and pad contract trainer does

`stream_order_experience_utilization_trainer.py`:

1. tokenizes the exact 10M pool once;
2. reads the exact materialized 100M stream as ten 10M blocks of raw-line hashes;
3. for exposure epoch `e`, uses stream block `e` as the row order and then applies the stage length for the selected arm;
4. builds word-boundary chunks in that stream order;
5. distributes each 10M block into exactly 253 optimizer updates;
6. uses full-effective-batch WWM, masked-token weighted microbatch loss, continuous optimizer/LR/masking RNG state, and 1M-word checkpoints;
7. records explicit seed mapping: initialization seed is `extra_init_seed` (43022 for the first run), masking/data RNG seed is `train_rng_seed` (43023), while `seed=43` is overwritten when those two are non-negative.

The trainer also pads with tokenizer `pad_token_id=3`, unlike the experience utilization trainer verified prototype's zero padding.

## Verified dry-run invariants

Full dry-runs completed for both legal40k stream-order arms:

- U256: `experiments/archive/representation_and_objectives/data/dryrun_stream_order_legal40k_U256_full/dryrun_metrics.json`
  - 10 epochs, 2,530 updates, exactly 100,000,000 charged words
  - 13,942,644 active tokens per epoch; 139,426,440 active tokens total
  - 76,164 chunks per epoch; 100 checkpoint records
  - realized dry-run masked targets at WWM 0.15: 20,916,090
- U64_128_256: `experiments/archive/representation_and_objectives/data/dryrun_stream_order_legal40k_U64_128_256_full/dryrun_metrics.json`
  - 3×L64 + 4×L128 + 3×L256, 2,530 updates, exactly 100,000,000 charged words
  - same 13,942,644 active tokens per epoch and 139,426,440 total
  - L64 chunks per epoch 252,270 with 1 overlong word and 1 continuation chunk; L128 chunks 140,303; L256 chunks 76,164
  - realized dry-run masked targets at WWM 0.15: 20,915,364

independent_review verification (`data/external/independent_review01_verifier1_integration.md`) agreed that the data-order confound is fixed and the accounting invariants are supported. It emphasized that the dry-run checks accounting, masking, and chunk construction, not full optimization dynamics.

## Mask-budget and pad/label contract

`experiments/archive/representation_and_objectives/scripts/mask_budget_and_pad_contract_check.py` compared the repaired trainer to the matched legal40k row256 baseline and checked pad labels:

- Matched row256 legal40k baseline used fixed WWM 0.15, AdamW LR 0.001, warmup fraction 0.06, and 2,529 updates.
- Baseline log masked targets: 20,568,519 over 100M charged words.
- U256 stream-order at mask_prob 0.15 produces +347,571 more realized targets (1.016898× baseline) because recovered active tokens are also eligible for masking.
- U64_128_256 at mask_prob 0.15 produces +346,845 targets (1.016863× baseline).
- Target-matched visibility should use mask probability about `0.14745584` by the chunk stream preflight active-token formula, or `0.14752237` if matching the exact realized row256 target count from the baseline log.
- Pad/label check is clean on first real stream-order batch for both arms: labels-on-pad `0`, pad-input changes `0`, active positions missing word group `0`.

The scientific choice is now explicit:

- `mask_prob=0.15` tests full recovered experience utilization: more visible linguistic material and proportionally more prediction pressure under the same nominal mask probability.
- `mask_prob≈0.14746` tests target-matched visibility, keeping expected realized target count closer to the old hidden-row baseline.

## Smoke-run status

A 12×384 one-step real GPU smoke was attempted only after the CPU/dry-run evidence, but it hit CUDA OOM because GPU0 had only 171 MB free at the moment. This was a resource collision, not evidence against the trainer. A tiny CPU non-dry one-step smoke succeeded using the same stream-order loop and emitted non-null loss:

- output: `experiments/archive/representation_and_objectives/data/stream_order_U256_tiny_cpu_1step_smoke/scientific_metrics.json`
- first non-dry step: loss `10.60994995959848`, words `40170`, active tokens `55799`, masked tokens `8393`, microbatches `5`, LR `6.622516556291391e-06`.

Before a full H100 launch, run a one-babysteps public method reading×384 GPU smoke on an actually free GPU if possible; this should not consume substantial compute and will confirm the real architecture's memory path for the repaired trainer.

## Next use after depth result

Do not launch this route until the depth vector is evaluated. If depth is coherent but below 41.8, first experience-utilization run should use the same architecture as indicated by depth:

- if depth is positive: legal40k 12×384/FFN1280, first run `U256`, with a deliberate choice between `--mask_prob 0.15` and `--mask_prob 0.14745584`;
- if depth is flat or worse: legal40k 8×480 `U256` as the clean visibility repair against the existing legal40k row256 baseline.

Only if U256 produces a real endpoint improvement should U64_128_256 be interpreted; if U64 beats U256, an additional length-mixture/permutation control is needed before attributing the gain specifically to chronological short-to-long order rather than context length/packing.

## Key artifacts

- Stream-order confound audit: `research/notes/representation_and_objectives/stream_order_confound_audit.md`, `experiments/archive/representation_and_objectives/data/stream_order_confound_audit/stream_order_confound_audit.json`
- Repaired trainer: `experiments/archive/representation_and_objectives/scripts/stream_order_experience_utilization_trainer.py`
- Dry-runs: `experiments/archive/representation_and_objectives/data/dryrun_stream_order_legal40k_U256_full/dryrun_metrics.json`, `experiments/archive/representation_and_objectives/data/dryrun_stream_order_legal40k_U64_128_256_full/dryrun_metrics.json`
- Mask and pad check: `research/notes/representation_and_objectives/mask_budget_and_pad_contract.md`, `experiments/archive/representation_and_objectives/data/mask_budget_and_pad_contract/mask_budget_and_pad_contract.json`
- independent_review: `data/external/independent_review01_verifier1_integration.md`
