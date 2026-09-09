# experience utilization trainer verified — Experience-utilization trainer built and verified

## What was built

`experiments/archive/representation_and_objectives/scripts/experience_utilization_trainer.py` — a faithful word-boundary chunking trainer implementing the chunk stream preflight experience-utilization experiment design. It replaces the prefix-slicing seq_len_schedule with a trainer that partitions each row into word-boundary chunks, so every tokenizer token from the 10M corpus is visible once per epoch.

Supports three arms:
- `U256`: 10 epochs at L256 (pure visibility repair)
- `U64_128_256`: 3 epochs L64, 4 L128, 3 L256 (visibility + context ordering)
- `U64x7_256x3`: 7 epochs L64, 3 L256 (extended short context)

Also supports `--dry_run` for full accounting verification without GPU.

## Verification against chunk stream preflight preflight

Both full 10-epoch dry-runs completed with all checks passed:

### U256 (264.4s CPU)

| Property | Expected | Actual | Match |
|---|---|---|---|
| Chunks per epoch | 76,164 | 76,164 | ✓ |
| Charged words per epoch | 10,000,000 | 10,000,000 | ✓ |
| Active tokens per epoch | 13,942,644 | 13,942,644 | ✓ |
| Total steps (10 epochs) | 2,530 | 2,530 | ✓ |
| Total words | 100,000,000 | 100,000,000 | ✓ |
| Checkpoints | 100 (1M–100M) | 100 | ✓ |
| All epoch steps 253 | yes | yes | ✓ |
| First-step chunks | 302 | 302 | ✓ |
| First-step masked tokens (wwm015) | 8,025 | 8,025 | ✓ |

### U64_128_256 (306.3s CPU)

| Property | Expected | Actual | Match |
|---|---|---|---|
| L64 chunks per epoch | 252,270 | 252,270 | ✓ |
| L64 overlong words | 1 | 1 | ✓ |
| L64 continuation chunks | 1 | 1 | ✓ |
| L128 chunks per epoch | 140,303 | 140,303 | ✓ |
| L256 chunks per epoch | 76,164 | 76,164 | ✓ |
| All epoch words 10M | yes | yes | ✓ |
| All epoch tokens 13,942,644 | yes | yes | ✓ |
| Total steps (10 epochs) | 2,530 | 2,530 | ✓ |
| Total words | 100,000,000 | 100,000,000 | ✓ |
| Checkpoints | 100 (1M–100M) | 100 | ✓ |
| First-step L64 chunks | 998 | 998 | ✓ |
| First-step L64 masked tokens | 8,031 | 8,031 | ✓ |
| Stage transitions correct | yes | yes | ✓ |

## Design invariants preserved

1. **Charged words**: exactly 10M per epoch, 100M total — words are counted from chunk metadata, not from row-level fields
2. **Active tokens**: 13,942,644 per epoch for legal40k — every raw tokenizer token visible once per epoch regardless of chunk length
3. **Optimizer updates**: 253 per epoch, 2,530 total — chunks distributed evenly, stage boundaries don't create partial epochs
4. **Masking**: full effective-batch WWM before microbatch forward/backward, masked-token weighted gradient accumulation
5. **RNG continuity**: single masking generator stream continuous across epochs and stages
6. **Overlong handling**: single L64 overlong whitespace word split into 2 chunks (1 normal + 1 continuation at charged_words=0)
7. **Checkpoint placement**: every 1M charged words

## Ready for GPU launch

The trainer is ready for actual GPU training once the depth vector is evaluated and the tokenizer/architecture choice is made.

Typical launch command:
```
python3 -B Scripts/experience_utilization_trainer.py \
  --arm U256 \
  --example_jsonl <10M_pool.jsonl> \
  --tokenizer_path <legal40k_path> \
  --output_dir training/runs/legal40k_U256_compact_view_reinvest_seed43022 \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --mask_prob 0.15 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023
```

## Artifacts

- Trainer: `experiments/archive/representation_and_objectives/scripts/experience_utilization_trainer.py`
- U256 dryrun: `experiments/archive/representation_and_objectives/data/dryrun_legal40k_U256_full/dryrun_metrics.json`
- U64_128_256 dryrun: `experiments/archive/representation_and_objectives/data/dryrun_legal40k_U64_128_256_full/dryrun_metrics.json`
- Step-level CSVs under the same directories
