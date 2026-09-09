# factorial construction report — Factorial stream construction report

## Scientific purpose

Four-arm ordinary-WWM factorial to separate contextual diversification from anchor recurrence in compact semantic views. The consequential outcome is the downstream interaction `(HS-LS)-(HD-LD)` in an independent RoBERTa masked-LM coordinate, not local NLL.

## Arms

| Arm | Second context | Source anchors | Row form |
|---|---|---|---|
| HS | compact faithful rewrite | same (own source) | `source_i + compact_i` |
| LS | hash-rotated source repeat | same (own source) | `source_i + repeat_i` |
| HD | compact from deranged donor | different (wrong source) | `source_i + compact_deranged(i)` |
| LD | repeat from deranged donor | different (wrong source) | `source_i + repeat_deranged(i)` |

## Construction verification

### Word counts — perfect factorial balance

All four 10M pools: **9,999,991 words**, **64,740 rows**.  
All four 100M streams: **99,999,910 words**, **647,400 rows**, **2,529 training steps**.  
All word deltas are **exactly zero** (HS-LS, HD-LD, HS-HD, LS-LD, interaction).

### BPE and supervised mass — negligible interaction confound

Full audit across all 3,006 changed rows (one pass):

| Contrast | BPE delta | BPE % | WG delta | WG % |
|---|---:|---:|---:|---:|
| HS-LS | +24,000 | +3.95% | -196 | -0.046% |
| HD-LD | +24,316 | +4.00% | -39 | -0.009% |
| HS-HD | -521 | -0.09% | -279 | -0.066% |
| LS-LD | -205 | -0.03% | -122 | -0.029% |
| **Interaction** | **-316** | **-0.05%** | **-157** | **-0.037%** |

Compact naturally uses ~4% more BPE pieces than repeat. This cancels almost exactly in the interaction (-0.05%). Expected WWM target interaction over 100M: ~236 targets out of ~635,000 total (0.037%).

Per-row interaction BPE: mean -0.1, median 0, stdev 8.1, range [-38, +32].

### Tensor parity — verified

- Init SHA identical across all 4 arms: `5397c302c18ee1a9...`
- Parameter count: 30,528,064 per arm
- LR schedule identical (6.622e-06 at babylm2026 live surface)
- All finite loss and gradients
- Parameters correctly diverge after training on different text
- **Ready for training: YES**

### Derangement quality

- 12,155 pairs, zero self-pairs
- 12,152/12,155 exact view-word length matches (99.975%)
- Source-content overlap collapse: own ~0.658 → wrong ~0.003
- Repeat-source overlap collapse: own ~0.634 → wrong ~0.003

## Pool and stream files

| File | SHA256 (first 16) |
|---|---|
| `factorial_hs_10M.jsonl` | `3127cba0e0a8b4d6` |
| `factorial_ls_10M.jsonl` | `ff15a80220daa366` |
| `factorial_hd_10M.jsonl` | `cb77a1508b9d7685` |
| `factorial_ld_10M.jsonl` | `8914940a1b4517fd` |
| `factorial_hs_100M.jsonl` | `741f19075195b027` |
| `factorial_ls_100M.jsonl` | `313c2157bd96c7f6` |
| `factorial_hd_100M.jsonl` | `da7d1098b5efcdf4` |
| `factorial_ld_100M.jsonl` | `1a1e26201ccf1067` |

All under `experiments/archive/representation_and_objectives/data/factorial_streams`.

## Exact training commands

Using `roberta_mlm_transfer_trainer.py` with matched config. Run as 2 waves × 2 GPUs.

### Wave 1: HS (gpu0) + LS (gpu1)

```bash
# HS arm — gpu 0
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py \
  --example_jsonl experiments/archive/representation_and_objectives/data/factorial_streams/factorial_hs_100M.jsonl \
  --example_jsonl_label "factorial_hs_compact_same_anchors" \
  --output_dir experiments/archive/representation_and_objectives/training/runs/factorial_hs_roberta_100M \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --model_family roberta \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 \
  --mask_prob 0.15 --grad_clip 1.0 \
  --max_word_exposure 99999910 \
  --lr_total_steps 2529 \
  --checkpoint_words 20000000 \
  --log_every 50 \
  --gradient_checkpointing

# LS arm — gpu 1
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py \
  --example_jsonl experiments/archive/representation_and_objectives/data/factorial_streams/factorial_ls_100M.jsonl \
  --example_jsonl_label "factorial_ls_repeat_same_anchors" \
  --output_dir experiments/archive/representation_and_objectives/training/runs/factorial_ls_roberta_100M \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --model_family roberta \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 \
  --mask_prob 0.15 --grad_clip 1.0 \
  --max_word_exposure 99999910 \
  --lr_total_steps 2529 \
  --checkpoint_words 20000000 \
  --log_every 50 \
  --gradient_checkpointing
```

### Wave 2: HD (gpu0) + LD (gpu1)

```bash
# HD arm — gpu 0
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py \
  --example_jsonl experiments/archive/representation_and_objectives/data/factorial_streams/factorial_hd_100M.jsonl \
  --example_jsonl_label "factorial_hd_compact_deranged_anchors" \
  --output_dir experiments/archive/representation_and_objectives/training/runs/factorial_hd_roberta_100M \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --model_family roberta \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 \
  --mask_prob 0.15 --grad_clip 1.0 \
  --max_word_exposure 99999910 \
  --lr_total_steps 2529 \
  --checkpoint_words 20000000 \
  --log_every 50 \
  --gradient_checkpointing

# LD arm — gpu 1
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py \
  --example_jsonl experiments/archive/representation_and_objectives/data/factorial_streams/factorial_ld_100M.jsonl \
  --example_jsonl_label "factorial_ld_repeat_deranged_anchors" \
  --output_dir experiments/archive/representation_and_objectives/training/runs/factorial_ld_roberta_100M \
  --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer \
  --tokenizer_label compliant16k_reinvest10M \
  --model_family roberta \
  --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 \
  --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 \
  --batch_size 256 --seq_length 256 --max_seq_length 256 \
  --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 \
  --mask_prob 0.15 --grad_clip 1.0 \
  --max_word_exposure 99999910 \
  --lr_total_steps 2529 \
  --checkpoint_words 20000000 \
  --log_every 50 \
  --gradient_checkpointing
```

## Checkpoint evaluation

After training, evaluate at chck_20M, chck_40M, chck_60M, chck_80M, chck_100M using the existing selected evaluator. The predeclared downstream estimands on official-compatible selected columns:

1. Same-anchor context diversity: `HS - LS` per column
2. Deranged-anchor compact marginal: `HD - LD` per column
3. **Interaction: `(HS - LS) - (HD - LD)`** per column

Scientific readings:
- **HS-LS > 0, HD-LD ≈ 0**: compact benefit requires shared anchors across source and context → contextual anchor diversification mechanism
- **HS-LS ≈ HD-LD > 0**: compact marginal helps regardless → generic dense-context effect
- **Positive main + positive interaction**: both matter

Local NLL alone cannot decide the route. The signal must appear on stable selected families (Supplement, EWoK, Entity, COMPS, excluding volatile GlobalPIQA/Reading).

## Notes

- Word counts per arm: exactly 99,999,910 (not 100,000,000) because the changed block reconstruction gives 423,511 words per pass instead of historical 423,520. This 9-word difference per pass is negligible and cancels perfectly across arms. Use `--max_word_exposure 99999910`.
- The trainer's word-count check accepts the exact stream with `--max_word_exposure 99999910`; no `--allow_larger_jsonl` flag is needed for the factorial construction report commands. earlier analysis later moved training to a repaired microbatch-accumulation trainer after full-batch GPU runs OOMed under the live memory envelope.
- lr_total_steps=2529 matches exactly: ceil(647,400 / 256) = 2,529.
- Gradient checkpointing is recommended to reduce H100 memory pressure at batch_size=256.
- Each run should take ~20-40 minutes on H100 (similar to 40M compact/repeat RoBERTa runs but scaled to 100M).

## Artifacts

- Pool files: `data/factorial_streams/factorial_{hs,ls,hd,ld}_10M.jsonl`
- Stream files: `data/factorial_streams/factorial_{hs,ls,hd,ld}_100M.jsonl`
- Materializer: `scripts/factorial_stream_materializer.py`
- Tensor parity smoke: `data/tensor_parity_smoke/tensor_parity_smoke.json`
- Full BPE audit: `data/full_bpe_audit/full_bpe_audit.json`
- This report: `notes/factorial_construction_report.md`
