# component gap analysis and leader reverse engineering — Experiment Configurations for Leader-Recipe Testing

## Reference: Our best run (Overall 41.34)

```
data: experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_100M.jsonl
tokenizer: baseline16k (no --tokenizer_path → uses default)
architecture: 8×480, ffn_mult=4 → intermediate=1920, 34.47M params
optimizer: AdamW, lr=1e-3
masking: wwm_fixed
seq_length: 256 (fixed)
word_exposure: 100,000,000 (10 epochs × 10M)
batch_size: 64
seed: 43 (extra_init_seed for seed43022 variant)
```

## Reference: Leader recipe (Overall 41.80)

```
data: FineWeb simplification pairs, 10M words
tokenizer: 40k SentencePiece BPE
architecture: 12×384, intermediate=1280, 34.68M params
optimizer: LAMB, lr=0.007
masking: wwm_to_token, switch_frac=0.7
seq_length: 64→256 curriculum
word_exposure: 100,000,000
```

## Pilot experiments (10M exposure = 1 epoch, ~15min each)

### Pilot A1: Full leader recipe on our clean-Qwen data

Tests: curriculum + LAMB + 12×384 architecture (isolates data effect only)

```bash
python experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_v2.py \
  --output_dir experiments/archive/frontier_consolidation/training/runs/pilot_A1_leader_recipe_10M \
  --example_jsonl experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl \
  --max_word_exposure 10000000 \
  --example_pool_words 10000000 \
  --masking_curriculum wwm_to_token \
  --switch_frac 0.7 \
  --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --seq_len_schedule "0.0:64,0.7:256" \
  --seq_length 64 --max_seq_length 256 \
  --optimizer lamb \
  --learning_rate 0.007 \
  --weight_decay 0.01 \
  --n_layer 12 --hidden_size 384 --n_head 12 \
  --intermediate_size 1280 \
  --batch_size 256 \
  --checkpoint_words 1000000 \
  --seed 43 --extra_init_seed 43022 \
  --log_every 20
```

### Pilot A2: Leader recipe but keep our 8×480 architecture

Tests: curriculum + LAMB (isolates architecture vs training recipe)

```bash
python experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_v2.py \
  --output_dir experiments/archive/frontier_consolidation/training/runs/pilot_A2_recipe_8x480_10M \
  --example_jsonl experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl \
  --max_word_exposure 10000000 \
  --example_pool_words 10000000 \
  --masking_curriculum wwm_to_token \
  --switch_frac 0.7 \
  --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --seq_len_schedule "0.0:64,0.7:256" \
  --seq_length 64 --max_seq_length 256 \
  --optimizer lamb \
  --learning_rate 0.007 \
  --weight_decay 0.01 \
  --n_layer 8 --hidden_size 480 --n_head 8 \
  --batch_size 256 \
  --checkpoint_words 1000000 \
  --seed 43 --extra_init_seed 43022 \
  --log_every 20
```

### Pilot A3: Only curriculum, keep AdamW and 8×480

Tests: curriculum alone (most conservative — just adds masking + seqlen schedule)

```bash
python experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_v2.py \
  --output_dir experiments/archive/frontier_consolidation/training/runs/pilot_A3_curriculum_only_10M \
  --example_jsonl experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl \
  --max_word_exposure 10000000 \
  --example_pool_words 10000000 \
  --masking_curriculum wwm_to_token \
  --switch_frac 0.7 \
  --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --seq_len_schedule "0.0:64,0.7:256" \
  --seq_length 64 --max_seq_length 256 \
  --optimizer adamw \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --n_layer 8 --hidden_size 480 --n_head 8 \
  --batch_size 256 \
  --checkpoint_words 1000000 \
  --seed 43 --extra_init_seed 43022 \
  --log_every 20
```

### Pilot A0: Baseline reproduction (no curriculum, match best run)

```bash
python experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_v2.py \
  --output_dir experiments/archive/frontier_consolidation/training/runs/pilot_A0_baseline_10M \
  --example_jsonl experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl \
  --max_word_exposure 10000000 \
  --example_pool_words 10000000 \
  --masking_curriculum wwm_fixed \
  --mask_prob_start 0.15 --mask_prob_end 0.15 \
  --seq_length 256 --max_seq_length 256 \
  --optimizer adamw \
  --learning_rate 1e-3 \
  --weight_decay 0.01 \
  --n_layer 8 --hidden_size 480 --n_head 8 \
  --batch_size 64 \
  --checkpoint_words 1000000 \
  --seed 43 --extra_init_seed 43022 \
  --log_every 20
```

## Decision logic

After 10M pilots complete:
- Compare loss_last across A0, A1, A2, A3
- If A1 or A2 loss is significantly lower (>0.05 lower), proceed to 100M run
- If A3 shows most of the gain, curriculum alone is the key
- If A1 >> A2, the 12×384 architecture matters
- If A2 >> A3, LAMB with high LR matters

## GPU allocation for pilots

Run A0+A1 on GPU0 (serial, ~30min total) and A2+A3 on GPU1 (serial, ~30min total).
All four pilots complete in ~30min wall time.

## Full run specification (after pilot results)

Best pilot config scaled to 100M exposure:
- Same settings but `--max_word_exposure 100000000`
- Use 100M JSONL: `qwen_aligned_100M.jsonl`
- Followed by full 9-column evaluation
