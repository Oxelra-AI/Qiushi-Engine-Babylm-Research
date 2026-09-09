# recgpt local reference consolidated — RecGPT Official-Corpus Training Implementation Plan

## Source: `serdardoesml/bblm26-recgpt` (MIT, commit 178566ec)

All hyperparameters and architecture details below are extracted directly from the public repository code.

## Architecture (model.py)

- **Model class:** `RecGPTForCausalLM` (HF-compatible causal LM)
- **Hidden size:** 768 (for 10M model card)
- **Embedding size:** 192 (factorized: embed → e_to_h linear → hidden)
- **Head dim:** 64 → num_heads = 768/64 = 12
- **Intermediate/FFN:** 12288 (16× multiplier with ReLU²)
- **Recursive depth:** 16 (shared attn+MLP block applied 16 times with depth-specific RMSNorm)
- **Vocab size:** 32768 (ByteLevel BPE)
- **Max position:** 512 (= sequence_len during training)
- **Tie embeddings:** False (untied lm_head in embedding_size=192 space)
- **Attention:** FlexAttention with causal + segment mask, RoPE θ=10000, QK RMSNorm (unparameterized), per-head sigmoid gate (zero-init weight, 0.0 bias)
- **Init:** Zero-init attention output projection, zero-init MLP down projection
- **Activation:** ReLU² (squared ReLU)
- **Parameters:** ~34.17M

## Optimizer (optimizer.py)

- **Type:** `SingleDeviceAuroraWithAuxAdam`
  - Aurora (modified Muon with polar decomposition + Newton-Schulz 12 iterations) for 2D weight matrices
  - Adam for everything else (embeddings, norms, lm_head, e_to_h, h_to_e, biases)
  - Cautious Weight Decay (decays only coordinates where update and parameter align)
- **Hyperparameters:**
  - lr_embed = 0.005 (Adam)
  - lr_block = 0.02 (Aurora/Muon)
  - adam_beta1 = 0.9, adam_beta2 = 0.997
  - muon_momentum = 0.95, nesterov = True
  - wd_adam = 0.005, wd_muon = 0.1
  - max_grad_norm = 2.0
- **LR schedule:** Linear warmup (0% by default), constant, linear cooldown (20%) to min_lr=0.0
  - Token-based scheduling (not step-based)

## Auxiliary Loss (nl_aux_model.py)

- **NextLat:** Predicts next-token hidden states from (current_hidden ∥ next_token_embedding)
- **Architecture:** input_proj(2×hidden → hidden) + 3-layer MLP (hidden→5120→5120→hidden) + residual + RMSNorm
- **Loss:** smooth_L1 on hidden state prediction, sum over hidden dim, mean over valid tokens
- **nl_mult:** 2.0 (auxiliary loss = 2× main CE loss contribution)
- **nl_depth:** 1 (single-step prediction, no multi-step rollout for 10M model)
- **Discarded after training** (not part of saved model)

## Data Pipeline

- **Tokenizer:** ByteLevel BPE, vocab 32768, trained on corpus JSONL with `<pad>` special token
- **Format:** Parquet with packed sequences, segment_ids for document boundaries
- **Batch:** microbatch_tok=32768 tokens, total_batch_tok=32768, no gradient accumulation
- **Sequence length:** 512
- **Epochs:** 10 (total ~100M word exposure for 10M-word corpus)
- **Checkpoint schedule (strict-small):** chck_1M through chck_9M at 1M-word intervals in epoch 1

## Training Settings

- **torch.compile:** True (critical for FlexAttention performance)
- **Precision:** BF16 autocast
- **Seed:** 0 (model), 0 (data)
- **Single GPU:** No DDP needed for 10M (microbatch = total_batch)

## What We Need to Adapt for Official Corpus

1. **Tokenizer:** Train a 32768 ByteLevel BPE on the official BabyLM Strict-Small corpus (JSONL format)
2. **Data:** Convert official corpus to tokenized parquet with document-level segment_ids
3. **Optimizer:** Implement Aurora (polar decomposition) locally — the code is self-contained in optimizer.py
4. **Training:** Adapt train.py to use our paths, HF cache, and checkpoint saving convention
5. **Evaluation:** Use `--backend causal` with our local evaluator after each epoch
6. **Comparison:** Train identical architecture on official data, compare to local RecGPT reference

## First Training Run Design

**Full RecGPT recipe on official data:**
- Same architecture (hidden 768, embed 192, FFN 12288, depth 16, heads 12)
- Same optimizer (Aurora + Adam + CWD)
- Same schedule (no warmup, 20% cooldown)
- Same auxiliary loss (NextLat nl_mult=2.0, nl_depth=1)
- Same BPE vocab size (32768) but trained on official corpus
- 10 epochs × official 10M words

**Target:** If this matches or exceeds local RecGPT on BLiMP/Supplement/COMPS/GlobalPIQA/EWoK, then architecture IS the value-add. If it falls substantially short, the custom corpus was load-bearing.

## Ablation Arms (after first run)

- **No recursion (depth=1):** Tests whether recursive weight reuse drives the phenotype
- **No NextLat (nl_mult=0):** Tests whether auxiliary hidden-state prediction is important
- **AdamW only (no Aurora):** Tests whether the optimizer is important
- **Tied embeddings:** Tests the factorized/untied claim
- **Smaller recursion (depth=4,8):** Tests depth-performance trade-off

## Evidence files
- RecGPT repository zip: `data/external/38424423a97c_serdardoesml-bblm26-recgpt-*.zip`
- Local reference: `notes/recgpt_local_reference_consolidated.md`
- Score JSON: `data/recgpt_public_causal_scores.json`
- Reading fix: `training/runs/recgpt_reading_spacefix/detailed_results.json`
