# execution state — MLM-primary same-corruption MNTP auxiliary: construction design

## Scientific hypothesis
The clean-Qwen DeBERTa-v2 8×480 MLM backbone can be strengthened by adding a
same-corruption directional prediction auxiliary that adds gradient-compatible
credit-assignment signal without displacing the bidirectional reconstruction
pressure responsible for Supplement, EWoK, Entity, and COMPS strength.

## Why this is different from causal attention verification
causal attention verification REPLACED whole MLM batches with causal batches:
- Removed 15% or 50% of the MLM reconstruction signal entirely
- Used uncorrupted input with lower-triangular attention (fundamentally different computation)
- Result: causal15 lost Supplement (-2.94) and EWoK (-1.62); causal50 broadly damaged

execution state ADDS to every MLM batch:
- Full 15% WWM-MLM loss preserved on every batch (zero displacement)
- Same corrupted input, same bidirectional attention, same forward pass
- At position j-1, predict what was masked at position j (adjacent reconstruction)
- Norm-calibrated so aux gradient is only 15% of MLM gradient magnitude

## Gradient probe evidence (mlm primary mntp gradient finding and route)
- Same-corruption MNTP is positively aligned with MLM at full-parameter level
  (mean cosine ~0.06-0.10 across trained checkpoints)
- No full-gradient conflict at any checkpoint (init, 10M, 50M, 100M)
- Aux/MLM gradient-norm ratio ~2-4x at trained checkpoints → norm calibration needed
- token_shift variant more aligned than word_start variant
- Layer structure: embeddings/lm_head/mid-upper layers consistently positive;
  earliest layers mildly negative at 10M, recovering by 50M

## Implementation
- Trainer: `scripts/mlm_mntp_auxiliary_trainer.py`
- Every batch: single forward → logits → compute both MLM loss and MNTP loss
- Two-backward per step: MLM.backward(retain_graph=True) → measure mlm_norm →
  store grads → zero → aux.backward() → measure aux_norm → combine
- EMA tracking: ema_mlm_norm and ema_aux_norm with β=0.9
- Lambda: target_ratio × ema_mlm / ema_aux (clamped ≤ 1.0)
- Combined grad: mlm_grad + λ × aux_grad → clip → optimizer step

## Smoke test results
- MLM loss earlier analysis = **9.811304092407227** — EXACT match to clean-Qwen loss_first
- This confirms byte-identical init, data, masking, and forward path
- Lambda stabilizes at ~0.167 (init ratio ~0.9, lower than trained ~3.0)
- effective_aux_ratio stays near 0.15 target

## What to expect during training
- As training progresses: MLM loss drops (→ ~2.5), aux loss stays higher (~12)
- Aux/MLM norm ratio will increase from ~0.9 to ~3-4x
- Lambda will decrease from ~0.17 to ~0.04-0.05 to maintain the 15% target
- The EMA with β=0.9 tracks this smoothly (10-step half-life)

## Evaluation plan
1. After training: verify scientific_metrics.json shows complete status
2. Run full nine-column evaluation on frozen chck_100M endpoint
3. Compare to clean-Qwen Overall 41.34429066479573
4. If positive delta with coherent column profile → proceed with SOTA route
5. If negative or neutral → close same-stack auxiliary route

## Fallback routes (if this fails)
Preserved from mlm primary mntp gradient finding and route route note and earlier research:
- Representation/architecture: attention gating, GEGLU FFN, layer weighting
  (GPT-BERT-grounded changes that don't touch the corpus)
- Tokenization: 40k vocab (leader uses 40k SentencePiece BPE; we use 16k)
- Same-window corpus preserved regardless of objective route outcome
