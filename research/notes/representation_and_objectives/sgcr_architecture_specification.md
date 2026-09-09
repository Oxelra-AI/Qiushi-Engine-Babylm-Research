# corrected entity gate stats — Support-Gated Compositional Residual (SGCR) Architecture Specification

## Scientific Motivation

Legal40k tokenization gives shorter segmentation and less sequence truncation than legal16k, improving BLiMP/Supplement/EWoK, but 63.2% of used legal40k tokens occur fewer than 50 times in the 10M training pool, causing GlobalPIQA/Entity/COMPS collapse (legal40k decision and next route). The support-sharing hypothesis (discriminating span support): most low-count legal40k tokens decompose into well-supported legal16k components from the same 10M pool (92.51% of count<20 tokens have all components ≥50). Low-support tokens are enriched in answer-discriminating spans for GlobalPIQA (ratio 1.87) and COMPS (ratio 2.92), the exact tasks where legal40k loses most ground.

SGCR preserves **exact** legal40k tokenization, labels, stream, masking, and data order while adding a jointly-trained low-dimensional legal16k component table and a support-dependent gate. Rare tokens borrow representation strength from well-supported components; common tokens use their standard learned embeddings.

## Architecture

### Base Model
DeBERTa-v2, either 12×384/FFN1280 (depth) or 8×480/FFN1920, with legal40k tokenizer. The depth model has 38,421,952 parameters.

### New Modules

1. **Component Embedding Table** — `nn.Embedding(16384, d_comp)`
   - Shared table indexed by legal16k token ids
   - d_comp is a hyperparameter (proposed: 64 for hidden=384, 96 for hidden=480)

2. **Component Projection** — `nn.Linear(d_comp, hidden_size, bias=True)`
   - Projects component space to model hidden dimension

3. **Static Buffers (no learnable parameters):**
   - `rho`: tensor of shape (40000,), where `rho[t] = n_t / (n_t + K)`
   - `decomp_ids`: tensor of shape (40000, max_comp_len), padded legal16k component ids
   - `decomp_lengths`: tensor of shape (40000,), valid component count per token
   - K is a hyperparameter (proposed: 50)

### Forward Pass — Input Embedding (Additive Residual Formulation)

```
W_std = word_embeddings.weight                 # (40000, hidden_size)
comp_embs = component_embeddings(decomp_ids)   # (40000, max_comp_len, d_comp)
mask = arange(max_comp_len) < decomp_lengths   # (40000, max_comp_len)
comp_mean = (comp_embs * mask.unsqueeze(-1)).sum(1) / decomp_lengths.clamp(min=1).unsqueeze(1)
comp_correction = component_projection(comp_mean)  # (40000, hidden_size)
W_eff = W_std + (1 - rho).unsqueeze(1) * comp_correction   # (40000, hidden_size)
output = F.embedding(input_ids, W_eff)         # (batch, seq, hidden_size)
```

**Why additive residual, not multiplicative:**
- Cold init (zeros): W_eff = W_std + 0 = W_std **exactly** (verified: max_diff=0.0)
- Gradient to W_std: always ∂L/∂W_eff (full gradient, like standard model)
- Gradient to comp: (1-rho) * ∂L/∂W_eff (stronger for rare tokens)
- After training: low-rho tokens deviate 17.9× more than high-rho tokens

Position embeddings, LayerNorm, and dropout proceed as standard.

### Forward Pass — Output Decoder

Weight tying means `decoder.weight = word_embeddings.weight`. Under SGCR, both input and output use the same effective embedding table `W_eff`:

```
# In cls.predictions:
hidden → transform.dense → GELU → transform.LayerNorm → logits
logits = hidden_transformed @ W_eff.T + decoder_bias   # (batch, seq, 40000)
```

This is the critical mechanism: rare tokens get output logits partially determined by their component compositions. A token the model has seen only 5 times can still produce reasonable output logits if its components (seen hundreds of times each) have learned good representations.

### Zero-Sharing Limit

When K = 0: rho[t] = n_t / (n_t + 0) = 1 for all t with n_t > 0.
→ W_eff = 1 · W_std + 0 · comp_proj = W_std exactly.
→ Model is **identically** the standard legal40k DeBERTa-v2.

When K → ∞: rho[t] → 0 for all t.
→ W_eff = comp_proj everywhere — all tokens represented by their components.

K = 50 interpolates: tokens with count 50 have rho = 0.5 (equal mix); count 200 → rho = 0.8; count 10 → rho = 0.167.

### Parameter-Matched Nonsharing Control

**Uniform-gate control:** Same architecture, same parameters, but `rho[t] = mean(rho)` for all tokens. This adds identical capacity without support-based routing. If SGCR outperforms the uniform control, the benefit is from support-dependent routing, not from added embedding capacity.

**Random-decomposition control:** Same architecture, but each legal40k token's decomposition is a random subset of legal16k ids with matching length distribution. Same capacity and gate values, but no meaningful compositional structure.

### Initialization Strategy (from pretrained checkpoint)

**Component embeddings:** Initialize `component_embeddings[c]` as the projection of the pretrained legal40k embedding for the legal40k token that most closely matches legal16k token c (cosine similarity in pretrained space, or directly the token that has the same surface form if it exists in both vocabularies).

**Component projection:** Initialize as a scaled identity-like mapping. For d_comp < hidden_size, initialize the weight as the first d_comp rows of an orthogonal matrix scaled by sqrt(hidden_size/d_comp), and bias as zero.

**Alternative cold start:** Initialize component_embeddings to zero and projection to zero. Since rho ≈ 1 for common tokens, the model starts identical to the pretrained state and gradually learns component representations.

**Preferred:** Cold start (zero initialization). It guarantees the model starts at exactly the pretrained loss and the component table learns from gradient signal, not from a heuristic initialization.

### Parameter Count

For 12×384 (hidden=384):

| d_comp | comp_emb | proj | total new | % of base |
|--------|----------|------|-----------|-----------|
| 32     | 524,288  | 12,672 | 536,960 | 1.40%   |
| 48     | 786,432  | 18,816 | 805,248 | 2.09%   |
| 64     | 1,048,576 | 24,960 | 1,073,536 | 2.79% |
| 96     | 1,572,864 | 37,248 | 1,610,112 | 4.19% |

d_comp=64 adds ~1.07M parameters (2.79% of base) — modest, much less than the 15.36M in word_embeddings alone.

For 8×480 (hidden=480):

| d_comp | comp_emb | proj | total new | % of base |
|--------|----------|------|-----------|-----------|
| 64     | 1,048,576 | 31,200 | 1,079,776 | ~2.4% |
| 96     | 1,572,864 | 46,560 | 1,619,424 | ~3.5% |

### Training Protocol

- **Same** legal40k tokenizer, compact_view_reinvest 100M stream, data order seed 43, fixed WWM 0.15, effective batch 256, AdamW LR 0.001, warmup 0.05, cosine decay
- **Same** initialization seeds (43022, 43023) for all standard parameters
- **New** component parameters: zero-initialized (cold start)
- **New** component parameters included in the same optimizer group with same LR/weight decay
- **Training from scratch** for full compliance: 100M words, 10 epochs, all parameters from random init + cold component start

### Implementation Notes

1. **Efficient W_eff computation:** Recompute W_eff once per forward pass, not per token. Since decomp_ids and rho are static buffers, the computation is a single batched embedding lookup + masked mean + linear projection + weighted sum. On H100 with vocab 40k and d_comp 64, this adds < 1ms per forward pass.

2. **Gradient flow:** Both word_embeddings and component_embeddings receive gradients through W_eff. For high-rho tokens, most gradient flows to word_embeddings; for low-rho tokens, most flows to component_embeddings and component_projection. This is the intended behavior: the model allocates learning capacity based on support.

3. **Tied output:** The effective embedding table W_eff is used for both input lookup and output projection in the same forward pass. No separate computation needed — just use `W_eff` as the decoder weight.

4. **Memory:** Extra memory for component_embeddings (16384 × d_comp) + projection + intermediate W_eff (40000 × hidden) + decomp buffers. For d_comp=64, hidden=384: ~62MB extra. Fits easily on H100.

## What This Construction Requires Before Launch

1. Corrected Entity discriminating-span measurement (with `nothing` skip) — running as s84_t10_tool1
2. Depth vector from s80_t21_tool1 — determines architecture choice and whether support-sharing addresses the remaining gap
3. Decision: if depth repairs GlobalPIQA/Entity already → protect depth endpoint, defer SGCR
4. If depth leaves GlobalPIQA/COMPS gap → SGCR is the indicated next route

## Files
- Architecture spec: this note
- Implementation: to be built after measurements confirm the route
- Gate statistics: `experiments/archive/representation_and_objectives/data/corrected_entity_gate_stats`
