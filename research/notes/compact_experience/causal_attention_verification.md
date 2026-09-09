# causal attention verification Causal Attention Verification

## Tests performed

1. **Future attention probability leak test**: On a 13-token sequence through all 8
   DeBERTa-v2 layers with output_attentions=True, verified that the maximum attention
   probability assigned to any future position (k > q) is **exactly 0.0**.

2. **Bidirectional forward equivalence test**: Confirmed that the manual decomposition
   `model.deberta.embeddings(ids, mask=am)` → `model.deberta.encoder(emb, am)` →
   `model.cls(hidden)` produces **identical logits** (max_diff = 0.0) to the standard
   `model(input_ids=ids, attention_mask=am).logits`.

## What this establishes

- The `[B, L, L]` boolean causal mask, processed by `DebertaV2Encoder.get_attention_mask`
  into `[B, 1, L, L]`, is correctly applied to the full disentangled-attention score
  (c2c + c2p + p2c) before softmax in transformers 4.57.6.
- Position `t` cannot read token `t+1` or later; the causal next-token target is genuinely
  unseen during forward computation.
- Splitting the model's forward into components (embeddings + encoder + cls) for the
  causal path introduces no hidden processing gap or mask normalization difference.

## Remaining interpretation constraints

The experiment tests **objective substitution** (replacing some MLM batches with causal
batches) at fixed word exposure and optimizer updates, not a clean mechanism isolation of
"consolidation." Confounded: target density, MLM update count, gradient statistics, and
masking RNG trajectory. These constrain interpretation but do not invalidate the experiment.

## Conclusion

The implementation is technically correct for its stated purpose. If training improves
downstream equal7 relative to the pure-MLM clean-Qwen reference (43.1129), the gain
is attributable to the mixed-objective recipe, not to a future information leak or
forward-path artifact.
