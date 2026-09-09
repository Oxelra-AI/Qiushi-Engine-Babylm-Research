# mntp hybrid trainer validation — MNTP hybrid trainer validation

## Fork

`training/scripts/babylm_masked_train_mntp.py`

## Key mechanism

GPT-BERT masked next-token prediction (MNTP): on deterministic every-Nth batch, mask random tokens with standard WWM corruption, then shift labels left so output at position k predicts masked token k+1, and apply a causal attention mask. The model always receives the 2D attention mask for its embedding layer; the causal mask is injected only at the encoder level via monkey-patching `encoder.get_attention_mask`.

## No-future-leak test

```
Causal: max diff pos 0-9 = 0.00e+00 (exactly zero)
Causal: max diff pos 10+ = 9.23e-01 (large, causal correctly allows self+past)
Bidir:  max diff pos 0-9 = 5.50e-03 (nonzero: bidirectional allows future influence)
```

Under the causal mask, changing a future token has ZERO effect on earlier positions. Under the standard bidirectional mask, future tokens DO influence earlier positions. Separation is verified.

## Design contract for matched S1 10M comparison

- Same official corpus, baseline16k tokenizer, data order, seed 42
- Same DeBERTa-v2 12×384/intermediate1280, p2c/c2p relative attention
- Same AdamW lr 0.001, cosine schedule, batch 256/micro 128
- Same flat WWM (mask_mode wwm, mask_prob 0.15)
- Same `max_word_exposure` and `lr_total_steps` as S1 10M
- ONLY DIFFERENCE: `--mntp_every 16` (every 16th batch uses MNTP instead of standard WWM)
- Expected: 15 MNTP steps + 230 WWM steps over ~245 total ≈ 1:15 ratio

## Comparison target

S1 10M scores (from notes/s1_10m_result_and_required_comparison.md):
- BLiMP 53.37, Supplement 52.34, Entity 17.73, COMPS 50.34
- GlobalPIQA mean 35.21, Reading mean 8.35, EWoK 50.60

The decisive test: does the hybrid improve any target-gap column (Entity, EWoK, GlobalPIQA, COMPS) without large Supplement/Reading damage?
