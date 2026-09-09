# causal gpt transfer experiment design: Causal GPT Architecture-Transfer Experiment Design

## Scientific Question

Does the compact semantic second-view mechanism transfer beyond DeBERTa-v2?
Specifically: does a decoder-only GPT2LMHeadModel trained on the compact-view pool
outperform the same model trained on the matched repeat pool?

## Two Implemented Corrections

1. **Tokenizer neutrality**: The spatial repair route status tokenizer was fitted on the compact-reinvest pool,
   giving the compact-view arm a tokenizer advantage. Solution: BPE 16k trained ONLY on
   the 9,576,489-word common filler block that is identical between both arms.
   SHA: `e6723383f9642945134a09a9641d52e595624d3c5a99622e2a47e4d912378102`

2. **Order balance**: Causal left-to-right models see source→view as continuation/compression.
   Solution: 50% source→view, 50% view→source order per pair (seed 171043: 6031/6124),
   applied identically to both arms.

## Experiment Setup

### Arms
- **compact-view**: Filler (9.58M words) + 12,155 order-balanced compact semantic views (0.42M words)
  Pool SHA: `fa2216ce8f376767f900447e1e44e1932177583229119223d0554a78414af1ed`
- **repeat**: Same filler + 12,155 order-balanced first-N-word repeats (matched 0.42M words)
  Pool SHA: `e40b3a8978a00897020e1075557b658196c1aeb2a0136b4cba871ccdc334a61d`

### Shared
- **Tokenizer**: Neutral BPE 16k on filler only
- **Model**: GPT2LMHeadModel, 8 layers, 480 hidden, 8 heads, 256 positions
  - 30,156,480 parameters (tied embeddings)
  - Dropout 0.1 (residual, embedding, attention)
- **Training**: Next-token prediction, AdamW lr=6e-4, warmup 5%, cosine decay
  - Batch 128 × 256 = 32,768 tokens per step
  - 10 epochs = 100M words exposure
  - Dense checkpoints every 2M words (50 per run)
- **Seed**: 43022 (matching DeBERTa reference)
- **Evaluation**: Official `--backend causal` zero-shot + Reading → cheap7

### Matched Controls
- Both pools exactly 10,000,000 words
- Identical filler block (61,735 rows, 9,576,489 words)
- Same tokenizer (filler-only, neutral)
- Same model architecture, seed, optimizer, schedule
- Same order-balance assignment (50/50 source↔view)
- Only difference: compact semantic view text vs first-N-word repeat text

## Launch Plan
```bash
# GPU0: compact-view
python causal_gpt_trainer.py \
  --pool data/causal_transfer_scaffold/causal_compact_10M.jsonl \
  --tokenizer data/causal_transfer_scaffold/neutral_tokenizer \
  --run-dir training/runs/causal_compact_seed43022 \
  --gpu 0 --seed 43022

# GPU1: repeat
python causal_gpt_trainer.py \
  --pool data/causal_transfer_scaffold/causal_repeat_10M.jsonl \
  --tokenizer data/causal_transfer_scaffold/neutral_tokenizer \
  --run-dir training/runs/causal_repeat_seed43022 \
  --gpu 1 --seed 43022
```

## Interpretation Contract (fixed before results)

### Positive result
Compact-view cheap7 > repeat cheap7 at multiple checkpoints (20M, 40M, 60M, 80M, 100M)
with broad family improvement (not just BLiMP/COMPS churn or few-example GlobalPIQA).
This would support a **transferable** data-efficient learning principle.

### Negative result  
Compact-view ≈ repeat or compact-view < repeat: the mechanism is DeBERTa-specific
(possibly requiring bidirectional attention for multi-view learning). Would redirect
toward DeBERTa-internal improvements only.

### Edge cases
- Only early advantage that fades: the mechanism provides early learning acceleration
  but not mature endpoint improvement (cf. U256 pattern).
- Family-specific advantage: similar to DeBERTa's allocation tradeoffs.
  Not sufficient for transferable principle.

## Relation to DeBERTa triangle
(msg 219) is training compact_repeat_reinvest and adjbreak_reinvest on DeBERTa.
causal GPT test is the architecture-transfer leg of the triangle:
- DeBERTa compact-view vs repeat → within-architecture evidence
- GPT compact-view vs repeat → cross-architecture evidence
- Together: whether compact semantic views are a general data-efficient principle
