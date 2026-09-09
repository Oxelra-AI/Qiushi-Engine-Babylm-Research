# mixture experiment design — Full SOTA Path Plan

## Current research state

**Inherited baseline:** 40.7028 Overall (9 columns)
**Target:** >41.8 (leader `wwm_curriculum_simplification_40k`)
**Gap:** ~1.1 Overall points = ~9.9 total column points needed

## What we now know

### Data experiments (related experiments, closed routes):
- Same-content ordering: net negative (-0.357 proxy)
- Residualized C/S selection: no broad gain
- Unconditional WWM→token switch: no net gain on strong backbone
- 100% aligned paired data: Entity gain +7.38, BUT aggregate flat vs baseline

### Entity/adjacency mechanism (paired alignment design):
- Coherent within-sequence adjacency → large Entity Tracking lift
- ALIGNED (WikiLarge+SynCSE): Entity 29.16 (fast full), exceeds leader's 28.45!
- But 100% paired data loses BLiMP (-3.22), Supplement (-4.4) vs baseline
- Effect is NOT specifically paraphrase — hard-neg also works (Entity +7.92)

### Leader decomposition:
The leader combines FOUR changes simultaneously. We've tested one (masking schedule)
and shown it doesn't help alone. The remaining three are likely interactive:

| Factor | Leader | Our baseline | Estimated impact |
|--------|--------|-------------|-----------------|
| Data | FineWeb simplification pairs | Official 10M | Entity/EWoK gain |
| Tokenizer | 40k SentencePiece BPE | 16k BPE | BLiMP/morphology |
| Optimizer | LAMB, LR 0.007, cosine | AdamW, LR 1e-3 | Convergence |
| Architecture | 12×384, 12 heads, shared keys | 8×480, 8 heads | Reasoning depth |

## Two-phase strategy

### Phase 1: Data mixture (running now)
- **Task:** Find optimal official/aligned ratio
- **Arms:** 25%, 50%, 75% aligned (+ existing 0% and 100%)
- **Expected result:** Identify the data composition that best balances Entity gain with BLiMP/Supplement retention
- **ETA:** ~3 hours from launch (Wave A ~96 min, Wave B ~96 min)

### Phase 2: Architecture + Tokenizer + Optimizer upgrade
- **Task:** Apply leader's proven architecture/training changes to best mixture data
- **Components:**
  1. Train 40k SentencePiece BPE tokenizer on best mixture data
  2. Build DeBERTa-v2 12×384 model (leader architecture)
  3. Use LAMB optimizer with LR 0.007
  4. Train on best mixture data for 100M exposure
  5. Evaluate on all 9 official columns

**Why this should work:**
- Our ALIGNED arm ALREADY beats the leader on Entity (29.16 vs 28.45) and Supplement (60.8 vs 56.01)
- The leader's BLiMP advantage (67.2 vs 64.12) likely comes from tokenizer+architecture, not data
- A mixture that retains Entity AND adds the tokenizer/architecture advantages could beat the leader on multiple columns

### Phase 2 technical details:

**Architecture (matching leader exactly):**
```json
{
  "hidden_size": 384,
  "intermediate_size": 1280,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "vocab_size": 40000,
  "max_position_embeddings": 1024,
  "relative_attention": true,
  "pos_att_type": ["p2c", "c2p"],
  "position_biased_input": false,
  "share_att_key": true,
  "norm_rel_ebd": "layer_norm",
  "hidden_act": "gelu",
  "dropout": 0.1
}
```

**Parameter budget:**
- Embeddings: 40000 × 384 = 15.36M
- 12 layers × (self-attention + FFN) ≈ 12 × (384² × 3 + 384 × 1280 × 2) ≈ 17.5M
- Total: ~33M (similar budget to current 34.5M)

**Optimizer:** LAMB (implemented in `scripts/lamb_optimizer.py`)
- LR: 0.007 (7× our current), cosine schedule
- Weight decay: 0.01
- Betas: (0.9, 0.999)
- Bias/LayerNorm excluded from decay and layer adaptation

**Tokenizer:** 40k SentencePiece BPE (script: `scripts/train_40k_tokenizer.py`)
- Trained on official+aligned combined text (~20M words)
- character_coverage=1.0, byte_fallback=True

**Training:**
- Best mixture data (from Phase 1)
- 100M word exposure (10 passes)
- Sequence length curriculum: 64 → 256 (matching leader)
- WWM for all epochs (our evidence: token switch doesn't help)
- Batch size: 256 (or scaled with sequence length)

## Decision rules

After Phase 1 evaluation:
- If best mixture equal-7 > baseline equal-7 (42.854): use that fraction for Phase 2
- If mixtures ≈ baseline: use 50% as default (balanced Entity/BLiMP)
- If mixtures < baseline everywhere: use 25% (minimal paired data for Entity insurance)

After Phase 2 evaluation:
- If new model > 41.8 Overall: SOTA achieved → proceed to full evaluation, submission prep
- If new model 41.0-41.8: close but not there → optimize specific weak columns
- If new model < 41.0: fundamental issue → return to Explore

## Timeline estimate

- Phase 1 training: ~3 hours (running now)
- Phase 1 evaluation: ~30 minutes
- Tokenizer training: ~5 minutes
- Phase 2 training: ~90 minutes
- Phase 2 evaluation: ~30 minutes (all 9 columns)
- Total: ~5.5 hours from now to first SOTA attempt

## Files and scripts

- LAMB optimizer: `scripts/lamb_optimizer.py`
- 40k tokenizer trainer: `scripts/train_40k_tokenizer.py`
- Mixture materializer: `scripts/mixture_materializer.py`
- Mixture launcher: `scripts/launch_mixture_training.sh`
- Mixture evaluator: `scripts/eval_mixture_fast.py`
- Mixture summary: `data/mixture/mixture_summary.json`
