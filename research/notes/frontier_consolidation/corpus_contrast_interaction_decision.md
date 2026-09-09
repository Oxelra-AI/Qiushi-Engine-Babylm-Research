# corpus contrast interaction decision: Corpus Contrast Interaction Test — Decision

## Purpose
Test whether crossed context-target interaction generalizes from EWoK to independently built corpus-derived contrasts and tracks within-coordinate broad transfer.

## Method
1. **Frozen construction** (CPU, no model information): Extracted 16,165 sentences from 2,647 held-out corpus rows (never used for training). Built 9,340 contrast pairs in two independent families:
   - Family A (within-source): 8,080 pairs from sentences sharing the same data source
   - Family B (cross-source): 1,260 pairs from sentences from different sources
   - SHA256: `2347d4b90c041824377f9283db9286af47b0cd2e1dd75bc5bdae1a0d58161839`

2. **Scoring** (GPU inference, no training): Four-way PLL on 3 legal16 80M models using the same official MLM pseudo-log-likelihood convention as earlier analysis.

3. **Within-coordinate analysis**: Tested whether interaction on corpus contrasts discriminates models that differ in broad cheap7.

## Within-Coordinate Results

| Model | broad cheap7 | corpus interaction | corpus accuracy |
|-------|-------------|-------------------|-----------------|
| a02_legal16_reinvest_80M | 42.9486 | 14.723 | 90.35% |
| a02_step075_strict_innov_80M | 41.8957 | 14.674 | 90.82% |
| a02_legal16_clean_80M | 41.6022 | 14.411 | 89.72% |

**Rank ordering**: Interaction ranks all three models in the same order as broad cheap7 across all families. Corpus accuracy ranks earlier analysis ABOVE reinvest — **backwards** from broad.

## Statistical Sensitivity

| Comparison | Δcheap7 | Δinteraction | z-score | Sensitivity ratio |
|-----------|---------|-------------|---------|-------------------|
| reinvest − clean | +1.3464 | +0.3116 | **5.03** | 0.231 |
| reinvest − earlier analysis | +1.0529 | +0.0487 | **0.88** (n.s.) | 0.046 |

**The reinvest→earlier analysis interaction difference is not statistically significant** (z=0.88 overall, z=0.14 on within-source family). The metric captures the clean→reinvest treatment signal but **fails to discriminate reinvest from the closed earlier analysis route** despite a 1.05 cheap7 gap.

Per-family reinvest→earlier analysis:
- Within-source (N=8080): Δinteraction=+0.009, z=0.14 — essentially zero
- Cross-source (N=1260): Δinteraction=+0.306, z=1.93 — marginal, driven by cross-register contrast only

## Convergent evidence against the interaction direction

1. **7-model aggregate** (earlier analysis): interaction vs broad Overall Pearson=0.198
2. **Domain-level within-coordinate** (EWoK): treatment delta correlation Pearson=0.448, p=0.167 (not significant)
3. **Corpus contrasts**: reinvest→earlier analysis z=0.88 (not significant), accuracy ranking reversed
4. **Independent natural-pair test**: natural-pair corpus interaction closed; target priors confound non-minimally-contrasting targets; models already choose observed continuations easily

## Decision

**The EWoK interaction direction is closed.** The metric captures the compact-view treatment effect (which is already protected in the corpus) but is operationally useless for discriminating training improvements. It cannot distinguish strong from mediocre interventions on the same corpus substrate.

No contextual self-contrast training route, interaction-shaped objective, or further interaction measurement refinement is authorized.

## What This Means for the Research

The remaining +0.542 Overall gap (41.258 → 41.8) is not addressable through:
- Masking credit reallocation (word-mean, innovation-biased, exact-swap)
- Tokenizer vocabulary changes (minfreq50, byte-alphabet, 40k)
- Source-view consistency optimization
- Depth alone
- Contextual interaction measurement

The compact-view reinvestment corpus is validated and protected. The gap is in the **legal representation/optimization package**, which has not been systematically optimized. The inherited recipe (AdamW lr=0.001, warmup 0.06, weight_decay 0.01, batch 256, mask 0.15, fixed seq256) was never tuned for the legal tokenizer. The leader uses LAMB lr=0.007 on a different architecture — a very different optimization point.

## Artifacts
- Frozen contrasts: `data/frozen_corpus_contrasts/`
- Scoring results: `data/corpus_contrast_scoring/`
- Construction script: `scripts/corpus_contrast_construction.py`
- Scoring script: `scripts/corpus_contrast_scoring.py`


## Optimization Screen Proposal

### Current recipe (never varied for legal tokenizer)
- Optimizer: AdamW
- Learning rate: 0.001
- Warmup fraction: 0.06 (~152 of 2529 steps)
- Weight decay: 0.01
- Batch size: 256
- Masking rate: 0.15 (fixed WWM)
- Sequence length: 256 (fixed)
- Architecture: DeBERTa-v2 8×480 (34.5M params)

### Training loss
- First loss: 9.838
- Final loss (100M): 2.553
- Clean control final loss (80M): 2.551

### Available resources
- 100 checkpoints saved (chck_1M through chck_100M)
- Cheap evaluation exists at 20M, 70M, 80M (treatment trajectory)
- Full evaluation exists at 100M only

### Cheapest decisive test: learning rate at 40M
1. Evaluate existing chck_40M with cheap columns → free reference baseline
2. Train lr=0.002 for 40M words (~30 min, 1 GPU) → candidate
3. Optionally train lr=0.0005 for 40M (~30 min, parallel on GPU1) → second candidate
4. **Decision rule**: if any candidate beats reference at 40M by ≥0.3 mean7, continue to 80M; if all worse, close lr and try next variable

### Other unexplored variables (prioritized)
- **Masking rate**: 0.20 or 0.25 instead of 0.15 (more signal per step on repeated data)
- **Warmup**: 0.10-0.15 instead of 0.06 (longer warmup on noisier legal embeddings)
- **Optimizer**: LAMB with higher lr (the leader uses LAMB lr=0.007)
- **Weight decay**: 0.005-0.05 range
- **Checkpoint selection**: full eval at 80M or 90M to check if 100M is optimal for SuperGLUE

### Related Comparisons
SGCR training/evaluation and EWoK interaction measurement remained pending. The natural-pair corpus interaction route was independently closed. Support-sharing representation and optimization/recipe search remain distinct scientific alternatives.
