# cross architecture item discrimination synthesis: Cross-Architecture Item-Level Discrimination — Synthesis

Created: 2026-09-02

## Core Finding

Structured semantic compression (compact views + source diversity reinvestment) does NOT 
systematically improve specific evaluation items across architectures. Instead, it creates 
**stochastic item-level churn** whose aggregate metric effect depends on the interaction 
between the evaluation metric's column-weighting structure and the architecture's baseline 
difficulty profile — a Simpson's paradox effect.

## Evidence

### 1. Item-level correlations are near zero (170,722 common items at 100M)

| Architecture pair | Pearson r | 95% CI | Agreement | Notes |
|---|---|---|---|---|
| full_deberta vs cc_nodis | 0.022 | [0.018, 0.027] | 0.534 | Both show positive column-mean |
| full_deberta vs plain_nodis | 0.012 | [0.006, 0.017] | 0.519 | Key discriminator |
| full_deberta vs roberta | −0.009 | [−0.015, −0.004] | 0.485 | Cross-architecture |
| cc_nodis vs plain_nodis | 0.064 | [0.058, 0.069] | 0.589 | Same arch, diff init |
| cc_nodis vs roberta | 0.028 | [0.023, 0.034] | 0.540 | |
| plain_nodis vs roberta | 0.008 | [0.002, 0.014] | 0.512 | |

Even between the two architectures where compact "works" (full DeBERTa and common-copy 
no-disentangle, both showing positive column-mean deltas), the item-level correlation is 
only 0.022. The items that benefit from compact training are **nearly completely different** 
across architectures.

### 2. Universal churn with architecture-independent magnitude

| Architecture | wrong→right flips | right→wrong flips | Net | Churn rate |
|---|---|---|---|---|
| full_deberta | 24,168 | 24,492 | **−324** | 28.5% |
| cc_nodis | 21,268 | 20,644 | +624 | 24.6% |
| plain_nodis | 22,470 | 21,957 | +513 | 26.1% |
| roberta | 20,181 | 19,892 | +289 | 23.5% |

All architectures show 24-29% item churn. The net direction is tiny (<0.4% of items) 
and **full DeBERTa is the only architecture where more items WORSEN than IMPROVE**.

### 3. Simpson's paradox: full DeBERTa's "advantage"

Full DeBERTa: net −324 items (more items worsened than improved under compact training).
Yet the column-mean evaluation metric shows positive improvement because:

- The improved items are concentrated in small evaluation columns 
  (Entity: 6,780 items, EWoK: 7,618 items, Supplement: 5,218 items)
- Equal column weighting gives them the same weight as large columns
  (BLiMP: 59,875 items, COMPS: 91,028 items)

Per-column item-level deltas (full DeBERTa, compact minus repeat):
| Column | Item Δ | Score Δ | Items | % wrong |
|--------|--------|---------|-------|---------|
| BLiMP | −0.0114 | ~−1.1 | 59,875 | 33% |
| Supplement | +0.0103 | ~+1.0 | 5,218 | 24% |
| EWoK | +0.0045 | ~+0.5 | 7,618 | 50% |
| Entity | +0.0100 | ~+1.0 | 6,780 | 74% |
| COMPS | +0.0023 | ~+0.2 | 91,028 | 48% |
| **Column mean** | — | **+0.3** | — | — |
| **Item-weighted** | **−0.0019** | **−0.2** | 170,519 | — |

The column mean is positive (+0.3) while the item-weighted measure is negative (−0.2).
This is exactly Simpson's paradox: the weighting reverses the sign.

### 4. Column-level patterns are architecture-dependent (not systematic)

| Column | full_deberta | cc_nodis | plain_nodis | roberta |
|--------|-------------|----------|-------------|---------|
| BLiMP | −0.011 | +0.009 | +0.000 | +0.004 |
| Supplement | +0.010 | −0.014 | +0.044 | +0.002 |
| EWoK | +0.005 | +0.006 | −0.013 | +0.005 |
| Entity | +0.010 | +0.008 | +0.001 | −0.002 |
| COMPS | +0.002 | +0.001 | +0.004 | −0.000 |

No column shows a consistent sign across all architectures (including same-architecture 
cc_nodis vs plain_nodis). This confirms the column-level effects are architecture/init-
dependent rather than systematic properties of the data treatment.

### 5. Flip overlap is slightly above chance but dominated by stochasticity

Positive-flip Jaccard overlap (observed / expected under independence):
| Pair | Jaccard | Expected | Ratio |
|------|---------|----------|-------|
| cc_nodis ∩ plain_nodis | 0.120 | 0.068 | 1.75× |
| cc_nodis ∩ roberta | 0.104 | 0.065 | 1.61× |
| plain_nodis ∩ roberta | 0.092 | 0.066 | 1.39× |
| cc_nodis ∩ full_deberta | 0.090 | 0.071 | 1.27× |
| full_deberta ∩ plain_nodis | 0.089 | 0.073 | 1.21× |
| full_deberta ∩ roberta | 0.079 | 0.069 | 1.15× |

Same-architecture pairs share slightly more flipped items (cc_nodis ∩ plain_nodis: 1.75×).
Full DeBERTa (with p2c/c2p) has the LOWEST overlap with all other architectures, confirming 
that the positional-score pathway creates a distinct optimization landscape — but this 
distinctness is not aligned with the compact treatment's aggregate benefit.

## What This Establishes

1. **The compact treatment does not teach specific linguistic competencies across architectures.**
   Items that benefit from compact training in one architecture are nearly uncorrelated with 
   items that benefit in another (r < 0.03). The treatment creates comparable amounts of 
   item-level churn (~25%) in all architectures.

2. **The "DeBERTa compact advantage" is substantially a metric-architecture alignment artifact.**
   Full DeBERTa compact harms more items than it helps (net −324), but the evaluation metric's 
   equal column-weighting amplifies improvements in small columns. Under item-weighted 
   evaluation, the advantage reverses.

3. **The compact treatment is a stochastic optimization perturbation, not a systematic learning
   improvement.** The specific items affected are determined by the interaction of the data 
   treatment with each architecture's random initialization and optimization dynamics. 
   Architecture-specific features determine ~10-15% of item flips (Jaccard 1.15-1.75×); 
   the rest is stochastic.

4. **Column-level effects are not systematic.** No evaluation column has a consistent compact 
   advantage across all architectures. This rules out the interpretation that compact training 
   systematically improves specific linguistic families (syntax, semantics, world knowledge).

## What This Changes About Prior Research

- The earlier analysis-250 narrative that compact views create a "data-efficiency mechanism" in DeBERTa 
  must be revised. The mechanism is primarily metric-architectural alignment, not architecture-
  specific learning.
  
- The GPT-2/RoBERTa transfer "failures" (related experiments) are not really failures of the 
  compact mechanism — the same churn occurs, but the metric interaction is different.
  
- The architecture interaction result/250 architecture-interaction causal chain (positional-score necessity → init 
  sensitivity → common-copy recovery) is correct but now has a simpler interpretation: 
  different initializations create different stochastic perturbation patterns.

## Implications for Data-Efficient Learning Principle

The generalizable principle is not about a specific learning mechanism but about the nature
of data treatment effects in data-limited settings:

**Structured semantic compression acts as a stochastic optimization perturbation that 
redistributes item-level competence. In data-limited training, models learn fragile, 
perturbation-sensitive representations where the specific items mastered depend heavily on 
initialization and training dynamics. Data treatments change the optimization trajectory but 
not systematically in the direction of improved competence — the aggregate effect on any 
evaluation metric is determined by the metric's weighting structure relative to the 
architecture's baseline difficulty profile.**

This has practical implications:
- Data treatment "effectiveness" should be evaluated on item-weighted metrics alongside 
  column-weighted metrics
- Apparent architecture-specificity of data treatments may reflect metric artifacts, not 
  genuine learning mechanisms
- In data-limited settings, the research priority should be treatments that create systematic 
  (non-stochastic) improvements, not treatments optimized for a specific metric

## Proposed Follow-Up Experiments

1. **Verify with extractive arms**: Check if source-only extractive data shows similar
   universal churn (same-direction redistribution) — if so, the stochastic perturbation is 
   not specific to semantic compression

2. **Test at different scales**: Does the stochastic finding persist at earlier checkpoints 
   (20M, 50M) or does it emerge only late?

3. **Cross-validate with the paired-world probe**: Similar role-assignment performance in compact-trained DeBERTa and RoBERTa would provide further support for the stochastic interpretation.

4. **Design metric-robust treatments**: Can we design data treatments whose item-level effects 
   ARE systematic (correlated across architectures)?

## Artifacts

- Main analysis: `experiments/archive/frontier_consolidation/data/cross_architecture_item_discrimination`
- Extension: `experiments/archive/frontier_consolidation/data/cross_architecture_item_discrimination/cross_architecture_extension.json`
- Scripts: `cross_architecture_item_discrimination.py`, `cross_architecture_extension.py`

## Boundary

CPU-only analysis on existing evaluation artifacts. No model loading, training, evaluation,
SuperGLUE, AoA, upload, or leaderboard submission. Protected assets unchanged.
