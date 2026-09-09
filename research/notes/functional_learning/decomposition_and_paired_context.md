# decomposition and paired context (U+R)/2 Decomposition and Paired-Context Objective

## Scientific finding: zero recipient dependence across all prior conditions

The analysis uses the following decomposition of existing margin metrics:
- **U** = d(update) = log P(new | update_context) - log P(source | update_context)
- **R** = -d(retain) = log P(source | retain_context) - log P(new | retain_context)

These decompose into two orthogonal components:
- **(U+R)/2** = recipient-dependent component = how differently the model treats update vs retain contexts
- **(U-R)/2** = shared preference = general preference for new_state over source_state regardless of context

### Decomposition table across all prior experimental conditions

| Condition | U | R | (U+R)/2 | (U-R)/2 | Recip? |
|---|---:|---:|---:|---:|:---:|
| Template baseline (held) | +0.065 | -0.056 | +0.005 | +0.061 | NO |
| earlier analysis answer-only seed43033 (held) | +4.965 | +2.599 | **+3.782** | +1.183 | **YES** |
| earlier analysis standard WWM (held) | +0.182 | -0.230 | -0.024 | +0.206 | NO |
| earlier analysis focused+bg (held) | +0.113 | -0.200 | -0.044 | +0.157 | NO |
| paired factorial result and bridge answer-only seed43035 (held) | +1.937 | -2.007 | **-0.035** | +1.972 | NO |
| paired factorial result and bridge answer+bg (held) | +1.284 | -0.979 | +0.153 | +1.132 | NO |
| Step034b interleave (held) | +1.822 | -1.157 | +0.333 | +1.490 | weak |
| Natural rows baseline | +1.935 | -1.954 | **-0.010** | +1.945 | NO |
| **paired factorial result and bridge answer-only (train)** | **+6.057** | **+3.207** | **+4.632** | +1.425 | YES |

### Key insight

Only ONE held-out condition achieved strong recipient dependence: earlier analysis answer-only with seed 43033.
All other conditions — including paired factorial result and bridge answer-only with a different seed — show (U+R)/2 ≈ 0 on held-out.

**The model learns shared new-phrase preference, not context-dependent entity tracking.**

Even paired factorial result and bridge answer-only, which learned excellent recipient dependence on TRAIN data ((U+R)/2 = +4.63),
failed to transfer this to held-out ((U+R)/2 = -0.035). The successful earlier analysis result was a lucky seed, 
not a reliable learning mechanism.

### Why standard CE fails

Standard cross-entropy on both contexts (update and retain) can be satisfied by:
1. **Context-dependent selection** (desirable): prefer new in update, source in retain
2. **Shared preference** (degenerate): prefer whichever candidate is generally more likely

The CE loss is indifferent between these strategies. Most seeds converge to strategy 2 because:
- The shared-preference gradient path is simpler (one direction applies to both contexts)
- The context-dependent path requires forming and using a query-entity → update-entity circuit

### The paired-context loss

L_paired = -log σ(d_update - d_retain) = -log σ(U + R)

This loss DIRECTLY optimizes the recipient-dependent component (U+R)/2:
- If d_update = d_retain (shared preference only), the argument is zero → loss = log(2)
- Only genuine context-dependent behavior (d_update >> d_retain) minimizes this loss
- Shared candidate preference cancels exactly because it affects both contexts equally

Combined objective: L = L_ce + λ * L_paired
- L_ce ensures absolute accuracy (can't succeed by making both wrong)
- L_paired ensures context dependence (can't succeed by shared preference)

### Experimental test

Template mode (3 arms × 500 epochs, seed 43036):
- ce_only: L = L_ce (standard both-context answer CE)
- paired_only: L = L_paired (context contrastive only)
- combined: L = L_ce + λ * L_paired

Natural-row mode (2 arms × 200 epochs on natural state-update pilot rows):
- ce_only: baseline comparison
- combined: paired intervention

### Expected outcome

If combined achieves higher held (U+R)/2 than ce_only across seeds, this establishes a specific
data-efficient learning mechanism: explicit paired-context supervision can install recipient-
dependent computation that standard answer CE fails to generalize reliably.

### Connection to the research goal

This is not yet a general data-efficient learning principle, but it addresses a concrete bottleneck:
the inherited model has zero entity-tracking ability despite correct update detection. The paired
objective tests whether the missing context-dependent circuit can be installed by changing the
training signal alone, without changing architecture, data volume, or model capacity.

If validated, the principle would be: **under limited data and compute, the form of supervision
matters more than the amount — paired-context contrasts that cancel degenerate solutions can
install computations that per-example supervision fails to reliably generalize.**

## Files

- Template experiment: `scripts/paired_context_objective.py`
  - Output: `data/paired_context_template/`
- Natural-row experiment: `scripts/paired_context_natural.py`
  - Output: `data/paired_context_natural/`
