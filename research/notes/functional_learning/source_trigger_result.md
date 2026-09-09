# source trigger experiment design Result: Source-trigger experiment reveals task-specification reversal

## Summary

source trigger experiment design crossed identity pairing (IDENT_FULL vs UNPAIRED_SRC) with attention
access (IDENT_FULL vs IDENT_BLOCKED) and task specification (TYPED vs UNINFORMATIVE)
on the shared-backbone causal-LM task with corrected T/U probes (matched token
multiset, entity-attribute swap). 3 seeds × 3 arms × 2 cue regimes × 300 epochs.

## Central finding: task specification reverses the sign

Under **typed cue** (REWRITE_CUE distinguishes backbone from SRC-target rows):

| Metric | D_ident_full | D_ident_blocked | Local access |
|---|---:|---:|---:|
| RWT total NLL | **+0.283 ± 0.331** | -0.194 ± 0.402 | **+0.477 ± 0.547** |
| Within-RWT NLL | **+0.395 ± 0.362** | -0.017 ± 0.285 | **+0.412 ± 0.517** |
| RWT family NLL | -0.112 ± 0.044 | -0.177 ± 0.119 | +0.065 ± 0.094 |
| p(SRC(a)) | -0.007 ± 0.010 | -0.004 ± 0.011 | -0.003 ± 0.002 |
| MRR | -0.020 ± 0.014 | +0.011 ± 0.037 | -0.031 ± 0.034 |

Under **uninformative cue** (CONST_CUE for all, no task info):

| Metric | D_ident_full | D_ident_blocked | Local access |
|---|---:|---:|---:|
| RWT total NLL | **-1.172 ± 0.522** | -0.234 ± 0.331 | **-0.938 ± 0.854** |
| Within-RWT NLL | -0.492 ± 0.565 | -0.028 ± 0.278 | -0.464 ± 0.832 |
| RWT family NLL | **-0.680 ± 0.048** | -0.206 ± 0.132 | -0.474 ± 0.136 |

**Task specification contribution** (TYPED minus UNINFORMATIVE):

| Metric | D_ident_full | D_ident_blocked |
|---|---:|---:|
| RWT total NLL | **+1.455 ± 0.205** | +0.040 ± 0.544 |
| Within-RWT NLL | **+0.888 ± 0.228** | +0.011 ± 0.316 |

## What the signs mean

**D_m > 0**: identity pairing creates extra true-source-specific damage.
**D_m < 0**: identity pairing creates extra true-source-specific facilitation.

Under typed cue:
- Identity pairing creates **T-specific within-RWT ranking disruption** (+0.395 wnll).
  The true source in context worsens identification of the correct rewrite token
  *more* for identity-paired than for unpaired-SRC training.
- The damage is **NOT mediated by source-token mass shift** (D_p_src_a ≈ 0).
  The typed cue successfully redirects family-level output away from SRC.
- **Blocking attention eliminates the effect** (D_ident_blocked ≈ 0).
- Identity-pairing creates a T-U gap of +0.524 on rwt_nll; unpaired creates +0.241.
  Both are positive (source presence hurts rewrite); identity doubles the gap.

Under uninformative cue:
- Identity pairing creates **T-specific facilitation** (D_rwt_nll = -1.172).
  Mostly at the family level: identity builds entity representations that put more
  mass on the RWT family when the true source is in context.
- The facilitation also requires attention access (D_ident_blocked = -0.234).

The **reversal across cue regimes is the strongest signal** (+1.455 ± 0.205).
Identity-paired models that can attend to the true source show:
- Uninformative: "I see the source, I know what family" → facilitation
- Typed: "I see the source, but cue says rewrite" → within-family disruption

## T-U gap per arm (mean ± std across 3 seeds, typed cue)

| Arm | T-U rwt_nll | T-U rwt_wnll | T-U p_src_a | T-U MRR |
|---|---:|---:|---:|---:|
| ident_full | +0.524±0.264 | +0.541±0.256 | -0.006±0.003 | -0.030±0.012 |
| ident_blocked | +0.047±0.288 | +0.128±0.262 | -0.003±0.004 | +0.001±0.022 |
| unpaired_src | +0.241±0.209 | +0.145±0.213 | +0.001±0.007 | -0.010±0.018 |

## Cue swap test (typed model, final epoch, seed-averaged)

Evaluating the same typed models with different cue tokens at inference:

| Arm | Cue | T_rwt_nll | T_p_src_a | T_mrr |
|---|---|---:|---:|---:|
| ident_full | rewrite | 7.030 | 0.0089 | 0.427 |
| ident_full | copy | 18.771 | **0.2248** | 0.400 |
| ident_full | const | 12.654 | 0.1178 | 0.406 |
| unpaired_src | rewrite | 7.931 | 0.0097 | 0.396 |
| unpaired_src | copy | 17.007 | 0.0746 | 0.404 |
| unpaired_src | const | 10.584 | 0.0372 | 0.402 |

Key observation: ident_full under copy cue puts **22.5%** of mass on SRC(a) tokens,
versus unpaired's 7.5%. This confirms identity pairing installs strong copy routing
that is CUE-GATED. Under rewrite cue, p_src_a drops to 0.89% (ident_full) and 0.97%
(unpaired). The typed cue successfully gates the copy output.

## Relation to BabyLM evidence

The synthetic result **partially bridges** toward but does NOT fully reproduce the
BabyLM source-triggered copy misfire because:

1. **BabyLM misfire mechanism**: mass shifts FROM correct targets TO source tokens.
   p(SRC) elevated under T, target probability suppressed. 11-14× specificity ratio.

2. **Synthetic mechanism** (this result): within-RWT ranking disrupted under T, but
   mass does NOT shift to SRC under typed cue. The typed cue GATES the copy output.

3. **MLM has no typed cue**: In BabyLM, all masked positions share the same objective.
   Some positions are identity-solvable (same token as source), others require
   transformation. Without a cue distinguishing them, the copy routing fires freely
   and produces the misfire measured in the BabyLM relation-learning study.

4. **The synthetic uninformative regime** (closer to MLM) shows facilitation, not misfire,
   because all predictions within one sequence use the same task (either all identity
   or all correspondence via backbone). There is no WITHIN-SEQUENCE heterogeneity
   where some positions are identity-solvable and others require transformation.

**The missing condition for reproducing the full BabyLM misfire is within-sequence
task heterogeneity**: mixed identity and non-identity prediction targets within the
same sequence, with no explicit task cue, so the copy routing generalizes from
identity-solvable positions to non-identity positions.

## What source trigger experiment design establishes

1. **Identity pairing (not just SRC exposure) creates source-specific effects.**
   IDENT_FULL and UNPAIRED_SRC share identical contexts and SRC-family targets.
   Only the source-target pairing differs. The interaction D_m is non-zero.

2. **Task specification modulates the sign** (+1.455 reversal, robust across seeds).
   This is the strongest finding: the same identity-practice installs computations
   that manifest as facilitation or damage depending on whether the evaluation
   disambiguates the requested relation.

3. **The effect requires attention access.** IDENT_BLOCKED consistently shows
   D_m ≈ 0 under both cue regimes.

4. **Copy routing is cue-gated.** Identity pairing installs strong copy routing
   (22.5% mass under copy cue vs 7.5% for unpaired) that is effectively suppressed
   by the rewrite cue. The BabyLM misfire requires the absence of such gating.

5. **Learning dynamics**: the T-U gap develops late (after epoch 100), suggesting
   source-specific effects require substantial identity practice to emerge.

## Connection to the mixture model

In the relation-learning mixture-model terms: the synthetic TYPED cue provides a reliable signal for
P_base to separate from P_id, so α → 0 under rewrite evaluation. But within the RWT
family, the identity-biased representations still degrade the P_content component,
producing within-RWT damage without mass-level misfire.

The uninformative regime cannot separate P_id from P_base, so identity practice
adds to both components, producing facilitation when the true source is useful.

## Evidence files
- Data: `data/source_trigger/results.json`
- Figure: `figures/source_trigger.png`
- Models: `data/source_trigger/models_*/`
- Script: `scripts/source_trigger.py`
- Analysis: `scripts/analyze.py`
- Design: `notes/source_trigger_experiment_design.md`
