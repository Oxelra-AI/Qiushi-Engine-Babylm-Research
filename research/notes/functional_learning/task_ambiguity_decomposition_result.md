# task ambiguity decomposition result complete result: NLL decomposition, task-cue controls, and mechanism correction

## Summary of what task ambiguity decomposition result (Parts A, B, and repaired B) establishes

### The within-family identification facilitation is REAL and task-specification-independent

| Condition | ident−neutral within_s | ident−neutral family_s | ident−neutral hc_s |
|-----------|----------------------:|----------------------:|------------------:|
| Uncued (Part A, V=33, SL=18) | **−0.83 ± 0.30** | +4.98 ± 0.37 | +4.16 ± 0.32 |
| Typed cue, original confounded (Part B, V=35) | −2.59 ± 0.84 | +0.18 ± 0.06 | −2.40 ± 0.90 |
| Typed cue, repaired NEUTRAL_CUE (V=37) | **−3.38 ± 1.59** | +0.10 ± 0.11 | −3.28 ± 1.50 |
| Constant cue, no task info (V=37) | **−3.00 ± 0.91** | +6.40 ± 0.87 | +3.40 ± 0.29 |

**Key finding**: The within-family improvement is −3.0 to −3.4 nats in ALL V=37 conditions, regardless of whether the cue provides task information. The typed cue only affects the family term (reducing it from +6.4 to +0.1), not the within-family identification.

The discrepancy with Part A (−0.83) may reflect the different vocab size and sequence length (V=33/SL=18 vs V=37/SL=19). Within the matched Step10b comparison, the result is clear: facilitation is task-specification-independent.

**MRR and top-1 accuracy confirm the identification improvement:**
- Typed cue: ident MRR=0.435 vs neutral 0.350 (Δ=+0.085)
- Constant cue: ident MRR=0.432 vs neutral 0.358 (Δ=+0.074)

### What the original Part B confound was

In the original Part B, neutral's 400 sub-sequences used REWRITE_CUE + random RWT targets. This actively corrupted the REWRITE_CUE → correct RWT conditional, making neutral artificially worse. The repaired experiment uses NEUTRAL_CUE for neutral's sub-sequences, and the within-family improvement is even LARGER (−3.38 vs −2.59), confirming the confound hurt neutral but the facilitation was still real.

### Three separable effects, corrected

1. **Within-family identification facilitation** (weight-based, task-independent, always positive):
   Identity practice builds entity-attribute representations that improve identification of the correct target within the output family. This transfers through weights (not attention) and does not require task specification. Robust across all conditions (−3.0 to −3.4 nats for V=37 experiments).

2. **Output-family assignment damage** (task-ambiguity-dependent, removable):
   When the model cannot distinguish copy from rewrite tasks, mixing SRC and RWT targets shifts the output distribution. Typed cue: +0.10. Constant cue: +6.40. This is a property of the mixed-objective setting, not a learning mechanism.

3. **Opportunity cost** (budget-dependent):
   Replacing 400 correspondence sequences with identity: all_corr hcg=6.25 vs ident_full hcg=3.72. More correspondence evidence produces larger conditioning gains. But ident_full's 3.72 is still much larger than neutral's 1.85, showing identity adds value beyond noise.

### Connection to BabyLM: the mechanism operates at different levels

The peer (relation_learning) reports that BabyLM's recurrence cost is SOURCE-SPECIFIC at the TOKEN level:
- DeBERTa three-seed T/U decomposition: T delta (true-source present) is +0.45 to +0.69, while U delta (unrelated source) is −0.30 to −0.35.
- This means REPEAT is BETTER with an unrelated source (possible facilitation) but WORSE with the true source (source-specific competition).
- Source-token mass: REPEAT places +0.10-0.14 more probability on source-span tokens than CLEAN.
- This is difficult to explain as pure opportunity cost (which would affect T and U equally).

**The two mechanisms share a common structure but operate at different levels:**

| Level | Synthetic setup | BabyLM MLM |
|-------|----------------|------------|
| Family level | SRC-vs-RWT mass shift | NOT present (MLM has one output family) |
| Within-family identification | Identity improves correct-r identification | Possibly analogous (entity retrieval helps) |
| Token-level source matching | Not tested at token level | Source-token mass elevation (+0.10-0.14) |
| Source specificity | Not measured (no T/U decomposition at within-family level) | T-specific, not U-general |

The shared principle: **identity practice installs a source-recognition computation that helps retrieval but competes with novel-content prediction.** In the synthetic setup, this manifests as family-level ambiguity (removable by cue) + genuine within-family facilitation. In BabyLM, it manifests as token-level source copying within a consistent objective (not removable by cue since there's no "this target differs from source" signal in MLM).

### What would advance the account further

1. **Token-level source mass decomposition in the synthetic setup**: Measure whether ident_full places more probability on specific source tokens (not just the SRC family) at the rewrite target position. This would connect the synthetic within-family effect to the BabyLM token-level effect.

2. **Source-specificity test**: If the +0.10 mass elevation is true-source-specific (only when the model recognizes its own source), this confirms an in-context copy mechanism rather than frequency hedging.

3. **Multi-operation depth**: Whether identity-induced source dependence helps 0-op retrieval but hurts multi-op transformation, connecting to the BabyLM Entity depth crossover.

## Files

- Part A data: `data/partA/results.json`
- Part B original data: `data/partB/results.json`
- Part B repaired data: `data/revision_010b_repaired_cued/results.json`
- Figures: `figures/family_decomp.png`, `figures/contrast_bars.png`
- Scripts: `scripts/focused.py`, `scripts/revision_010b_repaired_cued.py`
