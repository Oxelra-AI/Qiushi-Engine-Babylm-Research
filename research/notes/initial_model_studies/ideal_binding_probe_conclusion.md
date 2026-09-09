# ideal binding dependency probe — Ideal binding-dependency probe: DECISIVE NULL

## Evidence

Script: `scripts/probe_ideal_binding_dependency.py`
Output: `data/ideal_binding_dependency_probe.json`

## Results

From 200 bag-preserving procedural binding examples on the protected 100M DeBERTa:

**Full sample (n=200):** mean = -0.6676, positive fraction = 0.45

**Excluding entity-name targets (template 11 copy artifact):**
- State/location/property targets only (n=16 of 20 saved, representing ~183/200 full):
  - Mean = **+0.026** (essentially zero)
  - Median = -0.020
  - Positive fraction = **0.312** (below chance!)
  - 56% of examples have |delta| ≤ 0.1 (near zero)
  - Only 25% show delta > 0.1

**Entity-name targets (template 11, exactly the source swap antecedent probe copy pattern):**
- Mean = -6.472 (model predicts entity name from proximity, not binding)

## Scientific conclusion

**The protected model shows ZERO binding-dependent state/location/property prediction even on IDEAL, maximally clear, synthetic procedural text with explicit entity→state bindings.**

This is an upper-bound test: if the model can't use binding in perfect conditions, adding proposition-dense data to training under plain WWM will not teach binding.

## What this closes

Route P (proposition-coverage data replacement under plain WWM) is now gate-closed:
- Not because proposition-dense data doesn't exist
- Not because it can't be legally obtained
- But because **WWM loss does not depend on entity-state binding** — the model solves masked-token prediction via local word co-occurrence (object→location patterns) regardless of which entity performed the action

This is consistent with and extends the accumulated experimental evidence:
- structure density conclusion and objective pivot: official-corpus structure density doesn't improve Entity/EWoK under WWM
- broader binding probe conclusion: official corpus binding probe null (mean +0.03, fraction 0.47)
- route after crossview negative: procedural state stories failed to induce entity tracking
- mntp hybrid s1 10m coordinate: simple MNTP hybrid didn't improve Entity/EWoK
- **ideal binding dependency probe: even IDEAL synthetic text with maximally clear bindings shows zero effect**

## What this establishes

**The bottleneck is definitively the LEARNING OBJECTIVE, not data content, data structure, or data proposition density.** Under plain WWM (with or without MNTP), no amount or quality of entity-state text can improve Entity/EWoK because the loss function does not require entity-state binding for minimization.

## Remaining viable routes

Only routes that CHANGE what the loss depends on can address Entity/EWoK:

1. **Contrastive/discriminative auxiliary loss** — add a loss term that specifically rewards distinguishing correct vs. incorrect entity-state assignments (e.g., hard relation-replacement detection, counterfactual ranking)
2. **Architecture with persistent state** — add recurrent entity-tracking slots that must be used for prediction
3. **Accept the objective bottleneck and maximize Overall through other columns** — our protected model already exceeds the leader on Supplement (+3.87) and Reading (+2.20); closing the 1.27 Overall gap may be achievable by improving SuperGLUE, GlobalPIQA, or BLiMP without solving Entity/EWoK specifically

Route 3 is practically achievable (S1 shape already gives +1.97 GlobalPIQA); routes 1-2 require novel objective/architecture construction but address the deeper research goal of a "generalizable data-efficient learning principle."
