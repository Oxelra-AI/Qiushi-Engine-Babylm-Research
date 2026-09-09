# Calibrated research route options

Created UTC: 2026-09-07

Historical design note: the predictions, priorities and outstanding measurements below describe the state at this date, not later experimental outcomes.

## What the negative arm established (scientific value)

The plain-use state-update replacement arm failed as a practical SOTA route but produced two findings worth keeping in the paper:

1. **The SOTA base is a recency reader without entity gating.** Across two seeds, the base strongly takes the most recent state phrase under both true and foreign updates (+1.11 to +1.16 nats). On co-established distractor items, it lets a recent other-entity update override the target's source state. This is the first direct characterization of the base model's state-reading mechanism.

2. **Weak natural packets induce the cheapest sufficient relation, not the intended one.** The 1,666/type packet replacement shifted source-state retention on co-established distractor items (+0.03 to +0.13 nats) but did not install entity-gated binding. The binding contrast Δ[T(new-source)] − Δ[U(swapped-source)] was +0.02 at one seed and −0.09 at the other. Entity strata showed seed-basin-dependent policy overwrite: the arm pulled two different bases toward a common shallower policy rather than adding competence anywhere. This illustrates the proposed learning principle: the learner acquires the cheapest predictive relation from the training packet, not the latent compositional variable the designer intended.

## What is already measured in the ALN dose lineage

| Intervention | Word fraction | Overall effect | Ordinary heldout MLM | Notes |
|---|---|---|---|---|
| ALN (vs OFF) | 16.6% pair words | +0.35 Overall at 1 seed (AoA clean) | −0.0125, −0.0128 at 2 seeds | BLiMP +2.1, Supplement +1.6, EWoK +1.8, Entity +1.4, GlobalPIQA −4.5 |
| Compact-view-reinvest (vs clean base) | +4.2% of pool from filler | ~+0.07 Overall at 1 seed | modest | Inside seed floor; column vector not fully decomposed |
| SHUF (wrong correspondence) | same 16.6% | −0.95 to −0.71 source-recurring at 2 seeds | +0.012 to +0.014 ordinary heldout increase | Only source-recurring changed-form discounting replicated |
| DUP (exact duplication) | same 16.6% | stale retrieval +8.78 Entity at 0-relevant | NR | Changed-form cost −3.61 rel≥3; 1 seed |

The ALN dose−return curve: the first 16.6% was worth +0.35 Overall. The compact-view-reinvest added ~4.2% more aligned restatement from filler and gained ~+0.07. Marginal return is already diminishing. An additional 8−10% from filler would plausibly gain a few tenths of a point on Overall, resolvable by ordinary heldout MLM (SE ~0.002) but probably below Entity seed noise (2+ points between seeds).

## Option A: Restatement dose arm (practical bet, ~6 GPU-hours + evaluation)

**Design.** Resample rewrites for the ~70k originals whose first-pass Qwen rewrites were rejected in the compact-experience study (37,704 clean out of 110,608). Same source register by construction. Pack into rows that replace filler rather than ALN, matching the compact-view-reinvest topology and exact 100M accounting.

**Expected sign.** Positive on ordinary heldout MLM. Column vector: likely positive on BLiMP, Supplement, EWoK; possibly negative on GlobalPIQA (as ALN was). Entity: indeterminate at this dose.

**Expected magnitude.** Tenths of a point on Overall. The primary readout should be ordinary heldout loss (where the expected effect is −0.005 to −0.010 nats, resolvable against a 0.002 floor) and cheap7 as a column vector, not Entity or overall aggregate.

**Cost.** Two-seed training (~5h combined), two cheap7 evaluations (~1h). If cheap7 vector is positive, then coherent replay + full official evaluation adds ~8−10h.

**Risk.** The inherited compact-experience 25% cap on pair words suggests an expectation of diminishing or negative returns past that threshold. Failure to beat CLEAN on ordinary heldout at matched exposure was the proposed stopping criterion.

**Scientific value.** This is the first direct dose-response test on the strongest relation. Whether the strongest relation keeps paying, saturates, or crosses over when its words come from ordinary text is a first-class instance of the fixed-budget substitution question. Either answer belongs in the paper.

## Option B: Contrastive minimal-pair entity binding (mechanism experiment, ~8 GPU-hours + evaluation)

**Design.** Same-source packets with two variants: (a) target entity is updated, use sentence requires new target state; (b) another entity is updated, use sentence requires original target state. Entity identity is the only sufficient predictor. Added from filler, not replacing ALN.

**Expected sign.** If predictions hold, this is the first demonstration that fixed-budget experience installs entity-gated binding when entity identity is the only sufficient predictor. If not, it constrains the dose and complexity at which entity gating can be acquired.

**Expected magnitude.** Unknown. The T/U/N margin discrimination quantity is the primary readout.

**Cost.** Prompt design and generation (~2h), two-seed training (~5h), scoring (~1h). Higher design risk than Option A because contrastive packet generation quality requirements are strict.

**Scientific value.** Paper-level result either way. Directly tests the proposed central claim: fixed-budget experience converts to intended competence only when the intended variable is the only sufficient predictor.

## Option C: Close the practical loop and focus on the paper

**Argument.** The accumulated evidence—relation-typed readout, causal locality, asymmetric reach, recency-reader base, cheapest-sufficient-relation learner—already constitutes a paper-level contribution. The practical headroom for restatement dose is small (tenths of a point). Further practical arms may not meaningfully change the paper's scientific claims.

**Cost.** No additional GPU time for training was proposed for this option.

**Risk.** The paper lacks a positive practical demonstration of the principle improving training. The manuscript currently says "practical improvement remains unfinished." A dose arm would resolve this with either direction.

## Historical route priority

Option A was prioritized for its lower design risk and expected information per GPU-hour. Predictions were to be recorded before training so both directions would be interpretable. A positive cheap7 vector was the proposed criterion for coherent replay and full evaluation; a flat or negative result would instead support a dose-saturation finding.

Option B was deferred relative to Option A because it was scientifically stronger but higher in design risk and was not expected to contribute to Overall score.

State-margin relation-arm scoring was in progress at the time of this note and was intended as a fourth independent readout regardless of the practical route chosen.

## What needs seed43122 base SuperGLUE/AoA

Any two-seed official Overall claim requires the seed43122 base SuperGLUE and AoA, which had not been completed at the time of this note. The estimated cost was ~4 hours on one GPU. A cheap7/heldout signal from a practical arm was the proposed gate for this evaluation.
