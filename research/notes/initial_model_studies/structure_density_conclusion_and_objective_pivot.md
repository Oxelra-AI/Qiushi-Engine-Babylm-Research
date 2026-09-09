# structure density conclusion and objective pivot — Structure-density route conclusion and pivot to objective-side mechanism

## The complete three-arm screen (S1 10M, DeBERTa-v2 12×384, baseline16k, AdamW, flat WWM)

| column | high_entity_state | matched_low | uniform |
|---|---:|---:|---:|
| BLiMP | 53.75 | 53.67 | 54.17 |
| Supplement | 50.58 | 50.44 | 51.06 |
| Entity | 16.37 | 17.52 | 17.08 |
| COMPS | 50.34 | 49.55 | 49.74 |
| GlobalPIQA parallel | 17.48 | 18.45 | 17.48 |
| GlobalPIQA nonparallel | 55.00 | 51.00 | 55.00 |
| GlobalPIQA mean | 36.24 | 34.73 | 36.24 |
| Reading mean | 6.68 | 7.22 | 7.11 |
| Full EWoK | 49.13 | 49.25 | 48.95 |

Sources: `structure_density_three_arm_comparison.json`, `structure_arms_full_ewok.json`.

## The key finding: data selection worked, the objective did not

The structure density conclusion and objective pivot scorer audit (`structure_scorer_recoverability_audit.json`) shows the within-source selection genuinely separated recoverable entity/state structure, not just cue tokens:

| proxy | high_entity_state | matched_low | uniform |
|---|---:|---:|---:|
| coref candidate | 98.4% | 86.5% | 93.4% |
| entity+state recoverable | 69.0% | 42.2% | 57.3% |
| state-transition candidate | 45.1% | 19.4% | 32.4% |
| cue-without-anchor (false positive) | 1.6% | 11.3% | 6.1% |

high_entity_state has **1.6× more recoverable entity+state structure** than matched_low and the **lowest** cue-without-anchor rate. The scorer measured real structure, and matched_low is genuinely structure-poor.

Yet high_entity_state did **not** improve Entity (16.37 vs 17.52, worse) or EWoK (49.13 vs 49.25, flat). It even slightly lost Entity to random `uniform` (17.08).

## Scientific conclusion

**Under plain whole-word MLM, increasing the density of recoverable entity/state structure per training word does not improve entity tracking or world knowledge at the 10M budget.** The training objective does not force the model to use the cross-mention/state structure that is present in the data.

This falsifies the "density of recoverable structure per word is the lever, holding objective fixed" hypothesis in its simple form. It reframes the bottleneck: the missing ingredient is an **objective/credit-assignment mechanism that makes the loss depend on tracking entities and states across mentions**, not merely more structured raw text.

The only consistent positive signal across data arms remains GlobalPIQA nonparallel (high/uniform 55.0 vs matched_low 51.0), which tracks lexical/topic richness, not entity binding.

## What this rules out

- Scaling high_entity_state to 100M as an Entity/EWoK route (no signal at 10M, scorer verified).
- More within-source lexical-density data variants (matched_low already structure-poor; the contrast is real and still null on targets).
- Blaming the scorer: the scorer is validated; the data contrast is real.

## The pivot: objective-side entity/state mechanism on structure-rich data

The right next mechanism keeps the verified high-structure data but changes what the loss rewards. Candidate falsifiable objective, to be built and screened against a strict control:

**Entity-anchored masking with cross-mention recovery.** On high_entity_state windows:
- Detect repeated entity anchors (capitalized repeats, repeated content-noun types) and their coreferential pronouns.
- Preferentially mask *later* mentions/pronouns of an anchor while leaving *at least one* earlier mention of the same anchor visible in the window.
- The model must then recover the masked mention using the surviving co-referent mention — this makes the MLM loss explicitly depend on cross-mention identity binding.

Required relation-breaking control: identical windows, identical number of masked tokens, identical mask positions distribution, but assign the "anchor-visible" constraint at random so no true coreferent is guaranteed visible (relation-broken masking). If anchored masking beats relation-broken masking on Entity/EWoK with matched masking budget, the mechanism is localized to cross-mention binding rather than token difficulty.

Second candidate if binding masking is weak: **state-tracking span objective** — mask the post-transition state token (e.g., location after a move verb) and require prediction from the pre-transition context, versus a control that masks a random token of equal frequency.

## Why this is the load-bearing move

Every data-side route tested here (WikiAuto alignment, structure density) shifted mainly GlobalPIQA and never moved Entity/EWoK. The invariant across all of them is the plain WWM objective. The evidence now points at the objective as the true bottleneck for the Entity/EWoK gap to the leader. This is a transferable principle candidate: *in strict small-data pretraining, entity/world-knowledge ability is gated by whether the training objective forces cross-mention/state credit assignment, not by the density of such structure in the raw text.*
