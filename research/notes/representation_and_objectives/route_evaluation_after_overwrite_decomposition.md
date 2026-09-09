# official row association route evaluation after overwrite decomposition

Purpose: evaluate the ewok domain link interference ladder to test the confound that direct explicit-state override and action-mediated contradiction differ in evidence form. This analysis used only existing checkpoints, existing official EWoK row records, and no-training inference/aggregation.

## New artifacts

- Probe and scorer: `experiments/archive/representation_and_objectives/scripts/overwrite_decomposition.py`
- Frozen probe: `experiments/archive/representation_and_objectives/data/overwrite_decomposition/probe/overwrite_decomposition_frames.jsonl`
- Probe manifest: `experiments/archive/representation_and_objectives/data/overwrite_decomposition/probe/overwrite_decomposition_manifest.json`
- Decomposition summary: `experiments/archive/representation_and_objectives/data/overwrite_decomposition/overwrite_decomposition_summary.json`
- Decomposition note: `research/notes/representation_and_objectives/overwrite_decomposition.md`
- Order/distance table: `research/notes/representation_and_objectives/order_distance_residual_table.md`
- Official row association: `research/notes/representation_and_objectives/official_row_association.md`
- Official row content inspection: `research/notes/representation_and_objectives/update_sensitive_ewok_inspection.md`
- Independent scientific reading: `data/external/independent_review01_verifier1_integration.md`

The frozen decomposition probe has 1040 frames: 80 ewok domain link base frames times 13 conditions. It adds `prior_only_contradict`, matched action-only and prior+action variants, direct order/distance variants, and target-word-free paraphrase conditions. The target-word-free contexts contain zero literal `open/closed/empty/full` hits. Content/frame SHA256: `ce45c171554c5b468e0ef6bd26eb4b8b58330a6bac35e624949a38e7707bac68`.

## What survived decomposition

A real output-margin nonadditivity remains in many mature models. With signed margin `m`, the residual

`R = m(prior + action) - m(prior only) - m(action only) + m(no context)`

is strongly negative for direct adjacent prior+action contexts in mature checkpoints:

- legal16k_base_100M: action crossed 0.9875, combined crossed 0.0000, direct `R` median -1.2293
- scale1p75_82M: action crossed 0.5000, combined crossed 0.0000, direct `R` median -0.6761
- scale1p75_100M: action crossed 0.6125, combined crossed 0.0000, direct `R` median -0.7033
- legal40k_8x480_100M: action crossed 0.3625, combined crossed 0.0000, direct `R` median -0.5857
- fw_compact_100M: action crossed 0.0625, combined crossed 0.0000, direct `R` median -0.8489

The same qualitative suppression often survives target-word-free paraphrases:

- legal16k_base_100M: target-free action crossed 0.8250, target-free combined crossed 0.0500, target-free `R` median -0.5544
- scale1p75_82M: 0.3250 -> 0.2500, target-free `R` median -0.7987
- scale1p75_100M: 0.3750 -> 0.2375, target-free `R` median -0.7173
- legal40k_8x480_100M: 0.6750 -> 0.0625, target-free `R` median -0.9618
- fw_compact_100M: 0.6250 -> 0.0000, target-free `R` median -0.9372

This rules out a simple literal target-word repetition account for all of the effect.

## What weakened the route

The phenomenon is not form-invariant. Direct-distance variants often remove the negative residual: legal16k direct adjacent `R` median -1.2293 becomes +0.7698 at distance; scale1p75_82M -0.6761 becomes +1.0464; legal40k_8x480 -0.5857 becomes +1.5747. In contrast, target-free distance remains negative in many of the same models. This mixed order/distance behavior means the object is sensitive to surface realization, recency, token overlap, and paraphrase form. It is not yet an abstract transition-composition measurement.

The official-surface link is weak. Target-level association with update-sensitive EWoK material/physical dynamics rows is not stable, and the row-level association is essentially absent:

- `action_crossed__correct`: 849 rows with defined correlation, median row r = -0.0241, mean = -0.0038
- `direct_nonadditive_median__correct`: median row r = 0.0326, mean = 0.0156
- `targetfree_nonadditive_median__correct`: median row r = 0.0420, mean = 0.0100
- stacked model×row Pearson values for correctness are -0.0054, 0.0161, and 0.0092 respectively
- stacked model×row Pearson values for interaction sum are all small: -0.0359, 0.0437, 0.0297

The inspected EWoK material/physical dynamics rows are mostly material-affordance concept swaps such as `rigid/liquid -> break/drip`; they are not direct contradictory-prior state-update examples. Therefore the null association does not show that transition updating is irrelevant to BabyLM; it shows that the current synthetic overwrite object is not linked to the official rows we used as proxies.

## Scientific interpretation

The ewok domain link/187 sequence establishes a narrow and real behavioral fact: in small MLM checkpoints, an action can imply a result state when isolated, and contradictory earlier state evidence can suppress that implication beyond a simple additive margin model, including in some target-word-free paraphrases.

It does not establish that the model forms an entity-indexed result-state representation and then loses it. The observed residual could arise at several levels: weak event-to-result entailment, wrong entity or time binding, recency/copy bias, nonlinear contextualization before transition formation, or masked-token readout. Because the measurement is paraphrase-sensitive and not linked to official update rows, it is not a sufficient training target.

The protected `chck_82M` endpoint remains untouched and remains the practical above-frontier model. The overwrite ladder is not a reason to attach a tail, train a contrastive branch, tune templates, or reopen closed sparse20/ACS/role-switch/PVDM routes.

## Next scientific work

The stronger next object is to determine where the action-result relation fails:

1. Does the result state never form from the action?
2. Does it form but attach to the wrong entity or time?
3. Does it form and persist internally but get suppressed at retrieval/readout?

A useful next experiment should be no-training and use existing checkpoints first. It should build matched behavioral and representation probes that compare:

- explicit-state -> explicit-state supersession versus action -> action supersession, keeping evidence form constant;
- old/new order swaps with equal sentence counts and similar token counts;
- successful actions versus `tried/failed/may have` forms to separate result entailment from action lexical association;
- one-entity versus two-entity contexts where only the affected argument changes;
- changed-state queries versus unrelated-attribute queries;
- several transition families beyond only `open/closed` and `empty/full`.

If this matched probe shows a stable operation across forms, then an internal-state readout/patching experiment can test formation, binding, and readout. If the matched probe is also surface-bound or unlinked to official material, further research should leave this synthetic overwrite boundary and return to broader representation theory rooted in official EWoK/GlobalPIQA anatomy rather than training on the current ladder.
