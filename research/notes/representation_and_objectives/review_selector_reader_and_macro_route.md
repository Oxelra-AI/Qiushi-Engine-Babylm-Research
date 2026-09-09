# Selector-Reader Bridge and Macro-Route Review

## Files read

small-bridge sources:

- `notes/selector_reader_bridge_synthesis_and_macro_challenge.md`
- `data/selector_reader_groupsoftmax_seed27000/selector_reader_summary.md`
- `data/lexical_selector_baseline/lexical_selector_baseline.md`
- 

macro sources:

- `research/documents/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_readout_summary.md`
- `experiments/archive/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_rows.csv`
- `research/documents/frontier_consolidation/data/reference_decomposition_readout/reference_decomposition_summary.md`
- `experiments/archive/frontier_consolidation/data/reference_decomposition_readout/contrast_window_summaries.csv`
- `research/documents/frontier_consolidation/data/margin_domain_window_readout/margin_domain_window_summary.md`

## Selector-reader bridge reading

The selector reader bridge synthesis and macro challenge result is internally strong for a narrow operation: train a direct tag-conditioned reader `R`, hold it fixed, train a separate four-way selector `M` over candidate tags, and pass the selected tag into `R`. The construction was balanced and short enough to avoid truncation: 5,632 reader rows, 11,264 selector rows, 2,816 selector groups, four candidates per group, one positive per group, max token length 110.

The fixed reader fits the direct task. Tag-only reader training reaches 0.999645 overall with sparse changed-focal 0.996094; inline direct reader reaches 1.0 on all train families. Oracle-tag held readout is about 0.994–0.995 across exact, role-swap, and paraphrase suites. Thus the reader can use arbitrary local tags to retrieve changed and stable states once the address is explicitly supplied.

The repaired group-softmax selector also fits exact role wording: train top-1 1.0; held exact namespaces 0.996875 and 1.0; role-swap 1.0; train-changed exact 0.9984375. Composition with selected tags reaches the oracle reader ceiling on those same surfaces: held exact nsA 0.99375, nsB 0.9953125, role-swap 0.99375, trainChanged exact 0.9921875. This properly overturns the stronger reading of the earlier monolithic role-query failure: role-to-address-to-state computation is possible in this constructed interface when the selector is directly taught as a selector and the reader is protected from further updates.

The same files show why this bridge should now be set aside as a source of broad principle. Held temporal paraphrase fails in the address-identifying part of the task. Selector top-1 on `prior/revised` is 0.5109375 overall, decomposing into focal_before 0.0, focal_after 1.0, secondary_before 0.04375, secondary_after 1.0. The composed 0.75 number is inflated by stable secondary rows where before and after labels coincide. On changed-focal rows where the address matters, selected-tag composition is essentially 0.5: focal_before 0.00625 and focal_after 0.99375, while oracle tags give focal_before 0.98125 and focal_after 0.99375. The reader has the information; the selector chooses the wrong temporal address for one side of the paraphrase.

The lexical matcher baseline is decisive for interpretation. Exact string matching of query role words to the context line gives 1.0 on held exact namespaces, role swaps, and trainChanged exact, and 0.0 on `prior/revised` unless supplied a hand synonym map. This simple matcher explains the successful part of selector reader bridge synthesis and macro challenge without invoking temporal-role meaning. The bridge therefore supports exact surface-correspondence selection plus address routing, not semantic temporal transfer and not a BabyLM-scale pretraining law.

## Macro allocation reading

reference change is scientifically sound. Exact repeat `R` mixes duplicate recurrence with relatedness, so it is not a clean reference for broad source-companion value. Clean `C` places total stream quality, but a positive `V-C` does not say the companion belongs with its own source. Same-population breadth `B` is the first useful reference for the question: does the compact companion need to be related to its own source rather than merely adding independent text? If `V-B` survives beyond Entity or in balanced Entity splits, the tighter reference is the permuted companion because it preserves the compact companion multiset much better than `B` while breaking source pairing.

The current files give a mixed but informative picture.

First, Entity `V-R` is not a uniform state-record improvement. In both DeBERTa basins the late official all-18 Entity delta is positive, but neutral zero/nonzero balancing is negative:

- basin1 late mean: official all18 +4.018 pp, zero-op -9.661 pp, nonzero +6.753 pp, balanced zero/nonzero -1.454 pp.
- basin2 late mean: official all18 +2.539 pp, zero-op -9.115 pp, nonzero +4.870 pp, balanced zero/nonzero -2.123 pp.

This confirms that aggregate Entity `V-R` is strongly shaped by the benchmark mixture and a nonzero-operation tilt.

Second, breadth versus repeat has the same bad operation shape, only weaker on official aggregate: late `B-R` official +1.023 pp, zero-op -13.037 pp, nonzero +3.836 pp, balanced -4.601 pp. This is not a general finite-experience improvement of state records; it strengthens the idea that some added streams push the model toward changed-state answers while damaging no-change answers.

Third, view versus breadth inside Entity is different. Late `V-B` is robust and same-sign on both operation strata: official +2.994 pp, zero-op +3.376 pp, nonzero +2.918 pp, balanced +3.147 pp across 80/90/100M. This is the strongest macro result now visible. It says that, for Entity, aligned compact companions beat same-population unrelated breadth in a way that is not explained by the zero/nonzero tilt. That deserves a tighter source-pairing test, but only for the Entity/state-update family unless broader results change.

Fourth, broad ex-Entity `V-B` is not yet established. At the only visible 80M breadth point, exEntity5 `V-B` is -0.062 pp, while `B-C_old` is +0.896 pp. Per-family 80M `V-B` is mixed: BLiMP -0.870, COMPS -1.180, EWoK -0.150, Reading +0.070, Supplement +1.820. The cheap6 `V-B` +0.465 is Entity-dominated because Entity is +3.100. The 90M/100M broad breadth rows are largely missing, so the current data do not support a field-wide companion law.

Fifth, EWoK margins do not rescue broadness. The sampled margin readout is heterogeneous by domain and unstable across dose/window: all-domain common10_80 means are +0.0599, -0.2117, +0.0090 for the three doses, and endpoint80 MAX is -0.296. Some domains move positive and others negative. This keeps EWoK from serving as an independent broad carrier at present.

## Route judgment

The supplied-address bridge has reached a useful stopping point. It provides a clean decomposition of reader, selector, and composition in an explicit supervised interface, and it exposes the missing temporal-role map. It does not by itself answer the active goal, because the successful part is explainable by exact role-string correspondence and the semantic paraphrase test fails on the rows where address choice matters. More small supplied-address training would likely add variants of the same finding unless a new natural semantic interface is introduced.

The macro work now carries the highest value, but the next costly choice must be narrow. The present state justifies no broad BabyLM-scale training from the bridge. It also does not justify more clean totals as the next route, because totals cannot separate own-source correspondence from unrelated breadth or compact-text fertility. The only result strong enough to motivate a costly asymmetry is the balanced Entity `V-B` signal, not ex-Entity breadth. Therefore the next decisive macro question is: within Entity/state-update and binding-sensitive rows, does aligned compact view beat a permuted companion that preserves the compact rewrite multiset and token geometry while breaking source pairing?

This should not be launched blindly as another large run if cheaper evidence can decide. The low-cost first action is to recompute all existing Entity `V-B`, `B-C`, and `B-R` decompositions by zero/nonzero, affected/unaffected binding rows, and stable versus changed subfamilies using current prediction files, then incorporate incoming scorer outputs for the missing breadth columns. If balanced Entity `V-B` persists and broad ex-Entity remains small, the permuted companion should be restricted to Entity/EWoK/binding readout at the minimum checkpoint set rather than a full broad evaluation series. If broad ex-Entity `V-B` becomes robust across 90/100M, the permuted companion becomes a broader mechanism test because `B` will no longer be tight enough.

## Concrete next work

1. Use existing prediction files to build an operation- and binding-resolved triangle for `V`, `B`, `R`, and available `C`: Entity zero/nonzero, numops 0–5, split groups, and affected/unaffected binding rows. Report `V-B`, `B-R`, and `B-C`, not only `V-R`.

2. Merge incoming breadth scorer outputs into the reference decomposition. For each broad family at 80/90/100M, read `V-B` and `B-C`; separate Supplement from BLiMP/COMPS/EWoK/Reading because the 80M exEntity mean hides opposite signs.

3. If the only stable source-specific result remains balanced Entity `V-B`, route a minimal permuted-companion test that scores Entity and EWoK/binding panels first. It should decide whether aligned source pairing is necessary for the Entity gain: large `V-P` with small `P-B` supports source-conditioned correspondence; aligned≈permuted redirects toward compact-text distribution or regularization.

4. Do not reopen the supplied-address bridge or launch Strict-Small 100M training from it unless a new construction tests natural temporal-role language rather than exact tag/role copying.
