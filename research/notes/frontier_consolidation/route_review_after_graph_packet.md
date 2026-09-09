# Route Review After Generated Graph-Packet Failure

## Current research state

The submitted public fallback remains the legal scale1.75 `chck_82M` endpoint:

- local hardened Overall `41.942481167385985`
- public displayed Overall `41.94`
- HF repo `leslie721007/babylm-strict-small-scale1p75-chck82`, revision `f49775dc5eafbf3a14d6f2107ec5368188d2b1dd`
- trusted custom-code loading is required

The strongest local endpoint carrier remains coherent86 `alpha0.75`:

- cheap7 `44.18142857142857`
- SuperGLUE `69.81922238969935`
- Overall(AoA0) `42.1210247099666`
- carrier `data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
- carrier SHA `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`

## Graph-packet route conclusion

The graph packet v0 construction/166 generated graph-packet realization should stop. The reason is not only low accepted-example yield. The accepted v2 set had strong target priors before structured context was supplied: query-only target-vs-distractor margin was `+3.3781` and neutral context already preferred the target in `10/10` examples. The measurement also failed to align with the actual protected endpoint contrast: `chck100` was slightly stronger than `chck82` on structured-neutral lift (`+0.8635` vs `+0.8111`) and structured-reversed sensitivity (`+1.5110` vs `+1.4902`). An independent packet audit found that the graph packet v0 construction packet has no nonempty state slots, no `state_change` relation types, no `action` relation types, and no entity-state-update rows with explicit state/action structure.

Therefore enlarging this generated packet would refine a synthetic query-prior observable, not the BabyLM learning mechanism. It also risks repeating the already closed source-conditioning pathway under a more elaborate name.

## What evidence still protects the compact-view/reinvestment core

The protected empirical substrate is not compression alone. It is:

**redundancy-reduced semantic second views plus reinvested source diversity under fixed legal exposure.**

Evidence:

1. The old-tokenizer compact-view-reinvest endpoint reached Overall `42.086786` with all nine official-compatible columns. Relative to clean-Qwen it improved Overall by about `+0.742`, with large movement in EWoK (`+3.480`), Entity (`+1.990`), SuperGLUE (`+1.072`), and Reading (`+0.480`). That endpoint is not submittable because the tokenizer coordinate was later found illegal, but it remains causal evidence about data organization.
2. Under the legal tokenizer, compact-view reinvestment remained positive versus the clean legal control at maturity: earlier analysis/legal tokenizer clean control trajectory design showed reinvest-minus-clean around `+1.3464` mean7 at 80M with positive movement in BLiMP, Supplement, EWoK, Entity, COMPS, and Reading.
3. The matched FineWeb compact-versus-independent-breadth experiment showed compact views outperforming independent source breadth at 100M by `+0.5443` cheap7 and at 70M by `+0.6257`, even though the absolute endpoint movement was not enough for SOTA. This protects the semantic second-view part against the simple explanation that any extra FineWeb sentences work equally well.
4. The scale1.75 adapter endpoint shows the same legal compact-view/reinvest stream can reach the public SOTA surface when coupled to function-preserving residual capacity and stopped at the reproduced 82M point.

Limits of the core:

- The second seed for the old-tokenizer reinvest route was weak, especially EWoK and Supplement, so seed stability is not solved.
- The 82M point is reproducible but narrow; lower MLM loss and corpus-internal likelihood probes did not explain it.
- Several later interventions improved aggregate score by moving among official families rather than adding broad competence.

The next route should therefore make the compact-view/reinvestment signal more reliable or more broadly integrated, not replace it with a synthetic relational probe or another endpoint residual scale.

## Proposed Natural-Data Mechanisms

Four natural-data mechanisms were proposed: cross-document proposition recurrence, entity-centered natural histories, delayed semantic retrieval schedules, and complementary natural-clause views. An entity-keyed event-memory architecture was also under study. The proposed direct natural-data test changes the temporal organization of the same source/view evidence in the existing compact-view/reinvest corpus while preserving the text multiset.

### Proposed Route: Delayed Semantic Recurrence Scheduling

Scientific object:

The current compact-reinvest changed block contains `12,155` compact source-view pairs and `423,511` pair words inside each 10M pool, usually packed as adjacent source+view evidence inside row-level text. The route asks whether the same legal text builds more durable, seed-stable knowledge when each source and compact view are re-encountered after intervening natural experience rather than massed together. The variable is recurrence distance, not new text, new labels, masking, objective, or architecture.

Why this route is distinct and valuable:

- It is the closest possible perturbation of the replicated success: same sources, same compact views, same reinvested budget, same tokenizer, same objective.
- It cannot win by generated answer priors because it uses the already scored natural training text.
- It directly attacks the weak point of the core: the benefit is real but narrow and seed-sensitive. If spacing broadens the high-performing region or reduces seed variance, it upgrades the existing discovery instead of replacing it.
- It is cheaper and cleaner than generating new relational data. A CPU construction can prove exact text-multiset equality before any H100 use.

Main danger:

Spacing is easily confounded with row repacking and token visibility. U256 already showed that small row-boundary changes can redirect learning. Therefore the first real-training comparison must separate repacking from recurrence distance.

Required arms for the first exact-text construction:

1. `current_adjacent`: the existing compact-view-reinvest stream, current row packing and order.
2. `repacked_adjacent`: identical source and view strings, but source/view units are repacked as separate text units while kept adjacent. This isolates row-packing and local context changes.
3. `delayed_within_epoch`: same units and same word count, with each view placed after a fixed or stratified delay inside the same 10M pass.
4. Optional after those pass mechanical checks: `cross_epoch_delayed`, where source appears late in one pass and compact view early in the next pass, still keeping total exposure and pass count legal.

The first two non-current arms are enough to decide whether spacing is promising while controlling the row-boundary confound. If `repacked_adjacent` already differs strongly from `current_adjacent`, the research must interpret row packing before attributing anything to delay.

Construction work before any H100 training:

- Materialize schedule files from `data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl` and the exact common filler from `data/density_cleanqwen_overlay_medium_riskhard/common_filler_rows.jsonl`.
- Preserve every source string and rewrite string byte-for-byte; preserve all common filler rows byte-for-byte.
- Report exact word totals, row counts, max row length, source/view exposure counts, per-pass exposure, pair delay distribution, number of source-view pairs split across row boundaries, and SHA hashes for every 10M and 100M stream.
- Verify tokenizer and WWM candidate-token totals so a schedule win is not caused by a large masking-mass shift.
- Build a held-out natural recurrence measurement from training-only compact pairs: for a pair, mask content words in the source or view and compare same-pair counterpart versus unrelated matched counterpart, without official evaluation data. This measurement is not a substitute for training, but it gives a reusable natural-data readout for the screen.

Minimum reliable H100 screen after mechanical checks:

Use scale1.75 adapter architecture, legal same-pool 16k tokenizer, same seeds, same objective, same optimizer settings, and same checkpoint cadence. Train `repacked_adjacent` and `delayed_within_epoch` in parallel on the two H100s to 20M exposure, while comparing to the already existing `current_adjacent` scale1.75 `chck_20M` when coordinates match. This is the lowest cost reliable training action because order effects are training-dynamic and cannot be inferred from frozen scoring. It decides whether recurrence distance is a load-bearing variable beyond row repacking.

What would move the route forward:

- `delayed_within_epoch` beats `repacked_adjacent` on the natural same-pair transfer readout and shows nontrivial cheap7 improvement or reduced damage across BLiMP/Supplement/EWoK/Entity/COMPS/Reading/GlobalPIQA at 20M.
- The improvement is not carried by a few GlobalPIQA examples or one BLiMP family.
- If a 20M separation appears, continue only the most informative arm plus the repacked control to 50M/80M before considering full evaluation. Dense checkpoints should measure whether delayed recurrence widens the high-score window rather than creating another single-point endpoint.

What would stop it:

- `delayed_within_epoch` and `repacked_adjacent` are indistinguishable on natural transfer and cheap columns.
- `repacked_adjacent` differs from current mainly through row-boundary visibility, and delayed ordering adds nothing beyond that.
- Any improvement resembles known endpoint redistribution: positive aggregate with negative common-item balance or a small-column fluctuation.

## Secondary routes to keep for later

1. Cross-document proposition recurrence: high-value but needs a robust cluster-yield scan over legal sources and source-family separation. It remained a secondary proposal, including as an alternative if schedule manipulation is inert.
2. Complementary natural-clause views: promising for preserving relation/modality/consequence information that ordinary compaction may drop, but it needs source-derived extractive partitions and strong equality controls against redundant and random partitions. It is closer to another compaction variant and should not precede the exact-text schedule test.
3. Entity-centered natural histories: scientifically plausible as a natural-data counterpart to entity-memory studies, distinct from duplicate synthetic state training.

## Prerequisites for the Proposed Test

The proposed test requires a schedule materializer and natural recurrence measurement before training, together with exact text-multiset, word-count, token-count, WWM-mass, and delay-distribution reports. Training, if undertaken, would answer whether recurrence timing of the already validated compact semantic second views makes the benefit broader and more stable under identical legal exposure.
