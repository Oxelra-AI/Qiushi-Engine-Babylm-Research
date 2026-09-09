# route reopen conditional innovation — route reopened around conditional innovation in compact views

## Why the route changed

The two failed legal-coordinate interventions now define a repeated trade: changing global lexical credit (`word-mean`) or tokenizer support/segmentation (`minfreq50`) moved large GlobalPIQA-nonparallel scores while weakening the language and world-structure columns the legal endpoint must strengthen. The minfreq50 route is closed by its 80M surface: mean7 -0.1679 against the spatial repair route status token-mean reinvest reference, Supplement -2.29 and EWoK -1.44, with the main gain in GlobalPIQA_nonparallel. The support-sharing analysis also separates support-sharing from flat minfreq retokenization but provides no real sequence score; U256 remains an experience-utilization construction, not current score evidence.

The safest next scientific object is therefore not another global vocabulary/loss variant and not naive source-view agreement. The route should extend the validated compact-view mechanism itself: the model benefits from compact source/rewrite pairs plus reinvested source diversity, but the useful unsaturated quantity must be conditional information introduced by the rewrite, not pair identity. consistency decoy diagnostic and route correction already showed pair identity/retrieval is nearly saturated by 80M.

## New cheap evidence from route reopen conditional innovation

### 1. Conditional innovation probe

File: `research/documents/frontier_consolidation/data/conditional_innovation_probe/conditional_innovation_probe.md`

CPU-only, 512 changed rows, 600 selected rewrite word-group targets balanced between source-present copyable groups and source-absent innovation groups. For each masked rewrite target, the probe compared:

- true source span visible,
- source span masked,
- same-row decoy source span substituted from another pair.

Definitions in the file:

- `source_help = loss(source masked) - loss(true source)`
- `true_over_decoy = loss(same-row decoy source) - loss(true source)`

The mature legal16k token-mean compact-view model at 80M shows source-specific conditional information beyond generic pair identity:

| group | true loss | source help | true over decoy |
|---|---:|---:|---:|
| innovation | 4.4440 | 0.9717 | 1.3171 |
| copyable | 0.9308 | 4.8146 | 5.1955 |
| innovation: relation-cue | 3.0280 | 0.5826 | 0.7722 |
| innovation: content/other | 4.6681 | 1.0333 | 1.4034 |

At 100M the innovation true loss is still high (4.3550) and only modestly improved from 80M, while true-over-decoy remains high (1.3737). Clean_80M has the same structure but weaker: innovation true loss 5.3856, source_help 0.6206, true_over_decoy 0.8634. This supports a real compact-view treatment effect on source-specific innovation, not just source copying.

Important reading: copyable groups are easy and receive very large source help, so a naive pair objective could collapse into copying. But source-absent innovation groups are still high-loss while benefiting from the true source more than a same-row decoy. That is the most grounded unsaturated quantity now visible.

### 2. Contextual-computation comparison

File: `research/documents/frontier_consolidation/data/contextual_computation_probe/contextual_computation_probe.md`

CPU-only layerwise readout on 440 targets. The 8-layer token-mean model develops most source-specific conditional help in late layers, especially the final two layers:

| group | final true loss | final source help | final true over decoy | true loss drop last2 | source-help gain last2 | true-over-decoy gain last2 |
|---|---:|---:|---:|---:|---:|---:|
| innovation | 4.7178 | 0.9622 | 1.2665 | 1.1598 | 0.7848 | 0.8812 |
| copyable | 1.0125 | 4.1458 | 4.4140 | 4.4446 | 3.9575 | 4.1749 |
| innovation: relation-cue | 3.0576 | 0.3461 | 0.6375 | 0.3633 | 0.0777 | 0.2497 |
| innovation: content/other | 4.9599 | 1.0521 | 1.3582 | 1.2760 | 0.8879 | 0.9733 |

Clean_80M is weaker in final source help and true-over-decoy, especially for innovation relation cues (0.0793/0.1157). This does not by itself justify a deeper/iterative H100 run, but it keeps contextual computation as a real alternative: if the depth vector is strong, depth may be improving exactly this late conditional computation. If the depth result is weak, an 80M depth-imitation run is not justified; instead construct a small changed-span learning signal on the existing legal16k token-mean compact-view substrate.

### 3. Innovation target mass

File: `research/documents/frontier_consolidation/data/innovation_target_mass/innovation_target_mass.md`

Across all 3,005 changed rows and 12,145 both-visible pair records, one 10M pass contains:

- all rewrite groups: 158,110;
- copyable groups: 131,675 (83.3%);
- source-absent innovation groups: 26,435 (16.7%);
- source-absent relation-cue groups: 3,500 (2.2%).

Over ten passes, source-absent innovation gives 264,350 word-group targets inside the existing 100M stream. It is small relative to all training groups but not vanishing; a learning signal can target it without changing data, tokenizer, or legality. Relation-cue-only targeting is probably too narrow for the first run; broad innovation plus a monitored relation-cue slice is more plausible.

## Route judgment

The next route should be conditional-innovation learning on the legal16k token-mean compact-view reinvest substrate, not another global support/credit variant. The scientific purpose is to make the model use the true source view to learn source-absent rewrite innovations while preserving ordinary MLM as the main objective.

The first construction should be single-variable and cheap to validate mechanically:

1. Preserve the frozen 10M/100M compact-view reinvest stream, spatial repair route status legal16k tokenizer, 8x480 DeBERTa-v2 shape, seed mapping, optimizer schedule, WWM accounting, and 100M word charge.
2. Use only corpus-derived pair spans and normalized source/rewrite group metadata from `pair_span_map` and the fixed training file.
3. Bias a small fixed amount of mask target mass toward source-absent rewrite innovation groups and their local neighborhoods on changed rows, while reducing ordinary random target selection elsewhere so total selected WWM groups per step stays matched to the baseline within measured tolerance.
4. Avoid source/rewrite full-vector agreement as the primary objective; pair identity is already saturated.
5. Avoid a copy objective: exclude or separately cap source-present copyable rewrite groups, since they are already easy and would dominate.
6. Monitor true-source/decoy conditional losses with the route reopen conditional innovation probe at 20M/40M/80M and official-compatible cheap columns; the desired early signature is lower innovation true loss and preserved or improved Supplement/EWoK/BLiMP without replaying the word-mean/minfreq GlobalPIQA-nonparallel trade.

A more ambitious variant could add a small source-to-rewrite innovation decoder, but only if it is made leakage-safe: the source representation used by the auxiliary predictor must be computed while the target rewrite group is masked, and the loss must not reveal the target through unmasked target tokens or copied rewrite context. Because this is more complex and may require extra forward passes, the first build should probably be innovation-biased masking with target-mass matching.

## Relation to evidence

The support-sharing analysis and tail rephase experiment design state show no real U256 score yet. The support-sharing route is still a plausible legal40k construction, but the error-conditioned support probe did not connect it to EWoK/BLiMP, and minfreq50 shows that support/segmentation reshuffling can damage exactly those columns. The depth comparison remains the most relevant possible outside signal: if depth improves conditional/world-state columns, this would support interpreting route reopen conditional innovation's layerwise late emergence as a shared mechanism; if depth is weak, waiting for or imitating depth is not justified.

## Next work

Before a long run, the proposed training-side mechanism requires implementation and CPU/GPU correctness checks:

- build metadata that marks source-absent rewrite innovation WWM groups for every changed row;
- implement target-mass-matched innovation-biased WWM inside the existing trainer coordinate;
- measure selected-group counts, selected-token counts, source/category distribution, and collision with baseline WWM on representative batches;
- run a tiny CPU or one-step GPU smoke to verify masks/labels and no exposure/accounting change;
- prepare a 20M or 40M real-training screen only after depth status is read and the construction is mechanically clean.

A 20M/40M screen is lower-cost than another 80M global run and directly tests whether the innovation signal moves before full maturation. It should continue deeper only if it improves the conditional-innovation probe and broad cheap columns rather than only GlobalPIQA-nonparallel.

## Addendum after independent_review and repaired conditional-source probe

independent_review verified that the first route reopen conditional innovation probe identified a useful object but that the word `innovation` must be read operationally: source-dependent rewrite targets whose exact form is absent from the source, not guaranteed semantic novelty. It also pointed out two real fragilities: same-row decoy substitution can become malformed, and exact source absence does not prevent target leakage from elsewhere in the row.

I repaired both issues cheaply before any training run. New file: `research/documents/frontier_consolidation/data/conditional_innovation_repair_probe/conditional_innovation_repair_probe.md`.

The repaired probe uses unique source-absent targets whose normalized form is absent from the paired source and from the rest of the row, and fills same-row/cross-row source replacements to the true source-span length without adding extra mask tokens for short decoys. Results survive the main objections:

| checkpoint | bucket | n | true loss | source help | same-row decoy advantage | row-bootstrap source-help 5-95 | row-bootstrap decoy-advantage 5-95 |
|---|---|---:|---:|---:|---:|---|---|
| tokenmean_80M | unique_content_innovation | 220 | 5.3343 | 1.5248 | 2.2157 | [1.2643,1.7802] | [1.9263,2.5268] |
| tokenmean_100M | unique_content_innovation | 220 | 5.2162 | 1.5796 | 2.2902 | [1.3104,1.8420] | [1.9983,2.6051] |
| clean_80M | unique_content_innovation | 220 | 6.7185 | 0.8331 | 1.3696 | [0.5825,1.1070] | [1.0674,1.6343] |
| tokenmean_80M | unique_relation_or_function_innovation | 220 | 3.4461 | 0.2944 | 0.7876 | [0.0686,0.4995] | [0.4974,1.0853] |

This makes the current route stronger than it was before independent_review: the useful population is not only copyable rewrite groups or duplicated row words. Unique content innovations remain hard even at 100M, yet the true source helps them substantially more than a same-row replacement. Relation/function innovations are smaller and weaker; do not build a relation-only target on this evidence. Copyable targets are easy and should be capped or excluded from any new target-biased mask selection.

Depth training completed and guarded evaluation is running; a support-gated compositional residual prototype on legal40k was also built. Legal40k support sharing and depth/U256 are distinct from the compact-view legal16k line. The latter should turn the repaired source-dependent unique-content signal into a clean training construction rather than duplicate SGCR.

research Loop was restarted in exec mode with proposal `experiments/archive/frontier_consolidation/analysis/Research_Project_Proposal.md`; it is only asked to prototype and stress-test CPU construction artifacts under `experiments/archive/frontier_consolidation/analysis/work`. Do not rely on it until a later step checks status once and reads `Research_Report.md` if available.

Revised construction target:

1. Build a complete metadata file over the 100M stream or 10M pool that marks source-absent, row-unique, content-like rewrite WWM groups, with separate flags for relation/function innovations and copyable rewrite groups.
2. Implement a target-mass-matched biased WWM sampler on top of ordinary token-mean WWM: reserve a small fraction of selected WWM groups on changed rows for the unique-content innovation set, then reduce ordinary random selections elsewhere so selected group and token counts stay close to baseline.
3. Measure real selected group/token mass, changed-row frequency, source distribution, group length, lexical frequency, and displacement from ordinary WWM on representative batches; also measure frozen-batch loss and gradient scale so high-loss innovation groups do not dominate unintentionally.
4. Run only CPU or one-step GPU smokes in the construction step. A 20M or 40M screen becomes justified only after the sampler is mechanically clean, depth vector is read, and the expected score signature is explicit: lower repaired conditional-source true loss plus preserved/improved BLiMP/Supplement/EWoK, not merely another GlobalPIQA-nonparallel move.
