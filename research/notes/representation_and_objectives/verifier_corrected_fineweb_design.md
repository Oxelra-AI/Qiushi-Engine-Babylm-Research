# live fineweb core fact filter verifier-corrected FineWeb experiment design

Independent independent_review verification (`data/external/independent_review01_verifier1_integration.md`) confirmed the direction of the natural-arm design but found the simple three-arm family does not separate view utility from source diversity. This note records the corrected design and the low-cost checks that must precede a large H100 run.

## Corrected minimal causal family (four arms, one materialization family)

- `A_natural` — inherited clean-Qwen `qwen_pair_packed` rows preserved; a **predefined** coherent official/non-Qwen slice of the changed-block word budget, repacked to the fixed changed-block row-length sequence. Strong natural coordinate. Fix this slice before looking at any outcome.
- `B_breadth` — same common filler and same changed-block budget; **distinct** filtered FineWeb source rows only, same row-length sequence as A.
- `B_repeat` — same FineWeb source identities and multiplicity/adjacency as C, but literal repetition instead of a generated view. Representation-matched control.
- `C_view` — same FineWeb sources as `B_repeat`; one faithful near/compact view adjacent to its source.

Identified contrasts:
- `B_breadth − A_natural`: net value of FineWeb corpus substitution (breadth + register + factual density + quality), the deployment question.
- `C_view − B_repeat`: rewrite-vs-literal-repetition at fixed source identities (true view utility).
- `C_view − B_breadth`: semantic reinforcement vs spending the same budget on more distinct facts.
- `C_view − A_natural`: the load-bearing SOTA comparison over the strong coordinate.

The earlier three-arm plan (A/B/C only) measures only `B−A` and `C−A`; it cannot call `C−B` "view utility" because B spends the whole budget on unique sources while C spends part on alternate realizations. `B_repeat` fixes this.

If compact views free budget and it is reinvested into new sources, add the matched source-only reinvest arm, or source diversity and view effects are inseparable. Optional shuffled-view arm (same source+view texts, pairing broken) isolates the source-view adjacency effect.

## Confounds to control, not describe away

1. "Source breadth" is really corpus substitution: replacing CHILDES/subtitles/BNC/Gutenberg/SimpleWiki/Switchboard with factual web prose mixes breadth, register, factual density, and quality. This is the right deployment contrast but must not be labeled a pure breadth effect.
2. Equal words/row-lengths ≠ equal tokenizer exposure. Entity/number/acronym-dense FineWeb likely produces more subwords per whitespace word than dialogue/fiction. For every arm, materialization must report under the real baseline16k tokenizer/collator: non-padding tokens, tokens/word, truncation rate, padding fraction, masked targets, optimizer steps, and changed-block vs common-filler token totals. Word/epoch compliance stays fixed; token disparity is either matched or explicitly treated as part of the intervention.
3. Adjacency is a separate intervention (local same-window alignment vs corpus-level multi-view). `B_repeat` must match adjacency; a shuffled-view arm isolates it.
4. Scale changes the mixture: 1.75M→3.5M removes progressively more natural spoken/literary/child-directed material, so scale needs per-component curves, not aggregate equal7 extrapolation.

## near-view read against the trusted coordinate

`near_view − near_repeat`: Entity_full +3.57, Reading +0.775, Supplement +0.80; BLiMP −0.37, EWoK −0.27, COMPS −0.61, GlobalPIQA ~flat.

Against COMPACT_EXPERIENCE clean-Qwen (full-Entity equal7): `near_view` delta **−0.234**; `near_repeat` delta **−0.788**. So the view beats its repeat control but still sits below the trusted natural allocation, and this is a reduced screen (no SuperGLUE/AoA). companion analysis also materialized `cleanqwen_lengthmatched_near_core` but has not evaluated it — evaluating that same-family natural arm would be cleaner than the historical clean-Qwen checkpoint (controls init/order/run variance).

## Interpreting the pending 17.5% cached source-only run

Valid test of *cached* FineWeb substitution (matched 1,753,280-word block), not of the stricter live pool. Cached examples still contain "follow this link", broken article starts, citation/URL/index residue, software-blog prose.
- Broad balanced knowledge gain w/o eroding BLiMP/Supp/Reading → test larger matched family (3.5M) only if the gain is distributed, not Entity-only, and tokens are audited.
- Small/Entity-concentrated → 2.5M family, shift composition toward self-contained causal/relational/comparative/physical propositions rather than more named entities/dates.
- Flat/damaging → rejects scaling the *cached* source-only pool, not FineWeb in general; a cleaner live-source family is needed to separate source-quality failure from mixture mismatch.

## Source quality is not yet adequate for blind scaling

live fineweb core fact filter core filter kept 2,994 rows / 77,353 words (47.5% of strict words) across 617 docs; cap8 = 56,206 words, cap4 = 41,608 words — document/style redundancy is material, so per-doc caps and within-doc near-dedup must be selection policy. Accepted sample still admits headline/index residue ("PAT CAPUTO ... (805)"), discourse-dependent openers, quote/report fragments, and malformed notation. Heuristic domain labels are unreliable (Thatcher/Confucius→quant/science) and must not drive composition balancing without a precision audit.

## Low-cost checks before any large generation/training

1. **Source audit** (stratified over score bins, domains, doc-contribution counts, sentence position, admission path): label completeness, self-contained referents, explicit factual relation, absence of navigation/citation/list residue, target-component relevance, local duplication, questionable/temporal attribution. Report usable-sentence precision and final yield after cap-4/cap-8. Audit cached and live pools separately.
2. **Rewrite faithfulness** on stratified source/rewrite samples: subject-relation-object entailment, negation/modality, quantities/units/dates/comparison direction, temporal-causal order, coreference, unsupported additions, template artifacts. Near-view content recall ~0.856, compact ~0.662 (proposition deletion risk). Near view has 1,984/10,094 (~19.7%) copy-like pairs vs 25 for compact, so faithfulness/effect must be examined by lexical-overlap bin — "near view" currently mixes paraphrases with near-copies.
3. **Corpus invariants**: hash byte-identical common filler + Qwen block; one predefined changed-block row-length sequence; identical source identities/multiplicity between repeat and view; exact 10M words / ≤10 passes; fixed selection/order seeds; exact+fuzzy overlap checks vs benchmark text.
4. **Measure real end-to-end yield** Y_final = accepted+deduped+doc-capped core words / raw scanned FineWeb words on a representative shard before projecting 2.5M-3.5M generation volumes; the live fineweb core fact filter tables likely understate raw scanning needs.

## Files
- Design analysis: `experiments/archive/representation_and_objectives/data/natural_arm_design_analysis/natural_arm_scaled_fineweb_design_analysis.json`
- Core-fact filter: `experiments/archive/representation_and_objectives/data/live_fineweb_core_fact_filter/live_fineweb_core_fact_filter_summary.json`
- independent_review integration: `data/external/independent_review01_verifier1_integration.md`
