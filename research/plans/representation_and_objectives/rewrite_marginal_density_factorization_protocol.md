# lead reformulation after crossview closure protocol: rewrite marginal-density factorization after cross-view closure

## Why the object changed

The compact-view mechanism route has moved away from pairwise source--rewrite communication.
The decisive evidence is:

- earlier analysis: in the trusted DeBERTa-v2 masked-denoising coordinate, `compact_view_reinvest` beats `compact_repeat_reinvest` by **+2.4164 equal7** at 100M, above the same-recipe fast-equal7 seed band 1.3557. This remains the strongest real-scale compact-view result.
- correctness transition analysis: `adjbreak_reinvest` keeps the compact rewrite marginal but breaks source ownership and still beats repeat by +1.1836 equal7; the intact-vs-adjbreak profile contains large opposing BLiMP vs Supplement/EWoK movements but the aggregate gap is inside the seed band.
- adjusted source use residual: compact does **not** win by stronger ordered copied-token retrieval. On copy-opportunity-matched copied targets, compact's source-conditioned ordering interaction is lower than extractive families (compact minus mean extractives median about -0.491 nats, 95% [-0.605,-0.384]).
- crossview v2 20M result: the corrected own-visible/own-blocked/wrong-visible/wrong-blocked DeBERTa MLM screen found only a modest correct-partner effect on intrinsically source-absent content (`rw_abs_content` I_partner -0.038 to -0.048 nats), smaller than the copied-token interaction (-0.080 to -0.135). This stops direct source<->rewrite communication as the main mechanism route.

The remaining scientific object is therefore the **compact rewrite marginal as a high-density denoising experience**: what property of the compact side view makes it better training material than literal source repetition under a fixed word budget?

## Three factors to separate

A compact rewrite marginal combines at least three ingredients:

1. **Semantic/content selection**: the side view keeps high-utility propositions, entities, predicates, relations, and quantities from the source while dropping lower-utility words.
2. **Abstractive/compressive reformulation**: the side view realizes that retained content with different local syntax, function words, morphology, and word order; many per-pair targets are source-absent even though the pooled vocabulary mostly comes from the original corpus.
3. **Generic lexical/tokenizer coverage**: the side view changes unigram/subword frequency, tokenizer support, masked-target burden, tail coverage, and position distribution, independently of semantic utility.

The goal is not to name a winning rewrite style. The goal is to extract a transferable data-efficient learning principle expressible as a property of the training-event distribution.

## Critical correction from lead reformulation after crossview closure audit and independent_review

The existing earlier analysis `semantic_extract` and `random_extract` arms must **not** be launched as-is for this question.

- They are contiguous source spans selected by lexical overlap or random start, not a clean rewrite-guided semantic extract.
- Under the legal16k tokenizer, the existing arms are not support-matched:
  - compact side: 252,920 BPE tokens, 1.544 BPE/word, rare-support<=5 fraction 0.056;
  - semantic span: 227,078 BPE tokens, 1.384 BPE/word, rare-support<=5 fraction 0.042;
  - random span: 218,702 BPE tokens, 1.334 BPE/word, rare-support<=5 fraction 0.038.
  See `experiments/archive/representation_and_objectives/data/existing_extract_profile/existing_extract_profile.json`.
- A non-contiguous selected-word list would introduce an even stronger confound: it is not a fluent denoising sequence. Compact-vs-list would measure fluent compression versus incoherent word sequence, not abstraction alone.

Therefore the next experiment must first construct **fluent/order-preserving extractive controls** and audit their realized tokenizer/WWM geometry before any H100 training.

## Experimental family

Use the exact 12,155 compact-pair source population:

`experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`

and the shared filler when full 10M epochs are needed:

`experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl`

Every arm must use the same legal16k tokenizer as the triangle coordinate:

`experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_1M`

### Arm C: compact rewrite marginal

The compact side text only, preserving the compact rewrite's realized sentence-like form. For source-dose balancing, the source segment may still be present in the example with source--side attention blocked, but the readout should treat the side view as a standalone marginal. The exact packing choice must be shared across arms.

This arm contains semantic selection + abstractive/compressive reformulation + realized lexical coverage.

### Arm S: rewrite-guided fluent extractive compression

A literal-source control that tries to preserve the compact rewrite's retained content while avoiding abstractive noncopy reformulation.

Preferred construction hierarchy:

1. Select an ordered subsequence or small set of source spans maximizing overlap with the compact rewrite's content words, entities, predicates, relation markers, negation, numbers, and quantities.
2. Realize it as a fluent/order-preserving extractive compression. Avoid a bare list of selected words. Use minimal punctuation/connector insertion only if deterministic and non-language-learned; count any inserted words in the budget and audit them separately.
3. Match the compact rewrite length per row as closely as possible in whitespace words and BPE pieces; record all deviations.

This arm is intended to preserve semantic/content selection while removing noncopy rewriting. If it cannot be made fluent/order-preserving without heavy deterministic insertion, the arm identifies `rewrite-guided source span/phrase selection`, not semantic selection alone.

### Arm G: frequency/position/tokenizer-matched extractive compression

A literal-source control not guided by the compact rewrite, matched to Arm S/C on generic distributional variables.

Construction should sample ordered source words/spans to match, at minimum:

- side whitespace length per row;
- side BPE-piece length distribution;
- source-position distribution of selected words/spans;
- legal16k token-frequency/support bins;
- content/function/POS-proxy composition if implementable without a language-learned tagger;
- entity/number/punctuation fractions;
- WWM predicted-piece burden under deterministic masks;
- tail-token and repeated-token exposure.

This arm isolates generic lexical/tokenizer coverage and local extractive sequence exposure.

### Optional anchor R: literal-repeat side marginal

Include only if cheap construction shows the new standalone packing differs materially from the historical triangle geometry. R should use the literal source segment truncated/compressed to the same side-word lengths, with the same filler and source dose. It anchors the new family to the known repeat baseline and prevents interpreting an all-arms-null result as a factorization result when the packing itself destroyed the old effect.

## Static audit before training

Candidate corpora and their audit are prerequisites for a training decision; no H100 launch is justified by the design alone.

The audit must report, by arm and by row:

- exact source identity coverage and side-view word count;
- total words/epoch with shared filler;
- side BPE tokens, BPE/word, and pieces per masked word;
- unique word and BPE type counts;
- source-position histogram of selected source-derived words;
- compact-word overlap and source-word overlap;
- source-absent/noncopy mass for compact and inserted/non-source mass for extractive arms;
- legal16k support bins for side BPE tokens and for predicted WWM targets;
- deterministic WWM target counts per arm for a 10M or 20M short screen;
- simple coherence/pathology proxies: punctuation rate, average gap between selected source words, number of spans, fraction of rows with list-like or fragmentary output.

Pre-training continuation requirements:

- Arm S and Arm G must be close on all generic coverage variables; otherwise S-G is not semantic selection.
- Arm C and Arm S must be close enough on length, BPE/word, and target burden that C-S is not dominated by tokenizer burden.
- If Arm S is much less fluent/coherent than C, then C-S may only support `fluent reformulated compression as a package`, not abstraction alone.
- If matching cannot be achieved over all 12,155 pairs, create a fixed common-support subset and report how much compact-view mass is lost before any training decision.

## Training ladder and expensive-work discipline

### Stage 0: CPU construction/audit

Build C/S/G (and maybe R) candidate corpora and the audit above. Do not train if the audit shows obvious non-identification.

### Stage 1: short identical-init screen, 20M--40M words

Only after the audit passes, run the arms with:

- DeBERTa-v2 8x480;
- legal16k tokenizer;
- identical untrained initialization, preferably the crossview v2 20M result identical init if config/tokenizer-compatible;
- WWM 0.15;
- AdamW/LR geometry matching the triangle (`lr_total_steps=2529` for 100M-equivalent schedule);
- the same filler, row order, source-dose/packing, masks as far as possible;
- per-stratum loss logging for source, side-copied, side-noncopy/compact-absent, tail/support bins, filler.

The short screen is not a score result. It decides whether any contrast cleanly separates target ecology beyond local pathologies.

Stage 1 should stop a route if:

- all differences are explained by BPE/WWM burden or local coherence;
- Arm S and G differ mainly in token-support variables despite attempted matching;
- Arm C only improves local fluent-sequence loss/BLiMP-like proxy while the semantic/relational target strata do not move;
- no contrast survives beyond noise relative to same-init training variability.

### Stage 2: late-horizon 100M no-AoA endpoint only for a clean separation

Only if Stage 1 produces a clean and interpretable separation would the evidence justify 100M endpoint training and official-compatible no-AoA evaluation. Do not run three fresh 100M arms merely because the corpora exist.

Endpoint readouts:

- equal7 as secondary context, not the sole interpretation;
- correctness transition analysis-style item transitions for BLiMP, Supplement, EWoK, COMPS, GlobalPIQA;
- Supplement retained-plus-new competence;
- EWoK domains individually: social-properties, physical-dynamics, spatial-relations, physical-relations, not just pooled;
- COMPS `wugs_dist_in_between` vs `wugs_dist_before`;
- BLiMP movement, because prior compact/adjbreak contrasts showed grammar-relation opposition;
- association of endpoint movement with Stage-1 side-target strata.

## Interpretation patterns

Let C, S, and G denote endpoint competence vectors, not only scalar equal7.

### Generic coverage suffices

Pattern: C ≈ S ≈ G, and all three improve over a matched repeat/anchor with similar task profiles.

Meaning: compact rewriting is one way of changing token/support/position coverage, but semantic selection and abstraction are not necessary in this coordinate.

### Rewrite-guided semantic selection suffices

Pattern: C ≈ S and S > G by more than the relevant seed/noise band, with shared Supplement/EWoK/COMPS transition fingerprints and matched coverage variables.

Meaning: the transferable principle is content/proposition selection density per word; literal extractive compressions can carry the same useful training events.

### Fluent/abstractive reformulation has a residual

Pattern: C > S by more than the noise band after length, tokenizer support, WWM burden, content retention, and coherence have been controlled as far as possible; C's advantage appears in Supplement, several relational EWoK domains, and COMPS near-distinction readouts, not only in BLiMP or local MLM loss.

Meaning: only then is it justified to say compact reformulation/compression adds an invariance-building or relation-preserving signal beyond extractive selection. If Arm S remains fragmentary, the weaker conclusion is `fluent reformulated compression as a package`, not pure abstraction.

### Mixed vector pattern

If S-G moves BLiMP/coverage while C-S moves Supplement/EWoK, retain a multidimensional principle rather than collapsing it into one scalar. correctness transition analysis already showed opposing grammar and relational movements can hide in equal7.

### Null or ambiguous pattern

If all arms are close but task transitions differ, do not call it null; analyze competence-vector signatures. If all arms fall near repeat, suspect the new packing/source-dose geometry broke the earlier adjbreak marginal condition and add the R anchor before revising the compact-view theory.

## Relation to the general principle

A successful factorization should support a corpus-general formulation such as:

> Under a fixed word budget, sample-efficient denoising improves when the training stream increases competence-bearing event density per predicted token while controlling redundant/easy reconstruction, tokenizer support, and syntactic/relational coverage. Compact rewriting is one implementation; extractive selection or objective weighting may reproduce part or all of it.

This principle becomes transferable only if the measured corpus factors predict held-out competence movement across at least one out-of-family corpus or model after this within-family factorization. The present protocol is a necessary separation step, not the final principle.

## Next concrete work

Only Stage 0 construction is proposed:

1. Build candidate C/S/G (and maybe R) marginal corpora from the 12,155 pairs using legal, deterministic, non-language-learned algorithms.
2. Generate an audit table comparing length, BPE/word, support bins, target-burden, selected-position, compact-overlap, source-overlap, span/coherence proxies, and filler/word accounting.
3. Inspect examples from high-mismatch rows.
4. Decide from the audit whether the arms identify semantic selection versus abstraction versus coverage well enough for a 20M--40M screen.

No H100 training should launch until Stage 0 audit is read and judged. No 100M endpoint should launch until a Stage 1 short screen produces a clean separation.
