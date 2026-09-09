# lead reformulation after crossview closure Lead reformulation after crossview v2 20M result cross-view closure

## Research judgment

The crossview v2 20M result corrected four-arm DeBERTa MLM screen changed the mechanism object. The direct source--rewrite communication route should not organize the main line any longer. It found a real but small correct-partner effect on intrinsically source-absent content (`I_partner` about -0.038 to -0.048 nats), but a larger copied-token effect (`I_partner` about -0.080 to -0.135 nats). Together with adjusted source use residual's matched copied-target analysis, this means the compact-view advantage is not explained by stronger ordered copied-token retrieval and is not cleanly explained by direct source-to-source-absent target prediction either.

The strongest remaining object is the **rewrite marginal as a high-density denoising experience**. The real question is now: which property of the compact rewrite marginal carries the +2.4164 equal7 advantage over literal repetition in the DeBERTa masked-denoising coordinate?

Candidate properties:

1. **Semantic/content selection**: compact views preferentially retain propositions, entities, predicates, relations, quantities, and high-utility content per word.
2. **Abstractive/compressive reformulation**: compact views express retained content through alternate local syntax, function words, morphology, and word order, building invariance or relation-preserving compression beyond literal selection.
3. **Generic lexical/tokenizer coverage**: compact views alter subword support, tail/context exposure, position distribution, and WWM target ecology in a way that is useful independently of semantic selection.

The evidence supports the revised comparison: the next valuable line is a boundary-blocked/marginal comparison among compact rewrites, rewrite-guided semantic extracts, and frequency/position-matched extracts, not another pairwise source--rewrite attention experiment.

## Important correction before executing

The existing earlier analysis extractive arms cannot be used naively. Reading `prediction_geometry_scaffold_v2.py` and profiling the existing arms showed:

- earlier analysis `semantic_extract` is a contiguous source span chosen for lexical overlap with the compact rewrite, not a true rewrite-guided semantic extract over content units.
- earlier analysis `random_extract` is a random contiguous source span, so the contrast mostly changes span position and lexical overlap.
- Under the legal16k tokenizer the existing arms are not matched:
  - compact side: 252,920 side BPE tokens, 1.544 BPE/word, rare-support<=5 fraction 0.056;
  - semantic span: 227,078 side BPE tokens, 1.384 BPE/word, rare-support<=5 fraction 0.042;
  - random span: 218,702 side BPE tokens, 1.334 BPE/word, rare-support<=5 fraction 0.038.
  Static audit: `experiments/archive/representation_and_objectives/data/existing_extract_profile/existing_extract_profile.json`.

independent_review verifier identified the most dangerous confound: **within-view coherence/local conditional structure**. A non-contiguous selected-word list would be much less fluent than compact rewrites. Compact-vs-list would measure fluent compression versus fragmented source-word sequence, not abstraction alone. Therefore the next construction must use fluent/order-preserving extractive controls or else state a weaker conclusion.

## Why the marginal factorization is still worth doing

The compact-view line has one robust real-scale effect: view-repeat +2.4164 equal7. The adjbreak marginal already recovers about half of this gap (+1.1836), but the explanation is still not transferable. A factorized marginal study can convert `compact rewriting works here` into a general data-efficiency principle:

> At fixed word budget, useful denoising experience is governed by competence-bearing event density per predicted token, after controlling redundant local reconstruction, tokenizer support, and syntactic/relational coverage.

This is potentially portable beyond BabyLM if the factors can be measured and optimized on arbitrary corpora. It also connects to future objective design: if compact rewriting increases useful target frequency, selective masking/loss allocation may reproduce part of the effect without generation.

## Alternative routes considered

1. **Extend crossview v2 20M result to 100M or endpoint evaluation** — rejected. The predeclared source-absent-specific signal did not dominate; endpoint evaluation would turn the result into score fishing unless a new analysis supplies a stronger reason.
2. **Repeat causal-GPT/extractive controls from earlier analysis** — rejected. Causal topology was already closed as broad mechanism, and the earlier analysis extract controls are contiguous-span controls with tokenizer/support mismatches.
3. **Tie-breaking seed for view-vs-adjbreak** — lower value now. It would test the aggregate source-own gap but not explain the marginal property; crossview v2 20M result weakened the pairwise channel that made this seed tie-break central.
4. **Return immediately to entity/event/state memory** — preserved as an orthogonal route but not the best current move, because compact-view has a robust real-scale effect whose mechanism can still be made general. The entity-memory route has no valid positive training screen after the entity memory miniscreen synthesis flaw.
5. **Score endpoint packaging** — complete enough for the practical branch: coherent86 alpha0.75 measured-AoA Overall 42.1210247099666. Submission is outside this mechanism study. Packaging must not displace mechanism work.

## independent_review input integrated into the plan

independent_review generator ranked two high-value directions above local score-chasing:

- define/falsify a transferable competence-bearing event-density functional; and
- causally decompose the rewrite marginal through matched counterfactual corpora.

independent_review verifier supported standalone marginal comparison but warned that the proposed C/S/G arms do not isolate abstraction unless the extractive control is fluent/order-preserving and matched on tokenizer/WWM burden. It suggested treating any compact-over-extract advantage as `fluent reformulated compression as a package` unless content/coherence is controlled more tightly.

Full independent_review integrations:
- `data/external/independent_review01_generator1_integration.md`
- `data/external/independent_review01_verifier1_integration.md`

## Current protocol artifact

The corrected protocol is saved at:

`research/plans/representation_and_objectives/rewrite_marginal_density_factorization_protocol.md`

It calls for Stage 0 construction/audit only before any H100 work:

1. Build compact (C), rewrite-guided fluent extractive compression (S), and frequency/position/tokenizer-matched extractive compression (G), with optional repeat anchor (R) if packing changes the historical geometry.
2. Audit exact length, BPE/word, support bins, predicted WWM target burden, selected-position histograms, compact/source overlap, source-absent or inserted-token mass, span/coherence proxies, and filler/word accounting.
3. Inspect high-mismatch examples.
4. Decide whether the arms identify semantic selection vs reformulation vs generic coverage well enough for a 20M--40M short screen.

No H100 training is justified until this audit is read. No 100M endpoint training is justified until a short screen produces a clean separation.
