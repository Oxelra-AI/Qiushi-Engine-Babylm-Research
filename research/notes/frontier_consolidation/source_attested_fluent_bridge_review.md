# source attested fluent bridge review review of the source-attested fluent-bridge prototype

## Research question

The extractive selected readout result source-only extractive arms showed that broad source/tail access with telegraphic source-word views does not reproduce the natural compact view's stable selected competence. source attested fluent bridge prototype readout therefore tried a better-separated object: compact-like fluent views whose content lemmas are source-attested, preserving the source anchor while removing natural compact's novel content vocabulary.

This review asks whether the source attested fluent bridge prototype readout strict-accepted outputs are already a trustworthy data object for BabyLM training.

## Evidence read

- `data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_strict_validation.json`
- `data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_pair_level_summary.json`
- `data/source_attested_fluent_bridge_prototype/source_attested_fluent_bridge_strict_review_cases.jsonl`
- independent_review scientific reading: 

## Main judgment

The prototype is useful as a constructibility signal but is not ready to scale into a BabyLM pretraining pool. The measured strict pass gives 44/96 rows and 28/48 pairs with at least one accepted row. The best strict row per accepted pair has near-compact word and token geometry (mean word ratio 1.056, token ratio 1.039) and high source/natural lexical overlap (source recall 0.916, natural overlap 0.848). These numbers establish that constrained generation can often stay compact-like while avoiding unsupported content lemmas.

However, direct reading shows the present filter still admits many rows that are not fluent, not proposition-preserving, or not relation-preserving. The current success is therefore a mixed set: some good source-attested compact sentences, some near-source reductions, and some damaged sentences that lexical closure and length matching cannot detect.

## Good subset

Several rows are scientifically real examples of the intended object:

- Flies/zebras: `Having discovered flies' preference for dark coats, the team became interested in zebras.` preserves the discovery-before-interest relation with a source-attested compact surface.
- Money-market threshold: the generated sentence preserves the investor/fund/securities/amount relation and `$100,000`, though it weakens the word `only`.
- Emotional-cost agreement: the generated sentence preserves agreement, emotional cost, and supporting a personally different position, though it is less compressed than natural compact.
- Dr Cross/Mr Watts: the core hope/experiment/brain-activity/watching-dancing relation survives, with minor article and honorific deletion.
- Some ordinary rows are clean: e.g. `Regent Park Focus Youth Media Arts Centre nurtures youth leadership.` and `Knowing power bars were the same was critical.`

These rows show the route should not be abandoned. A high-precision source-attested fluent bridge may be constructible.

## Load-bearing weaknesses

### The strict accept flag still admits grammar damage

Examples that passed automatic strict acceptance but are not good training rows:

- `Resistance temperature coefficient found increasing with oxygen content from −0.1 to −3.5%/°C.` lacks a finite auxiliary.
- `The sculpture visualization shows the fire perimeter grew and changed direction over the first 114 hours burned.` has a malformed final participle attachment.
- `Note G describes Analytical Engine computing Bernoulli numbers, is complete rigorous, recognized first computer program...` is a compressed fragment with missing articles, auxiliaries, and coordination.
- `Since oxygen carried by blood, pores benefit from less nutrients.` has a bare passive and an unclear deprivation relation.
- `Flippers, legs, vertebrae, bones are not easily distinguishable...` damages the source appositive/list structure.
- `Saber Marine feels the expense justified by the superior hull result.` lacks the copular passive.
- `Regent Park Focus Youth Media Arts Centre dedicated nurturing youth leadership.` lacks the main predicate form.

The current fluency proxy catches only a few named patterns (`happy the bird`, bare `advised`, `from tray`, plural-subject `says`) and misses many analogous failures.

### Source-attested lemmas do not preserve propositions

Examples that passed lexical and geometry tests but changed or dropped central content:

- `Keep healthy foods safe by giving them.` changes the object: the source is about keeping the pup healthy, not keeping foods safe.
- `According to loop gravity equations, gravity reverses at extremely high densities.` deletes the `becomes a repulsive force` payload.
- The money-market output drops `only`, weakening exclusivity.
- `What are alternative data kinds lenders use?` drops the `what is alternative data` part of the question.
- `The critical ingredient was knowing power bars.` drops `were the same`, the central complement.
- Some relation-zero rows retain entities but drop a key participant or condition, e.g. the Tesla/SpaceX relation is lost in the short Elon Musk variant.

This is central for BabyLM: the intended data object is not merely a compact sequence of source words; it must preserve who did what to whom, under what polarity, modality, comparison, quantity, and cause/time relation.

### The relation proxy is often vacuous

Many accepted rows have `source_relation_count_proxy = 0` but are counted as retaining relations. This makes the reported relation-retention number too optimistic. Relation-bearing quality must be judged by actual predicate-argument and connector preservation, not by a coarse word-list count.

### The hard cases remain hardest

The aggregate 58.3% pair pass hides the important split. Hard source-absent + relation rows have only 7/32 strict row accepts (21.9%). Ordinary rows are much easier. The bridge matters scientifically only if it works on the hard natural compact cases where source-attested fluency has to replace novel compact wording without damaging relations.

## Implication for expensive work

Do not launch BabyLM training from source attested fluent bridge prototype readout outputs. The present generator plus automatic selector would inject enough malformed or proposition-damaged rows to make a full run hard to interpret: a negative training result would not tell whether source-attested fluent bridges are wrong or whether the pool was noisy; a positive result would still be entangled with near-source copying and hidden semantic loss.

The next work should build a high-precision construction-and-selection pipeline at small scale. GPU generation is cheap enough for prototype batches, but full DeBERTa/RoBERTa training should wait until retained rows are demonstrably fluent and proposition-preserving on stratified source/bridge/natural examples.

## Construction requirements before a larger pool

A larger source-attested bridge pool should require each retained row to satisfy the following tests:

1. Complete English sentence: finite predicate where needed; no dropped copula/auxiliary; no dangling participle; no malformed noun pile.
2. Predicate-argument preservation: subject, object, complement, coreference, and attachment match the source proposition.
3. Polarity, modality, quantification, comparison, causality, temporal order, attribution, numbers, signs, units, ranges, currencies, and named entities preserved when central.
4. Compact geometry close to natural compact at row and pool level: word and active-token ratios controlled, with no systematic long-copy drift.
5. Content lemma closure measured conservatively, but with exact protected surface checks for numbers/units/signs and names.
6. Nonvacuous relation reading: rows with no detected relation words should not automatically count as relation-preserving; relation identity and arguments must be read.
7. Source-copy degree tracked separately: conservative deletion can be useful, but the pool needs enough contextual re-expression rather than simple near-source truncation.
8. Human or independent_review reading on a stratified sample before training, with labels for fluency, proposition preservation, relation preservation, compactness, and source-copy degree.

## Better next construction

A source-attested bridge comparison requires repair of the generator and selector before pretraining. Another telegraphic extractive arm is not the proposed direction:

- Generate multiple candidates per pair under both regimes, with prompts that explicitly require a finite verb and all central complements.
- Add a second-pass self-repair prompt for failed rows, especially hard source-absent + relation rows.
- Build an automatic structural selector that rejects missing complements (`were the same`, `becomes repulsive`, `only`, negative/unit surfaces), bare passives, missing copulas, and malformed coordination.
- Produce a 200-300 pair stratified review set from the full compact pool, not only 48 pairs, and read a blinded sample against source and natural compact.
- Only if retained precision is high on hard relation rows should a small pool build proceed; training should remain downstream of quality, not a way to average over noise.

This keeps the research aligned with the strongest validated DeBERTa data result: compact generated second views plus reinvested source diversity. The current prototype strengthens the possibility that fluent source-attested re-expression can be built, but it has not yet shown that the source-attested bridge is a clean, scalable, training-worthy object.
