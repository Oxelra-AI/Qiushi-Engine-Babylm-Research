# data substrate mechanism scale — FW1p75M source–compact mechanism construction

## Why this route exists

The active target is a compliant BabyLM Strict-Small SOTA, not a narrower score artifact. The completed compliant endpoints remain below the 41.80 public frontier, and the partial SGCR endpoint columns already make a large SGCR breakthrough unlikely unless the missing EWoK and SuperGLUE columns are unusually strong. U256 is a lower-priority successor because it changes only about 1.7% realized masked targets and exposes almost no FineWeb compact material.

The surviving data mechanism with replicated positive evidence is source-aligned compact-view learning: an original sentence and a shorter faithful rendering of the same proposition(s) give the learner two views of stable content under a fixed word budget. The current corpus uses only 423,511 FineWeb source+compact words. A mechanism-scale successor should test this principle at a substantially larger fraction of the 10M word budget while preserving enough original BabyLM material for broad language columns.

## First construction scale

Use **FW1p75M** as the first serious materialization target:

| component | words |
|---|---:|
| original BabyLM official-source rows | 7,919,680 |
| FineWeb source+compact or matched source-repeat block | 1,750,000 |
| retained inherited official-source Qwen paraphrase-pair rows | 330,311 |
| neutral top-up | 9 |
| total | 10,000,000 |

This replaces the inherited official-source Qwen paraphrase block before displacing official originals. It raises FineWeb source+compact mass from 423,511 words to 1,750,000 words, a 4.13x increase, while preserving the 7.92M original official-source foundation. It is not the public leader recipe: the leader model card describes about 10M FineWeb simplification-pair words plus a 64→256 and WWM→token schedule, whereas FW1p75M keeps the official-source foundation and tests a mixed-corpus compact-view mechanism.

At the current compact rewrite/source ratio 0.6177, FW1p75M needs about 1,081,802 FineWeb source words and 668,198 compact rewrite words. Existing cached sources union to 875,891 source words, so the source deficit is about 206k words, roughly 9k sentences at current lengths. If new-source generation cannot reach this quality quickly, FW1p5M is the lower-scale fallback: it still gives 3.54x current FineWeb pair mass and needs only about 927k source words, close to the cached pool.

## Required arms for attribution

Construct matched 10M corpora, not final deliverables.

1. **compact_view**
   - Same selected FineWeb sources as all arms.
   - Source sentence followed locally by faithful compact rewrite.
   - Compact rewrite length target: weighted rewrite/source ratio near 0.60–0.65, with accepted range roughly 0.50–0.75 unless the source demands longer wording to preserve proposition structure.

2. **source_repeat**
   - Same selected FineWeb sources, same source positions, same total FineWeb block word count, same row-length sequence, same official-source rows and retained old Qwen rows.
   - The second span is deterministic source material matched to the rewrite word count at word boundaries.
   - This is the indispensable comparison for compact re-expression beyond FineWeb content and same-source repetition.

3. **source_diversity** (construct if feasible before training; train after treatment/repeat if the first paired result is scientifically ambiguous or promising)
   - Same first-span selected sources.
   - The second-span budget is filled with different FineWeb sources matched by domain, document concentration, source length, and row length.
   - This separates aligned compact consolidation from seeing more independent propositions.

Because tokenizer learning counts as part of the legal data package, each arm must train its own tokenizer only on that arm's exact 10M corpus. The resulting comparison is a compliant corpus-tokenizer-system comparison. Tokenizer movement is not a nuisance to ignore; it must be measured as part of the mechanism chain.

## Source selection

Build the source list independent of evaluation text. Use evaluation text only later for overlap removal, never to choose examples that resemble benchmark wording.

Prioritize self-contained, stable propositions:

- physical affordance, material, object property, function and process;
- causal, conditional and mechanistic relations;
- spatial/geographic, part-whole and classification relations;
- quantities, units, comparison direction, dates, rates and order;
- social/institutional roles and stable processes;
- entity–attribute and entity–relation facts with clear arguments.

Preserve diversity over document, domain, entity, relation pattern and semantic cluster. Current cached union has 38,167 unique source sentences, 875,891 source words and at least 8,469 document ids. The new 206k source words needed for FW1p75M should come mainly from new documents and underrepresented physical/causal/spatial/quantity domains rather than further exploiting the top cached documents.

## Rewrite form

Compact rewrites must be faithful reductions, not factual expansions. They should preserve:

- named entities, numbers, units and comparison direction;
- polarity, modality and causal direction;
- subject, predicate and core arguments;
- independent propositions in subordinate clauses when dropping them would change the source meaning.

They should not add explanations, new world facts, new entities, changed units, changed time order, or softened/strengthened modality. Existing compact pairs have mean content recall 0.663, entity recall 0.996 and number recall 1.000; at FW1p75M scale this must be strengthened with source-pair records for entity, number, unit, predicate, argument, polarity, modality, comparison and causal direction retention.

## Research assets to build before any H100 training

The proposed construction uses `data/fw1p75m_source_compact_mechanism/` for the following scientific artifacts:

- frozen selected FineWeb source list with text, word count, doc id, domain tags, normalized hash and provenance path;
- generated/reused compact rewrite file with source hash, generation model/prompt/version, rewrite text, word count and retention measurements;
- matched `compact_view`, `source_repeat` and, if practical, `source_diversity` 10M corpora with exact word counts, row counts, row-length sequences, source accounting and SHA256 hashes;
- sidecar listing which inherited official-source Qwen rows are retained and which are displaced;
- per-arm tokenizer-only measurements after legal tokenizer training: vocab support spectrum, tokens/word, visible tokens, WWM groups, truncation, source/rewrite joint visibility and realized masked target counts;
- overlap scans against official evaluation text and old source pools, used only to remove near copies and record provenance.

Do not start full 100M training until the resumed SGCR collation is read and these corpus/tokenizer assets show that the arms are mechanically matched and scientifically interpretable.

## Expensive-work meaning if this route is launched

If SGCR stays sub-frontier after full official-compatible collation, FW1p75M is the strongest successor because it attacks the largest unresolved score mass (EWoK/GlobalPIQA/COMPS/Entity) through a source-content and compact-consolidation mechanism, rather than through small suffix visibility, pure optimizer change, or flat vocabulary size.

The first expensive comparison should be `compact_view` versus `source_repeat` on the best completed compliant backbone unless the completed SGCR vector changes the backbone choice. The result is meaningful if the whole official-compatible vector shows compact_view improving over source_repeat in EWoK and GlobalPIQA with support from COMPS or Entity, while preserving BLiMP, Supplement, Reading and SuperGLUE. If both FineWeb arms improve similarly, the result points to source substrate rather than compact consolidation. If source_diversity exceeds compact_view, broader independent proposition coverage is stronger than local aligned views. If compact_view exceeds both repeat and diversity and shows increased source–rewrite prediction consistency, the result supports a general source-aligned compact consolidation principle.

## Relation to frontier_consolidation

Legal 80M restarts test fresh optimizer momentum/variance. The mechanism-scale data construction above is a distinct comparison: source-repeat and diversity controls separate compact alignment and FineWeb content from restart dynamics.
