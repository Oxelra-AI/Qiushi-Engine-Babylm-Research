# aligned rewrite mechanism result — aligned-rewrite mechanism result at 10M

## Scientific question

The experiment tested whether **correct semantic correspondence within one MLM window** produces surface-invariant entity/relation representations beyond identical text exposure with wrong correspondence.

The intended mechanism was not simply "use simplification data". The falsifiable mechanism was:

> source_i + target_i in the same window should make WWM MLM learn semantic invariants useful for Entity Tracking and EWoK, more than source_i + target_perm[i] with the same source/target text multisets.

## Data and training equality

Both arms used the same fixed training base:

- S1 DeBERTa-v2 12×384, intermediate 1280;
- baseline16k tokenizer;
- AdamW, LR 0.001, flat WWM, sequence length 256;
- effective exposure 9,999,996 words;
- 829 optimizer steps;
- checkpoint `chck_9999996w`.

Materialization invariants from `data/aligned_rewrite_revision_150/manifest_w9999996_n212001_sel15001_shuf15002_side128-128.json`:

- 212,001 selected pairs;
- same source and target text multisets for aligned and shuffled;
- same source+target text multiset for unpaired_mix;
- zero aligned-vs-shuffled token-count and WWM-group deltas;
- no examples over 256 baseline16k tokens;
- no identity target adjacency in shuffled.

Training dynamics:

| arm | final MLM loss | note |
|---|---:|---|
| aligned | 6.7672 | correct source-target pairing |
| shuffled | 6.5663 | wrong target pairing, same text multiset |
| aligned - shuffled | +0.2010 | aligned was harder / worse pretraining loss |

## 10M evaluation results

Evidence files:

- available-coordinate comparison: `data/aligned_vs_shuffled_10m_available_comparison.json`
- full EWoK comparison: `data/aligned_vs_shuffled_10m_full_ewok_comparison.json`
- S1 official-corpus 10M full EWoK: `data/s1_10m_full_ewok_word_tokenize_score.json`
- public leader local scores: `data/public_leader_available_coordinate.json`, `data/public_leader_full_ewok_word_tokenize_score.json`

| column | aligned | shuffled | aligned - shuffled | S1 official 10M | public leader |
|---|---:|---:|---:|---:|---:|
| BLiMP | 53.65 | 51.83 | +1.82 | 53.37 | 67.20 |
| Supplement | 44.02 | 43.84 | +0.18 | 52.34 | 56.04 |
| Entity | 16.00 | 16.42 | -0.42 | 17.73 | 28.45 |
| COMPS | 50.03 | 50.18 | -0.15 | 50.34 | 53.57 |
| GlobalPIQA parallel | 10.68 | 13.59 | -2.91 | 19.42 | 22.33 |
| GlobalPIQA nonparallel | 51.00 | 52.00 | -1.00 | 51.00 | 57.00 |
| GlobalPIQA mean | 30.84 | 32.795 | -1.955 | 35.21 | 39.665 |
| Reading mean | 2.715 | 2.615 | +0.10 | 8.35 | 5.425 |
| full EWoK | 49.42 | 51.17 | -1.75 | 50.60 | 56.07 |

## Mechanism interpretation

The primary prediction fails for this data/source/scale:

- aligned does **not** improve Entity over shuffled; it is lower by 0.42;
- aligned does **not** improve EWoK over shuffled; it is lower by 1.75;
- aligned also lowers GlobalPIQA mean by 1.955 versus shuffled;
- aligned only improves BLiMP and Reading slightly, which are not the target mechanism columns.

The comparison with S1 official-corpus 10M is also unfavorable:

- both aligned and shuffled rewrite-pair corpora reduce Entity relative to S1 official 10M (16.00/16.42 vs 17.73);
- aligned reduces EWoK relative to S1 official 10M (49.42 vs 50.60);
- shuffled is only slightly above S1 EWoK (51.17 vs 50.60), while still damaging Entity, Supplement, GlobalPIQA, and Reading substantially.

Therefore the simple mechanism

> correct WikiAuto simplification correspondence inside the MLM window creates entity/world-knowledge invariance

is negative at 10M under the S1 base.

This does **not** imply all semantic redundancy is useless. It means the accessible `GEM/wiki_auto_asset_turk` sentence-pair construction lacks the specific capability-producing structure needed for the public leader profile, or that naive same-window adjacency makes the MLM objective learn local text-distribution and sentence-level simplification artifacts rather than reusable entity/relation abstraction.

## What the negative result rules out

It rules out directly scaling this exact aligned WikiAuto pair construction to 100M as the next main route. Scaling would require a new reason, such as a mechanistic probe showing late-stage emergence or a revised data source with much higher entity/relation density.

It also weakens the idea that the public leader's gains come from alignment alone. If the public leader's simplification-pair data is load-bearing, its useful property may instead be one or more of:

1. **source quality and diversity**: FineWeb-derived source text may contain broader real-world entities and commonsense relations than WikiAuto/Turk simplifications;
2. **entity density**: leader data may contain many named/common entities and factual predicates, while WikiAuto pairs may be mostly sentence simplification style;
3. **semantic compression**: simplification may help only when the target is a compressed abstraction of the source rather than a loose sentence-level rewrite;
4. **curriculum/data interaction**: leader's length/mask curriculum and 40k tokenizer may only help under the paired FineWeb distribution, not under official or WikiAuto data;
5. **distributional regularization rather than correspondence**: wrong pairing or unpaired mixture may be enough if simplification text distribution is the useful part.

## Next scientific work

The next step should not automatically train unpaired/source/rewrite controls or scale aligned to 100M. First, decide what question would most reduce uncertainty.

Concrete next options:

1. **Probe the current corpus for mechanism failure**: quantify entity density, pronoun/coreference density, relation-bearing predicates, source-target lexical overlap, target compression ratio, and whether aligned examples actually contain cross-surface entity/relation correspondences. If the corpus lacks the needed structure, rebuild data rather than train more arms.
2. **Train `unpaired_mix` only if it answers a still-live question**: since shuffled already beats aligned on EWoK and shares same text multiset, unpaired_mix can tell whether same-window wrong pairing itself helps or hurts, but it will not rescue the correct-correspondence mechanism.
3. **Design a higher-density semantic-redundancy source**: official-corpus-derived or open-source pairs where entity, relation, and state correspondences are explicit and checked by automatic filters, not generic simplification pairs.
4. **Reopen broader data-efficient learning mechanisms**: the public leader gap may require structured data selection/compression, not passive aligned adjacency.

This negative mechanism evidence constrains the next experimental design; it does not establish that all approaches to improving Overall will fail.
