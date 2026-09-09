# earlier analysis training-corpus treatment geometry under the spatial repair route status tokenizer

CPU-only substrate comparison. Uses only allowed training corpus files and the fixed spatial repair route status legal tokenizer; no evaluation text, no GPU work, no route selection by itself.

## Integrity

- clean_10m_sha256: `True`
- clean_100m_sha256: `True`
- reinvest_10m_sha256: `True`
- reinvest_100m_sha256: `True`
- tokenizer_sha256: `True`

## Headline corpus geometry

| corpus | rows | words | raw tok/word | visible tok/word | visible groups/word | over256 rows | truncated tokens | any-cue frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| clean_qwen_10m | 64381 | 10000000 | 1.468864 | 1.430053 | 0.979209 | 15872 | 388105 | 0.259732 |
| lengthmatched_compact_reinvest_10m | 64740 | 10000000 | 1.468864 | 1.431499 | 0.979995 | 15305 | 373647 | 0.259732 |
| compact_repeat_reinvest_10m | 64740 | 10000000 | 1.464012 | 1.427089 | 0.980239 | 15085 | 369228 | 0.255734 |
| compact_view_reinvest_10m | 64740 | 10000000 | 1.466452 | 1.429489 | 0.980219 | 15117 | 369626 | 0.255834 |

## Compact-view reinvest minus clean-Qwen

- Δ rows: `359`
- Δ raw_tokens_step35: `-24117`
- Δ visible_tokens_seq256_step35: `-5638`
- Δ visible_groups_seq256_step35: `10100`
- Δ over256_rows_step35: `-755`
- Δ truncated_tokens_step35: `-18479`
- Δ raw_tokens_per_declared_word_step35: `-0.002411700`
- Δ visible_tokens_per_declared_word_step35: `-0.000563800`
- Δ visible_groups_per_declared_word_step35: `0.001010000`
- Δ mean_tokens_per_visible_group_step35: `-0.002079963`

## Source-class shift (reinvest minus clean)

| source class | Δ rows | Δ words | clean frac | reinvest frac | Δ frac |
|---|---:|---:|---:|---:|---:|
| fineweb_source_compact_view_pair | 3005 | 423511 | 0.000000 | 0.042351 | 0.042351 |
| childes | -861 | -137760 | 0.271264 | 0.257488 | -0.013776 |
| gutenberg | -622 | -99520 | 0.195888 | 0.185936 | -0.009952 |
| open_subtitles | -611 | -97760 | 0.192720 | 0.182944 | -0.009776 |
| simple_wiki | -354 | -56640 | 0.111568 | 0.105904 | -0.005664 |
| bnc_spoken | -192 | -30720 | 0.060656 | 0.057584 | -0.003072 |
| switchboard | -7 | -1120 | 0.002224 | 0.002112 | -0.000112 |
| neutral_topup | 1 | 9 | 0.000000 | 0.000001 | 0.000001 |
| inherited_qwen_pair_packed | 0 | 0 | 0.165680 | 0.165680 | 0.000000 |

## Coarse cue-rate shift (reinvest minus clean)

| cue class | clean frac | reinvest frac | Δ frac | Δ words |
|---|---:|---:|---:|---:|
| any_cue | 0.259732 | 0.255834 | -0.003898 | -38979 |
| mental_social_dialogue | 0.103012 | 0.099448 | -0.003564 | -35638 |
| negation_modality | 0.026468 | 0.025859 | -0.000608 | -6084 |
| material_property | 0.015474 | 0.015707 | 0.000234 | 2339 |
| quantity_measure | 0.019579 | 0.019787 | 0.000208 | 2084 |
| spatial_state | 0.058837 | 0.058705 | -0.000132 | -1321 |
| causal_temporal | 0.033170 | 0.033104 | -0.000066 | -661 |
| physical_dynamics | 0.013430 | 0.013390 | -0.000040 | -402 |

## Interpretation

The mature clean-vs-reinvest vector, when it lands, should be read as a treatment changing two broad substrate properties: it replaces 423,520 official clean-Qwen words with 423,511 FineWeb source+compact-view pair words plus 9 neutral top-up words, and it changes tokenizer-visible target volume/context geometry under the same spatial repair route status tokenizer.
This script does not turn official weak subtasks into a new objective. It compares only training-corpus composition and token geometry, without further endpoint microscopy.
If reinvest beats clean broadly at 70M/80M, the source-diversity plus compact second-view mechanism survives in legal coordinates and should be combined with representation/optimization evidence rather than relation-only masking.
If reinvest gains only where the corpus adds broad factual/physical/source diversity but loses in dialogue/syntax columns, the next mechanism may be how compact second views trade lexical/syntactic consolidation for semantic/source breadth, which source-view consistency or representation changes could address.
If reinvest is broadly below clean despite old-tokenizer gains, the old mechanism depended on the inherited representation coordinate and broader hypotheses must be reopened instead of mining benchmark-specific token classes.

Full JSON: `experiments/archive/frontier_consolidation/data/corpus_treatment_geometry/corpus_treatment_geometry.json`
