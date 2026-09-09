# clean natural nearduplicate adjacency CLEAN_NATURAL_SPLIT materialization

This is a ready-to-train extent arm, not evidence yet. It begins the natural-composition test motivated by the refined CLEAN adjacency audit.

## Construction

Selected natural-source rows with a refined sentence-span content-overlap coefficient at least `0.5` and at least `8` content tokens per span. `qwen_pair_packed` was excluded and left unchanged. For each selected row, the later span of its strongest high-overlap pair was swapped with a same-source neutral span from a row without a selected hit when possible, using maximum raw word-count ratio `1.8`. This preserves the text multiset within natural subcorpora and keeps total word accounting exact, while removing one relation-bearing adjacency per selected row.

## Audit

- Selected natural rows: `2107`; performed swaps: `2106`.
- Natural-only τ=0.50 hit pairs before/after: `3508` -> `917`; row hits `2107` -> `405`.
- All-row τ=0.50 hit pairs before/after: `25625` -> `23034`; the dense qwen-pair block remains intentionally unchanged.
- Word count: 10M exact `True` (10000000); 100M exact `True` (100000000).
- 100M stream SHA prefix: `388da005a00f`.

## Pre-stated readout

If this arm is trained, score it against CLEAN on the same DeBERTa seed43022 coordinate with target-form-separated readouts. The removed natural dose is mixed: before splitting there were 79 exact-sequence hits and 3420 nonidentical high-overlap hits at τ=0.50. Therefore the strongest pre-stated signal is not a symmetric replay of the designed REPEAT arm. If baseline natural exact recurrences matter, splitting should reduce natural-copy gain and can relieve source-absent changed/substituted-token cost. If baseline natural nonidentical adjacency matters, splitting should reduce true-source advantage on source-recurring restatement tokens. Because the refined natural dose is about 3.5k selected hits per 10M versus 33.3k designed compact relations per 10M, a small or null measurement would mostly bound sensitivity to this natural dose rather than overturn the compact locality result.

## Files

- stream_10M: `experiments/archive/relation_learning/data/clean_natural_split_pools/cleanqwen_natural_sentence_split_tau050_10M.jsonl`
- stream_100M: `experiments/archive/relation_learning/data/clean_natural_split_pools/cleanqwen_natural_sentence_split_tau050_100M.jsonl`
- swaps: `experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_swaps.csv`
- by_source: `experiments/archive/relation_learning/data/clean_natural_split_pools/natural_split_before_after_by_source.csv`
- summary: `research/documents/relation_learning/data/clean_natural_split_pools/clean_natural_split_summary.md`
- metadata: `experiments/archive/relation_learning/data/clean_natural_split_pools/clean_natural_split_metadata.json`
