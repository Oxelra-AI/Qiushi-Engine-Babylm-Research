# closing ladder partial synthesis GlobalPIQA ladder item anatomy

This table checks exact GlobalPIQA item identities for coherent86, matched ordinary continuation, exact dense-mask/sparse-label acquisition, and clean preservation.

## Overall split counts

| endpoint | group | parallel | nonparallel | mean | delta vs coherent86 | contribution to Overall |
|---|---|---:|---:|---:|---:|---:|
| coherent86 | parent | 30/103 (Δ0) | 48/100 (Δ0) | 38.563107 | 0.000000 | 0.000000 |
| ordinary64 | ordinary | 32/103 (Δ2) | 48/100 (Δ0) | 39.533981 | 0.970874 | 0.107875 |
| ordinary65 | ordinary | 32/103 (Δ2) | 48/100 (Δ0) | 39.533981 | 0.970874 | 0.107875 |
| ms64 | ms | 31/103 (Δ1) | 50/100 (Δ2) | 40.048544 | 1.485437 | 0.165049 |
| ms65 | ms | 31/103 (Δ1) | 50/100 (Δ2) | 40.048544 | 1.485437 | 0.165049 |
| clean64 | clean | 31/103 (Δ1) | 50/100 (Δ2) | 40.048544 | 1.485437 | 0.165049 |
| clean65 | clean | 31/103 (Δ1) | 50/100 (Δ2) | 40.048544 | 1.485437 | 0.165049 |

## Deterministic item sets

Ordinary seed-shared changed items: 2
MS/clean seed-shared changed items: 5

### Ordinary shared items

- parallel parallel_ex000000_eng_latn: delta 1 / 1; gold='The amount of air in the bag stays the same'; prompt='A plastic bag is filled with air and then sealed. When an object is placed on the bag, what happens?'
- parallel parallel_ex000071_eng_latn: delta 1 / 1; gold='The wick should be longer than the height of the candle wax'; prompt='How long is the wick of a candle, compared to the height of the candle wax?'

### MS/clean shared items

- nonparallel group0123_ex000032_eng_latn_0_v1: deltas=[1, 1, 1, 1]; gold='Place the clay sticks onto a baking sheet and bake at 350°F for one minute.'; prompt='How do you bake small polymer clay sticks to make clay sprinkles?'
- nonparallel group0123_ex000084_eng_latn_0_v1: deltas=[1, 1, 1, 1]; gold='Place a dry towel on the counter first.'; prompt='How do I protect the counter from the heat of the iron ?'
- parallel parallel_ex000000_eng_latn: deltas=[1, 1, 1, 1]; gold='The amount of air in the bag stays the same'; prompt='A plastic bag is filled with air and then sealed. When an object is placed on the bag, what happens?'
- parallel parallel_ex000071_eng_latn: deltas=[1, 1, 1, 1]; gold='The wick should be longer than the height of the candle wax'; prompt='How long is the wick of a candle, compared to the height of the candle wax?'
- parallel parallel_ex000094_eng_latn: deltas=[-1, -1, -1, -1]; gold='Drying the laundry'; prompt='You put some cookies in the oven before doing some housework. What housework would not be possible to finish before the cookies need to be taken out of the oven?'

Outputs:
{
  "item_correctness": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/item_correctness.csv",
  "flips": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/flips_vs_coherent86.csv",
  "patterns": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/pattern_rows_vs_coherent86.csv",
  "ordinary_seed_shared": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ordinary_seed_shared_changed_items.csv",
  "ms_clean_seed_shared": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/ms_clean_seed_shared_changed_items.csv",
  "counts": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/counts.csv",
  "overall": "experiments/archive/relation_learning/data/globalpiqa_ladder_item_anatomy/overall.csv"
}
