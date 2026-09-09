# route3 correspondence closure semantic Route 3 pair rebuild

Corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Corpus SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
True semantic pairs: 526
Target/family shuffle controls: 526
Family/length matched controls: 526
Target-swapped nulls: 526

## True-pair quality summary
- same entity head fraction: 0.105
- same entity category fraction: 0.618
- mean word-length difference: 3.50
- median word-length difference: 2.00

## Family yields
- bright/dark: kept 66 / candidate_edges 551 (semantic=14, cross_fill=52, same_head_kept=8)
- deep/shallow: kept 9 / candidate_edges 24 (semantic=6, cross_fill=3, same_head_kept=6)
- flat/sharp: kept 14 / candidate_edges 43 (semantic=0, cross_fill=14, same_head_kept=0)
- frozen/melted: kept 6 / candidate_edges 16 (semantic=0, cross_fill=6, same_head_kept=0)
- full/empty: kept 48 / candidate_edges 1107 (semantic=19, cross_fill=29, same_head_kept=2)
- heavy/light: kept 27 / candidate_edges 182 (semantic=2, cross_fill=25, same_head_kept=0)
- hot/cold: kept 45 / candidate_edges 284 (semantic=11, cross_fill=34, same_head_kept=2)
- large/small: kept 69 / candidate_edges 1232 (semantic=9, cross_fill=60, same_head_kept=1)
- new/old: kept 30 / candidate_edges 130 (semantic=0, cross_fill=30, same_head_kept=0)
- open/closed: kept 80 / candidate_edges 2276 (semantic=31, cross_fill=49, same_head_kept=28)
- smooth/rough: kept 9 / candidate_edges 21 (semantic=1, cross_fill=8, same_head_kept=1)
- soft/hard: kept 18 / candidate_edges 155 (semantic=0, cross_fill=18, same_head_kept=0)
- strong/weak: kept 42 / candidate_edges 366 (semantic=3, cross_fill=39, same_head_kept=2)
- thick/thin: kept 15 / candidate_edges 88 (semantic=5, cross_fill=10, same_head_kept=4)
- tight/loose: kept 18 / candidate_edges 36 (semantic=0, cross_fill=18, same_head_kept=0)
- warm/cool: kept 15 / candidate_edges 29 (semantic=2, cross_fill=13, same_head_kept=1)
- wet/dry: kept 15 / candidate_edges 45 (semantic=2, cross_fill=13, same_head_kept=0)

## Files
- true_semantic: `experiments/archive/representation_and_objectives/data/semantic_route3/route3_semantic_true_pairs.jsonl`
- target_family_shuffle: `experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_family_shuffle.jsonl`
- family_length_matched: `experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_family_length_matched.jsonl`
- target_swapped_null: `experiments/archive/representation_and_objectives/data/semantic_route3/route3_control_target_swapped_null.jsonl`
- JSON summary: `experiments/archive/representation_and_objectives/data/semantic_route3/semantic_route3_summary.json`
