# earlier analysis dose original base-presence audit

Base 10M: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`

## dose21
- pairs: 9299
- exact original text already in same-source/example base row: 7256 pairs (0.780)
- original words already in base rows: 173390 / 225164 (0.770)
- pair words whose original was already in base: 341292 / 443200 (0.770)
- present originals not themselves replaced by dose25 same-key row: 1859

## dose25_extra
- pairs: 8340
- exact original text already in same-source/example base row: 6534 pairs (0.783)
- original words already in base rows: 156546 / 203368 (0.770)
- pair words whose original was already in base: 308029 / 400000 (0.770)
- present originals not themselves replaced by dose25 same-key row: 1676

## dose25_superset
- pairs: 17639
- exact original text already in same-source/example base row: 13790 pairs (0.782)
- original words already in base rows: 329936 / 428532 (0.770)
- pair words whose original was already in base: 649321 / 843200 (0.770)
- present originals not themselves replaced by dose25 same-key row: 3535

Presence means the selected pair's normalized original sentence is contained in a base 10M ordinary row with the same source and example_id. Because the 100M stream repeats the base pool ten times, such originals are cross-row duplicates when inserted in dose rows unless their same source/example_id row was also replaced. This does not test semantic novelty of the rewrite.
