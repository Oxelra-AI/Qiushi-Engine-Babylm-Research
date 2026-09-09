# route reconstruction — candidate signal inventory

CPU-only inventory of two legal candidate signals before any new training.

## Directed source→compact edit structure
- pair records: `12155`; text pairs parsed: `12155`; missing text fields: `0`
- usable changed-span probes (>=3 changed target tokens and >=1 equal anchor length>=3): `7027`
- directional edit targets: `8004`
- overlap_multiset: mean `0.4793`, median `0.4688`, p10 `0.3214`, p90 `0.6500`
- rewrite_recall_from_source: mean `0.8315`, median `0.8421`, p10 `0.6667`, p90 `1.0000`
- changed_target_tokenish: mean `4.0545`, median `4.0000`, p10 `1.0000`, p90 `8.0000`
- changed_target_frac: mean `0.2653`, median `0.2500`, p10 `0.0833`, p90 `0.4762`
- n_equal_anchors_ge3: mean `1.5594`, median `1.0000`, p10 `0.0000`, p90 `3.0000`
- overlap bins: `{'0p4_0p6': 6567, 'lt_0p4': 3322, '0p6_0p8': 2122, 'ge_0p8': 144}`
- changed-target fraction bins: `{'0p2_0p4': 5325, 'lt_0p2': 4363, '0p4_0p6': 2046, 'ge_0p6': 421}`

## Intra-row natural adjacency
Cross-row order is not available from the packed pool; only within-row adjacency is valid for this route.

| source | rows | words | segments | rows>=3seg | adj pairs | adj triples | mean seg/row | median seg words | natural? | exclude? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| childes | 16093 | 2574880 | 348939 | 16085 | 332846 | 316758 | 21.68 | 6.0 | True | False |
| gutenberg | 11621 | 1859360 | 90580 | 11508 | 78959 | 67370 | 7.79 | 16.0 | True | False |
| open_subtitles | 11434 | 1829440 | 222445 | 11334 | 211011 | 199639 | 19.45 | 6.0 | True | False |
| qwen_pair_packed | 12236 | 1656800 | 73631 | 12047 | 61395 | 49159 | 6.02 | 21.0 | False | True |
| simple_wiki | 6619 | 1059040 | 59633 | 6527 | 53014 | 46447 | 9.01 | 13.0 | True | False |
| bnc_spoken | 3599 | 575840 | 33297 | 3505 | 29698 | 26119 | 9.25 | 11.0 | True | False |
| cleanqwen_fineweb_compact_view_reinvest | 3005 | 423511 | 24321 | 3005 | 21316 | 18311 | 8.09 | 16.0 | False | True |
| switchboard | 132 | 21120 | 1795 | 132 | 1663 | 1531 | 13.60 | 9.0 | True | False |
| neutral_cleanqwen_topup_compact_reinvest::open_subtitles | 1 | 9 | 1 | 0 | 0 | 0 | 1.00 | 8.0 | False | False |

Natural totals excluding synthetic rows:
- rows `49498`, words `7919680`, segments `756689`, adjacent pairs `707191`, adjacent triples `657864`

## Route implications
- directed edit-state zero-training probe supported: `True`
- natural discourse-state zero-training probe supported: `True`
- Discourse route must exclude `qwen_pair_packed` and FineWeb compact rows from the natural adjacency auxiliary, and use true/reversed/shuffled within-source controls.
- Transformation route must measure conditional token NLL on changed spans with overlap/edit-matched decoys; pair retrieval or pooled alignment is saturated and not a target.

JSON: `experiments/archive/frontier_consolidation/data/candidate_signal_inventory/candidate_signal_inventory.json`
