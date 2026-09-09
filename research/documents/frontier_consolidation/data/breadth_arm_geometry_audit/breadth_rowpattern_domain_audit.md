# pre result reading order and replication plan MAX breadth row-pattern and domain audit

Changed rows: 7923; seq_length 256.

## Sequence hashes

| arm | tokens_full_sha | tokens_visible_sha | wwm_groups_sha |
|---|---|---|---|
| view | ab2ce41d40981b478338f2bda208066ce76d9e563632699ea3099ad5fa4489b2 | 4e224193fd0f499bb3cf295e0e691525864e1e8e49d233c47d8d254c40c121f1 | 5ae63cc770f533abf29dd0a5094d5187e8d7052a4594231f7f0b087f273a0ebb |
| repeat | 81e22812412454248b264fe7529fc96174648750739f1d228add095003229a5f | 4f98fe8004ddf38670c8e89d47743da7b4ac52f174f2bf7e2bb3f44a8ea11df7 | 74af57c0bed7b360c27e7e7fbc5d00492b946211cbde409b58b628c6d659fcd1 |
| breadth | 47eeb21063dbd9df2b8da64cdf1b94a2c2ca0ca26607a5fd18a2e257ed9e71c0 | 577b01f788f8ee6311c14c17f1ea8562b115b2eb2b52a4ce2927b0f3465a6a10 | f1680b0a3e2712f824c147b1d791e46d7f7ecb9bb4568bc4e5c6cf7dd5af11a7 |

## Equality and pattern distance vs view

| comparison | token-count exact | visible-count exact | WWM-count exact | word-start pattern exact | mean pattern distance |
|---|---|---|---|---|---|
| repeat_vs_view | 0.0321 | 0.0414 | 0.9816 | 0.0000 | 0.3348 |
| breadth_vs_view | 0.0303 | 0.0332 | 0.9797 | 0.0000 | 0.3746 |

## Visible WWM group size

| arm | mean | sd | p05 | median | p95 |
|---|---|---|---|---|---|
| view | 1.43927 | 0.90282 | 1.0 | 1.0 | 3.0 |
| repeat | 1.38693 | 0.85994 | 1.0 | 1.0 | 3.0 |
| breadth | 1.40200 | 0.89030 | 1.0 | 1.0 | 3.0 |

## Domain and quality

Common text-domain JS rewrite vs breadth: `0.04092502211084107`
Metadata-domain JS rewrite vs breadth: `0.3820765822075588`

Rewrite domains by pair metadata: `{'no_domain': 20838, 'science_technical': 3922, 'quant_numeric': 2831, 'causal_relational': 2660, 'institutions_society': 2368, 'geography_places': 1834, 'media_culture': 832, 'people_history': 790}`
Breadth domains by source metadata: `{'entity_factual': 5817, 'quant_numeric': 3797, 'no_domain': 3096, 'causal_relational': 1564, 'science_technical': 849}`
Rewrite flags: `{'no_soft_flags': 31803, 'source_risk_long_parenthetical': 467, 'source_risk_deictic_time': 279, 'near_copy_view': 239, 'source_risk_apostle_or_title_apposition': 239, 'source_risk_heading_dash_chain': 163, 'source_risk_probability_hedge': 138, 'not_shorter_than_source': 15, 'source_risk_web_or_reference_phrase': 10}`
Breadth flags: `{'no_flags': 14326, 'flagged': 797, 'url_or_email': 406, 'tableish': 292, 'bad_substring': 49, 'mojibake': 25, 'nonlatin': 21, 'html': 21, 'symbol_or_index_like': 4}`

## Conditions to carry

- visible word-start pattern exact match is only 0.0000; rowwise subword-boundary geometry differs even though WWM group totals match
- mean WWM group token span differs by -2.590%: selected groups expose different subword span lengths
- common text-domain distribution differs (JS=0.0409 bits)
