# earlier analysis static token mask prior

Prepare a compliance-safe learning-signal repair: redistribute MLM mask pressure toward relation and information-bearing corpus tokens under the legal spatial repair route status tokenizer, without reading evaluation examples or changing the training corpus.

## Inputs
- 10M pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- tokenizer SHA256: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- row limit: 500

## Corpus surface
- rows seen: 500
- words seen: 70003
- counted token occurrences: 99998
- observed token ids: 8704 / 16384
- changed rows seen: 500; high source/rewrite role alignment: 500
- rows reaching token length 256 under trainer truncation: 11

## Feature occurrence fractions
- causal: 979 tokens (0.9790%), distinct ids 66
- changed_block: 99017 tokens (99.0190%), distinct ids 8691
- changed_domain: 90068 tokens (90.0698%), distinct ids 8361
- dynamic: 1290 tokens (1.2900%), distinct ids 89
- entity: 15485 tokens (15.4853%), distinct ids 2736
- numeric: 3527 tokens (3.5271%), distinct ids 117
- paired_source: 58717 tokens (58.7182%), distinct ids 8304
- rewrite: 40300 tokens (40.3008%), distinct ids 7614
- rewrite_relation: 4979 tokens (4.9791%), distinct ids 311
- spatial: 5324 tokens (5.3241%), distinct ids 74
- temporal: 1226 tokens (1.2260%), distinct ids 48

## Scheme `relation_only_v1`
- occurrence-weighted mean: 1.000000
- min/max on observed ids: 0.8442 / 1.9839
- top weighted observed tokens:
  - id 4240 `Ġbeside` weight=1.983934 count=2 features={'spatial': 1.0, 'rewrite': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 1.0}
  - id 13725 `ĠThroughout` weight=1.941723 count=1 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 1.0, 'changed_block': 1.0, 'rewrite_relation': 1.0}
  - id 12835 `ĠAround` weight=1.920617 count=2 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.5, 'paired_source': 0.5, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.5}
  - id 4416 `ĠOver` weight=1.912499 count=13 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.4615, 'paired_source': 0.5385, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.4615}
  - id 1862 `Ġinside` weight=1.895994 count=12 features={'spatial': 1.0, 'rewrite': 0.5833, 'paired_source': 0.4167, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.5833}
  - id 1908 `Ġacross` weight=1.874184 count=25 features={'spatial': 1.0, 'rewrite': 0.48, 'paired_source': 0.52, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.48}
  - id 2856 `ĠSince` weight=1.874184 count=15 features={'causal': 1.0, 'temporal': 1.0, 'entity': 1.0, 'rewrite': 0.5333, 'paired_source': 0.4667, 'changed_block': 1.0, 'changed_domain': 0.8667, 'rewrite_relation': 0.5333}
  - id 1761 `Ġbehind` weight=1.870288 count=13 features={'spatial': 1.0, 'rewrite': 0.4615, 'paired_source': 0.5385, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.4615}
  - id 3858 `Ġbeyond` weight=1.864335 count=6 features={'spatial': 1.0, 'rewrite': 0.5, 'paired_source': 0.5, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.5}
  - id 1881 `Ġoutside` weight=1.863965 count=19 features={'spatial': 1.0, 'rewrite': 0.4737, 'paired_source': 0.5263, 'changed_block': 1.0, 'changed_domain': 0.8947, 'rewrite_relation': 0.4737}
  - id 2274 `ĠBy` weight=1.859645 count=18 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.2778, 'paired_source': 0.7222, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.2778}
  - id 388 `Ġfrom` weight=1.858943 count=334 features={'spatial': 1.0, 'rewrite': 0.4401, 'paired_source': 0.5599, 'changed_block': 1.0, 'changed_domain': 0.9192, 'rewrite_relation': 0.4401}
  - id 1208 `Ġagainst` weight=1.8573 count=24 features={'spatial': 1.0, 'rewrite': 0.4167, 'paired_source': 0.5833, 'changed_block': 1.0, 'changed_domain': 0.9583, 'rewrite_relation': 0.4167}
  - id 14843 `ĠAlong` weight=1.8573 count=5 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.2, 'paired_source': 0.8, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.2}
  - id 1229 `Ġnear` weight=1.846747 count=24 features={'spatial': 0.9583, 'rewrite': 0.5833, 'paired_source': 0.4167, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.5833}
  - id 3841 `Ġbelow` weight=1.846747 count=12 features={'spatial': 1.0, 'rewrite': 0.4167, 'paired_source': 0.5833, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.4167}
  - id 731 `Ġthrough` weight=1.845655 count=58 features={'spatial': 1.0, 'rewrite': 0.3793, 'paired_source': 0.6207, 'changed_block': 1.0, 'changed_domain': 0.9138, 'rewrite_relation': 0.3793}
  - id 601 `Ġinto` weight=1.842688 count=78 features={'spatial': 1.0, 'rewrite': 0.3462, 'paired_source': 0.6538, 'changed_block': 1.0, 'changed_domain': 0.9615, 'rewrite_relation': 0.3462}
  - id 1962 `Ġamong` weight=1.841065 count=26 features={'spatial': 1.0, 'rewrite': 0.3846, 'paired_source': 0.6154, 'changed_block': 1.0, 'changed_domain': 0.8462, 'rewrite_relation': 0.3846}
  - id 1084 `ĠAt` weight=1.840032 count=22 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.1364, 'paired_source': 0.8636, 'changed_block': 1.0, 'changed_domain': 0.9545, 'rewrite_relation': 0.1364}
  - id 3350 `Ġonto` weight=1.839209 count=7 features={'spatial': 1.0, 'rewrite': 0.4286, 'paired_source': 0.5714, 'changed_block': 1.0, 'changed_domain': 0.7143, 'rewrite_relation': 0.4286}
  - id 1733 `Ġwithin` weight=1.834435 count=24 features={'spatial': 1.0, 'rewrite': 0.2917, 'paired_source': 0.7083, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.2917}
  - id 213 `Ġto` weight=1.828443 count=1691 features={'spatial': 0.997, 'rewrite': 0.3134, 'paired_source': 0.6866, 'changed_block': 1.0, 'changed_domain': 0.9072, 'rewrite_relation': 0.3122}
  - id 2390 `ĠFrom` weight=1.825641 count=8 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.25, 'paired_source': 0.75, 'changed_block': 1.0, 'changed_domain': 0.5, 'rewrite_relation': 0.25}
  - id 250 `Ġon` weight=1.82375 count=424 features={'spatial': 0.9788, 'rewrite': 0.3679, 'paired_source': 0.6321, 'changed_block': 1.0, 'changed_domain': 0.9245, 'rewrite_relation': 0.3561}

## Scheme `relation_info_v1`
- occurrence-weighted mean: 1.000000
- min/max on observed ids: 0.8148 / 1.8161
- top weighted observed tokens:
  - id 13725 `ĠThroughout` weight=1.816095 count=1 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 1.0, 'changed_block': 1.0, 'rewrite_relation': 1.0}
  - id 12835 `ĠAround` weight=1.80933 count=2 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.5, 'paired_source': 0.5, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.5}
  - id 4416 `ĠOver` weight=1.777992 count=13 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.4615, 'paired_source': 0.5385, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.4615}
  - id 14843 `ĠAlong` weight=1.753826 count=5 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.2, 'paired_source': 0.8, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.2}
  - id 4240 `Ġbeside` weight=1.750746 count=2 features={'spatial': 1.0, 'rewrite': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 1.0}
  - id 3622 `yond` weight=1.742866 count=1 features={'spatial': 1.0, 'entity': 1.0, 'paired_source': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0}
  - id 7851 `ĠBetween` weight=1.742866 count=1 features={'spatial': 1.0, 'entity': 1.0, 'paired_source': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0}
  - id 2856 `ĠSince` weight=1.7399 count=15 features={'causal': 1.0, 'temporal': 1.0, 'entity': 1.0, 'rewrite': 0.5333, 'paired_source': 0.4667, 'changed_block': 1.0, 'changed_domain': 0.8667, 'rewrite_relation': 0.5333}
  - id 2274 `ĠBy` weight=1.733778 count=18 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.2778, 'paired_source': 0.7222, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.2778}
  - id 6895 `ĠThrough` weight=1.721962 count=6 features={'spatial': 1.0, 'entity': 1.0, 'paired_source': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0}
  - id 1084 `ĠAt` weight=1.718755 count=22 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.1364, 'paired_source': 0.8636, 'changed_block': 1.0, 'changed_domain': 0.9545, 'rewrite_relation': 0.1364}
  - id 2390 `ĠFrom` weight=1.717768 count=8 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.25, 'paired_source': 0.75, 'changed_block': 1.0, 'changed_domain': 0.5, 'rewrite_relation': 0.25}
  - id 2182 `ĠBecause` weight=1.699485 count=2 features={'causal': 1.0, 'entity': 1.0, 'rewrite': 1.0, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 1.0}
  - id 1029 `ĠOn` weight=1.697163 count=18 features={'spatial': 0.8889, 'entity': 1.0, 'rewrite': 0.5, 'paired_source': 0.5, 'changed_block': 1.0, 'changed_domain': 0.9444, 'rewrite_relation': 0.4444}
  - id 1507 `ĠTo` weight=1.666114 count=29 features={'spatial': 0.8966, 'entity': 1.0, 'rewrite': 0.2759, 'paired_source': 0.7241, 'changed_block': 1.0, 'changed_domain': 0.9655, 'rewrite_relation': 0.2414}
  - id 1862 `Ġinside` weight=1.665253 count=12 features={'spatial': 1.0, 'rewrite': 0.5833, 'paired_source': 0.4167, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.5833}
  - id 3858 `Ġbeyond` weight=1.651173 count=6 features={'spatial': 1.0, 'rewrite': 0.5, 'paired_source': 0.5, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.5}
  - id 1761 `Ġbehind` weight=1.646179 count=13 features={'spatial': 1.0, 'rewrite': 0.4615, 'paired_source': 0.5385, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.4615}
  - id 9640 `ĠDue` weight=1.645865 count=3 features={'causal': 1.0, 'entity': 1.0, 'rewrite': 0.6667, 'paired_source': 0.3333, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.6667}
  - id 565 `ĠIn` weight=1.640287 count=181 features={'spatial': 0.9061, 'entity': 1.0, 'rewrite': 0.3039, 'paired_source': 0.6961, 'changed_block': 1.0, 'changed_domain': 0.9282, 'rewrite_relation': 0.2486}
  - id 1908 `Ġacross` weight=1.638553 count=25 features={'spatial': 1.0, 'rewrite': 0.48, 'paired_source': 0.52, 'changed_block': 1.0, 'changed_domain': 1.0, 'rewrite_relation': 0.48}
  - id 1881 `Ġoutside` weight=1.634298 count=19 features={'spatial': 1.0, 'rewrite': 0.4737, 'paired_source': 0.5263, 'changed_block': 1.0, 'changed_domain': 0.8947, 'rewrite_relation': 0.4737}
  - id 3350 `Ġonto` weight=1.629766 count=7 features={'spatial': 1.0, 'rewrite': 0.4286, 'paired_source': 0.5714, 'changed_block': 1.0, 'changed_domain': 0.7143, 'rewrite_relation': 0.4286}
  - id 3841 `Ġbelow` weight=1.628639 count=12 features={'spatial': 1.0, 'rewrite': 0.4167, 'paired_source': 0.5833, 'changed_block': 1.0, 'changed_domain': 0.8333, 'rewrite_relation': 0.4167}
  - id 1208 `Ġagainst` weight=1.626881 count=24 features={'spatial': 1.0, 'rewrite': 0.4167, 'paired_source': 0.5833, 'changed_block': 1.0, 'changed_domain': 0.9583, 'rewrite_relation': 0.4167}

Full JSON: `experiments/archive/frontier_consolidation/data/static_token_mask_prior_smoke/static_token_mask_prior.json`
