# earlier analysis static token mask prior

Prepare a compliance-safe learning-signal repair: redistribute MLM mask pressure toward relation and information-bearing corpus tokens under the legal spatial repair route status tokenizer, without reading evaluation examples or changing the training corpus.

## Inputs
- 10M pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- pool SHA256: `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- tokenizer SHA256: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`
- row limit: full

## Corpus surface
- rows seen: 64740
- words seen: 10000000
- counted token occurrences: 14294893
- observed token ids: 16335 / 16384
- changed rows seen: 3005; high source/rewrite role alignment: 3005
- rows reaching tokenizer max length 256 (possible trainer truncation): 15548

## Feature occurrence fractions
- causal: 75275 tokens (0.5266%), distinct ids 210
- changed_block: 601461 tokens (4.2075%), distinct ids 13112
- changed_domain: 473640 tokens (3.3134%), distinct ids 12675
- dynamic: 143966 tokens (1.0071%), distinct ids 382
- entity: 3569435 tokens (24.9700%), distinct ids 7852
- numeric: 368739 tokens (2.5795%), distinct ids 1499
- paired_source: 356746 tokens (2.4956%), distinct ids 12904
- rewrite: 244715 tokens (1.7119%), distinct ids 12406
- rewrite_relation: 29552 tokens (0.2067%), distinct ids 566
- spatial: 701383 tokens (4.9065%), distinct ids 311
- temporal: 153818 tokens (1.0760%), distinct ids 228

## Scheme `relation_only_v1`
- occurrence-weighted mean: 1.000000
- min/max on observed ids: 0.9258 / 1.9096
- top weighted observed tokens:
  - id 6895 `ĠThrough` weight=1.909644 count=173 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0116, 'paired_source': 0.0867, 'changed_block': 0.0983, 'changed_domain': 0.0983, 'rewrite_relation': 0.0116}
  - id 13725 `ĠThroughout` weight=1.904709 count=88 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0114, 'paired_source': 0.0455, 'changed_block': 0.0568, 'changed_domain': 0.0455, 'rewrite_relation': 0.0114}
  - id 12511 `ĠWithin` weight=1.904678 count=102 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0098, 'paired_source': 0.0686, 'changed_block': 0.0784, 'changed_domain': 0.049, 'rewrite_relation': 0.0098}
  - id 12835 `ĠAround` weight=1.904552 count=97 features={'spatial': 0.9897, 'entity': 1.0, 'rewrite': 0.0412, 'paired_source': 0.0619, 'changed_block': 0.1031, 'changed_domain': 0.0722, 'rewrite_relation': 0.0412}
  - id 2390 `ĠFrom` weight=1.904244 count=1242 features={'spatial': 0.9984, 'entity': 1.0, 'rewrite': 0.0169, 'paired_source': 0.0338, 'changed_block': 0.0507, 'changed_domain': 0.0427, 'rewrite_relation': 0.0169}
  - id 7851 `ĠBetween` weight=1.903956 count=213 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0094, 'paired_source': 0.0329, 'changed_block': 0.0423, 'changed_domain': 0.0423, 'rewrite_relation': 0.0094}
  - id 12235 `ĠFROM` weight=1.897871 count=88 features={'spatial': 1.0, 'entity': 1.0}
  - id 3622 `yond` weight=1.872486 count=62 features={'spatial': 0.9677, 'entity': 0.9194, 'rewrite': 0.0161, 'paired_source': 0.0645, 'changed_block': 0.0806, 'changed_domain': 0.0484, 'rewrite_relation': 0.0161}
  - id 4640 `Ġthroughout` weight=1.869234 count=472 features={'spatial': 1.0, 'rewrite': 0.0339, 'paired_source': 0.0953, 'changed_block': 0.1292, 'changed_domain': 0.1059, 'rewrite_relation': 0.0339}
  - id 3841 `Ġbelow` weight=1.867616 count=638 features={'spatial': 0.9922, 'rewrite': 0.058, 'paired_source': 0.0878, 'changed_block': 0.1458, 'changed_domain': 0.1066, 'rewrite_relation': 0.058}
  - id 1962 `Ġamong` weight=1.864097 count=1509 features={'spatial': 1.0, 'rewrite': 0.0278, 'paired_source': 0.0583, 'changed_block': 0.0861, 'changed_domain': 0.0656, 'rewrite_relation': 0.0278}
  - id 1908 `Ġacross` weight=1.863729 count=1749 features={'spatial': 1.0, 'rewrite': 0.0292, 'paired_source': 0.0389, 'changed_block': 0.068, 'changed_domain': 0.0583, 'rewrite_relation': 0.0292}
  - id 1122 `Ġbetween` weight=1.863637 count=3778 features={'spatial': 0.9997, 'rewrite': 0.0257, 'paired_source': 0.0598, 'changed_block': 0.0855, 'changed_domain': 0.0693, 'rewrite_relation': 0.0254}
  - id 388 `Ġfrom` weight=1.863077 count=30772 features={'spatial': 0.9998, 'rewrite': 0.0291, 'paired_source': 0.0389, 'changed_block': 0.068, 'changed_domain': 0.0533, 'rewrite_relation': 0.0291}
  - id 2856 `ĠSince` weight=1.862366 count=970 features={'causal': 0.9928, 'temporal': 0.9928, 'entity': 1.0, 'rewrite': 0.0423, 'paired_source': 0.0515, 'changed_block': 0.0938, 'changed_domain': 0.0794, 'rewrite_relation': 0.0423}
  - id 1733 `Ġwithin` weight=1.862319 count=1996 features={'spatial': 0.9995, 'rewrite': 0.0225, 'paired_source': 0.0601, 'changed_block': 0.0827, 'changed_domain': 0.0646, 'rewrite_relation': 0.0225}
  - id 7999 `ĠAmong` weight=1.859593 count=208 features={'spatial': 0.9471, 'entity': 1.0, 'rewrite': 0.0096, 'paired_source': 0.1154, 'changed_block': 0.125, 'changed_domain': 0.0913, 'rewrite_relation': 0.0096}
  - id 3858 `Ġbeyond` weight=1.859565 count=632 features={'spatial': 1.0, 'rewrite': 0.0206, 'paired_source': 0.0269, 'changed_block': 0.0475, 'changed_domain': 0.0348, 'rewrite_relation': 0.0206}
  - id 1208 `Ġagainst` weight=1.858723 count=3422 features={'spatial': 0.9997, 'rewrite': 0.0175, 'paired_source': 0.0263, 'changed_block': 0.0438, 'changed_domain': 0.0362, 'rewrite_relation': 0.0175}
  - id 1029 `ĠOn` weight=1.858177 count=3221 features={'spatial': 0.9509, 'numeric': 0.0003, 'entity': 0.9997, 'rewrite': 0.0152, 'paired_source': 0.027, 'changed_block': 0.0422, 'changed_domain': 0.0335, 'rewrite_relation': 0.0112}
  - id 731 `Ġthrough` weight=1.858003 count=7727 features={'spatial': 0.9988, 'rewrite': 0.0168, 'paired_source': 0.0333, 'changed_block': 0.0501, 'changed_domain': 0.0396, 'rewrite_relation': 0.0166}
  - id 5572 `Ġbeneath` weight=1.857996 count=368 features={'spatial': 1.0, 'rewrite': 0.0136, 'paired_source': 0.0272, 'changed_block': 0.0408, 'changed_domain': 0.0353, 'rewrite_relation': 0.0136}
  - id 2707 `Ġtoward` weight=1.856396 count=1048 features={'spatial': 0.9981, 'rewrite': 0.0181, 'paired_source': 0.0239, 'changed_block': 0.042, 'changed_domain': 0.0258, 'rewrite_relation': 0.0181}
  - id 601 `Ġinto` weight=1.85638 count=12066 features={'spatial': 0.9981, 'rewrite': 0.0137, 'paired_source': 0.0323, 'changed_block': 0.046, 'changed_domain': 0.0371, 'rewrite_relation': 0.0135}
  - id 2345 `Ġabove` weight=1.855796 count=1274 features={'spatial': 0.9906, 'rewrite': 0.0298, 'paired_source': 0.0581, 'changed_block': 0.0879, 'changed_domain': 0.0651, 'rewrite_relation': 0.0298}

## Scheme `relation_info_v1`
- occurrence-weighted mean: 1.000000
- min/max on observed ids: 0.8655 / 1.8296
- top weighted observed tokens:
  - id 13725 `ĠThroughout` weight=1.829568 count=88 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0114, 'paired_source': 0.0455, 'changed_block': 0.0568, 'changed_domain': 0.0455, 'rewrite_relation': 0.0114}
  - id 12835 `ĠAround` weight=1.827993 count=97 features={'spatial': 0.9897, 'entity': 1.0, 'rewrite': 0.0412, 'paired_source': 0.0619, 'changed_block': 0.1031, 'changed_domain': 0.0722, 'rewrite_relation': 0.0412}
  - id 12511 `ĠWithin` weight=1.82736 count=102 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0098, 'paired_source': 0.0686, 'changed_block': 0.0784, 'changed_domain': 0.049, 'rewrite_relation': 0.0098}
  - id 12235 `ĠFROM` weight=1.82406 count=88 features={'spatial': 1.0, 'entity': 1.0}
  - id 6895 `ĠThrough` weight=1.823566 count=173 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0116, 'paired_source': 0.0867, 'changed_block': 0.0983, 'changed_domain': 0.0983, 'rewrite_relation': 0.0116}
  - id 7851 `ĠBetween` weight=1.815511 count=213 features={'spatial': 1.0, 'entity': 1.0, 'rewrite': 0.0094, 'paired_source': 0.0329, 'changed_block': 0.0423, 'changed_domain': 0.0423, 'rewrite_relation': 0.0094}
  - id 3622 `yond` weight=1.802007 count=62 features={'spatial': 0.9677, 'entity': 0.9194, 'rewrite': 0.0161, 'paired_source': 0.0645, 'changed_block': 0.0806, 'changed_domain': 0.0484, 'rewrite_relation': 0.0161}
  - id 14843 `ĠAlong` weight=1.792232 count=78 features={'spatial': 0.9359, 'entity': 1.0, 'rewrite': 0.0128, 'paired_source': 0.0897, 'changed_block': 0.1026, 'changed_domain': 0.1026, 'rewrite_relation': 0.0128}
  - id 2390 `ĠFrom` weight=1.788623 count=1242 features={'spatial': 0.9984, 'entity': 1.0, 'rewrite': 0.0169, 'paired_source': 0.0338, 'changed_block': 0.0507, 'changed_domain': 0.0427, 'rewrite_relation': 0.0169}
  - id 7999 `ĠAmong` weight=1.783563 count=208 features={'spatial': 0.9471, 'entity': 1.0, 'rewrite': 0.0096, 'paired_source': 0.1154, 'changed_block': 0.125, 'changed_domain': 0.0913, 'rewrite_relation': 0.0096}
  - id 8986 `ĠBY` weight=1.767791 count=137 features={'spatial': 0.927, 'entity': 1.0, 'paired_source': 0.0073, 'changed_block': 0.0073, 'changed_domain': 0.0073}
  - id 2856 `ĠSince` weight=1.755531 count=970 features={'causal': 0.9928, 'temporal': 0.9928, 'entity': 1.0, 'rewrite': 0.0423, 'paired_source': 0.0515, 'changed_block': 0.0938, 'changed_domain': 0.0794, 'rewrite_relation': 0.0423}
  - id 11787 `By` weight=1.753691 count=110 features={'spatial': 0.9091, 'entity': 0.9636, 'paired_source': 0.0091, 'changed_block': 0.0091, 'changed_domain': 0.0091}
  - id 15441 `ĠDOWN` weight=1.746314 count=57 features={'spatial': 0.8772, 'entity': 1.0}
  - id 1029 `ĠOn` weight=1.739778 count=3221 features={'spatial': 0.9509, 'numeric': 0.0003, 'entity': 0.9997, 'rewrite': 0.0152, 'paired_source': 0.027, 'changed_block': 0.0422, 'changed_domain': 0.0335, 'rewrite_relation': 0.0112}
  - id 13759 `ĠOVER` weight=1.72944 count=70 features={'spatial': 0.8571, 'entity': 1.0}
  - id 9092 `On` weight=1.726663 count=88 features={'spatial': 0.8523, 'entity': 0.9773, 'rewrite': 0.0227, 'paired_source': 0.0227, 'changed_block': 0.0455, 'changed_domain': 0.0455, 'rewrite_relation': 0.0227}
  - id 7884 `To` weight=1.7074 count=209 features={'spatial': 0.8517, 'numeric': 0.0191, 'entity': 0.9569}
  - id 1507 `ĠTo` weight=1.706283 count=2483 features={'spatial': 0.8953, 'entity': 1.0, 'rewrite': 0.0141, 'paired_source': 0.035, 'changed_block': 0.0491, 'changed_domain': 0.0419, 'rewrite_relation': 0.0117}
  - id 10677 `At` weight=1.706079 count=132 features={'spatial': 0.8409, 'numeric': 0.0076, 'entity': 0.9697, 'paired_source': 0.0076, 'changed_block': 0.0076}
  - id 565 `ĠIn` weight=1.705801 count=10993 features={'spatial': 0.9209, 'numeric': 0.0004, 'entity': 0.9996, 'rewrite': 0.0287, 'paired_source': 0.068, 'changed_block': 0.0968, 'changed_domain': 0.0758, 'rewrite_relation': 0.0239}
  - id 2274 `ĠBy` weight=1.705633 count=1328 features={'spatial': 0.8735, 'entity': 1.0, 'rewrite': 0.0286, 'paired_source': 0.0565, 'changed_block': 0.0851, 'changed_domain': 0.0693, 'rewrite_relation': 0.0271}
  - id 1084 `ĠAt` weight=1.701658 count=3179 features={'spatial': 0.8946, 'numeric': 0.0013, 'entity': 0.9987, 'rewrite': 0.0123, 'paired_source': 0.0337, 'changed_block': 0.0459, 'changed_domain': 0.0384, 'rewrite_relation': 0.0104}
  - id 3940 `ĠON` weight=1.693445 count=365 features={'spatial': 0.8411, 'entity': 1.0, 'paired_source': 0.0027, 'changed_block': 0.0027, 'changed_domain': 0.0027}
  - id 8127 `ĠUP` weight=1.68622 count=176 features={'spatial': 0.8125, 'numeric': 0.0057, 'entity': 0.9943, 'rewrite': 0.0057, 'paired_source': 0.0057, 'changed_block': 0.0114, 'changed_domain': 0.0114}

Full JSON: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
