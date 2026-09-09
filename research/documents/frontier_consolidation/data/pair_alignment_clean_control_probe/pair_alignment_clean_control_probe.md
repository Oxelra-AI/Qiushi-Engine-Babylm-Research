# successor route comparison while minfreq runs pair alignment clean-control probe

CPU-only comparison on changed-block pair texts. No training, official evaluation, corpus change, or route selection.

Sampled `160` changed examples and `635` both-visible pair records.

## Last-layer centered geometry
- clean_20M: margin=0.7910, actual=0.7708, shuffled=-0.0202, same-row-other=0.2621, top1=0.9858
- clean_80M: margin=0.8247, actual=0.8065, shuffled=-0.0182, same-row-other=0.1273, top1=0.9984
- reinvest_20M: margin=0.7949, actual=0.7799, shuffled=-0.0150, same-row-other=0.2676, top1=0.9827
- reinvest_80M: margin=0.8474, actual=0.8232, shuffled=-0.0242, same-row-other=0.0318, top1=1.0000

## Contrasts
- {'layer': 'embedding', 'reinvest_minus_clean_20M_margin': 0.0007214602082967758, 'reinvest_minus_clean_20M_top1': 0.022047244094488105, 'reinvest_minus_clean_80M_margin': 0.010514823719859123, 'reinvest_minus_clean_80M_top1': 0.015748031496063075}
- {'layer': 'layer_4', 'reinvest_minus_clean_20M_margin': 0.012614112347364426, 'reinvest_minus_clean_20M_top1': -0.0047244094488189115, 'reinvest_minus_clean_80M_margin': 0.018303263932466507, 'reinvest_minus_clean_80M_top1': 0.009448818897637823}
- {'layer': 'last', 'reinvest_minus_clean_20M_margin': 0.0038855234161019325, 'reinvest_minus_clean_20M_top1': -0.0031496062992125706, 'reinvest_minus_clean_80M_margin': 0.0227573961019516, 'reinvest_minus_clean_80M_top1': 0.0015748031496063408}

## Interpretation
- At 80M last layer, reinvest-minus-clean paired-geometry margin delta is +0.0228 and retrieval-top1 delta is +0.0016 on the same held-out changed-block pair texts. This quantifies whether pair co-training adds geometry beyond generic same-tokenizer language learning.
- Because the clean model never trained on the compact FineWeb source/rewrite paired rows, this is a route-design probe for shared-subspace consistency, not an official score measurement and not a contamination or compliance test.

Full JSON: `experiments/archive/frontier_consolidation/data/pair_alignment_clean_control_probe/pair_alignment_clean_control_probe.json`
