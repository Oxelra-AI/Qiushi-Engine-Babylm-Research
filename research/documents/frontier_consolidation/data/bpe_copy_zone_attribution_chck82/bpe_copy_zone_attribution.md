# source wide skeleton recurrence integrated BPE copy-zone attribution

Tokenizer-offset re-annotation of existing reciprocal multiview mechanism and scaffold lift records; no model scoring.

Records: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_records.jsonl` SHA `900487f93aa03222e18fb1e8ede512070fdd41001bee4c078b6eb79a44b31f8f`
Tokenizer checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`

## all_by_pair_target_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.674222 | 1.072609 | 0.878788 | 0.726029 | 3.400252 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=not_in_source | 97 | 0.099026 | -0.139214 | 0.443299 | 6.558724 | 6.657751 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=prefix_only | 222 | 5.816492 | 6.374584 | 0.950450 | 0.977083 | 6.793575 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=tail_only | 160 | 6.084254 | 6.257601 | 0.981250 | 0.489972 | 6.574226 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=both_prefix_and_tail | 53 | 0.894466 | 0.042930 | 0.679245 | 1.086437 | 1.980903 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=prefix_only | 311 | 3.281293 | 1.558393 | 0.758842 | 2.173768 | 5.455061 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=tail_only | 148 | 3.882549 | 3.889151 | 0.797297 | 1.680582 | 5.563131 |
| pair_type=repeat, target_segment=other, source_bpe_zone_by_id=both_prefix_and_tail | 46 | 2.678110 | 1.531026 | 0.978261 | 0.211791 | 2.889901 |
| pair_type=repeat, target_segment=other, source_bpe_zone_by_id=prefix_only | 466 | 5.361147 | 5.444611 | 0.982833 | 0.332591 | 5.693738 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=both_prefix_and_tail | 50 | 0.427747 | 0.147509 | 0.800000 | 0.736884 | 1.164632 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=prefix_only | 296 | 5.063384 | 5.121746 | 0.996622 | 0.301948 | 5.365332 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=tail_only | 166 | -0.313832 | -0.095506 | 0.427711 | 5.742074 | 5.428243 |

## compact_other_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.674222 | 1.072609 | 0.878788 | 0.726029 | 3.400252 |
| source_bpe_zone_by_id=not_in_source | 97 | 0.099026 | -0.139214 | 0.443299 | 6.558724 | 6.657751 |
| source_bpe_zone_by_id=prefix_only | 222 | 5.816492 | 6.374584 | 0.950450 | 0.977083 | 6.793575 |
| source_bpe_zone_by_id=tail_only | 160 | 6.084254 | 6.257601 | 0.981250 | 0.489972 | 6.574226 |

## compact_other_content_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 10 | 4.741742 | 4.756427 | 1.000000 | 1.083536 | 5.825278 |
| source_bpe_zone_by_id=not_in_source | 46 | 0.435714 | 0.128334 | 0.500000 | 6.890688 | 7.326402 |
| source_bpe_zone_by_id=prefix_only | 104 | 6.731483 | 7.050069 | 0.980769 | 1.180202 | 7.911685 |
| source_bpe_zone_by_id=tail_only | 91 | 6.330639 | 6.518514 | 0.967033 | 0.743658 | 7.074297 |

## compact_other_step178_copy_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.674222 | 1.072609 | 0.878788 | 0.726029 | 3.400252 |
| source_bpe_zone_by_id=prefix_only | 222 | 5.816492 | 6.374584 | 0.950450 | 0.977083 | 6.793575 |
| source_bpe_zone_by_id=tail_only | 160 | 6.084254 | 6.257601 | 0.981250 | 0.489972 | 6.574226 |

## compact_other_step178_copy_text_by_normzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_norm=both_prefix_and_tail | 43 | 2.708979 | 1.222197 | 0.883721 | 0.741963 | 3.450942 |
| source_bpe_zone_by_norm=prefix_only | 238 | 5.383989 | 5.802707 | 0.928571 | 1.439685 | 6.823674 |
| source_bpe_zone_by_norm=tail_only | 158 | 6.077360 | 6.305114 | 0.962025 | 0.732296 | 6.809656 |

## compact_other_by_content_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| token_is_content_norm=False, source_bpe_zone_by_id=both_prefix_and_tail | 23 | 1.775300 | 0.416429 | 0.826087 | 0.570592 | 2.345892 |
| token_is_content_norm=False, source_bpe_zone_by_id=not_in_source | 51 | -0.204653 | -0.139214 | 0.392157 | 6.259306 | 6.054653 |
| token_is_content_norm=False, source_bpe_zone_by_id=prefix_only | 118 | 5.010058 | 5.238843 | 0.923729 | 0.798064 | 5.808122 |
| token_is_content_norm=False, source_bpe_zone_by_id=tail_only | 69 | 5.759311 | 5.770392 | 1.000000 | 0.155401 | 5.914712 |
| token_is_content_norm=True, source_bpe_zone_by_id=both_prefix_and_tail | 10 | 4.741742 | 4.756427 | 1.000000 | 1.083536 | 5.825278 |
| token_is_content_norm=True, source_bpe_zone_by_id=not_in_source | 46 | 0.435714 | 0.128334 | 0.500000 | 6.890688 | 7.326402 |
| token_is_content_norm=True, source_bpe_zone_by_id=prefix_only | 104 | 6.731483 | 7.050069 | 0.980769 | 1.180202 | 7.911685 |
| token_is_content_norm=True, source_bpe_zone_by_id=tail_only | 91 | 6.330639 | 6.518514 | 0.967033 | 0.743658 | 7.074297 |

## Scientific reading
The key rows are compact rewrite/other targets. A high mean lift in `tail_only` or `both_prefix_and_tail` same-id strata means the compact view makes the model exploit source BPE tokens beyond the matched repeat prefix, strengthening the source-wide skeleton recurrence interpretation. If high lift were confined to `prefix_only`, copied-token lift would be less distinct from exact prefix recurrence. This remains a small attribution of a 128-pair probe, not official competence evidence.

Annotated CSV: `experiments/archive/frontier_consolidation/data/bpe_copy_zone_attribution_chck82/bpe_copy_zone_annotated_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/bpe_copy_zone_attribution_chck82/bpe_copy_zone_attribution.json`
