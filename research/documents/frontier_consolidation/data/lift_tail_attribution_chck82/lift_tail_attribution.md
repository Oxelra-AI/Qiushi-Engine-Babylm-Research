# source wide skeleton recurrence integrated lift tail-attribution

Existing reciprocal multiview mechanism and scaffold chck82 DeBERTa lift records joined to source-prefix/tail positions; no new scoring.

Records: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_records.jsonl` SHA `900487f93aa03222e18fb1e8ede512070fdd41001bee4c078b6eb79a44b31f8f`

## all_by_pair_target_zone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| pair_type=compact, target_segment=other, source_copy_zone_by_text=both_prefix_and_tail | 28 | 2.245115 | 1.147403 | 0.928571 | 0.943780 | 3.188896 |
| pair_type=compact, target_segment=other, source_copy_zone_by_text=not_in_source | 261 | 5.026742 | 5.816849 | 0.804598 | 2.335285 | 7.362027 |
| pair_type=compact, target_segment=other, source_copy_zone_by_text=prefix_only | 128 | 3.937742 | 2.849816 | 0.882812 | 1.859885 | 5.797628 |
| pair_type=compact, target_segment=other, source_copy_zone_by_text=tail_only | 95 | 5.091816 | 5.133307 | 0.957895 | 0.857504 | 5.949320 |
| pair_type=compact, target_segment=source, source_copy_zone_by_text=both_prefix_and_tail | 53 | 0.695596 | 0.056691 | 0.679245 | 1.701897 | 2.397493 |
| pair_type=compact, target_segment=source, source_copy_zone_by_text=not_in_source | 151 | 5.680148 | 6.677096 | 0.874172 | 1.613176 | 7.293325 |
| pair_type=compact, target_segment=source, source_copy_zone_by_text=prefix_only | 201 | 2.198755 | 0.425414 | 0.711443 | 2.205743 | 4.404498 |
| pair_type=compact, target_segment=source, source_copy_zone_by_text=tail_only | 107 | 2.859693 | 2.150364 | 0.738318 | 1.917801 | 4.777493 |
| pair_type=repeat, target_segment=other, source_copy_zone_by_text=both_prefix_and_tail | 43 | 2.330656 | 1.318220 | 0.976744 | 0.462640 | 2.793297 |
| pair_type=repeat, target_segment=other, source_copy_zone_by_text=not_in_source | 157 | 7.568065 | 7.722771 | 0.980892 | 0.111702 | 7.679767 |
| pair_type=repeat, target_segment=other, source_copy_zone_by_text=prefix_only | 311 | 4.265198 | 3.784965 | 0.983923 | 0.409225 | 4.674423 |
| pair_type=repeat, target_segment=other, source_copy_zone_by_text=unknown | 1 | 6.606728 | 6.606728 | 1.000000 | 0.030002 | 6.636730 |
| pair_type=repeat, target_segment=source, source_copy_zone_by_text=both_prefix_and_tail | 42 | 0.500489 | 0.179006 | 0.857143 | 0.718305 | 1.218794 |
| pair_type=repeat, target_segment=source, source_copy_zone_by_text=not_in_source | 159 | 4.278445 | 5.086099 | 0.748428 | 3.022308 | 7.300753 |
| pair_type=repeat, target_segment=source, source_copy_zone_by_text=prefix_only | 203 | 3.829020 | 3.121226 | 0.990148 | 0.487834 | 4.316854 |
| pair_type=repeat, target_segment=source, source_copy_zone_by_text=tail_only | 107 | -0.098716 | -0.008073 | 0.457944 | 4.236664 | 4.137948 |
| pair_type=repeat, target_segment=source, source_copy_zone_by_text=unknown | 1 | 0.031429 | 0.031429 | 1.000000 | 16.335968 | 16.367397 |

## compact_other_content_by_zone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_copy_zone_by_text=both_prefix_and_tail | 7 | 6.227639 | 5.991652 | 1.000000 | 1.519767 | 7.747406 |
| source_copy_zone_by_text=not_in_source | 109 | 4.106378 | 4.389094 | 0.788991 | 3.030893 | 7.137271 |
| source_copy_zone_by_text=prefix_only | 66 | 6.344465 | 6.629442 | 0.954545 | 1.851119 | 8.195584 |
| source_copy_zone_by_text=tail_only | 69 | 6.285496 | 6.233583 | 0.971014 | 0.797701 | 7.083197 |

## compact_source_content_by_zone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_copy_zone_by_text=both_prefix_and_tail | 7 | 4.024389 | 3.485524 | 0.857143 | 1.300634 | 5.325023 |
| source_copy_zone_by_text=not_in_source | 55 | 5.999291 | 6.949876 | 0.890909 | 1.436610 | 7.435901 |
| source_copy_zone_by_text=prefix_only | 98 | 4.223937 | 3.764091 | 0.826531 | 2.442284 | 6.666221 |
| source_copy_zone_by_text=tail_only | 59 | 4.520138 | 4.523306 | 0.813559 | 2.263319 | 6.783457 |

## compact_other_by_content_zone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| token_is_content_norm=False, source_copy_zone_by_text=both_prefix_and_tail | 21 | 0.917608 | 0.570898 | 0.904762 | 0.751785 | 1.669393 |
| token_is_content_norm=False, source_copy_zone_by_text=not_in_source | 152 | 5.686740 | 6.483471 | 0.815789 | 1.836461 | 7.523201 |
| token_is_content_norm=False, source_copy_zone_by_text=prefix_only | 62 | 1.375747 | 1.033604 | 0.806452 | 1.869217 | 3.244965 |
| token_is_content_norm=False, source_copy_zone_by_text=tail_only | 26 | 1.923974 | 1.511225 | 0.923077 | 1.016212 | 2.940185 |
| token_is_content_norm=True, source_copy_zone_by_text=both_prefix_and_tail | 7 | 6.227639 | 5.991652 | 1.000000 | 1.519767 | 7.747406 |
| token_is_content_norm=True, source_copy_zone_by_text=not_in_source | 109 | 4.106378 | 4.389094 | 0.788991 | 3.030893 | 7.137271 |
| token_is_content_norm=True, source_copy_zone_by_text=prefix_only | 66 | 6.344465 | 6.629442 | 0.954545 | 1.851119 | 8.195584 |
| token_is_content_norm=True, source_copy_zone_by_text=tail_only | 69 | 6.285496 | 6.233583 | 0.971014 | 0.797701 | 7.083197 |

## Scientific reading
If compact rewrite-side lift for content tokens is substantial in `tail_only` or `both_prefix_and_tail` strata, then the copy-lift is not simply the first-N repeat prefix; compact views reintroduce source-wide content keys that the matched repeat prefix would not expose. If lift remains concentrated in `prefix_only`, the skeleton view is less distinct from exact recurrence. Counts are small because reciprocal multiview mechanism and scaffold sampled 128 pairs and at most a few targets per side, so this is attribution of existing probe evidence rather than final proof.

Annotated records CSV: `experiments/archive/frontier_consolidation/data/lift_tail_attribution_chck82/lift_tail_annotated_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/lift_tail_attribution_chck82/lift_tail_attribution.json`
