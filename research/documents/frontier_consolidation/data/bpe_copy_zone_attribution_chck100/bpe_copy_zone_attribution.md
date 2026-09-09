# source wide skeleton recurrence integrated BPE copy-zone attribution

Tokenizer-offset re-annotation of existing reciprocal multiview mechanism and scaffold lift records; no model scoring.

Records: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck100_128/reciprocal_view_lift_records.jsonl` SHA `16a04052e5b28e0c18d03aeba7fc25b5eb614fb226dd8b32e0b093ea5ee143f4`
Tokenizer checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`

## all_by_pair_target_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.689774 | 1.496117 | 0.909091 | 0.645784 | 3.335559 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=not_in_source | 97 | 0.079276 | -0.162886 | 0.463918 | 6.432235 | 6.511512 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=prefix_only | 222 | 5.745843 | 6.046185 | 0.950450 | 0.952212 | 6.698055 |
| pair_type=compact, target_segment=other, source_bpe_zone_by_id=tail_only | 160 | 6.029999 | 6.174821 | 0.968750 | 0.474058 | 6.504058 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=both_prefix_and_tail | 53 | 0.881713 | 0.092938 | 0.679245 | 1.047501 | 1.929214 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=prefix_only | 311 | 3.266512 | 1.418221 | 0.758842 | 2.151061 | 5.417573 |
| pair_type=compact, target_segment=source, source_bpe_zone_by_id=tail_only | 148 | 3.855431 | 3.716107 | 0.804054 | 1.643073 | 5.498504 |
| pair_type=repeat, target_segment=other, source_bpe_zone_by_id=both_prefix_and_tail | 46 | 2.699497 | 1.687970 | 0.978261 | 0.205687 | 2.905184 |
| pair_type=repeat, target_segment=other, source_bpe_zone_by_id=prefix_only | 466 | 5.384248 | 5.430126 | 0.987124 | 0.288614 | 5.672862 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=both_prefix_and_tail | 50 | 0.489558 | 0.125969 | 0.800000 | 0.643873 | 1.133432 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=prefix_only | 296 | 4.990251 | 5.101924 | 0.996622 | 0.259326 | 5.249576 |
| pair_type=repeat, target_segment=source, source_bpe_zone_by_id=tail_only | 166 | -0.312964 | -0.136329 | 0.373494 | 5.656391 | 5.343427 |

## compact_other_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.689774 | 1.496117 | 0.909091 | 0.645784 | 3.335559 |
| source_bpe_zone_by_id=not_in_source | 97 | 0.079276 | -0.162886 | 0.463918 | 6.432235 | 6.511512 |
| source_bpe_zone_by_id=prefix_only | 222 | 5.745843 | 6.046185 | 0.950450 | 0.952212 | 6.698055 |
| source_bpe_zone_by_id=tail_only | 160 | 6.029999 | 6.174821 | 0.968750 | 0.474058 | 6.504058 |

## compact_other_content_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 10 | 4.601765 | 4.324082 | 1.000000 | 0.955594 | 5.557359 |
| source_bpe_zone_by_id=not_in_source | 46 | 0.454763 | 0.524019 | 0.521739 | 6.771061 | 7.225824 |
| source_bpe_zone_by_id=prefix_only | 104 | 6.759740 | 7.311715 | 0.971154 | 1.117941 | 7.877681 |
| source_bpe_zone_by_id=tail_only | 91 | 6.270454 | 6.262302 | 0.967033 | 0.728917 | 6.999371 |

## compact_other_step178_copy_by_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_id=both_prefix_and_tail | 33 | 2.689774 | 1.496117 | 0.909091 | 0.645784 | 3.335559 |
| source_bpe_zone_by_id=prefix_only | 222 | 5.745843 | 6.046185 | 0.950450 | 0.952212 | 6.698055 |
| source_bpe_zone_by_id=tail_only | 160 | 6.029999 | 6.174821 | 0.968750 | 0.474058 | 6.504058 |

## compact_other_step178_copy_text_by_normzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| source_bpe_zone_by_norm=both_prefix_and_tail | 43 | 2.730526 | 1.496117 | 0.906977 | 0.664979 | 3.395505 |
| source_bpe_zone_by_norm=prefix_only | 238 | 5.313924 | 5.445500 | 0.928571 | 1.408467 | 6.722391 |
| source_bpe_zone_by_norm=tail_only | 158 | 6.023932 | 6.187004 | 0.949367 | 0.708021 | 6.731953 |

## compact_other_by_content_idzone
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| token_is_content_norm=False, source_bpe_zone_by_id=both_prefix_and_tail | 23 | 1.858474 | 0.771250 | 0.869565 | 0.511085 | 2.369559 |
| token_is_content_norm=False, source_bpe_zone_by_id=not_in_source | 51 | -0.259398 | -0.331899 | 0.411765 | 6.126628 | 5.867230 |
| token_is_content_norm=False, source_bpe_zone_by_id=prefix_only | 118 | 4.852239 | 5.230643 | 0.932203 | 0.806146 | 5.658385 |
| token_is_content_norm=False, source_bpe_zone_by_id=tail_only | 69 | 5.712878 | 6.010837 | 0.971014 | 0.137940 | 5.850818 |
| token_is_content_norm=True, source_bpe_zone_by_id=both_prefix_and_tail | 10 | 4.601765 | 4.324082 | 1.000000 | 0.955594 | 5.557359 |
| token_is_content_norm=True, source_bpe_zone_by_id=not_in_source | 46 | 0.454763 | 0.524019 | 0.521739 | 6.771061 | 7.225824 |
| token_is_content_norm=True, source_bpe_zone_by_id=prefix_only | 104 | 6.759740 | 7.311715 | 0.971154 | 1.117941 | 7.877681 |
| token_is_content_norm=True, source_bpe_zone_by_id=tail_only | 91 | 6.270454 | 6.262302 | 0.967033 | 0.728917 | 6.999371 |

## Scientific reading
The key rows are compact rewrite/other targets. A high mean lift in `tail_only` or `both_prefix_and_tail` same-id strata means the compact view makes the model exploit source BPE tokens beyond the matched repeat prefix, strengthening the source-wide skeleton recurrence interpretation. If high lift were confined to `prefix_only`, copied-token lift would be less distinct from exact prefix recurrence. This remains a small attribution of a 128-pair probe, not official competence evidence.

Annotated CSV: `experiments/archive/frontier_consolidation/data/bpe_copy_zone_attribution_chck100/bpe_copy_zone_annotated_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/bpe_copy_zone_attribution_chck100/bpe_copy_zone_attribution.json`
