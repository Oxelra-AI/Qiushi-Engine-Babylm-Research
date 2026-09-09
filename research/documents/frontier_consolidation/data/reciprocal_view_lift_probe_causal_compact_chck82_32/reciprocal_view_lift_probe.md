# reciprocal multiview mechanism and scaffold reciprocal view-lift probe

Status: `RECIPROCAL_VIEW_LIFT_PROBE`
Model type: `causal`
Pairs file: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`
Sampled pairs: 32 / available 12155
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/causal_gpt_compact_seed43022_100M/hf_model/chck_82M`
Model SHA256: `6dfcee180952c542d1d77a99ab7c8399f6aa325e371c4f4fcf208ba857172001`


## Summary groups

### model_type/pair_type/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other | 256 | 0.855414 | 0.000000 | 0.460938 | 4.825434 | 5.680848 |
| model_type=causal, pair_type=compact, target_segment=source | 256 | 0.880697 | 0.000000 | 0.414062 | 4.387296 | 5.267993 |
| model_type=causal, pair_type=repeat, target_segment=other | 256 | 1.375382 | 0.000000 | 0.488281 | 4.656589 | 6.031971 |
| model_type=causal, pair_type=repeat, target_segment=source | 256 | 0.893439 | 0.000000 | 0.453125 | 4.144897 | 5.038337 |

### model_type/pair_type/order/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, order=other_to_source, target_segment=other | 128 | 0.000000 | 0.000000 | 0.000000 | 5.570965 | 5.570965 |
| model_type=causal, pair_type=compact, order=other_to_source, target_segment=source | 128 | 1.761393 | 0.712564 | 0.828125 | 3.361516 | 5.122910 |
| model_type=causal, pair_type=compact, order=source_to_other, target_segment=other | 128 | 1.710828 | 0.909517 | 0.921875 | 4.079903 | 5.790731 |
| model_type=causal, pair_type=compact, order=source_to_other, target_segment=source | 128 | 0.000000 | 0.000000 | 0.000000 | 5.413077 | 5.413077 |
| model_type=causal, pair_type=repeat, order=other_to_source, target_segment=other | 128 | 0.000000 | 0.000000 | 0.000000 | 5.991548 | 5.991548 |
| model_type=causal, pair_type=repeat, order=other_to_source, target_segment=source | 128 | 1.786879 | 0.737941 | 0.906250 | 3.290062 | 5.076941 |
| model_type=causal, pair_type=repeat, order=source_to_other, target_segment=other | 128 | 2.750764 | 1.847553 | 0.976562 | 3.321629 | 6.072393 |
| model_type=causal, pair_type=repeat, order=source_to_other, target_segment=source | 128 | 0.000000 | 0.000000 | 0.000000 | 4.999733 | 4.999733 |

### model_type/pair_type/target_segment/copy_by_id
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other, copy_by_id=False | 48 | 1.151428 | 0.109817 | 0.520833 | 6.130796 | 7.282224 |
| model_type=causal, pair_type=compact, target_segment=other, copy_by_id=True | 208 | 0.787103 | 0.000000 | 0.447115 | 4.524196 | 5.311300 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_id=False | 102 | 0.227970 | 0.000000 | 0.323529 | 4.848291 | 5.076261 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_id=True | 154 | 1.313022 | 0.000000 | 0.474026 | 4.081962 | 5.394985 |
| model_type=causal, pair_type=repeat, target_segment=other, copy_by_id=True | 256 | 1.375382 | 0.000000 | 0.488281 | 4.656589 | 6.031971 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_id=False | 75 | 0.062367 | 0.000000 | 0.373333 | 4.718070 | 4.780437 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_id=True | 181 | 1.237806 | 0.000000 | 0.486188 | 3.907395 | 5.145201 |

### model_type/pair_type/target_segment/copy_by_text
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other, copy_by_text=False | 39 | 0.844264 | 0.073027 | 0.512821 | 5.864467 | 6.708731 |
| model_type=causal, pair_type=compact, target_segment=other, copy_by_text=True | 217 | 0.857418 | 0.000000 | 0.451613 | 4.638695 | 5.496113 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_text=False | 95 | 0.168863 | 0.000000 | 0.315789 | 4.722779 | 4.891642 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_text=True | 161 | 1.300723 | 0.000000 | 0.472050 | 4.189341 | 5.490064 |
| model_type=causal, pair_type=repeat, target_segment=other, copy_by_text=True | 256 | 1.375382 | 0.000000 | 0.488281 | 4.656589 | 6.031971 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_text=False | 73 | 0.062100 | 0.000000 | 0.369863 | 4.816095 | 4.878195 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_text=True | 183 | 1.225066 | 0.000000 | 0.486339 | 3.877152 | 5.102218 |

## Interpretation note
For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_causal_compact_chck82_32/reciprocal_view_lift_probe.json`
