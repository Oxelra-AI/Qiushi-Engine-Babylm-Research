# reciprocal multiview mechanism and scaffold reciprocal view-lift probe

Status: `RECIPROCAL_VIEW_LIFT_PROBE`
Model type: `causal`
Pairs file: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`
Sampled pairs: 32 / available 12155
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/causal_gpt_repeat_seed43022_100M/hf_model/chck_82M`
Model SHA256: `bc04b5336dc676cd9e89f9c22c7fff3899abcbfcebd045998c98524985588e37`


## Summary groups

### model_type/pair_type/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other | 256 | 0.719289 | 0.000000 | 0.429688 | 5.254898 | 5.974187 |
| model_type=causal, pair_type=compact, target_segment=source | 256 | 0.820703 | 0.000000 | 0.414062 | 4.432327 | 5.253029 |
| model_type=causal, pair_type=repeat, target_segment=other | 256 | 1.314101 | 0.000000 | 0.480469 | 4.643985 | 5.958085 |
| model_type=causal, pair_type=repeat, target_segment=source | 256 | 0.857330 | 0.000000 | 0.433594 | 4.116427 | 4.973757 |

### model_type/pair_type/order/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, order=other_to_source, target_segment=other | 128 | 0.000000 | 0.000000 | 0.000000 | 5.879952 | 5.879952 |
| model_type=causal, pair_type=compact, order=other_to_source, target_segment=source | 128 | 1.641405 | 0.760479 | 0.828125 | 3.439245 | 5.080651 |
| model_type=causal, pair_type=compact, order=source_to_other, target_segment=other | 128 | 1.438578 | 0.679756 | 0.859375 | 4.629844 | 6.068422 |
| model_type=causal, pair_type=compact, order=source_to_other, target_segment=source | 128 | 0.000000 | 0.000000 | 0.000000 | 5.425408 | 5.425408 |
| model_type=causal, pair_type=repeat, order=other_to_source, target_segment=other | 128 | 0.000000 | 0.000000 | 0.000000 | 5.931585 | 5.931585 |
| model_type=causal, pair_type=repeat, order=other_to_source, target_segment=source | 128 | 1.714659 | 0.907950 | 0.867188 | 3.286952 | 5.001611 |
| model_type=causal, pair_type=repeat, order=source_to_other, target_segment=other | 128 | 2.628201 | 1.698307 | 0.960938 | 3.356384 | 5.984586 |
| model_type=causal, pair_type=repeat, order=source_to_other, target_segment=source | 128 | 0.000000 | 0.000000 | 0.000000 | 4.945902 | 4.945902 |

### model_type/pair_type/target_segment/copy_by_id
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other, copy_by_id=False | 48 | 1.044125 | 0.001092 | 0.500000 | 6.666529 | 7.710655 |
| model_type=causal, pair_type=compact, target_segment=other, copy_by_id=True | 208 | 0.644327 | 0.000000 | 0.413462 | 4.929137 | 5.573463 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_id=False | 102 | 0.236563 | 0.000000 | 0.274510 | 4.741773 | 4.978336 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_id=True | 154 | 1.207600 | 0.037829 | 0.506494 | 4.227369 | 5.434969 |
| model_type=causal, pair_type=repeat, target_segment=other, copy_by_id=True | 256 | 1.314101 | 0.000000 | 0.480469 | 4.643985 | 5.958085 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_id=False | 75 | 0.004505 | 0.000000 | 0.306667 | 4.916280 | 4.920785 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_id=True | 181 | 1.210710 | 0.000000 | 0.486188 | 3.784996 | 4.995706 |

### model_type/pair_type/target_segment/copy_by_text
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=causal, pair_type=compact, target_segment=other, copy_by_text=False | 39 | 0.717870 | 0.000000 | 0.487179 | 6.422704 | 7.140574 |
| model_type=causal, pair_type=compact, target_segment=other, copy_by_text=True | 217 | 0.719544 | 0.000000 | 0.419355 | 5.045016 | 5.764559 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_text=False | 95 | 0.166439 | 0.000000 | 0.263158 | 4.622616 | 4.789055 |
| model_type=causal, pair_type=compact, target_segment=source, copy_by_text=True | 161 | 1.206759 | 0.034034 | 0.503106 | 4.320044 | 5.526803 |
| model_type=causal, pair_type=repeat, target_segment=other, copy_by_text=True | 256 | 1.314101 | 0.000000 | 0.480469 | 4.643985 | 5.958085 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_text=False | 73 | 0.001780 | 0.000000 | 0.301370 | 5.023483 | 5.025263 |
| model_type=causal, pair_type=repeat, target_segment=source, copy_by_text=True | 183 | 1.198615 | 0.000000 | 0.486339 | 3.754596 | 4.953210 |

## Interpretation note
For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_causal_repeat_chck82_32/reciprocal_view_lift_probe.json`
