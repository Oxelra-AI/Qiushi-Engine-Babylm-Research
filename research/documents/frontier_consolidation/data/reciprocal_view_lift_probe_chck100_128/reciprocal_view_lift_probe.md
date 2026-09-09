# reciprocal multiview mechanism and scaffold reciprocal view-lift probe

Status: `RECIPROCAL_VIEW_LIFT_PROBE`
Model type: `mlm`
Pairs file: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`
Sampled pairs: 128 / available 12155
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`
Model SHA256: `7349475846ef2a193df850cc23876b2a86f99e82b9f051816064e4561b0f6f52`


## Summary groups

### model_type/pair_type/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other | 512 | 4.564120 | 4.679864 | 0.861328 | 1.821246 | 6.385366 |
| model_type=mlm, pair_type=compact, target_segment=source | 512 | 3.189882 | 1.661337 | 0.763672 | 1.889985 | 5.079868 |
| model_type=mlm, pair_type=repeat, target_segment=other | 512 | 5.143040 | 5.108778 | 0.986328 | 0.281163 | 5.424203 |
| model_type=mlm, pair_type=repeat, target_segment=source | 512 | 2.831328 | 1.196098 | 0.775391 | 2.046709 | 4.878037 |

### model_type/pair_type/order/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=other | 512 | 4.564120 | 4.679864 | 0.861328 | 1.821246 | 6.385366 |
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=source | 512 | 3.189882 | 1.661337 | 0.763672 | 1.889985 | 5.079868 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=other | 512 | 5.143040 | 5.108778 | 0.986328 | 0.281163 | 5.424203 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=source | 512 | 2.831328 | 1.196098 | 0.775391 | 2.046709 | 4.878037 |

### model_type/pair_type/target_segment/copy_by_id
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=False | 97 | 0.079276 | -0.162886 | 0.463918 | 6.432235 | 6.511512 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=True | 415 | 5.612385 | 5.877721 | 0.954217 | 0.743497 | 6.355882 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=False | 200 | -0.099830 | -0.060494 | 0.470000 | 4.074148 | 3.974318 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=True | 312 | 5.298673 | 5.804005 | 0.951923 | 0.489881 | 5.788553 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_id=True | 512 | 5.143040 | 5.108778 | 0.986328 | 0.281163 | 5.424203 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=False | 166 | -0.312964 | -0.136329 | 0.373494 | 5.656391 | 5.343427 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=True | 346 | 4.339862 | 3.825353 | 0.968208 | 0.314896 | 4.654758 |

### model_type/pair_type/target_segment/copy_by_text
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=False | 73 | 0.040022 | -0.313785 | 0.424658 | 6.257556 | 6.297578 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=True | 439 | 5.316419 | 5.536277 | 0.933941 | 1.083546 | 6.399964 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=False | 188 | -0.112485 | -0.060494 | 0.473404 | 4.012750 | 3.900265 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=True | 324 | 5.106071 | 5.532516 | 0.932099 | 0.658257 | 5.764328 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_text=True | 512 | 5.143040 | 5.108778 | 0.986328 | 0.281163 | 5.424203 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=False | 162 | -0.310210 | -0.145867 | 0.364198 | 5.639843 | 5.329632 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=True | 350 | 4.285412 | 3.768547 | 0.965714 | 0.383601 | 4.669013 |

## Interpretation note
For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck100_128/reciprocal_view_lift_probe.json`
