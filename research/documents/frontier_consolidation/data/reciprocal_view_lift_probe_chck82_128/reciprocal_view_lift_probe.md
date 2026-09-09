# reciprocal multiview mechanism and scaffold reciprocal view-lift probe

Status: `RECIPROCAL_VIEW_LIFT_PROBE`
Model type: `mlm`
Pairs file: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`
Sampled pairs: 128 / available 12155
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
Model SHA256: `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`


## Summary groups

### model_type/pair_type/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other | 512 | 4.614446 | 4.657592 | 0.859375 | 1.866139 | 6.480586 |
| model_type=mlm, pair_type=compact, target_segment=source | 512 | 3.208019 | 1.659734 | 0.761719 | 1.918651 | 5.126670 |
| model_type=mlm, pair_type=repeat, target_segment=other | 512 | 5.120093 | 5.055516 | 0.982422 | 0.321738 | 5.441831 |
| model_type=mlm, pair_type=repeat, target_segment=source | 512 | 2.867291 | 1.348728 | 0.792969 | 2.108213 | 4.975504 |

### model_type/pair_type/order/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=other | 512 | 4.614446 | 4.657592 | 0.859375 | 1.866139 | 6.480586 |
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=source | 512 | 3.208019 | 1.659734 | 0.761719 | 1.918651 | 5.126670 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=other | 512 | 5.120093 | 5.055516 | 0.982422 | 0.321738 | 5.441831 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=source | 512 | 2.867291 | 1.348728 | 0.792969 | 2.108213 | 4.975504 |

### model_type/pair_type/target_segment/copy_by_id
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=False | 97 | 0.099026 | -0.139214 | 0.443299 | 6.558724 | 6.657751 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=True | 415 | 5.669858 | 6.039464 | 0.956627 | 0.769318 | 6.439176 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=False | 200 | -0.102212 | -0.052173 | 0.465000 | 4.105439 | 4.003226 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=True | 312 | 5.329963 | 5.902292 | 0.951923 | 0.516864 | 5.846827 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_id=True | 512 | 5.120093 | 5.055516 | 0.982422 | 0.321738 | 5.441831 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=False | 166 | -0.313832 | -0.095506 | 0.427711 | 5.742074 | 5.428243 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=True | 346 | 4.393495 | 4.084913 | 0.968208 | 0.364800 | 4.758295 |

### model_type/pair_type/target_segment/copy_by_text
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=False | 73 | 0.061620 | -0.314681 | 0.397260 | 6.372756 | 6.434375 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=True | 439 | 5.371522 | 5.721597 | 0.936219 | 1.116748 | 6.488270 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=False | 188 | -0.106480 | -0.061860 | 0.462766 | 4.034307 | 3.927827 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=True | 324 | 5.131248 | 5.630049 | 0.935185 | 0.691048 | 5.822295 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_text=True | 512 | 5.120093 | 5.055516 | 0.982422 | 0.321738 | 5.441831 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=False | 162 | -0.312812 | -0.104476 | 0.419753 | 5.728329 | 5.415516 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=True | 350 | 4.339225 | 3.996528 | 0.965714 | 0.432617 | 4.771842 |

## Interpretation note
For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck82_128/reciprocal_view_lift_probe.json`
