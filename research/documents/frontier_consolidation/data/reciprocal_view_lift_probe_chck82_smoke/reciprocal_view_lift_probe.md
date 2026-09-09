# reciprocal multiview mechanism and scaffold reciprocal view-lift probe

Status: `RECIPROCAL_VIEW_LIFT_PROBE`
Model type: `mlm`
Pairs file: `experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`
Sampled pairs: 16 / available 12155
Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
Model SHA256: `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`


## Summary groups

### model_type/pair_type/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other | 64 | 3.948000 | 3.255975 | 0.875000 | 1.350031 | 5.298031 |
| model_type=mlm, pair_type=compact, target_segment=source | 64 | 3.229711 | 1.471169 | 0.812500 | 1.820594 | 5.050305 |
| model_type=mlm, pair_type=repeat, target_segment=other | 64 | 5.630478 | 5.422520 | 1.000000 | 0.262480 | 5.892958 |
| model_type=mlm, pair_type=repeat, target_segment=source | 64 | 2.731503 | 1.184893 | 0.812500 | 2.022227 | 4.753730 |

### model_type/pair_type/order/target_segment
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=other | 64 | 3.948000 | 3.255975 | 0.875000 | 1.350031 | 5.298031 |
| model_type=mlm, pair_type=compact, order=source_to_other, target_segment=source | 64 | 3.229711 | 1.471169 | 0.812500 | 1.820594 | 5.050305 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=other | 64 | 5.630478 | 5.422520 | 1.000000 | 0.262480 | 5.892958 |
| model_type=mlm, pair_type=repeat, order=source_to_other, target_segment=source | 64 | 2.731503 | 1.184893 | 0.812500 | 2.022227 | 4.753730 |

### model_type/pair_type/target_segment/copy_by_id
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=False | 11 | 0.662960 | 0.452330 | 0.545455 | 4.181010 | 4.843970 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_id=True | 53 | 4.629801 | 4.824276 | 0.943396 | 0.762469 | 5.392270 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=False | 23 | 0.099320 | 0.338316 | 0.608696 | 3.940494 | 4.039814 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_id=True | 41 | 4.985784 | 5.337915 | 0.926829 | 0.631382 | 5.617167 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_id=True | 64 | 5.630478 | 5.422520 | 1.000000 | 0.262480 | 5.892958 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=False | 23 | -0.051297 | -0.009782 | 0.478261 | 5.366199 | 5.314902 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_id=True | 41 | 4.292585 | 3.291203 | 1.000000 | 0.146341 | 4.438926 |

### model_type/pair_type/target_segment/copy_by_text
| key | n | mean lift | median lift | positive frac | mean paired NLL | mean side-only NLL |
|---|---:|---:|---:|---:|---:|---:|
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=False | 9 | 0.199770 | -0.189144 | 0.444444 | 3.893351 | 4.093120 |
| model_type=mlm, pair_type=compact, target_segment=other, copy_by_text=True | 55 | 4.561347 | 4.753227 | 0.945455 | 0.933851 | 5.495198 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=False | 19 | 0.289172 | 0.338316 | 0.631579 | 3.653449 | 3.942620 |
| model_type=mlm, pair_type=compact, target_segment=source, copy_by_text=True | 45 | 4.471272 | 3.615795 | 0.888889 | 1.046722 | 5.517995 |
| model_type=mlm, pair_type=repeat, target_segment=other, copy_by_text=True | 64 | 5.630478 | 5.422520 | 1.000000 | 0.262480 | 5.892958 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=False | 21 | -0.065527 | -0.009782 | 0.476190 | 5.834233 | 5.768706 |
| model_type=mlm, pair_type=repeat, target_segment=source, copy_by_text=True | 43 | 4.097494 | 3.243752 | 0.976744 | 0.160550 | 4.258043 |

## Interpretation note
For MLM, positive lift in both source and rewrite target segments is evidence of reciprocal cross-view conditioning. For causal models, lift should appear only for the second segment of a particular order because future tokens are masked by the objective. A full mechanistic conclusion requires enough sampled pairs and comparison to repeat controls; tiny smoke runs only validate the probe.

JSON: `experiments/archive/frontier_consolidation/data/reciprocal_view_lift_probe_chck82_smoke/reciprocal_view_lift_probe.json`
