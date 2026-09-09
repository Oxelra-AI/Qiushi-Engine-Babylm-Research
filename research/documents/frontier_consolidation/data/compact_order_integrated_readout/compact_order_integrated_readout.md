# compact order evaluation and channel readout plan compact ordered-vs-scrambled integrated readout

Combined readout for the matched compact ordered-vs-scrambled 40M experiment. Official-family deltas alone are not sufficient. Because the channel probe evaluates ordered compact text, the source_absent_content fixed-event row must show an ordered advantage materially larger than retained_content and function_other before it identifies the source-absent compact channel rather than native-order familiarity.

## Official-compatible selected cheap scores
### chck_20M
| metric | ordered-scrambled |
|---|---:|
| cheap7 | 0.172143 |
| cheap6_no_GlobalPIQA | 0.195833 |
| cheap5_no_GlobalPIQA_Reading | 0.262 |
| EWoK_Entity | 1.675 |
| BLiMP | -0.06 |
| Supplement | -1.43 |
| EWoK | 2.3 |
| Entity | 1.05 |
| COMPS | -0.55 |
| GlobalPIQA | 0.03 |
| Reading | -0.135 |

### chck_40M
| metric | ordered-scrambled |
|---|---:|
| cheap7 | -0.67 |
| cheap6_no_GlobalPIQA | -0.288333 |
| cheap5_no_GlobalPIQA_Reading | -0.298 |
| EWoK_Entity | -0.02 |
| BLiMP | -0.67 |
| Supplement | -1.04 |
| EWoK | -0.53 |
| Entity | 0.49 |
| COMPS | 0.26 |
| GlobalPIQA | -2.96 |
| Reading | -0.24 |

## Source-absent channel probe
Negative ordered_minus_scrambled means ordered training has lower NLL on the fixed ordered compact event. Because the fixed probe text is ordered, the source_absent_content row is mechanism-specific only when it is more ordered-favorable than retained_content and function_other.
### chck_20M
| category | piece Δ | 95% CI | P(ordered better) |
|---|---:|---:|---:|
| retained_content | -0.080398 | [-0.097026, -0.064006] | 1.0000 |
| source_absent_content | -0.053718 | [-0.072235, -0.0361] | 1.0000 |
| function_other | -0.079314 | [-0.1002, -0.059057] | 1.0000 |

Category interaction: negative means source_absent_content has a larger ordered advantage than the comparison category.
| contrast | Δ difference | 95% interval | P(diff<0) |
|---|---:|---:|---:|
| source_absent_minus_retained_content | 0.026679 | [0.001362, 0.05132] | 0.016 |
| source_absent_minus_function_other | 0.025595 | [-0.002694, 0.049602] | 0.047 |
| source_absent_minus_controls_mean | 0.026137 | [0.002996, 0.047116] | 0.015 |

### chck_40M
| category | piece Δ | 95% CI | P(ordered better) |
|---|---:|---:|---:|
| retained_content | -0.010249 | [-0.057007, 0.035521] | 0.6640 |
| source_absent_content | -0.25524 | [-0.303417, -0.209492] | 1.0000 |
| function_other | -0.024953 | [-0.060397, 0.013351] | 0.9050 |

Category interaction: negative means source_absent_content has a larger ordered advantage than the comparison category.
| contrast | Δ difference | 95% interval | P(diff<0) |
|---|---:|---:|---:|
| source_absent_minus_retained_content | -0.244991 | [-0.316237, -0.180352] | 1.0 |
| source_absent_minus_function_other | -0.230287 | [-0.289624, -0.175748] | 1.0 |
| source_absent_minus_controls_mean | -0.237639 | [-0.295849, -0.183309] | 1.0 |

## Readout
- chck_20M: {'official_delta_cheap7': 0.172143, 'official_delta_cheap6_no_GlobalPIQA': 0.195833, 'official_delta_cheap5_no_GlobalPIQA_Reading': 0.262, 'official_delta_EWoK_Entity': 1.675, 'source_absent_ordered_minus_scrambled_nll': -0.053718, 'source_absent_95pct_interval': [-0.072235, -0.0361], 'source_absent_ordered_better': True, 'source_absent_minus_controls_mean_nll': 0.026137, 'source_absent_minus_controls_mean_interval': [0.002996, 0.047116], 'p_source_absent_more_ordered_favorable_than_controls': 0.015, 'source_absent_category_selective': False, 'channel_interpretation': 'native_order_or_general_order_effect_not_source_absent_specific', 'reading_for_maturation': 'not-yet-justified'}
- chck_40M: {'official_delta_cheap7': -0.67, 'official_delta_cheap6_no_GlobalPIQA': -0.288333, 'official_delta_cheap5_no_GlobalPIQA_Reading': -0.298, 'official_delta_EWoK_Entity': -0.02, 'source_absent_ordered_minus_scrambled_nll': -0.25524, 'source_absent_95pct_interval': [-0.303417, -0.209492], 'source_absent_ordered_better': True, 'source_absent_minus_controls_mean_nll': -0.237639, 'source_absent_minus_controls_mean_interval': [-0.295849, -0.183309], 'p_source_absent_more_ordered_favorable_than_controls': 1.0, 'source_absent_category_selective': True, 'channel_interpretation': 'source_absent_selective', 'reading_for_maturation': 'not-yet-justified'}

## Realized BPE/WWM differences to keep in interpretation
Changed-block ordered-minus-scrambled: Δ active tokens 0.0, Δ candidate groups 8.0, Δ view-active tokens 165.0, CPU-proxy Δ masked tokens 557.0, Δ masked view/source 472.0/183.0.

JSON: `experiments/archive/frontier_consolidation/data/compact_order_integrated_readout/compact_order_integrated_readout.json`
