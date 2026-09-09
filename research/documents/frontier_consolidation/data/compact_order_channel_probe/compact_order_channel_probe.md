# compact order evaluation and channel readout plan compact ordered-vs-scrambled source-absent channel probe

Fixed ordered compact-side denoising readout on the compact order experiment design arms. Negative ordered_minus_scrambled means ordered training gives lower NLL on the same ordered source+compact masked event. Because this probe uses ordered compact text, source_absent_content only identifies the compact source-absent channel when its ordered advantage is materially larger than retained_content and function_other; category_interactions report that comparison.

Events: 12288 from experiments/archive/frontier_consolidation/data/factorial_view_candidate_audit/compact_candidate_pairs.jsonl with selected counts {'retained_content': 4096, 'source_absent_content': 4096, 'function_other': 4096}.

## Realized loader/WWM small differences from lead route assessment after source use probe
compact-minus-scrambled changed block: Δ active tokens 0.0, Δ candidate groups 8.0, Δ view active tokens 165.0, Δ CPU-proxy masked tokens 557.0, Δ CPU-proxy masked view/source 472.0/183.0.

## chck_20M: ordered minus scrambled fixed-event NLL
Positive means ordered arm has *higher/worse* loss than scrambled; negative means ordered arm has lower/better fixed-event loss.
Because this fixed probe uses ordered compact text, the ordered arm is on its native sequence distribution and the scrambled arm is not; source-absent mechanism evidence requires a category interaction, not only a negative source_absent_content row.
| category | piece Δ | 95% CI | P(Δ<0) | events | pairs |
|---|---:|---:|---:|---:|---:|
| retained_content | -0.080398 | [-0.097026, -0.064006] | 1.0000 | 4096 | 3534 |
| source_absent_content | -0.053718 | [-0.072235, -0.0361] | 1.0000 | 4096 | 3459 |
| function_other | -0.079314 | [-0.1002, -0.059057] | 1.0000 | 4096 | 3520 |

Category interaction: negative means the source_absent_content ordered advantage is larger than the comparison category.
| contrast | Δ difference | 95% interval | P(diff<0) |
|---|---:|---:|---:|
| source_absent_minus_retained_content | 0.026679 | [0.001362, 0.05132] | 0.016 |
| source_absent_minus_function_other | 0.025595 | [-0.002694, 0.049602] | 0.047 |
| source_absent_minus_controls_mean | 0.026137 | [0.002996, 0.047116] | 0.015 |

## chck_40M: ordered minus scrambled fixed-event NLL
Positive means ordered arm has *higher/worse* loss than scrambled; negative means ordered arm has lower/better fixed-event loss.
Because this fixed probe uses ordered compact text, the ordered arm is on its native sequence distribution and the scrambled arm is not; source-absent mechanism evidence requires a category interaction, not only a negative source_absent_content row.
| category | piece Δ | 95% CI | P(Δ<0) | events | pairs |
|---|---:|---:|---:|---:|---:|
| retained_content | -0.010249 | [-0.057007, 0.035521] | 0.6640 | 4096 | 3534 |
| source_absent_content | -0.25524 | [-0.303417, -0.209492] | 1.0000 | 4096 | 3459 |
| function_other | -0.024953 | [-0.060397, 0.013351] | 0.9050 | 4096 | 3520 |

Category interaction: negative means the source_absent_content ordered advantage is larger than the comparison category.
| contrast | Δ difference | 95% interval | P(diff<0) |
|---|---:|---:|---:|
| source_absent_minus_retained_content | -0.244991 | [-0.316237, -0.180352] | 1.0 |
| source_absent_minus_function_other | -0.230287 | [-0.289624, -0.175748] | 1.0 |
| source_absent_minus_controls_mean | -0.237639 | [-0.295849, -0.183309] | 1.0 |

JSON: `experiments/archive/frontier_consolidation/data/compact_order_channel_probe/compact_order_channel_probe.json`
