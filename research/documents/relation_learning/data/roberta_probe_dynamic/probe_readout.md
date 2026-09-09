# probe results interpretation held-out copy, rewrite-conditioning, and Entity cue-ablation probes

## Record counts

- Held-out natural copy records: 4000 from `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl`.
- Held-out rewrite-conditioning records: 11784 from unselected accepted pairs in `experiments/archive/frontier_consolidation/data/expansion_analysis/combined_all_accepted_pairs.jsonl` after excluding `experiments/archive/frontier_consolidation/data/dose_distribution_select/selected_matched_max_pairs.jsonl`.
- Entity cue-ablation records: 9768 on stale-non-gold relevant-update Entity items.

## Key late contrasts

For copy and rewrite probes, positive gain contrast means the first arm benefits more from the relevant context cue. For Entity cue ablations, a positive `effect_no_initial` means removing the queried-box initial clause raises the gold-over-stale margin; a negative `effect_no_last/all_relevant` means removing update evidence lowers the gold-over-stale margin.

### Held-out natural copy gain

| seed | contrast | late delta gain | n |
|---:|---|---:|---:|
| 43022 | RminusC | -0.0487 | 2000 |
| 43022 | RminusV | -0.0569 | 2000 |
| 43022 | VminusC | +0.0082 | 2000 |

### Held-out rewrite content-conditioning gain

| seed | group | contrast | late delta gain | n |
|---:|---|---|---:|---:|
| 43022 | ALL | CminusR | +0.3214 | 5892 |
| 43022 | ALL | VminusC | +0.1058 | 5892 |
| 43022 | ALL | VminusR | +0.4272 | 5892 |
| 43022 | token_nonoverlap | CminusR | +0.3983 | 2732 |
| 43022 | token_nonoverlap | VminusC | +0.0666 | 2732 |
| 43022 | token_nonoverlap | VminusR | +0.4649 | 2732 |
| 43022 | token_overlap | CminusR | +0.2549 | 3160 |
| 43022 | token_overlap | VminusC | +0.1397 | 3160 |
| 43022 | token_overlap | VminusR | +0.3946 | 3160 |

### Entity gold-vs-stale cue ablations

| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |
|---:|---|---|---:|---:|---:|---:|
| 43022 | ALL | RminusC | +0.0476 | -0.1155 | -0.0852 | 1221 |
| 43022 | ALL | RminusV | +0.0234 | -0.0335 | +0.0010 | 1221 |
| 43022 | ALL | VminusC | +0.0242 | -0.0820 | -0.0862 | 1221 |
| 43022 | ALL | VminusR | -0.0234 | +0.0335 | -0.0010 | 1221 |
| 43022 | rel_ge2 | RminusC | +0.0489 | -0.1158 | -0.0851 | 1205 |
| 43022 | rel_ge2 | RminusV | +0.0228 | -0.0341 | +0.0008 | 1205 |
| 43022 | rel_ge2 | VminusC | +0.0261 | -0.0817 | -0.0859 | 1205 |
| 43022 | rel_ge2 | VminusR | -0.0228 | +0.0341 | -0.0008 | 1205 |
| 43022 | rel_ge3 | RminusC | +0.0703 | -0.1112 | -0.1140 | 716 |
| 43022 | rel_ge3 | RminusV | +0.0202 | -0.0456 | -0.0183 | 716 |
| 43022 | rel_ge3 | VminusC | +0.0501 | -0.0656 | -0.0956 | 716 |
| 43022 | rel_ge3 | VminusR | -0.0202 | +0.0456 | +0.0183 | 716 |

## Scientific reading

This note is generated mechanically from the probe results interpretation CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.
