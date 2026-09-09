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
| 43022 | RminusC | +0.4996 | 2000 |
| 43022 | RminusV | +0.1725 | 2000 |
| 43022 | VminusC | +0.3271 | 2000 |
| 43122 | RminusC | +0.6678 | 2000 |
| 43122 | RminusV | +0.3575 | 2000 |
| 43122 | VminusC | +0.3103 | 2000 |

### Held-out rewrite content-conditioning gain

| seed | group | contrast | late delta gain | n |
|---:|---|---|---:|---:|
| 43022 | ALL | CminusR | +0.4922 | 5892 |
| 43022 | ALL | VminusC | +0.6126 | 5892 |
| 43022 | ALL | VminusR | +1.1049 | 5892 |
| 43022 | token_nonoverlap | CminusR | +0.7517 | 2732 |
| 43022 | token_nonoverlap | VminusC | +0.6837 | 2732 |
| 43022 | token_nonoverlap | VminusR | +1.4354 | 2732 |
| 43022 | token_overlap | CminusR | +0.2679 | 3160 |
| 43022 | token_overlap | VminusC | +0.5512 | 3160 |
| 43022 | token_overlap | VminusR | +0.8191 | 3160 |
| 43122 | ALL | CminusR | +0.6357 | 5892 |
| 43122 | ALL | VminusC | +0.7675 | 5892 |
| 43122 | ALL | VminusR | +1.4032 | 5892 |
| 43122 | token_nonoverlap | CminusR | +1.0433 | 2732 |
| 43122 | token_nonoverlap | VminusC | +0.8938 | 2732 |
| 43122 | token_nonoverlap | VminusR | +1.9371 | 2732 |
| 43122 | token_overlap | CminusR | +0.2833 | 3160 |
| 43122 | token_overlap | VminusC | +0.6582 | 3160 |
| 43122 | token_overlap | VminusR | +0.9416 | 3160 |

### Entity gold-vs-stale cue ablations

| seed | group | contrast | d effect no-initial | d effect no-last-update | d effect no-all-updates | n |
|---:|---|---|---:|---:|---:|---:|
| 43022 | ALL | RminusC | +0.0315 | -0.0019 | -0.7838 | 1221 |
| 43022 | ALL | RminusV | +0.6213 | +0.2719 | -0.9315 | 1221 |
| 43022 | ALL | VminusC | -0.5899 | -0.2739 | +0.1477 | 1221 |
| 43022 | ALL | VminusR | -0.6213 | -0.2719 | +0.9315 | 1221 |
| 43022 | rel_ge2 | RminusC | +0.0092 | -0.0015 | -0.7938 | 1205 |
| 43022 | rel_ge2 | RminusV | +0.5857 | +0.2628 | -0.9567 | 1205 |
| 43022 | rel_ge2 | VminusC | -0.5765 | -0.2643 | +0.1629 | 1205 |
| 43022 | rel_ge2 | VminusR | -0.5857 | -0.2628 | +0.9567 | 1205 |
| 43022 | rel_ge3 | RminusC | -0.0508 | +0.1423 | -0.6032 | 716 |
| 43022 | rel_ge3 | RminusV | +0.4325 | +0.4280 | -0.8656 | 716 |
| 43022 | rel_ge3 | VminusC | -0.4832 | -0.2857 | +0.2624 | 716 |
| 43022 | rel_ge3 | VminusR | -0.4325 | -0.4280 | +0.8656 | 716 |
| 43122 | ALL | RminusC | +0.4078 | +0.1357 | -1.1230 | 1221 |
| 43122 | ALL | RminusV | +0.6150 | +0.2922 | -1.6450 | 1221 |
| 43122 | ALL | VminusC | -0.2072 | -0.1565 | +0.5220 | 1221 |
| 43122 | ALL | VminusR | -0.6150 | -0.2922 | +1.6450 | 1221 |
| 43122 | rel_ge2 | RminusC | +0.3908 | +0.1179 | -1.1575 | 1205 |
| 43122 | rel_ge2 | RminusV | +0.5860 | +0.2772 | -1.6857 | 1205 |
| 43122 | rel_ge2 | VminusC | -0.1953 | -0.1593 | +0.5282 | 1205 |
| 43122 | rel_ge2 | VminusR | -0.5860 | -0.2772 | +1.6857 | 1205 |
| 43122 | rel_ge3 | RminusC | +0.3221 | +0.2729 | -1.1934 | 716 |
| 43122 | rel_ge3 | RminusV | +0.4393 | +0.4665 | -1.8356 | 716 |
| 43122 | rel_ge3 | VminusC | -0.1172 | -0.1936 | +0.6422 | 716 |
| 43122 | rel_ge3 | VminusR | -0.4393 | -0.4665 | +1.8356 | 716 |

## Scientific reading

This note is generated mechanically from the probe results interpretation CSV outputs. The important interpretation is not any single NLL level but whether the held-out context gains and the causal Entity ablation effects separate the arms in the same direction as the official relevant-update crossover.
