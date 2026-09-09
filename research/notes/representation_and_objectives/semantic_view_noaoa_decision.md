# semantic full eval route update semantic-view no-AoA decision

The completed paired no-AoA trajectory resolves the blocked semantic-view question enough to choose the next evaluation action. The treatment is not a final result, but it has a real downstream signal under the packet-local control.

Source JSON: `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json`

## Endpoint comparison

| ckpt | treat equal7 | ctrl equal7 | Δ equal7 | Δ EWoK+Entity+COMPS+GPIQA | SG+AoA needed for 41.8 | AoA needed if SG=COMPACT_EXPERIENCE |
|---|---:|---:|---:|---:|---:|---:|
| chck_10M | 38.4157 | 38.0121 | 0.4036 | -0.290 | 107.290 | 36.981 |
| chck_20M | 39.5857 | 39.9379 | -0.3521 | -1.280 | 99.100 | 28.791 |
| chck_30M | 40.6814 | 40.1950 | 0.4864 | -0.415 | 91.430 | 21.121 |
| chck_40M | 41.5229 | 39.8029 | 1.7200 | 7.355 | 85.540 | 15.231 |
| chck_50M | 41.0071 | 40.5914 | 0.4157 | 0.715 | 89.150 | 18.841 |
| chck_60M | 41.7486 | 41.6121 | 0.1364 | -0.055 | 83.960 | 13.651 |
| chck_70M | 41.2414 | 41.3614 | -0.1200 | -1.095 | 87.510 | 17.201 |
| chck_80M | 41.8429 | 41.5800 | 0.2629 | 3.025 | 83.300 | 12.991 |
| chck_90M | 41.7086 | 41.4971 | 0.2114 | 2.825 | 84.240 | 13.931 |
| chck_100M | 41.6029 | 41.4307 | 0.1721 | 2.675 | 84.980 | 14.671 |

Best treatment equal7: `chck_80M` = 41.8429. This is the selected endpoint for official-style full completion.

Best causal Δ equal7: `chck_40M` = 1.7200; it has a large transient GlobalPIQA contribution and is not the best absolute endpoint.

## Scientific reading

Same-source generated views are not weak: at 80M the treatment beats packet-local by +0.2629 equal7 with +1.27 EWoK, +2.10 Entity, +0.60 COMPS, and -0.945 GlobalPIQA. At 40M the causal signal is much larger (+1.72 equal7) but partly transient and below the best absolute treatment endpoint. Late checkpoints show persistent Entity/EWoK improvement but Supplement and GlobalPIQA tradeoffs.

The 80M treatment still trails the public strict-small leader by about -6.56 EWoK and -10.20 Entity despite better Supplement and Reading. Therefore, even if the selected full evaluation is competitive through SuperGLUE/AoA, the mechanism does not remove the source-breadth problem; it shows that faithful second views can be useful enough to test as a factor in a broader FineWeb design.

Selected full official-style evaluation is in progress for treatment/control at `chck_80M`. No completed full result is established here.

JSON: `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_decision/semantic_view_noaoa_decision_table.json`
