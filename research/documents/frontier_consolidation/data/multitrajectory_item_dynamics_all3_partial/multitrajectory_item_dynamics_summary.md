# multitrajectory item dynamics ready multi-trajectory DeBERTa item dynamics

CPU/file-only analysis of already written selected-grid prediction payloads. It does not run model inference.

## Trajectories

| label | status | usable | best | best cheap7 | 82M | 84M | 100M | final-best Δ | near-best band | local excess cheap7 | peak spread M |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| scale1p75_seed43022_reference | ok | 16 | chck_84M | 44.123626 | 43.959634 | 44.123626 | 43.543183 | -0.580443 | chck_82M,chck_84M | 0.258062 | 18.000000 |
| scale1p75_seed43122_dense | partial | 11 | chck_88M | 43.910285 | 43.584588 | 43.726547 |  | -0.244851 | chck_88M | 0.262868 | 18.000000 |
| scale1p25_seed43022_dense | ok | 16 | chck_86M | 43.538267 | 43.331929 | 43.500911 | 43.348989 | -0.189277 | chck_84M,chck_86M | 0.153455 | 18.000000 |

## Column peak vectors

| label | BLiMP | Supplement | EWoK | Entity | COMPS | GlobalPIQA | Reading |
|---|---:|---:|---:|---:|---:|---:|---:|
| scale1p75_seed43022_reference | 94 | 76 | 86 | 88 | 78 | 84 | 78 |
| scale1p75_seed43122_dense | 88 | 70 | 78 | 80 | 76 | 88 | 88 |
| scale1p25_seed43022_dense | 88 | 96 | 78 | 86 | 88 | 84 | 94 |

## Dynamics around each aggregate best endpoint

### scale1p75_seed43022_reference

Best endpoint: `chck_84M`.  Local excess cheap7: 0.258062; relation/state: 0.051195; cheap5 no GP/Reading: 0.102375. Max |item-recomputed minus selected| cheap7: 0.002456.

| column | n | net prev→best | gains prev→best | losses prev→best | net best→next | gain-best lost at next | net best→final | mean flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | -139 | 1015 | 1154 | 127 | 359 | 225 | 0.479282 |
| Supplement | 5218 | -10 | 51 | 61 | 17 | 20 | 10 | 0.376389 |
| EWoK | 7618 | -3 | 237 | 240 | 28 | 84 | -19 | 0.861775 |
| Entity | 6780 | 11 | 134 | 123 | 10 | 57 | -48 | 0.554130 |
| COMPS | 91028 | 82 | 3001 | 2919 | -2 | 1055 | 115 | 0.938755 |
| GlobalPIQA | 203 | 1 | 5 | 4 | -4 | 1 | -4 | 0.586207 |

### scale1p75_seed43122_dense

Best endpoint: `chck_88M`.  Local excess cheap7: 0.262868; relation/state: 0.152968; cheap5 no GP/Reading: 0.057753. Max |item-recomputed minus selected| cheap7: 0.001937.

| column | n | net prev→best | gains prev→best | losses prev→best | net best→next | gain-best lost at next | net best→final | mean flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 32 | 724 | 692 | -82 | 245 | -82 | 0.442405 |
| Supplement | 5218 | -4 | 50 | 54 | 14 | 12 | 14 | 0.326179 |
| EWoK | 7618 | -12 | 145 | 157 | 3 | 47 | 3 | 0.760305 |
| Entity | 6780 | 14 | 88 | 74 | -5 | 28 | -5 | 0.531711 |
| COMPS | 91028 | 29 | 1858 | 1829 | 5 | 548 | 5 | 0.821165 |
| GlobalPIQA | 203 | 4 | 5 | 1 | -2 | 2 | -2 | 0.339901 |

### scale1p25_seed43022_dense

Best endpoint: `chck_86M`.  Local excess cheap7: 0.153455; relation/state: 0.027291; cheap5 no GP/Reading: 0.068206. Max |item-recomputed minus selected| cheap7: 0.002799.

| column | n | net prev→best | gains prev→best | losses prev→best | net best→next | gain-best lost at next | net best→final | mean flips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BLiMP | 59875 | 235 | 1164 | 929 | 82 | 298 | 0 | 0.523073 |
| Supplement | 5218 | 25 | 72 | 47 | -8 | 22 | -10 | 0.425834 |
| EWoK | 7618 | -41 | 203 | 244 | -10 | 51 | -35 | 0.889997 |
| Entity | 6780 | 36 | 178 | 142 | -4 | 62 | -47 | 0.550442 |
| COMPS | 91028 | 131 | 3224 | 3093 | 161 | 821 | 148 | 1.011733 |
| GlobalPIQA | 203 | 0 | 3 | 3 | -3 | 1 | -2 | 0.497537 |

## Pairwise item-overlap vs reference

| other | kind | ref endpoint | other endpoint | column | net other-ref items | net pct | disagreement | correct-set Jaccard |
|---|---|---|---|---|---:|---:|---:|---:|
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | BLiMP | -779 | -1.301044 | 0.189144 | 0.754184 |
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | Supplement | -144 | -2.759678 | 0.148716 | 0.814176 |
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | EWoK | 40 | 0.525072 | 0.308742 | 0.529318 |
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | Entity | -122 | -1.799410 | 0.154277 | 0.557530 |
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | COMPS | -328 | -0.360329 | 0.394736 | 0.453788 |
| scale1p75_seed43122_dense | best_vs_best | chck_84M | chck_88M | GlobalPIQA | 6 | 2.955665 | 0.167488 | 0.649485 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | BLiMP | -1161 | -1.939040 | 0.171942 | 0.773063 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | Supplement | 102 | 1.954772 | 0.128018 | 0.842638 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | EWoK | -59 | -0.774481 | 0.282358 | 0.556220 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | Entity | -24 | -0.353982 | 0.145428 | 0.586236 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | COMPS | 106 | 0.116448 | 0.348288 | 0.503749 |
| scale1p25_seed43022_dense | best_vs_best | chck_84M | chck_86M | GlobalPIQA | 4 | 1.970443 | 0.187192 | 0.612245 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | BLiMP | -867 | -1.448017 | 0.189044 | 0.754063 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | Supplement | -114 | -2.184745 | 0.145650 | 0.818312 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | EWoK | 67 | 0.879496 | 0.305198 | 0.534721 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | Entity | -149 | -2.197640 | 0.151475 | 0.561299 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | COMPS | -302 | -0.331766 | 0.394340 | 0.454294 |
| scale1p75_seed43122_dense | same_endpoint | chck_84M | chck_84M | GlobalPIQA | 3 | 1.477833 | 0.172414 | 0.635417 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | BLiMP | -1396 | -2.331524 | 0.173127 | 0.771084 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | Supplement | 77 | 1.475661 | 0.134343 | 0.835020 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | EWoK | -18 | -0.236282 | 0.273825 | 0.568563 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | Entity | -60 | -0.884956 | 0.146608 | 0.580414 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | COMPS | -25 | -0.027464 | 0.354935 | 0.496148 |
| scale1p25_seed43022_dense | same_endpoint | chck_84M | chck_84M | GlobalPIQA | 4 | 1.970443 | 0.177340 | 0.628866 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | BLiMP | -938 | -1.566597 | 0.187891 | 0.755865 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | Supplement | -157 | -3.008816 | 0.143158 | 0.820949 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | EWoK | 24 | 0.315043 | 0.312155 | 0.527237 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | Entity | -146 | -2.153392 | 0.160177 | 0.544081 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | COMPS | -355 | -0.389990 | 0.392044 | 0.456372 |
| scale1p75_seed43122_dense | same_endpoint | chck_86M | chck_86M | GlobalPIQA | 6 | 2.955665 | 0.177340 | 0.617021 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | BLiMP | -1288 | -2.151148 | 0.168585 | 0.777312 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | Supplement | 85 | 1.628977 | 0.127060 | 0.844037 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | EWoK | -87 | -1.142032 | 0.276844 | 0.564256 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | Entity | -34 | -0.501475 | 0.145428 | 0.587102 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | COMPS | 108 | 0.118645 | 0.348179 | 0.503859 |
| scale1p25_seed43022_dense | same_endpoint | chck_86M | chck_86M | GlobalPIQA | 8 | 3.940887 | 0.187192 | 0.604167 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | BLiMP | -1386 | -2.314823 | 0.169754 | 0.776183 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | Supplement | 82 | 1.571483 | 0.133384 | 0.836581 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | EWoK | -75 | -0.984510 | 0.274744 | 0.563139 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | Entity | -23 | -0.339233 | 0.139676 | 0.591105 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | COMPS | 139 | 0.152700 | 0.345729 | 0.507512 |
| scale1p25_seed43022_dense | same_endpoint | chck_100M | chck_100M | GlobalPIQA | 6 | 2.955665 | 0.167488 | 0.634409 |
| scale1p75_seed43122_dense | same_endpoint | chck_90M | chck_90M | GlobalPIQA | 13 | 6.403941 | 0.201970 | 0.568421 |
| scale1p75_seed43122_dense | same_endpoint | chck_88M | chck_88M | GlobalPIQA | 11 | 5.418719 | 0.182266 | 0.614583 |
| scale1p25_seed43022_dense | same_endpoint | chck_90M | chck_90M | GlobalPIQA | 10 | 4.926108 | 0.177340 | 0.604396 |
| scale1p75_seed43122_dense | same_endpoint | chck_70M | chck_70M | GlobalPIQA | 8 | 3.940887 | 0.206897 | 0.557895 |
| scale1p25_seed43022_dense | same_endpoint | chck_92M | chck_92M | GlobalPIQA | 8 | 3.940887 | 0.187192 | 0.591398 |
| scale1p75_seed43122_dense | same_endpoint | chck_76M | chck_76M | Supplement | -168 | -3.219624 | 0.148716 | 0.815502 |
| scale1p75_seed43122_dense | same_endpoint | chck_76M | chck_76M | GlobalPIQA | 6 | 2.955665 | 0.187192 | 0.608247 |
| scale1p25_seed43022_dense | same_endpoint | chck_70M | chck_70M | GlobalPIQA | 6 | 2.955665 | 0.128079 | 0.697674 |

## Reading the result

Endpoint arithmetic should still use the selected_trajectory/payload task scores.  This tool recomputes item-level official-like accuracies from prediction files to study churn and overlap, so small rounding/tie differences are reported explicitly in each trajectory's `trajectory_score_alignment` block.

This analysis is designed to be rerun on seed43122 after its selected-grid scoring delivers.  Score-level reproduction is not enough: a robust late-phase result should show related column peak structure and item/family dynamics, not only a single volatile endpoint.

JSON: `experiments/archive/frontier_consolidation/data/multitrajectory_item_dynamics_all3_partial/multitrajectory_item_dynamics_summary.json`
