# signed transition readout ready signed transition signature analyzer

CPU/file-only analysis of already written selected-grid prediction payloads. It compares signed item transitions, not static endpoint overlap.

## Trajectory windows

| label | status | best metric | prev→best | best→next | n classification items | selected cheap7(prev,best,next) |
|---|---|---|---|---|---:|---|
| scale1p75_seed43022_reference | ok | cheap7 | chck_82M→chck_84M | chck_84M→chck_86M | 170722 | 43.958571, 44.123571, 43.770714 |
| scale1p25_seed43022_dense | ok | cheap7 | chck_84M→chck_86M | chck_86M→chck_88M | 170722 | 43.500714, 43.537857, 43.267143 |

## Column signed transitions around each trajectory's own peak

### scale1p75_seed43022_reference

| column | n | rise net pct | rise gains/losses | fall net pct | fall gains/losses |
|---|---:|---:|---|---:|---|
| BLiMP | 59875 | -0.240112 | 1015/1154 | 0.222349 | 986/859 |
| Supplement | 5218 | 0.545878 | 51/61 | -0.808011 | 64/47 |
| EWoK | 7618 | 0.018014 | 237/240 | 0.066870 | 234/206 |
| Entity | 6780 | 0.261068 | 134/123 | 0.007431 | 119/109 |
| COMPS | 91028 | 0.014408 | 3001/2919 | 0.086869 | 2820/2822 |
| GlobalPIQA | 203 | 0.543689 | 5/4 | -1.985437 | 2/6 |

### scale1p25_seed43022_dense

| column | n | rise net pct | rise gains/losses | fall net pct | fall gains/losses |
|---|---:|---:|---|---:|---|
| BLiMP | 59875 | 0.391830 | 1164/929 | 0.133812 | 751/669 |
| Supplement | 5218 | -0.060258 | 72/47 | -0.232664 | 36/44 |
| EWoK | 7618 | -0.846942 | 203/244 | -0.368463 | 140/150 |
| Entity | 6780 | 0.472413 | 178/142 | -0.115231 | 97/101 |
| COMPS | 91028 | 0.274451 | 3224/3093 | 0.131980 | 2297/2136 |
| GlobalPIQA | 203 | 0.000000 | 3/3 | -1.456311 | 2/5 |

## Pairwise signed-transition comparison to reference

### scale1p25_seed43022_dense

Item-level transition similarity over all classification items:

| window | ref window | other window | ref net pct | other net pct | ref churn pct | other churn pct | sign agreement on ref-changed | opposite on ref-changed | cosine | gain J | loss J |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rise | chck_82M->chck_84M | chck_84M->chck_86M | -0.033973 | 0.226099 | 5.238926 | 5.448624 | 0.035331 | 0.041369 | -0.005920 | 0.017865 | 0.017375 |
| fall | chck_84M->chck_86M | chck_86M->chck_88M | 0.103092 | 0.127693 | 4.846476 | 3.765186 | 0.026227 | 0.029973 | -0.004251 | 0.014653 | 0.015328 |

Official-like column and subtask-level signed net-vector similarity:

| level | window | groups | Pearson | weighted Pearson | cosine | sign agreement on ref movers | ref mover groups |
|---|---|---:|---:|---:|---:|---:|---:|
| column_official_subtask_mean | rise | 6 | -0.077226 | -0.248842 | -0.016266 | 0.333333 | 3 |
| column_official_subtask_mean | fall | 6 | 0.915932 | 0.773866 | 0.933815 | 1.000000 | 2 |
| column_official_subtask_mean | rise+fall_pattern | 6 | 0.698183 | 0.266155 | 0.706816 | 0.333333 | 3 |
| column_subtask | rise | 107 | -0.367627 | -0.376681 | -0.369552 | 0.448276 | 87 |
| column_subtask | fall | 107 | 0.190754 | 0.147471 | 0.189981 | 0.505747 | 87 |
| column_subtask | rise+fall_pattern | 107 | -0.202960 | -0.241053 | -0.202040 | 0.359223 | 103 |

Top reference moving subtask families (showing whether the other trajectory moves the same families in the same directions):

| column | subtask | n | ref rise | other rise | ref fall | other fall | pattern same? |
|---|---|---:|---:|---:|---:|---:|---|
| BLiMP | existential_there_quantifiers_2 | 911 | 7.574094 | -7.793633 | -5.268935 | 1.866081 | False |
| BLiMP | npi_present_2 | 914 | -5.689278 | 2.188184 | 1.094092 | 0.656455 | False |
| BLiMP | only_npi_licensor_present | 882 | -5.102041 | -0.226757 | 3.174603 | 2.721088 | True |
| BLiMP | npi_present_1 | 909 | -4.180418 | 2.200220 | 0.330033 | 0.440044 | False |
| GlobalPIQA | global_piqa_nonparallel | 100 | 4.000000 | 0.000000 | -3.000000 | 0.000000 | False |
| BLiMP | left_branch_island_echo_question | 947 | 3.801478 | -1.689546 | -1.900739 | -1.161563 | False |
| BLiMP | left_branch_island_simple_question | 951 | 0.210305 | -0.420610 | -3.049422 | -0.946372 | True |
| Entity | move_contents_3_ops | 406 | 1.477833 | 1.231527 | -2.955665 | -0.985222 | True |
| BLiMP | principle_A_domain_1 | 914 | 0.109409 | 2.625821 | 2.954048 | 0.656455 | True |
| GlobalPIQA | global_piqa_parallel | 103 | -2.912621 | 0.000000 | -0.970874 | -2.912621 | False |
| BLiMP | irregular_plural_subject_verb_agreement_2 | 892 | -2.802691 | 1.121076 | 0.896861 | -0.896861 | False |
| BLiMP | only_npi_scope | 837 | -2.747909 | 3.345281 | 1.911589 | 1.433692 | False |

## Reading the result

For seed43122, use this signed-transition readout before interpreting static correct-set overlap. A robust residual-capacity late phase should show the same task/subtask families strengthening from pre-peak to peak and eroding from peak to post-peak. The column table is official-like (subtask mean) so it aligns with score-coordinate family movement; item-weighted column files are also saved for churn anatomy. If transition signatures do not recur, treat seed43022 `chck_84M` as an endpoint asset and make stochastic competence stabilization the next target rather than tuning the original peak.

JSON: `experiments/archive/frontier_consolidation/data/signed_transition_signature_smoke_reference_scale125_v2/signed_transition_signature_summary.json`
