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
| BLiMP | 59875 | -0.232150 | 1015/1154 | 0.212109 | 986/859 |
| Supplement | 5218 | -0.191644 | 51/61 | 0.325795 | 64/47 |
| EWoK | 7618 | -0.039380 | 237/240 | 0.367551 | 234/206 |
| Entity | 6780 | 0.162242 | 134/123 | 0.147493 | 119/109 |
| COMPS | 91028 | 0.090082 | 3001/2919 | -0.002197 | 2820/2822 |
| GlobalPIQA | 203 | 0.492611 | 5/4 | -1.970443 | 2/6 |

### scale1p25_seed43022_dense

| column | n | rise net pct | rise gains/losses | fall net pct | fall gains/losses |
|---|---:|---:|---|---:|---|
| BLiMP | 59875 | 0.392484 | 1164/929 | 0.136952 | 751/669 |
| Supplement | 5218 | 0.479111 | 72/47 | -0.153315 | 36/44 |
| EWoK | 7618 | -0.538199 | 203/244 | -0.131268 | 140/150 |
| Entity | 6780 | 0.530973 | 178/142 | -0.058997 | 97/101 |
| COMPS | 91028 | 0.143912 | 3224/3093 | 0.176869 | 2297/2136 |
| GlobalPIQA | 203 | 0.000000 | 3/3 | -1.477833 | 2/5 |

## Pairwise signed-transition comparison to reference

### scale1p25_seed43022_dense

Item-level transition similarity over all classification items:

| window | ref window | other window | ref net pct | other net pct | ref churn pct | other churn pct | sign agreement on ref-changed | opposite on ref-changed | cosine | gain J | loss J |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rise | chck_82M->chck_84M | chck_84M->chck_86M | -0.033973 | 0.226099 | 5.238926 | 5.448624 | 0.035331 | 0.041369 | -0.005920 | 0.017865 | 0.017375 |
| fall | chck_84M->chck_86M | chck_86M->chck_88M | 0.103092 | 0.127693 | 4.846476 | 3.765186 | 0.026227 | 0.029973 | -0.004251 | 0.014653 | 0.015328 |

Subtask-level signed net-vector similarity:

| window | groups | Pearson | weighted Pearson | cosine | sign agreement on ref movers | ref mover groups |
|---|---:|---:|---:|---:|---:|---:|
| rise | 107 | -0.367627 | -0.376681 | -0.369552 | 0.448276 | 87 |
| fall | 107 | 0.190754 | 0.147471 | 0.189981 | 0.505747 | 87 |
| rise+fall_pattern | 107 | -0.202960 | -0.241053 | -0.202040 | 0.359223 | 103 |

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

For seed43122, use this signed-transition readout before interpreting static correct-set overlap. A robust residual-capacity late phase should show the same task/subtask families strengthening from pre-peak to peak and eroding from peak to post-peak. If transition signatures do not recur, treat seed43022 `chck_84M` as an endpoint asset and make stochastic competence stabilization the next target rather than tuning the original peak.

JSON: `experiments/archive/frontier_consolidation/data/signed_transition_signature_smoke_reference_scale125/signed_transition_signature_summary.json`
