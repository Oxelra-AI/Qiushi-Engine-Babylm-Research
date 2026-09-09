# route reopen conditional innovation — repaired conditional-innovation probe

CPU-only; no model update/evaluation, no corpus/tokenizer change, no H100 work.

Stricter leakage-aware conditional-source probe using unique source-absent targets and no extra mask tokens in decoy source fills.

- changed rows loaded: `768`
- selected targets: `660`; counts: `{'unique_content_innovation': 220, 'copyable': 220, 'unique_relation_or_function_innovation': 220}`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Results
| checkpoint | bucket | n | true loss | source help | same-row decoy advantage | cross-row decoy advantage | masked-same | source help bootstrap 5-95 | same decoy bootstrap 5-95 |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| tokenmean_80M | all | 660 | 3.3152 | 2.0451 | 2.6364 | 1.9766 | -0.5913 | [1.8373,2.2411] | [2.4014,2.8823] |
| tokenmean_80M | copyable | 220 | 1.1651 | 4.3159 | 4.9058 | 4.2950 | -0.5899 | [3.9839,4.6701] | [4.4701,5.2895] |
| tokenmean_80M | unique_content_innovation | 220 | 5.3343 | 1.5248 | 2.2157 | 1.4477 | -0.6909 | [1.2643,1.7802] | [1.9263,2.5268] |
| tokenmean_80M | unique_relation_or_function_innovation | 220 | 3.4461 | 0.2944 | 0.7876 | 0.1869 | -0.4932 | [0.0686,0.4995] | [0.4974,1.0853] |
| tokenmean_100M | all | 660 | 3.2249 | 2.0806 | 2.6978 | 2.0035 | -0.6172 | [1.8703,2.2652] | [2.4559,2.9432] |
| tokenmean_100M | copyable | 220 | 1.0810 | 4.3686 | 4.9833 | 4.3491 | -0.6147 | [4.0331,4.7192] | [4.5329,5.3651] |
| tokenmean_100M | unique_content_innovation | 220 | 5.2162 | 1.5796 | 2.2902 | 1.5064 | -0.7107 | [1.3104,1.8420] | [1.9983,2.6051] |
| tokenmean_100M | unique_relation_or_function_innovation | 220 | 3.3776 | 0.2935 | 0.8197 | 0.1550 | -0.5262 | [0.0623,0.5011] | [0.5162,1.1246] |
| clean_80M | all | 660 | 4.3216 | 1.5608 | 2.0343 | 1.4249 | -0.4735 | [1.3559,1.7606] | [1.8118,2.2659] |
| clean_80M | copyable | 220 | 2.2952 | 3.6593 | 4.0725 | 3.6228 | -0.4131 | [3.2881,4.1039] | [3.6286,4.5331] |
| clean_80M | unique_content_innovation | 220 | 6.7185 | 0.8331 | 1.3696 | 0.6862 | -0.5365 | [0.5825,1.1070] | [1.0674,1.6343] |
| clean_80M | unique_relation_or_function_innovation | 220 | 3.9509 | 0.1899 | 0.6607 | -0.0342 | -0.4707 | [-0.0098,0.3828] | [0.3975,0.9356] |

## Reading
- Unique innovation buckets exclude target norms seen in the paired source or elsewhere in the same row; this reduces exact-form leakage but does not prove semantic novelty.
- Filled decoys remove the previous extra-[MASK] artifact for short decoys, though any source substitution remains a grammatical corruption control rather than a perfect semantic negative.
- If unique_content_innovation keeps positive source_help and same_decoy_advantage with bootstrap intervals away from zero, the conditional-source signal is not only copyability or duplicated function-word entropy.

Full JSON: `experiments/archive/frontier_consolidation/data/conditional_innovation_repair_probe/conditional_innovation_repair_probe.json`
