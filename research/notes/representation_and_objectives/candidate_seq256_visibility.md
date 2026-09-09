# cached fineweb broad source candidate candidate seq256 visibility audit

Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`; seq_len=256.

| candidate | rows | trunc frac | focus source | focus rows | focus trunc frac | focus token p95 | focus token max |
|---|---:|---:|---|---:|---:|---:|---:|
| cached_fineweb_raw3m_treatment | 64381 | 0.2239 | fineweb_edu_random_quality_cached_initial_model_studies | 18750 | 0.1972 | 294.0 | 759.0 |
|  |  | 0.2239 | qwen_pair_packed | 12236 | 0.0087 | 231.0 | 363.0 |
| cached_fineweb_raw3m_control | 64381 | 0.2583 | official_lengthmatched_to_cached_fineweb | 18750 | 0.3153 | 297.0 | 784.0 |
|  |  | 0.2583 | qwen_pair_packed | 12236 | 0.0087 | 231.0 | 363.0 |
| cached_fineweb_single_doc_treatment | 64381 | 0.2325 | fineweb_edu_random_quality_single_doc_cached_initial_model_studies | 11032 | 0.1643 | 287.0 | 759.0 |
|  |  | 0.2325 | qwen_pair_packed | 12236 | 0.0087 | 231.0 | 363.0 |
| cached_fineweb_single_doc_control | 64381 | 0.2583 | official_lengthmatched_to_cached_fineweb | 11032 | 0.3147 | 296.0 | 784.0 |
|  |  | 0.2583 | official_lengthmatched_to_cached_fineweb_single_doc | 11032 | 0.3147 | 296.0 | 784.0 |
|  |  | 0.2583 | qwen_pair_packed | 12236 | 0.0087 | 231.0 | 363.0 |

JSON: `experiments/archive/representation_and_objectives/training/data/candidate_seq256_visibility/candidate_seq256_visibility.json`
