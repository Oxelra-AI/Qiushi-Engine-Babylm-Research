# cached fineweb quality audit seqsafe96 FineWeb candidate seq256 visibility

Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`; seq_len=256.

| arm | whole trunc frac | whole visible word frac | focus | rows | words | trunc frac | visible word frac | hidden words | token p95 | token max |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| treatment | 0.1754 | 0.9818 | fineweb_edu_random_quality_single_doc_seqsafe_cached_initial_model_studies | 21916 | 1753280 | 0.0015 | 0.9998 | 332 | 163.0 | 370.0 |
|  | 0.1754 | 0.9818 | qwen_pair_packed | 12236 | 1656800 | 0.0087 | 0.9995 | 897 | 231.0 | 363.0 |
|  | 0.1754 | 0.9818 | official_identical_tail_after_seqsafe_fineweb_block | 41187 | 6589920 | 0.3175 | 0.9725 | 181217 | 296.0 | 670.0 |
| control | 0.1754 | 0.9817 | official_lengthmatched_to_seqsafe_fineweb | 21916 | 1753280 | 0.0014 | 0.9997 | 501 | 172.0 | 626.0 |
|  | 0.1754 | 0.9817 | qwen_pair_packed | 12236 | 1656800 | 0.0087 | 0.9995 | 897 | 231.0 | 363.0 |
|  | 0.1754 | 0.9817 | official_identical_tail_after_seqsafe_fineweb_block | 41187 | 6589920 | 0.3175 | 0.9725 | 181217 | 296.0 | 670.0 |

Replacement block delta: FineWeb-control visible word fraction +0.000096; hidden words FineWeb 332, control 501.

JSON: `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/seq256_visibility_audit.json`
