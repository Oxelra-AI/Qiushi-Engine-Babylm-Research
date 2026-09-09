# preservation admission standard raw AoA refits

CDI: `experiments/archive/initial_model_studies/repos/babylm-eval/strict/evaluation_data/full_eval/aoa/cdi_human.csv`
Tokenizer: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model`

| endpoint | raw r | p | fitted words | stored if official | harmful negative? |
|---|---:|---:|---:|---:|---:|
| dense_seed62064 | -0.021837 | 0.716002 | 280 | 0.000000 | False |
| dense_seed62065 | -0.022479 | 0.707521 | 281 | 0.000000 | False |
| clean_pres_seed62064 | 0.005714 | 0.923625 | 284 | 0.000000 | False |
| clean_pres_seed62065 | -0.017343 | 0.770656 | 285 | 0.000000 | False |
| densemask_sparselabel_seed62064 | -0.022823 | 0.703259 | 281 | 0.000000 | False |

Missing files:
[
  {
    "endpoint": "coherent86",
    "path": "experiments/archive/functional_learning/data/batched_aoa_clean_eval_measured/coherent86/full/AoA_word/surprisal.json"
  }
]

Word-fit and overlap records are saved beside this summary.
