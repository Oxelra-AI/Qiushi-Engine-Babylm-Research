# preservation admission standard raw AoA refits with official semantics

Official AoAEvaluator: bounded child sigmoid to threshold 0.5; mean surprisal per word/checkpoint; subword length by prefix subtraction; model AoA is log10(step+1) at halfway from random surprisal to min surprisal; raw Pearson before p>0.1 clipping.

| endpoint | raw r | p | fitted words | official clipped | harmful negative p<=0.2 |
|---|---:|---:|---:|---:|---:|
| coherent86 | -0.040198 | 0.545925 | 228 | 0.000000 | False |
| dense_seed62064 | -0.033354 | 0.622688 | 220 | 0.000000 | False |
| dense_seed62065 | -0.029103 | 0.668426 | 219 | 0.000000 | False |
| clean_pres_seed62064 | -0.026870 | 0.690512 | 222 | 0.000000 | False |
| clean_pres_seed62065 | -0.041864 | 0.534926 | 222 | 0.000000 | False |
| densemask_sparselabel_seed62064 | -0.032574 | 0.630856 | 220 | 0.000000 | False |

Outputs:
{
  "summary_csv": "experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_endpoint_summary.csv",
  "word_fits_csv": "experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_all_word_fits.csv",
  "overlap_csv": "experiments/archive/relation_learning/data/raw_aoa_refits_official/raw_aoa_fit_word_overlap.csv"
}
