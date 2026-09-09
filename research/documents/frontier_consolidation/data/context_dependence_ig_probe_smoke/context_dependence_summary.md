# context dependence ig probe result context-dependence information-gain probe

Forward-only masked-token NLL comparison on legal-corpus text. No training, upload, or leaderboard submission was performed.

Status: `CONTEXT_DEPENDENCE_IG_PROBE`
Corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Rows sampled: `4` from scan `100`; items `8`
Checkpoints: `{'chck82': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M', 'chck100': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'}`
Radii: `[4, 8]`; device `cpu`; max_seq_len `256`

## Per-checkpoint summaries
### chck82
- full NLL mean/median: `2.155534434132278` / `0.4298592135310173`
- r=4: IG mean `3.328552913852036`, corr IG~fullNLL pearson `-0.4889078590223395`, partial control logfreq `-0.3686484081094494`, highIG top20 fullNLL `0.23334108293056488`, lowIG bottom20 fullNLL `7.702022075653076`
- r=8: IG mean `3.23721241671592`, corr IG~fullNLL pearson `-0.5020759305231309`, partial control logfreq `-0.3418901638022824`, highIG top20 fullNLL `0.23334108293056488`, lowIG bottom20 fullNLL `7.702022075653076`
### chck100
- full NLL mean/median: `2.2044392364332452` / `0.5121102556586266`
- r=4: IG mean `3.248255976825021`, corr IG~fullNLL pearson `-0.5470976516128115`, partial control logfreq `-0.41574450186714473`, highIG top20 fullNLL `0.1756788045167923`, lowIG bottom20 fullNLL `7.541102409362793`
- r=8: IG mean `3.098809481947683`, corr IG~fullNLL pearson `-0.5334154788745072`, partial control logfreq `-0.3920138004045353`, highIG top20 fullNLL `0.1756788045167923`, lowIG bottom20 fullNLL `7.541102409362793`

## Between-checkpoint relation
### chck100_minus_chck82
- radius_4: mean full-NLL delta `0.048904802300967276`, corr refIG~delta `-0.3272997162927398`, partial control logfreq `-0.6791583685801319`
- radius_8: mean full-NLL delta `0.048904802300967276`, corr refIG~delta `-0.3078841467922126`, partial control logfreq `-0.6973465218932625`

CSV: `experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_smoke/context_dependence_items.csv`
JSON: `experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_smoke/context_dependence_summary.json`
