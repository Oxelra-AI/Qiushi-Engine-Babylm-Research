# context dependence ig probe result context-dependence information-gain probe

Forward-only masked-token NLL comparison on legal-corpus text. No training, upload, or leaderboard submission was performed.

Status: `CONTEXT_DEPENDENCE_IG_PROBE`
Corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Rows sampled: `3006` from scan `3006`; items `12024`
Checkpoints: `{'clean80': 'experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M', 'reinvest80': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'}`
Radii: `[4, 8, 16]`; device `cuda`; max_seq_len `256`

## Per-checkpoint summaries
### clean80
- full NLL mean/median: `2.2745203808508045` / `0.39633187651634216`
- r=4: IG mean `3.992054435990843`, corr IG~fullNLL pearson `-0.4953346665137569`, partial control logfreq `-0.6811650515900814`, highIG top20 fullNLL `0.38667469301593016`, lowIG bottom20 fullNLL `4.732079872335947`
- r=8: IG mean `3.5851100745893167`, corr IG~fullNLL pearson `-0.49543309529155966`, partial control logfreq `-0.680353711844598`, highIG top20 fullNLL `0.38786997621294583`, lowIG bottom20 fullNLL `5.130411731883105`
- r=16: IG mean `2.944187017771661`, corr IG~fullNLL pearson `-0.44366542310043705`, partial control logfreq `-0.5907746163356805`, highIG top20 fullNLL `0.3943144394614201`, lowIG bottom20 fullNLL `5.290127235033957`
### reinvest80
- full NLL mean/median: `1.697075923574598` / `0.12554440647363663`
- r=4: IG mean `4.225718681144896`, corr IG~fullNLL pearson `-0.4871239955070117`, partial control logfreq `-0.6605202371856519`, highIG top20 fullNLL `0.21382074860508046`, lowIG bottom20 fullNLL `3.844197391817946`
- r=8: IG mean `3.7217606724424974`, corr IG~fullNLL pearson `-0.4815440377982254`, partial control logfreq `-0.6561285308147726`, highIG top20 fullNLL `0.23169747036366495`, lowIG bottom20 fullNLL `4.075675948627119`
- r=16: IG mean `3.045744362999264`, corr IG~fullNLL pearson `-0.4255162664039678`, partial control logfreq `-0.5592067873374353`, highIG top20 fullNLL `0.23660302497876556`, lowIG bottom20 fullNLL `4.173671394079244`

## Between-checkpoint relation
### reinvest80_minus_clean80
- radius_4: mean full-NLL delta `-0.5774444572762065`, corr refIG~delta `0.2028531727283292`, partial control logfreq `0.2983274872436188`
- radius_8: mean full-NLL delta `-0.5774444572762065`, corr refIG~delta `0.20527209831262483`, partial control logfreq `0.300550339317795`
- radius_16: mean full-NLL delta `-0.5774444572762065`, corr refIG~delta `0.19934500559411636`, partial control logfreq `0.27733846467211926`

CSV: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_changed_block/context_dependence_items.csv`
JSON: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_changed_block/context_dependence_summary.json`
