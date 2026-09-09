# context dependence ig probe result context-dependence information-gain probe

Forward-only masked-token NLL comparison on legal-corpus text. No training, upload, or leaderboard submission was performed.

Status: `CONTEXT_DEPENDENCE_IG_PROBE`
Corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Rows sampled: `1024` from scan `64740`; items `4096`
Checkpoints: `{'clean80': 'experiments/archive/frontier_consolidation/training/runs/complianttok_cleanqwen_seed43022_80M/hf_model/chck_80M', 'reinvest80': 'experiments/archive/frontier_consolidation/training/runs/complianttok_reinvest_seed43022_r2/hf_model/chck_80M'}`
Radii: `[4, 8, 16]`; device `cuda`; max_seq_len `256`

## Per-checkpoint summaries
### clean80
- full NLL mean/median: `2.631845228125613` / `1.2666993141174316`
- r=4: IG mean `1.8318865142945597`, corr IG~fullNLL pearson `-0.27303073054138594`, partial control logfreq `-0.5194988372485747`, highIG top20 fullNLL `1.2130679091787187`, lowIG bottom20 fullNLL `4.190451216812317`
- r=8: IG mean `1.3611919781731245`, corr IG~fullNLL pearson `-0.26945663293545935`, partial control logfreq `-0.49243512038806125`, highIG top20 fullNLL `1.4113088556768894`, lowIG bottom20 fullNLL `4.371303519322758`
- r=16: IG mean `1.01847422926787`, corr IG~fullNLL pearson `-0.2464301156204152`, partial control logfreq `-0.4429150119280462`, highIG top20 fullNLL `1.7315902210614542`, lowIG bottom20 fullNLL `4.204226274884809`
### reinvest80
- full NLL mean/median: `2.5389978014329557` / `1.175542950630188`
- r=4: IG mean `1.8753483034360414`, corr IG~fullNLL pearson `-0.2947393891757567`, partial control logfreq `-0.53083131080937`, highIG top20 fullNLL `1.070775228801271`, lowIG bottom20 fullNLL `4.170076165298257`
- r=8: IG mean `1.3715470060797372`, corr IG~fullNLL pearson `-0.2908110556144798`, partial control logfreq `-0.5058403592756711`, highIG top20 fullNLL `1.202413566567752`, lowIG bottom20 fullNLL `4.402631111852416`
- r=16: IG mean `1.0108100663517448`, corr IG~fullNLL pearson `-0.26653144974550425`, partial control logfreq `-0.45862081496552165`, highIG top20 fullNLL `1.467460446430982`, lowIG bottom20 fullNLL `4.278891656534139`

## Between-checkpoint relation
### reinvest80_minus_clean80
- radius_4: mean full-NLL delta `-0.09284742669265711`, corr refIG~delta `0.06342487648040425`, partial control logfreq `0.08986471641467629`
- radius_8: mean full-NLL delta `-0.09284742669265711`, corr refIG~delta `0.06773442992734922`, partial control logfreq `0.09142285915830421`
- radius_16: mean full-NLL delta `-0.09284742669265711`, corr refIG~delta `0.048755724995839494`, partial control logfreq `0.0686662969306337`

CSV: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_common_sample/context_dependence_items.csv`
JSON: `experiments/archive/frontier_consolidation/data/ig_matched_clean80_reinvest80_common_sample/context_dependence_summary.json`
