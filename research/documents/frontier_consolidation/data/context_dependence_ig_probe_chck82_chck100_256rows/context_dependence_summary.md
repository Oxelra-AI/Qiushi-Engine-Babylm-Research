# context dependence ig probe result context-dependence information-gain probe

Forward-only masked-token NLL comparison on legal-corpus text. No training, upload, or leaderboard submission was performed.

Status: `CONTEXT_DEPENDENCE_IG_PROBE`
Corpus: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
Rows sampled: `256` from scan `64740`; items `1024`
Checkpoints: `{'chck82': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M', 'chck100': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M'}`
Radii: `[4, 8, 16]`; device `cuda`; max_seq_len `256`

## Per-checkpoint summaries
### chck82
- full NLL mean/median: `2.611631627239106` / `1.3105310797691345`
- r=4: IG mean `1.5995864004253235`, corr IG~fullNLL pearson `-0.29112833785428954`, partial control logfreq `-0.52415797769889`, highIG top20 fullNLL `1.087633361622198`, lowIG bottom20 fullNLL `4.27820863614918`
- r=8: IG mean `1.2211366250929956`, corr IG~fullNLL pearson `-0.27980297001092547`, partial control logfreq `-0.477342349988334`, highIG top20 fullNLL `1.446234027756944`, lowIG bottom20 fullNLL `4.368977121570531`
- r=16: IG mean `0.8782079076179343`, corr IG~fullNLL pearson `-0.2473118889747269`, partial control logfreq `-0.430275796001961`, highIG top20 fullNLL `1.8230487564555826`, lowIG bottom20 fullNLL `4.366573502181792`
### chck100
- full NLL mean/median: `2.5809037562887625` / `1.2678457498550415`
- r=4: IG mean `1.5994970785584997`, corr IG~fullNLL pearson `-0.29475327759482584`, partial control logfreq `-0.524317465141616`, highIG top20 fullNLL `1.1712854401911517`, lowIG bottom20 fullNLL `4.024875393420385`
- r=8: IG mean `1.215543184925167`, corr IG~fullNLL pearson `-0.28536001555791907`, partial control logfreq `-0.47837762235710785`, highIG top20 fullNLL `1.4164891096799834`, lowIG bottom20 fullNLL `4.366098503856098`
- r=16: IG mean `0.8537723277638065`, corr IG~fullNLL pearson `-0.24883777855365138`, partial control logfreq `-0.4257917719188438`, highIG top20 fullNLL `1.8294507699114655`, lowIG bottom20 fullNLL `4.359134706942474`

## Between-checkpoint relation
### chck100_minus_chck82
- radius_4: mean full-NLL delta `-0.030727870950343572`, corr refIG~delta `0.03623598629507455`, partial control logfreq `0.0670239776639445`
- radius_8: mean full-NLL delta `-0.030727870950343572`, corr refIG~delta `0.05528324404839251`, partial control logfreq `0.08148419268982901`
- radius_16: mean full-NLL delta `-0.030727870950343572`, corr refIG~delta `0.08177760941688744`, partial control logfreq `0.10742657214527467`

CSV: `experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_chck82_chck100_256rows/context_dependence_items.csv`
JSON: `experiments/archive/frontier_consolidation/data/context_dependence_ig_probe_chck82_chck100_256rows/context_dependence_summary.json`
