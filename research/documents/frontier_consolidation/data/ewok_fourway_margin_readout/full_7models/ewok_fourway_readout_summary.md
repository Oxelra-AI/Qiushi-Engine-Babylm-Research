# earlier analysis EWoK four-way context-target readout

This is a mechanistic measurement over existing checkpoints, not a pretraining target. Official EWoK uses only C1T1 vs C2T1; this readout additionally scores C1T2 and C2T2 and reports interaction `(s11+s22)-(s12+s21)`.

Rows: 7618 from `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict/evaluation_data/full_eval/ewok_filtered`; rows/domain=0; seed=80080.

## Model table
- a02_oldtok_reinvest_100M: acc_from_scores=51.720, mean_official_margin=0.0890, mean_interaction=0.1780, interaction_positive_rate=52.42, context_insensitive_frac=0.214, broad={'Overall_non_submittable': 42.0331347900748, 'EWoK': 53.54}
- a02_legal16_reinvest_80M: acc_from_scores=50.788, mean_official_margin=0.0481, mean_interaction=0.0965, interaction_positive_rate=51.69, context_insensitive_frac=0.197, broad={'cheap7_mean80M': 42.9486}
- a01_legal40k_8x480_100M: acc_from_scores=51.063, mean_official_margin=0.0446, mean_interaction=0.0892, interaction_positive_rate=50.89, context_insensitive_frac=0.174, broad={'Overall': 41.1406}
- a02_legal16_reinvest_100M: acc_from_scores=50.709, mean_official_margin=0.0410, mean_interaction=0.0822, interaction_positive_rate=51.85, context_insensitive_frac=0.200, broad={'Overall': 41.257770896404615, 'EWoK': 50.39}
- a01_legal40k_depth12_100M: acc_from_scores=51.221, mean_official_margin=0.0360, mean_interaction=0.0720, interaction_positive_rate=52.61, context_insensitive_frac=0.197, broad={'Overall': 41.0276}
- a02_step075_strict_innov_80M: acc_from_scores=49.541, mean_official_margin=0.0109, mean_interaction=0.0217, interaction_positive_rate=50.56, context_insensitive_frac=0.216, broad={'cheap7_mean80M': 41.8957}
- a02_legal16_clean_80M: acc_from_scores=49.737, mean_official_margin=-0.0164, mean_interaction=-0.0329, interaction_positive_rate=50.08, context_insensitive_frac=0.228, broad={'cheap7_mean80M': 41.6022}

## Validation against saved official predictions
- a02_legal16_reinvest_100M: available=True, match_rate=0.9843791021265424, matches=7499/7618
  first mismatch: {'uid': 'agent-properties_144', 'domain': 'agent-properties', 'index': 144, 'computed_pred': 'Ali thinks that playing cards requires more effort than playing hockey. Ali is tired. Ali chooses cards over hockey.', 'saved_pred': 'Ali thinks that playing cards requires less effort than playing hockey. Ali is tired. Ali chooses cards over hockey.', 's11': -44.34375, 's21': -44.291015625, 'official_margin': -0.052734375}
- a02_legal16_clean_80M: available=False, match_rate=None, matches=0/0
- a02_legal16_reinvest_80M: available=True, match_rate=0.9863481228668942, matches=7514/7618
  first mismatch: {'uid': 'agent-properties_113', 'domain': 'agent-properties', 'index': 113, 'computed_pred': 'Chao knows that the wheel is not in the shop. Chao imagines that the wheel is in the shop.', 'saved_pred': 'Chao knows that the wheel is in the shop. Chao imagines that the wheel is in the shop.', 's11': -22.81365966796875, 's21': -22.923126220703125, 'official_margin': 0.109466552734375}
- a02_step075_strict_innov_80M: available=True, match_rate=0.9841165660278288, matches=7497/7618
  first mismatch: {'uid': 'agent-properties_9', 'domain': 'agent-properties', 'index': 9, 'computed_pred': 'Mohammed is in the gallery. Mohammed sees the volleyball inside. Mohammed doubts that the volleyball is in the gallery.', 'saved_pred': 'Mohammed is in the gallery. Mohammed sees the volleyball outside. Mohammed doubts that the volleyball is in the gallery.', 's11': -22.60455322265625, 's21': -22.530242919921875, 'official_margin': -0.074310302734375}
- a02_oldtok_reinvest_100M: available=True, match_rate=0.9843791021265424, matches=7499/7618
  first mismatch: {'uid': 'agent-properties_29', 'domain': 'agent-properties', 'index': 29, 'computed_pred': 'Chao recently saw the wheel inside the shop. Later, Chao saw the wheel inside the zoo. Chao doubts that the wheel is in the shop.', 'saved_pred': 'Chao recently saw the wheel inside the shop. Earlier, Chao saw the wheel inside the zoo. Chao doubts that the wheel is in the shop.', 's11': -39.208038330078125, 's21': -39.36383056640625, 'official_margin': 0.155792236328125}
- a01_legal40k_8x480_100M: available=True, match_rate=0.9872669992123917, matches=7521/7618
  first mismatch: {'uid': 'agent-properties_125', 'domain': 'agent-properties', 'index': 125, 'computed_pred': 'Jesse sees that the truck is in the school. Jesse imagines that the truck is in the school.', 'saved_pred': 'Jesse pretends that the truck is in the school. Jesse imagines that the truck is in the school.', 's11': -27.95501708984375, 's21': -27.933868408203125, 'official_margin': -0.021148681640625}
- a01_legal40k_depth12_100M: available=True, match_rate=0.9898923601995274, matches=7541/7618
  first mismatch: {'uid': 'agent-properties_28', 'domain': 'agent-properties', 'index': 28, 'computed_pred': 'Chao recently saw the wheel inside the shop. Later, Chao saw the wheel inside the zoo. Chao believes that the wheel is in the shop.', 'saved_pred': 'Chao recently saw the wheel inside the shop. Earlier, Chao saw the wheel inside the zoo. Chao believes that the wheel is in the shop.', 's11': -26.780059814453125, 's21': -26.692718505859375, 'official_margin': -0.08734130859375}

## Correlations across scored models
- pearson_mean_interaction_vs_sampled_official_accuracy: 0.8913030866906815
- spearman_mean_interaction_vs_sampled_official_accuracy: 0.7142857142857143
- pearson_mean_interaction_vs_broad_reference_scalar: 0.1984903792157824
- spearman_mean_interaction_vs_broad_reference_scalar: 0.39285714285714285
- note: Small-N descriptive correlations only; used to decide whether the interaction is worth testing on legal corpus-derived contrasts.

CSV records: `experiments/archive/frontier_consolidation/data/ewok_fourway_margin_readout/full_7models/ewok_fourway_records.csv`
Full JSON: `experiments/archive/frontier_consolidation/data/ewok_fourway_margin_readout/full_7models/ewok_fourway_readout_summary.json`
