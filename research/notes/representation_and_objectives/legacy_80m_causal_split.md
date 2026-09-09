# continuation training dynamics — standard staged-WWM 80M causal split

This note merges only completed files. No scoring is launched here.


## Main table

- `reference`: Supplement=59.21, Entity=27.72; GP_parallel=24.271844660194176, GP_nonparallel=53.0, GP_hard52_margin=1.7169032966268565, GP_hard52_ranks={'3': 21, '2': 8, '4': 21, '1': 2}; EWoK_acc=0.5010501443948543, EWoK_stable_frac_wrong=0.6927124440936596, EWoK_wrong_median=-0.9279484115540981

- `standard`: Supplement=58.72, Entity=27.7; GP_parallel=27.184466019417474, GP_nonparallel=53.0, GP_hard52_margin=1.800280800260969, GP_hard52_ranks={'4': 17, '2': 8, '3': 25, '1': 2}; EWoK_acc=0.5090574954056183, EWoK_stable_frac_wrong=0.679144385026738, EWoK_wrong_median=-0.8804604583419859

- `control`: Supplement=61.49, Entity=27.73; GP_parallel=24.271844660194176, GP_nonparallel=51.0, GP_hard52_margin=1.602562760702558, GP_hard52_ranks={'4': 15, '2': 14, '3': 20, '1': 3}; EWoK_acc=0.4985560514570753, EWoK_stable_frac_wrong=0.6979057591623037, EWoK_wrong_median=-1.1994800232350826

- `treatment`: Supplement=58.83, Entity=28.78; GP_parallel=22.33009708737864, GP_nonparallel=48.0, GP_hard52_margin=1.820028181437216, GP_hard52_ranks={'4': 23, '3': 20, '2': 7, '1': 2}; EWoK_acc=0.49553688632186926, EWoK_stable_frac_wrong=0.70752016653656, EWoK_wrong_median=-1.200680741108954


## Standard deltas

- `standard_minus_reference`: {'Supplement': -0.490000000000002, 'Entity': -0.019999999999999574, 'GlobalPIQA_parallel': 2.912621359223298, 'GlobalPIQA_parallel_hard52_mean_top_minus_correct': 0.08337750363411245, 'GlobalPIQA_nonparallel': 0.0, 'EWoK_fourcell_accuracy': 0.008007351010763997, 'EWoK_saved_wrong': -61.0, 'EWoK_stable_failure': -93.0, 'EWoK_stable_failure_frac_wrong': -0.0135680590669216, 'EWoK_interaction_sum_wrong_median': 0.04748795321211219, 'EWoK_interaction_sum_wrong_mean': -0.8028852084710505}

- `standard_minus_control`: {'Supplement': -2.770000000000003, 'Entity': -0.030000000000001137, 'GlobalPIQA_parallel': 2.912621359223298, 'GlobalPIQA_parallel_hard52_mean_top_minus_correct': 0.19771803955841083, 'GlobalPIQA_nonparallel': 2.0, 'EWoK_fourcell_accuracy': 0.010501443948542966, 'EWoK_saved_wrong': -80.0, 'EWoK_stable_failure': -126.0, 'EWoK_stable_failure_frac_wrong': -0.018761374135565667, 'EWoK_interaction_sum_wrong_median': 0.3190195648930967, 'EWoK_interaction_sum_wrong_mean': -0.7119483522442969}

- `standard_minus_treatment`: {'Supplement': -0.10999999999999943, 'Entity': -1.0800000000000018, 'GlobalPIQA_parallel': 4.854368932038835, 'GlobalPIQA_parallel_hard52_mean_top_minus_correct': -0.01974738117624697, 'GlobalPIQA_nonparallel': 5.0, 'EWoK_fourcell_accuracy': 0.013520609083749024, 'EWoK_saved_wrong': -103.0, 'EWoK_stable_failure': -179.0, 'EWoK_stable_failure_frac_wrong': -0.028375781509822007, 'EWoK_interaction_sum_wrong_median': 0.320220282766968, 'EWoK_interaction_sum_wrong_mean': -0.7222208421265779}


## Row movement

- GP hard52 standard vs reference: standard better rank 11, same 33, reference better rank 8; standard lower margin 24, reference lower margin 27; mean margin delta 0.0833775036341127.

- EWoK standard vs reference: standard correct 3878, reference correct 3817, standard-only correct 1076, reference-only correct 1015; stable standard 2540, stable reference 2633, stable both 1657.

- GP hard52 standard vs control: standard better rank 9, same 26, control better rank 17; standard lower margin 18, control lower margin 33; mean margin delta 0.19771803955841097.

- EWoK standard vs control: standard correct 3878, control correct 3798, standard-only correct 1248, control-only correct 1168; stable standard 2540, stable control 2666, stable both 1535.

- GP hard52 standard vs treatment: standard better rank 13, same 29, treatment better rank 10; standard lower margin 23, treatment lower margin 28; mean margin delta -0.01974738117624677.

- EWoK standard vs treatment: standard correct 3878, treatment correct 3775, standard-only correct 1234, treatment-only correct 1131; stable standard 2540, stable treatment 2719, stable both 1550.


## Interpretation

- Standard staged WWM is close to the uninterrupted compact EWoK reference, so the shared EWoK weakening in PVDM/control is mainly attributable to target redistribution and anchor swapping rather than staged replay.

- Standard staged WWM preserves nonparallel GlobalPIQA relative to compact reference, unlike PVDM treatment and control; this favors preserving ordinary WWM while adding only sparse relation-specific pressure if later design proceeds.


Files: `experiments/archive/representation_and_objectives/data/legacy_80m_causal_split/legacy_80m_causal_split.json`
