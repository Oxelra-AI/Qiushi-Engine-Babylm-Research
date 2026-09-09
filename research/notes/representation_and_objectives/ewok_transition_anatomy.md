# comparative query binding pilot EWoK transition anatomy

Summary JSON: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/ewok_transition_anatomy_summary.json`. Row-level transition files are under `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy`.

This uses official-format EWoK rows as a diagnostic only, not a training source. It asks whether the scale1.75 EWoK cost is row-localized enough to target with a natural fork, or distributed across the trajectory.

## live_minus_matched: `matched_legal16k_80M` -> `scale1p75_live`
n=7618; acc_a=0.509189; acc_b=0.492780; net_correct=-125; net_stable_failure=193.
lost_correct=1129, gained_correct=1004, added_stable_failure=1091, removed_stable_failure=898, lost_correct_to_stable=711, stable_to_correct=553.
delta interaction_sum mean=-0.12505790355640448, median=-0.11973453685641289, p10=-3.533694273850415, p90=3.547463558614254.
Top domain/context transition tables: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/live_minus_matched/by_domain_transition.csv`, `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/live_minus_matched/by_contextdiff_transition.csv`.

## disabled_minus_matched: `matched_legal16k_80M` -> `scale1p75_disabled`
n=7618; acc_a=0.509189; acc_b=0.494487; net_correct=-112; net_stable_failure=113.
lost_correct=1138, gained_correct=1026, added_stable_failure=1054, removed_stable_failure=941, lost_correct_to_stable=673, stable_to_correct=569.
delta interaction_sum mean=-0.12701798234351896, median=-0.10574781335890293, p10=-3.6120711360126734, p90=3.5368779953569174.
Top domain/context transition tables: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/disabled_minus_matched/by_domain_transition.csv`, `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/disabled_minus_matched/by_contextdiff_transition.csv`.

## live_minus_disabled: `scale1p75_disabled` -> `scale1p75_live`
n=7618; acc_a=0.494487; acc_b=0.492780; net_correct=-13; net_stable_failure=80.
lost_correct=575, gained_correct=562, added_stable_failure=613, removed_stable_failure=533, lost_correct_to_stable=363, stable_to_correct=295.
delta interaction_sum mean=0.0019600787871144765, median=0.009923217818140984, p10=-1.8674147225683555, p90=1.9247519169002771.
Top domain/context transition tables: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/live_minus_disabled/by_domain_transition.csv`, `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/live_minus_disabled/by_contextdiff_transition.csv`.

## scale0p5_minus_live: `scale1p75_live` -> `scale1p75_scale0p5`
n=7618; acc_a=0.492780; acc_b=0.498950; net_correct=47; net_stable_failure=-81.
lost_correct=392, gained_correct=439, added_stable_failure=370, removed_stable_failure=451, lost_correct_to_stable=199, stable_to_correct=262.
delta interaction_sum mean=-0.003969309469561407, median=-0.008015338331460953, p10=-1.4117076992988586, p90=1.3903183892834932.
Top domain/context transition tables: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/scale0p5_minus_live/by_domain_transition.csv`, `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/scale0p5_minus_live/by_contextdiff_transition.csv`.

## scale2p5_minus_live: `scale1p75_live` -> `scale1p75_scale2p5`
n=7618; acc_a=0.492780; acc_b=0.496718; net_correct=30; net_stable_failure=-31.
lost_correct=245, gained_correct=275, added_stable_failure=230, removed_stable_failure=261, lost_correct_to_stable=125, stable_to_correct=140.
delta interaction_sum mean=-0.007013650947551718, median=0.001200169324874878, p10=-0.8849591246980708, p90=0.8561286889016628.
Top domain/context transition tables: `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/scale2p5_minus_live/by_domain_transition.csv`, `experiments/archive/representation_and_objectives/data/ewok_transition_anatomy/scale2p5_minus_live/by_contextdiff_transition.csv`.

