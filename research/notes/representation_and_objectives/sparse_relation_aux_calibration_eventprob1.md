# sparse relation aux calibration — sparse relation auxiliary calibration

Created: 2026-08-31T02:17:05Z  Runtime: 214.6 s  Device: cuda  Dropout mode: train

## Coverage on exact pvdm 80m ewok fourcell reader/121 70M→80M segment prefix

Batches/rows/words scanned: 250 / 64000 / 9971289

Selected before cross-target filtering: 24000; used after microbatch-local cross-target filtering: 5505; used fraction: 0.229375

Used events per 256-row batch: 22.02

Used categories: {'temporal': 1407, 'causal_connector': 1768, 'spatial': 1818, 'physical_change': 138, 'negation': 374}

Match levels: {'4': 2792, '3': 1791, '2': 304, '0': 572, '1': 46}

## Pre-update margin comparison

- ALL: {'head_success_semantic': 0.4637462235649547, 'head_success_anchor_permuted': 0.4788519637462236, 'head_success_delta_sem_minus_anchor': -0.015105740181268867, 'raw_cosine_success_semantic': 0.8867069486404834, 'raw_cosine_success_anchor_permuted': 0.8277945619335347, 'raw_cosine_success_delta_sem_minus_anchor': 0.05891238670694865, 'raw_cosine_margin_mean_semantic': 0.1237177007263273, 'raw_cosine_margin_mean_anchor_permuted': 0.0999573640626633}

- spatial: {'head_success_semantic': 0.45, 'head_success_anchor_permuted': 0.46458333333333335, 'head_success_delta_sem_minus_anchor': -0.014583333333333337, 'raw_cosine_success_semantic': 0.9083333333333333, 'raw_cosine_success_anchor_permuted': 0.8416666666666667, 'raw_cosine_success_delta_sem_minus_anchor': 0.06666666666666665, 'raw_cosine_margin_mean_semantic': 0.13324639175940925, 'raw_cosine_margin_mean_anchor_permuted': 0.10304716873991614}

- causal_connector: {'head_success_semantic': 0.44761904761904764, 'head_success_anchor_permuted': 0.5, 'head_success_delta_sem_minus_anchor': -0.05238095238095236, 'raw_cosine_success_semantic': 0.8666666666666667, 'raw_cosine_success_anchor_permuted': 0.8, 'raw_cosine_success_delta_sem_minus_anchor': 0.06666666666666665, 'raw_cosine_margin_mean_semantic': 0.11606125068806467, 'raw_cosine_margin_mean_anchor_permuted': 0.09149493779100124}

- temporal: {'head_success_semantic': 0.5140845070422535, 'head_success_anchor_permuted': 0.45774647887323944, 'head_success_delta_sem_minus_anchor': 0.056338028169014065, 'raw_cosine_success_semantic': 0.8450704225352113, 'raw_cosine_success_anchor_permuted': 0.8028169014084507, 'raw_cosine_success_delta_sem_minus_anchor': 0.04225352112676051, 'raw_cosine_margin_mean_semantic': 0.112523099906008, 'raw_cosine_margin_mean_anchor_permuted': 0.10209670747486009}

- negation: {'head_success_semantic': 0.4326923076923077, 'head_success_anchor_permuted': 0.49038461538461536, 'head_success_delta_sem_minus_anchor': -0.057692307692307654, 'raw_cosine_success_semantic': 0.9423076923076923, 'raw_cosine_success_anchor_permuted': 0.9230769230769231, 'raw_cosine_success_delta_sem_minus_anchor': 0.019230769230769162, 'raw_cosine_margin_mean_semantic': 0.1345268476109665, 'raw_cosine_margin_mean_anchor_permuted': 0.11575162779683104}

- physical_change: {'head_success_semantic': 0.5277777777777778, 'head_success_anchor_permuted': 0.5555555555555556, 'head_success_delta_sem_minus_anchor': -0.02777777777777779, 'raw_cosine_success_semantic': 1.0, 'raw_cosine_success_anchor_permuted': 0.8888888888888888, 'raw_cosine_success_delta_sem_minus_anchor': 0.11111111111111116, 'raw_cosine_margin_mean_semantic': 0.14308027509186003, 'raw_cosine_margin_mean_anchor_permuted': 0.09498335938486788}

## Gradient summary

- aux::anchor_permuted::ALL: {'batches': 4, 'events_used': 75, 'terms': 150, 'model_l2': {'n': 4, 'mean': 0.09362475769173874, 'median': 0.09342946496428392, 'min': 0.07554396293730578, 'max': 0.1120961379010813}, 'encoder_l2': {'n': 4, 'mean': 0.0882337944573734, 'median': 0.08772105005708052, 'min': 0.07108594960772736, 'max': 0.10640712810760519}, 'head_l2': {'n': 4, 'mean': 0.09801807904313138, 'median': 0.0973799257142928, 'min': 0.08026178163984644, 'max': 0.11705068310409344}, 'ratios_to_mlm_model': {'n': 4, 'mean': 0.140276096822467, 'median': 0.13872135786839845, 'min': 0.11369758092664732, 'max': 0.16996409062642376}, 'ratios_to_mlm_encoder': {'n': 4, 'mean': 0.16805828176855828, 'median': 0.16545791150758055, 'min': 0.13643213636154805, 'max': 0.20488516769752396}}

- aux::anchor_permuted::physical_change: {'batches': 2, 'events_used': 4, 'terms': 8, 'model_l2': {'n': 2, 'mean': 0.03148336211570397, 'median': 0.03148336211570397, 'min': 0.02949162420242656, 'max': 0.033475100028981374}, 'encoder_l2': {'n': 2, 'mean': 0.029364557969870662, 'median': 0.029364557969870662, 'min': 0.027716340673844347, 'max': 0.03101277526589698}, 'head_l2': {'n': 2, 'mean': 0.03224289362483155, 'median': 0.03224289362483155, 'min': 0.03045806326731607, 'max': 0.034027723982347026}, 'ratios_to_mlm_model': {'n': 2, 'mean': 0.0467702878068288, 'median': 0.0467702878068288, 'min': 0.04321510457489506, 'max': 0.05032547103876254}, 'ratios_to_mlm_encoder': {'n': 2, 'mean': 0.05540622674039045, 'median': 0.05540622674039045, 'min': 0.051666799052671755, 'max': 0.05914565442810915}}

- aux::semantic::ALL: {'batches': 4, 'events_used': 75, 'terms': 150, 'model_l2': {'n': 4, 'mean': 0.10221664997878242, 'median': 0.10346105472019552, 'min': 0.0865184241194599, 'max': 0.11542606635527874}, 'encoder_l2': {'n': 4, 'mean': 0.09496373327613403, 'median': 0.09584190381978913, 'min': 0.08025878442407226, 'max': 0.10791234104088557}, 'head_l2': {'n': 4, 'mean': 0.1029708163224027, 'median': 0.10130522342188439, 'min': 0.08887032880575856, 'max': 0.12040248964008343}, 'ratios_to_mlm_model': {'n': 4, 'mean': 0.15307799732690192, 'median': 0.15354211526376232, 'min': 0.1302147139955048, 'max': 0.17501304478457824}, 'ratios_to_mlm_encoder': {'n': 4, 'mean': 0.1808036738033686, 'median': 0.18069705505850397, 'min': 0.15403715475676524, 'max': 0.20778343033970115}}

- aux::semantic::physical_change: {'batches': 2, 'events_used': 4, 'terms': 8, 'model_l2': {'n': 2, 'mean': 0.028175725327084636, 'median': 0.028175725327084636, 'min': 0.021999222086374202, 'max': 0.034352228567795066}, 'encoder_l2': {'n': 2, 'mean': 0.026123383051011174, 'median': 0.026123383051011174, 'min': 0.020241025167957567, 'max': 0.03200574093406478}, 'head_l2': {'n': 2, 'mean': 0.026695121046397687, 'median': 0.026695121046397687, 'min': 0.021866022416728494, 'max': 0.03152421967606688}, 'ratios_to_mlm_model': {'n': 2, 'mean': 0.04194017335621463, 'median': 0.04194017335621463, 'min': 0.03223622668265178, 'max': 0.05164412002977748}, 'ratios_to_mlm_encoder': {'n': 2, 'mean': 0.049385616776083704, 'median': 0.049385616776083704, 'min': 0.03773185617392258, 'max': 0.06103937737824483}}

- mlm: {'batches': 4, 'events_used': 0, 'terms': 0, 'model_l2': {'n': 4, 'mean': 0.6678918223983694, 'median': 0.6648005266481083, 'min': 0.6595283597137317, 'max': 0.6824378765835296}, 'encoder_l2': {'n': 4, 'mean': 0.5252937587236759, 'median': 0.5226905133424764, 'min': 0.5193500793805442, 'max': 0.5364439288292062}, 'head_l2': {'n': 4, 'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}, 'ratios_to_mlm_model': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'ratios_to_mlm_encoder': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}}

## Calibration readout

Problems for full launch: ['low cross-target retention 0.229', 'sparse comparative coverage 0 used events', 'semantic raw hidden cosine already high 0.887', 'anchor_permuted raw hidden cosine already high 0.828']


Files: `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration_eventprob1/calibration_summary.json`, `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration_eventprob1/batch_calibration_records.jsonl`, `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration_eventprob1/gradient_records.jsonl`

