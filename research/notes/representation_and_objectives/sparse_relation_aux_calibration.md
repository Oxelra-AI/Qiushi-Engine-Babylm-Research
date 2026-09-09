# sparse relation aux calibration — sparse relation auxiliary calibration

Created: 2026-08-31T02:12:25Z  Runtime: 217.04 s  Device: cuda  Dropout mode: train

## Coverage on exact pvdm 80m ewok fourcell reader/121 70M→80M segment prefix

Batches/rows/words scanned: 250 / 64000 / 9971289

Selected before cross-target filtering: 19847; used after microbatch-local cross-target filtering: 4000; used fraction: 0.20154179472968206

Used events per 256-row batch: 16.0

Used categories: {'spatial': 1370, 'temporal': 953, 'causal_connector': 1280, 'physical_change': 110, 'negation': 287}

Match levels: {'3': 1265, '2': 204, '0': 421, '4': 2072, '1': 38}

## Pre-update margin comparison

- ALL: {'head_success_semantic': 0.4881656804733728, 'head_success_anchor_permuted': 0.48520710059171596, 'head_success_delta_sem_minus_anchor': 0.00295857988165682, 'raw_cosine_success_semantic': 0.883629191321499, 'raw_cosine_success_anchor_permuted': 0.8047337278106509, 'raw_cosine_success_delta_sem_minus_anchor': 0.07889546351084809, 'raw_cosine_margin_mean_semantic': 0.11113255621802937, 'raw_cosine_margin_mean_anchor_permuted': 0.08975909882329801}

- spatial: {'head_success_semantic': 0.4971751412429379, 'head_success_anchor_permuted': 0.5084745762711864, 'head_success_delta_sem_minus_anchor': -0.011299435028248539, 'raw_cosine_success_semantic': 0.8813559322033898, 'raw_cosine_success_anchor_permuted': 0.7853107344632768, 'raw_cosine_success_delta_sem_minus_anchor': 0.096045197740113, 'raw_cosine_margin_mean_semantic': 0.12101633438451143, 'raw_cosine_margin_mean_anchor_permuted': 0.09274408029192417}

- causal_connector: {'head_success_semantic': 0.47865853658536583, 'head_success_anchor_permuted': 0.5213414634146342, 'head_success_delta_sem_minus_anchor': -0.04268292682926833, 'raw_cosine_success_semantic': 0.8597560975609756, 'raw_cosine_success_anchor_permuted': 0.8048780487804879, 'raw_cosine_success_delta_sem_minus_anchor': 0.05487804878048774, 'raw_cosine_margin_mean_semantic': 0.09788325967880465, 'raw_cosine_margin_mean_anchor_permuted': 0.07849748784721625}

- temporal: {'head_success_semantic': 0.5043859649122807, 'head_success_anchor_permuted': 0.43859649122807015, 'head_success_delta_sem_minus_anchor': 0.06578947368421056, 'raw_cosine_success_semantic': 0.9298245614035088, 'raw_cosine_success_anchor_permuted': 0.8070175438596491, 'raw_cosine_success_delta_sem_minus_anchor': 0.1228070175438597, 'raw_cosine_margin_mean_semantic': 0.11369927523232866, 'raw_cosine_margin_mean_anchor_permuted': 0.09345827483686439}

- negation: {'head_success_semantic': 0.44047619047619047, 'head_success_anchor_permuted': 0.35714285714285715, 'head_success_delta_sem_minus_anchor': 0.08333333333333331, 'raw_cosine_success_semantic': 0.8809523809523809, 'raw_cosine_success_anchor_permuted': 0.8809523809523809, 'raw_cosine_success_delta_sem_minus_anchor': 0.0, 'raw_cosine_margin_mean_semantic': 0.10131366604140826, 'raw_cosine_margin_mean_anchor_permuted': 0.10369192141418655}

- physical_change: {'head_success_semantic': 0.5, 'head_success_anchor_permuted': 0.55, 'head_success_delta_sem_minus_anchor': -0.050000000000000044, 'raw_cosine_success_semantic': 0.8, 'raw_cosine_success_anchor_permuted': 0.8, 'raw_cosine_success_delta_sem_minus_anchor': 0.0, 'raw_cosine_margin_mean_semantic': 0.16545688789337873, 'raw_cosine_margin_mean_anchor_permuted': 0.12092688539996743}

## Gradient summary

- aux::anchor_permuted::ALL: {'batches': 4, 'events_used': 63, 'terms': 126, 'model_l2': {'n': 3, 'mean': 0.09042915807694037, 'median': 0.09142775077376303, 'min': 0.08643424907851636, 'max': 0.09342547437854172}, 'encoder_l2': {'n': 3, 'mean': 0.08490793545919494, 'median': 0.08562307166499862, 'min': 0.08149739100673047, 'max': 0.08760334370585568}, 'head_l2': {'n': 3, 'mean': 0.09485357550355282, 'median': 0.09507618613762062, 'min': 0.09190603624051168, 'max': 0.09757850413252613}, 'ratios_to_mlm_model': {'n': 3, 'mean': 0.13484171145281842, 'median': 0.13397226899461578, 'min': 0.12994268262057904, 'max': 0.14061018274326048}, 'ratios_to_mlm_encoder': {'n': 3, 'mean': 0.1610574631534758, 'median': 0.15961234168848132, 'min': 0.15542680343661774, 'max': 0.16813324433532836}}

- aux::anchor_permuted::physical_change: {'batches': 1, 'events_used': 2, 'terms': 4, 'model_l2': {'n': 1, 'mean': 0.025790214967137348, 'median': 0.025790214967137348, 'min': 0.025790214967137348, 'max': 0.025790214967137348}, 'encoder_l2': {'n': 1, 'mean': 0.024173730624567097, 'median': 0.024173730624567097, 'min': 0.024173730624567097, 'max': 0.024173730624567097}, 'head_l2': {'n': 1, 'mean': 0.025650752849776335, 'median': 0.025650752849776335, 'min': 0.025650752849776335, 'max': 0.025650752849776335}, 'ratios_to_mlm_model': {'n': 1, 'mean': 0.03779130064739403, 'median': 0.03779130064739403, 'min': 0.03779130064739403, 'max': 0.03779130064739403}, 'ratios_to_mlm_encoder': {'n': 1, 'mean': 0.045062921445166666, 'median': 0.045062921445166666, 'min': 0.045062921445166666, 'max': 0.045062921445166666}}

- aux::semantic::ALL: {'batches': 4, 'events_used': 63, 'terms': 126, 'model_l2': {'n': 3, 'mean': 0.09579997937348693, 'median': 0.09731511061698209, 'min': 0.08584454851327074, 'max': 0.10424027899020798}, 'encoder_l2': {'n': 3, 'mean': 0.08864216969679023, 'median': 0.0899559985536084, 'min': 0.07917180189514829, 'max': 0.09679870864161402}, 'head_l2': {'n': 3, 'mean': 0.09383513494719319, 'median': 0.09370177488943043, 'min': 0.0868686975536055, 'max': 0.1009349323985436}, 'ratios_to_mlm_model': {'n': 3, 'mean': 0.14284746374005156, 'median': 0.14259922251702692, 'min': 0.12905614430726203, 'max': 0.15688702439586572}, 'ratios_to_mlm_encoder': {'n': 3, 'mean': 0.1681541859255754, 'median': 0.16768947082677996, 'min': 0.15099158315219943, 'max': 0.18578150379774688}}

- aux::semantic::physical_change: {'batches': 1, 'events_used': 2, 'terms': 4, 'model_l2': {'n': 1, 'mean': 0.03230451096610648, 'median': 0.03230451096610648, 'min': 0.03230451096610648, 'max': 0.03230451096610648}, 'encoder_l2': {'n': 1, 'mean': 0.0302443010301873, 'median': 0.0302443010301873, 'min': 0.0302443010301873, 'max': 0.0302443010301873}, 'head_l2': {'n': 1, 'mean': 0.03058148463886629, 'median': 0.03058148463886629, 'min': 0.03058148463886629, 'max': 0.03058148463886629}, 'ratios_to_mlm_model': {'n': 1, 'mean': 0.04733692556431904, 'median': 0.04733692556431904, 'min': 0.04733692556431904, 'max': 0.04733692556431904}, 'ratios_to_mlm_encoder': {'n': 1, 'mean': 0.056379240037622874, 'median': 0.056379240037622874, 'min': 0.056379240037622874, 'max': 0.056379240037622874}}

- mlm: {'batches': 4, 'events_used': 0, 'terms': 0, 'model_l2': {'n': 4, 'mean': 0.6678918223983694, 'median': 0.6648005266481083, 'min': 0.6595283597137317, 'max': 0.6824378765835296}, 'encoder_l2': {'n': 4, 'mean': 0.5252937587236759, 'median': 0.5226905133424764, 'min': 0.5193500793805442, 'max': 0.5364439288292062}, 'head_l2': {'n': 4, 'mean': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0}, 'ratios_to_mlm_model': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}, 'ratios_to_mlm_encoder': {'n': 0, 'mean': None, 'median': None, 'min': None, 'max': None}}

## Calibration readout

Problems for full launch: ['low cross-target retention 0.202', 'sparse comparative coverage 0 used events', 'semantic raw hidden cosine already high 0.884', 'anchor_permuted raw hidden cosine already high 0.805']


Files: `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration/calibration_summary.json`, `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration/batch_calibration_records.jsonl`, `experiments/archive/representation_and_objectives/data/sparse_relation_aux_calibration/gradient_records.jsonl`

