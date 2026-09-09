# full context pivot substitution probe — full-context pivot-substitution readiness analysis

## Repaired donor/readiness facts

- chck80_standard: ALL true-shuffle mean=0.4640306826616269 success=0.6429832303618711; inferred shuffle levels={'0': 2228, '1': 24, '2': 14}

  - physical_change n=381 true-anchor=0.7610173493127517 / true-shuffle=0.44769467740366115 success_shuffle=0.6902887139107612

  - comparative n=383 true-anchor=1.6275370916120364 / true-shuffle=1.3309591182914269 success_shuffle=0.7389033942558747

  - comparative without greater/higher/lower: {'true_anchor': {'n': 341, 'mean': 1.5755139692634266, 'std': 2.431876561421539, 'min': -2.3217668533325195, 'max': 13.118413984775543}, 'true_pivotmasked': {'n': 341, 'mean': 1.1246238902323509, 'std': 2.060906566781932, 'min': -4.073864459991455, 'max': 9.484508834779263}, 'true_shuffle': {'n': 341, 'mean': 1.3189952840343735, 'std': 2.409811090965563, 'min': -4.43615198135376, 'max': 13.742071472108364}, 'true_anchor_succ': 252, 'true_anchor_den': 341, 'true_shuffle_succ': 250, 'true_shuffle_den': 341, 'true_pivotmasked_succ': 231, 'true_pivotmasked_den': 341, 'n': 341, 'true_anchor_success_rate': 0.7390029325513197, 'true_shuffle_success_rate': 0.7331378299120235, 'true_pivotmasked_success_rate': 0.6774193548387096}

  - comparative more/better/worse/less: {'true_anchor': {'n': 169, 'mean': 1.0822220943787566, 'std': 2.035595473557047, 'min': -2.3217668533325195, 'max': 11.01802921295166}, 'true_pivotmasked': {'n': 169, 'mean': 0.8995138340283146, 'std': 1.7574259414369524, 'min': -4.073864459991455, 'max': 9.484508834779263}, 'true_shuffle': {'n': 169, 'mean': 0.7349069448533749, 'std': 1.9255952499458382, 'min': -4.43615198135376, 'max': 13.742071472108364}, 'true_anchor_succ': 119, 'true_anchor_den': 169, 'true_shuffle_succ': 114, 'true_shuffle_den': 169, 'true_pivotmasked_succ': 111, 'true_pivotmasked_den': 169, 'n': 169, 'true_anchor_success_rate': 0.7041420118343196, 'true_shuffle_success_rate': 0.6745562130177515, 'true_pivotmasked_success_rate': 0.6568047337278107}

  - target-length counts ALL: {'1': 1386, '2': 656, '3': 118, '4': 59, '5': 36, '6': 11}

- chck70_compact: ALL true-shuffle mean=0.4784486257188172 success=0.6663724624889673; inferred shuffle levels={'0': 2228, '1': 24, '2': 14}

  - physical_change n=381 true-anchor=0.7329256384782591 / true-shuffle=0.4685206749498492 success_shuffle=0.6745406824146981

  - comparative n=383 true-anchor=1.6239977816272375 / true-shuffle=1.3520939666106107 success_shuffle=0.7911227154046997

  - comparative without greater/higher/lower: {'true_anchor': {'n': 341, 'mean': 1.6199319368061715, 'std': 2.54469273287956, 'min': -5.187401056289673, 'max': 13.386422991752625}, 'true_pivotmasked': {'n': 341, 'mean': 1.097127089160836, 'std': 1.963165282598275, 'min': -4.65849757194519, 'max': 9.593829989433289}, 'true_shuffle': {'n': 341, 'mean': 1.3588428806890447, 'std': 2.3702284181221422, 'min': -3.7608275413513184, 'max': 12.553614020347595}, 'true_anchor_succ': 274, 'true_anchor_den': 341, 'true_shuffle_succ': 269, 'true_shuffle_den': 341, 'true_pivotmasked_succ': 247, 'true_pivotmasked_den': 341, 'n': 341, 'true_anchor_success_rate': 0.8035190615835777, 'true_shuffle_success_rate': 0.7888563049853372, 'true_pivotmasked_success_rate': 0.7243401759530792}

  - comparative more/better/worse/less: {'true_anchor': {'n': 169, 'mean': 1.0055439944334945, 'std': 1.884239410719136, 'min': -1.7341880798339844, 'max': 9.775066614151001}, 'true_pivotmasked': {'n': 169, 'mean': 0.8546215550692962, 'std': 1.6296367072766296, 'min': -2.441575527191162, 'max': 7.45198130607605}, 'true_shuffle': {'n': 169, 'mean': 0.6793272141256982, 'std': 1.630114629281714, 'min': -3.7608275413513184, 'max': 8.256013870239258}, 'true_anchor_succ': 126, 'true_anchor_den': 169, 'true_shuffle_succ': 117, 'true_shuffle_den': 169, 'true_pivotmasked_succ': 115, 'true_pivotmasked_den': 169, 'n': 169, 'true_anchor_success_rate': 0.7455621301775148, 'true_shuffle_success_rate': 0.6923076923076923, 'true_pivotmasked_success_rate': 0.6804733727810651}

  - target-length counts ALL: {'1': 1386, '2': 656, '3': 118, '4': 59, '5': 36, '6': 11}

## Trainable event density on 80M→90M segment

Events passing full-context substitution filters by category: {'temporal': 29709, 'negation': 15994, 'causal_connector': 30136, 'physical_change': 23390, 'spatial': 26949, 'comparative': 1441} total=127619

Per 256-row batch all-passing: {'n_batches': 250, 'mean': 510.476, 'median': 511.0, 'min': 461, 'p10': 483.9, 'p90': 535.0, 'max': 553, 'zero_batches': 0}

One-per-row per batch: {'n_batches': 250, 'mean': 219.392, 'median': 219.0, 'min': 202, 'p10': 213.0, 'p90': 227.0, 'max': 235, 'zero_batches': 0}

Cap up to 8/family, 48 total per batch: {'n_batches': 250, 'mean': 45.388, 'median': 45.0, 'min': 40, 'p10': 43.0, 'p90': 48.0, 'max': 48, 'zero_batches': 0}

Per-category batch stats: {'causal_connector': {'n_batches': 250, 'mean': 120.544, 'median': 121.0, 'min': 92, 'p10': 105.0, 'p90': 136.0, 'max': 160, 'zero_batches': 0}, 'comparative': {'n_batches': 250, 'mean': 5.764, 'median': 5.0, 'min': 0, 'p10': 3.0, 'p90': 10.0, 'max': 15, 'zero_batches': 1}, 'negation': {'n_batches': 250, 'mean': 63.976, 'median': 64.0, 'min': 43, 'p10': 53.0, 'p90': 76.0, 'max': 87, 'zero_batches': 0}, 'physical_change': {'n_batches': 250, 'mean': 93.56, 'median': 94.0, 'min': 68, 'p10': 81.0, 'p90': 105.0, 'max': 122, 'zero_batches': 0}, 'spatial': {'n_batches': 250, 'mean': 107.796, 'median': 107.0, 'min': 76, 'p10': 93.9, 'p90': 122.0, 'max': 134, 'zero_batches': 0}, 'temporal': {'n_batches': 250, 'mean': 118.836, 'median': 118.0, 'min': 83, 'p10': 104.0, 'p90': 135.0, 'max': 154, 'zero_batches': 0}}

Top rejects: {'anchor_token_length_mismatch': 54911, 'anchor_token_length_mismatch::negation': 21161, 'anchor_token_length_mismatch::physical_change': 21033, 'control_distance_mismatch_gt8::physical_change': 13441, 'anchor_token_length_mismatch::causal_connector': 4315, 'anchor_token_length_mismatch::temporal': 3979, 'truncated_or_missing': 3847, 'anchor_token_length_mismatch::spatial': 2832, 'control_freq_mismatch_gt2::physical_change': 2152, 'anchor_token_length_mismatch::comparative': 1591, 'control_distance_mismatch_gt8::temporal': 1523, 'control_distance_mismatch_gt8::causal_connector': 1499, 'control_distance_mismatch_gt8::negation': 1479, 'truncated_or_missing::negation': 1235, 'control_distance_mismatch_gt8::comparative': 1150, 'truncated_or_missing::physical_change': 871, 'truncated_or_missing::temporal': 732, 'control_distance_mismatch_gt8::spatial': 607, 'truncated_or_missing::causal_connector': 510, 'truncated_or_missing::spatial': 425, 'control_freq_mismatch_gt2::causal_connector': 177, 'control_freq_mismatch_gt2::comparative': 168, 'target_len_gt6::temporal': 139, 'target_len_gt6::spatial': 80, 'truncated_or_missing::comparative': 74, 'target_len_gt6::physical_change': 49, 'target_len_gt6::causal_connector': 24, 'target_len_gt6::comparative': 10, 'target_len_gt6::negation': 8}

## Files

- summary: `experiments/archive/representation_and_objectives/data/fullctx_readiness_analysis/fullctx_readiness_analysis.json`

- batch records: `experiments/archive/representation_and_objectives/data/fullctx_readiness_analysis/batch_density_records.jsonl`

