# topology 2x2 scaffold and deberta pending state pair copy-overlap audit

No training/evaluation was run. This quantifies copied-token confounding for any future topology 2×2 screen.

## all_pairs

pairs=12155, dual_pair_words=847,022, mean overlap=0.8303, median overlap=0.8400, p25=0.7500, p75=0.9091, mean noncopy=0.1697.

bins: {'overlap_lt_0p50': 29, 'overlap_0p50_0p70': 1476, 'overlap_0p70_0p85': 4926, 'overlap_ge_0p85': 5724}

## current_step179_selected

pairs=6071, dual_pair_words=423,512, mean overlap=0.8295, median overlap=0.8333, p25=0.7500, p75=0.9091, mean noncopy=0.1705.

bins: {'overlap_lt_0p50': 22, 'overlap_0p50_0p70': 744, 'overlap_0p70_0p85': 2459, 'overlap_ge_0p85': 2846}

## greedy_low_overlap_dose_subset

pairs=6020, dual_pair_words=423,586, mean overlap=0.7395, median overlap=0.7500, p25=0.6989, p75=0.8000, mean noncopy=0.2605.

bins: {'overlap_lt_0p50': 29, 'overlap_0p50_0p70': 1476, 'overlap_0p70_0p85': 4515, 'overlap_ge_0p85': 0}

## length_near70_then_low_overlap_subset

pairs=6184, dual_pair_words=423,544, mean overlap=0.8271, median overlap=0.8333, p25=0.7500, p75=0.9167, mean noncopy=0.1729.

bins: {'overlap_lt_0p50': 17, 'overlap_0p50_0p70': 812, 'overlap_0p70_0p85': 2637, 'overlap_ge_0p85': 2718}

Scientific reading: the current 2×2 subset has high lexical overlap; if independent evidence revives reciprocal topology, a lower-overlap matched subset can test non-copy semantic interaction more sharply, but it changes the data distribution and therefore should be interpreted separately.

JSON: `experiments/archive/frontier_consolidation/data/pair_copy_overlap_audit/pair_copy_overlap_audit.json`
