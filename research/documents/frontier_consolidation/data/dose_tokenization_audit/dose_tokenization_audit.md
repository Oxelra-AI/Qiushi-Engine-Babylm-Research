# matched max dose execution note dose tokenization audit

Fixed spatial repair route status tokenizer prevents per-dose tokenizer refit confounding; these numbers measure whether the new matched increment still changes subword burden enough to qualify the dose result.

- matched_max_all: pairs 33291, words 1118587, tokens {'n': 33291, 'mean': 48.37370460484816, 'median': 45, 'q10': 27, 'q25': 34, 'q75': 59, 'q90': 75, 'min': 16, 'max': 176, 'sum': 1610409}, weighted tokens/word 1.4397, unk 3, unique token ids 14478
- medium_unused_accepted: pairs 5839, words 204541, tokens {'n': 5839, 'mean': 50.74841582462751, 'median': 47, 'q10': 28, 'q25': 35, 'q75': 62, 'q90': 80, 'min': 17, 'max': 162, 'sum': 296320}, weighted tokens/word 1.4487, unk 2, unique token ids 11640
- old_selected_1x_prefix: pairs 12155, words 423511, tokens {'n': 12155, 'mean': 50.01217605923488, 'median': 47, 'q10': 28, 'q25': 36, 'q75': 61, 'q90': 77, 'min': 17, 'max': 165, 'sum': 607898}, weighted tokens/word 1.4354, unk 0, unique token ids 13129
- war_expansion_accepted: pairs 15297, words 490535, tokens {'n': 15297, 'mean': 46.16532653461463, 'median': 42, 'q10': 26, 'q25': 32, 'q75': 56, 'q90': 72, 'min': 16, 'max': 176, 'sum': 706191}, weighted tokens/word 1.4396, unk 1, unique token ids 13348

## Shifts vs old selected 1x prefix
- matched_max_all: tokens/word shift +0.0043 (ratio 1.0030), pair-token mean shift -1.638
- medium_unused_accepted: tokens/word shift +0.0133 (ratio 1.0093), pair-token mean shift +0.736
- war_expansion_accepted: tokens/word shift +0.0043 (ratio 1.0030), pair-token mean shift -3.847

JSON: `experiments/archive/frontier_consolidation/data/dose_tokenization_audit/dose_tokenization_audit.json`
