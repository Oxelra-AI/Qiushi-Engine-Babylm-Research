# compact aoa support and exposure localization compact-view AoA support and corpus-exposure localization

This is passive analysis of existing AoA files for the compact and inherited controls and already materialized 10M corpora. It uses official AoA words only to interpret an observed failure, not to choose future training examples.

## Compact-core full result now available
full official-compatible `compact_view_core` scored Overall 39.9877: BLiMP 67.15, Supplement 61.85, EWoK 51.26, Entity 27.85, COMPS 52.18, GlobalPIQA 35.135, SuperGLUE 68.9012, Reading 8.25, AoA -12.687.
Against COMPACT_EXPERIENCE clean-Qwen, NLP_average is essentially tied (-0.0018) but Overall is -1.3566, dominated by AoA -12.687 and SuperGLUE -1.407/GlobalPIQA -1.485 losses.

## What changed in the AoA score
Clean support: n=180, r=-0.1109, p=0.1382, official leaderboard AoA=0.000.
Compact support: n=193, r=-0.1269, p=0.07871, official leaderboard AoA=-12.687.
Same fitted words common to clean and compact: n=159. On those common words compact r=-0.0553, p=0.489; clean r=-0.0737, p=0.3561.
On common words, compact shifts fitted model-AoA earlier by mean -0.0262 log10 words, but child-AoA vs shift is r=0.0251, p=0.7539; this does not support a clean same-word developmental inversion caused by compact views.
The nonzero compact official AoA partly comes from fit-support composition: compact has 34 fitted words not fitted in clean, clean has 21 words not fitted in compact. Compact-only bins: {'late_gt25': 12, 'middle_20to25': 19, 'early_le20': 3}; clean-only bins: {'middle_20to25': 11, 'late_gt25': 8, 'early_le20': 2}.
Among compact-only fitted words, r=-0.3415, p=0.04807; this small support should be read as a sensitivity clue rather than a robust mechanism.

## Training-corpus AoA-word exposure
Core replacement block vs the held-out clean-Qwen slice: target-word occurrence delta -12318; bin deltas {'middle_20to25': -6313, 'late_gt25': -4725, 'early_le20': -1280}; per-million deltas {'early_le20': -3022.2893842085386, 'late_gt25': -11156.497922176044, 'middle_20to25': -14906.025689459766}.
Reinvest replacement block vs the same held-out slice: target-word occurrence delta -14644; bin deltas {'middle_20to25': -7462, 'late_gt25': -5634, 'early_le20': -1548}; per-million deltas {'early_le20': -3655.0812240272007, 'late_gt25': -13302.795617680385, 'middle_20to25': -17619.002644503213}.
At full 10M-pool scale, compact-core vs clean-Qwen changes AoA target occurrences by -12318 total, bin deltas {'middle_20to25': -6313, 'late_gt25': -4725, 'early_le20': -1280}; reinvest vs clean-Qwen changes -14644, bin deltas {'middle_20to25': -7462, 'late_gt25': -5634, 'early_le20': -1548}.

## Scientific reading for the next action
The compact-density intervention is not simply shifting the same AoA words into an anti-child order. On harmonized word support, compact and clean are both statistically nonzero only after thresholding, and the compact-minus-clean per-word timing shifts are not child-AoA structured. The dangerous part is that compact FineWeb replacement changes which words yield valid fitted acquisition curves and it substitutes a small but vocabulary-visible slice of the clean-Qwen developmental substrate. Because the full `compact_view_core` endpoint loses 1.36 Overall despite tied NLP_average, no new 100M compact-dose or test-shaped patch should be launched before the pending reinvest endpoint is actually seen. If reinvest inherits negative AoA, the compact FineWeb overlay should be treated as a mechanism probe rather than the SOTA route; a better repair would need to preserve the clean-Qwen AoA-neutral substrate while testing compact density inside that substrate or via a schedule/content form justified independently of the AoA word list.

Machine-readable JSON: `experiments/archive/representation_and_objectives/data/compact_aoa_localization/compact_aoa_support_and_exposure_localization.json`
