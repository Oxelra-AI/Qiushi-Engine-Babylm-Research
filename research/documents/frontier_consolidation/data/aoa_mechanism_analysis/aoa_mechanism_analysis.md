# full core route localization and existing ladder plan AoA mechanism analysis from existing surprisals

This analysis recomputes unthresholded fitted model-AoA from existing official AoA surprisal files. It is passive analysis of already-trained checkpoint ladders; it does not use AoA words to build or train any model.

## Available model fits
- clean_qwen_seed43022: n=180/290, r=-0.1109, p=0.1382, leaderboard_AoA=0.000, late-minus-early model AoA=-0.13552980682069027
- clean_qwen_seed43122: n=197/290, r=-0.1019, p=0.1543, leaderboard_AoA=0.000, late-minus-early model AoA=-0.09795309699752686
- density_compact_view_core: n=193/290, r=-0.1269, p=0.07871, leaderboard_AoA=-12.687, late-minus-early model AoA=-0.1506335281005553

## Compact-view core relative to inherited clean-Qwen
Common fitted words: 159. Mean delta in fitted log10 word exposure (compact minus clean): -0.0262.
Correlation of child AoA with compact-minus-clean delta: r=0.0251, p=0.7539. Negative here means compact views advance human-late words relative to human-early words more than clean-Qwen.
- late_gt25: n=72, mean_delta=-0.016119759393185502
- middle_20to25: n=77, mean_delta=-0.03725415788676758
- early_le20: n=10, mean_delta=-0.01296478233396341
Largest advances of compact vs clean: tongue(-0.939), wash(-0.813), look(-0.800), picture(-0.631), bus(-0.552), beach(-0.506), see(-0.452), tired(-0.419)
Largest delays of compact vs clean: fast(0.714), chair(0.703), stand(0.658), blow(0.592), school(0.576), cake(0.557), flag(0.500), dinner(0.349)

Machine-readable JSON: `experiments/archive/frontier_consolidation/data/aoa_mechanism_analysis/aoa_mechanism_analysis.json`
