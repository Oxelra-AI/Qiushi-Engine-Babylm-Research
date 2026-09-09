# aoa developmental signal audit — official AoA curve-fit audit

This CPU-only check reproduces the official BabyLM AoA curve-fit logic for existing surprisal files. It preserves the unthresholded model-AoA/child-AoA Pearson correlation and p-value, which the official `aoa_score.json` collapses to 0.0 when p>0.1. It does not use AoA words as a training signal.

## Four existing models
- clean_qwen_seed43022: n_valid=180/290, r=-0.1109, p=0.1382, official_thresholded=0.0000, Overall=41.34429066479573
- clean_qwen_seed43122: n_valid=197/290, r=-0.1019, p=0.1543, official_thresholded=0.0000, Overall=40.65005195633467
- devcurr_firstpass_seed43022: n_valid=155/290, r=-0.0572, p=0.4794, official_thresholded=0.0000, Overall=40.64750216341753
- devcurr_firstpass_seed43122: n_valid=174/290, r=-0.1047, p=0.1691, official_thresholded=0.0000, Overall=40.56562101511466

## Matched seed broad-score deltas for the prior legal first-pass source order
- seed 43022: Overall 41.3443 → 40.6475 (Δ -0.6968); Entity 25.76 → 23.72; GlobalPIQA 36.620 → 34.225
- seed 43122: Overall 40.6501 → 40.5656 (Δ -0.0844); Entity 25.26 → 24.86; GlobalPIQA 34.620 → 35.635
- mean Overall across the two matched seeds: 40.9972 → 40.6066 (Δ -0.3906).

## Common-subset check
Common valid fitted-word set across all four models: 118 words. On this common set, no model has a positive significant model-AoA/child-AoA correlation; clean seed43022 is significantly negative (r=-0.1900, p=0.03933) and the others remain negative but non-significant. Thus the 0.0 official AoA result is not just caused by different valid-word sets.
- common clean_qwen_seed43022: r=-0.1900, p=0.03933, n=118
- common clean_qwen_seed43122: r=-0.1225, p=0.1864, n=118
- common devcurr_firstpass_seed43022: r=-0.1211, p=0.1916, n=118
- common devcurr_firstpass_seed43122: r=-0.0321, p=0.7298, n=118

## Interpretation
The safe first-pass source-order intervention was enough to alter early surprisal timing proxies (see `aoa_developmental_signal_audit.json`), but its official fitted model-AoA/child-AoA correlation remains small and non-significant for both seeds. Together with the broad-score decrease, this argues against spending H100 time on another pure presentation-order AoA run. It does not rule out a genuinely new objective or corpus design, but such a design must be justified without official AoA target conditioning and evaluated on the full broad task surface.

Machine-readable audit: `experiments/archive/frontier_consolidation/data/aoa_developmental_audit/official_aoa_curvefit_audit.json`
