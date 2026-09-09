# full core route localization and existing ladder plan harmonized AoA support

This CPU note uses fitted per-word model AoAs from `analyze_aoa_from_surprisals.py`. It checks same-word support, fit-entry/fit-exit composition, word influence, and bootstrap uncertainty so small raw AoA changes are not overinterpreted.

Fit-done models: clean_qwen_seed43022, clean_qwen_seed43122, density_compact_view_core
Common fitted-word set across fit-done models: 145 words.

## Same-word correlations
- clean_qwen_seed43022: r=-0.10765629135517821, p=0.19743886561090462, Spearman=-0.14337899543378993, late-minus-early=-0.07025998918092657
- clean_qwen_seed43122: r=-0.07312264118867799, p=0.38208386407845035, Spearman=-0.09533538025507794, late-minus-early=-0.08273099268954365
- density_compact_view_core: r=-0.09125056554178912, p=0.27502161588802815, Spearman=-0.09355613289245789, late-minus-early=-0.08086092285390922

## Pairwise comparisons to clean_qwen_seed43022
- clean_qwen_seed43122_minus_clean_qwen_seed43022: common=155, mean_delta=-0.011535125366802436, child-vs-delta r=0.0840443851853308, p=0.2984722445737905; only_anchor=25, only_other=42
- density_compact_view_core_minus_clean_qwen_seed43022: common=159, mean_delta=-0.02615623054673015, child-vs-delta r=0.025054611859353825, p=0.7539136392089772; only_anchor=21, only_other=34

JSON: `experiments/archive/frontier_consolidation/data/harmonized_aoa_support/harmonized_aoa_support.json`
