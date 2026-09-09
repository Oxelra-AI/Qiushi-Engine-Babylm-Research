# adjusted source use residual adjusted source-use probe analysis

CPU-only analysis of hypothesis comparison frozen chck82 source-use records. This is a mechanism-sharpening analysis, not training evidence.

## Family composition
| family | n | absent frac | unique frac | tail-only frac | contentlike frac | complete-BPE-copy frac | raw mean I_f | copied mean I_f |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 14999 | 0.217 | 0.646 | 0.307 | 0.634 | 0.647 | 2.748 | 3.257 |
| prefix_fluent_vs_prefix_scrambled | 14999 | 0.005 | 0.728 | 0.000 | 0.457 | 0.994 | 3.465 | 3.451 |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 14997 | 0.005 | 0.799 | 0.412 | 0.556 | 0.994 | 3.595 | 3.582 |

## Reweighted copied-target contrasts
### copied_main_pooled_common
Retained strata 112/499; retained fractions: compact_vs_compact_scrambled=0.515, prefix_fluent_vs_prefix_scrambled=0.937, sourcewide_onegap_vs_sourcewide_onegap_scrambled=0.542
Weighted means: compact_vs_compact_scrambled=2.838, prefix_fluent_vs_prefix_scrambled=3.462, sourcewide_onegap_vs_sourcewide_onegap_scrambled=3.203
Contrasts: compact_minus_prefix=-0.624, compact_minus_onegap=-0.365, compact_minus_mean_extracts=-0.495, onegap_minus_prefix=-0.259

### copied_main_compact_distribution
Retained strata 112/499; retained fractions: compact_vs_compact_scrambled=0.515, prefix_fluent_vs_prefix_scrambled=0.937, sourcewide_onegap_vs_sourcewide_onegap_scrambled=0.542
Weighted means: compact_vs_compact_scrambled=3.023, prefix_fluent_vs_prefix_scrambled=3.569, sourcewide_onegap_vs_sourcewide_onegap_scrambled=3.330
Contrasts: compact_minus_prefix=-0.546, compact_minus_onegap=-0.307, compact_minus_mean_extracts=-0.427, onegap_minus_prefix=-0.239

### copied_coarser_pooled_common
Retained strata 25/59; retained fractions: compact_vs_compact_scrambled=0.604, prefix_fluent_vs_prefix_scrambled=0.995, sourcewide_onegap_vs_sourcewide_onegap_scrambled=0.583
Weighted means: compact_vs_compact_scrambled=2.814, prefix_fluent_vs_prefix_scrambled=3.542, sourcewide_onegap_vs_sourcewide_onegap_scrambled=3.218
Contrasts: compact_minus_prefix=-0.728, compact_minus_onegap=-0.403, compact_minus_mean_extracts=-0.566, onegap_minus_prefix=-0.325

### copied_no_gap_pooled_common
Retained strata 11/20; retained fractions: compact_vs_compact_scrambled=0.608, prefix_fluent_vs_prefix_scrambled=1.000, sourcewide_onegap_vs_sourcewide_onegap_scrambled=0.586
Weighted means: compact_vs_compact_scrambled=2.818, prefix_fluent_vs_prefix_scrambled=3.550, sourcewide_onegap_vs_sourcewide_onegap_scrambled=3.223
Contrasts: compact_minus_prefix=-0.732, compact_minus_onegap=-0.405, compact_minus_mean_extracts=-0.568, onegap_minus_prefix=-0.327

## Bootstrap intervals for main copied reweighting
### copied_main_pooled_common (ok 0/0)
### copied_coarser_pooled_common (ok 0/0)

## Pair-matched deltas over common pair IDs
### all_targets: common pairs 181
- compact_minus_prefix: mean -0.882, 95% quantile [-6.014, 3.483], p>0 0.348
- compact_minus_onegap: mean -1.160, 95% quantile [-5.748, 3.169], p>0 0.304
- compact_minus_mean_extracts: mean -1.021, 95% quantile [-5.417, 2.837], p>0 0.293
- onegap_minus_prefix: mean 0.278, 95% quantile [-4.133, 5.097], p>0 0.552
### copied_targets: common pairs 181
- compact_minus_prefix: mean -0.432, 95% quantile [-5.861, 4.934], p>0 0.420
- compact_minus_onegap: mean -0.704, 95% quantile [-6.124, 4.010], p>0 0.365
- compact_minus_mean_extracts: mean -0.568, 95% quantile [-5.180, 3.928], p>0 0.392
- onegap_minus_prefix: mean 0.272, 95% quantile [-4.133, 5.097], p>0 0.552
### unique_copied_content: common pairs 160
- compact_minus_prefix: mean -0.693, 95% quantile [-9.850, 7.839], p>0 0.412
- compact_minus_onegap: mean -0.953, 95% quantile [-9.284, 6.129], p>0 0.419
- compact_minus_mean_extracts: mean -0.823, 95% quantile [-9.448, 5.828], p>0 0.419
- onegap_minus_prefix: mean 0.260, 95% quantile [-8.300, 7.403], p>0 0.531
### tail_only_copied: common pairs 0
### prefix_only_copied: common pairs 177
- compact_minus_prefix: mean -0.660, 95% quantile [-7.785, 7.704], p>0 0.429
- compact_minus_onegap: mean -0.876, 95% quantile [-9.198, 6.777], p>0 0.418
- compact_minus_mean_extracts: mean -0.768, 95% quantile [-7.376, 6.387], p>0 0.412
- onegap_minus_prefix: mean 0.216, 95% quantile [-7.395, 7.541], p>0 0.542

## Compact source-absent targets
Compact absent-only: n_targets=3251.000, n_pairs=2624.000, mean_I_f=0.908, median_I_f=0.651, positive_frac=0.613, q10=-2.329, q90=4.449, mean_source_effect_ord=0.872, mean_source_effect_scr=-0.036, mean_nll_ord_s=5.025, mean_nll_scr_s=9.324, mean_nll_ord=5.898, mean_nll_scr=9.288

## Interpretation
- Raw compact mean I_f is lower than prefix/onegap, but compact has a large source-absent target share; raw family means mostly mix retrieval-opportunity composition with family effects.
- On copied targets with common-support reweighting over copy/multiplicity/lexical/source-position/gap bins, compact-minus-mean-extractive I_f is -0.495 nats; this measures whether compact has a residual ordered-source interaction after copy-opportunity adjustment.
- Compact source-absent targets retain positive but much weaker ordering interaction (mean I_f 0.908 nats) than copied compact targets; this supports weak semantic/context association, not strong direct retrieval.
- This analysis remains a frozen reconstruction probe; it can sharpen predictions for a boundary-blocked MLM visibility intervention but cannot settle which training signal caused downstream Supplement/EWoK transitions.

JSON: `experiments/archive/representation_and_objectives/data/adjusted_source_use_probe/adjusted_source_use_probe.json`
