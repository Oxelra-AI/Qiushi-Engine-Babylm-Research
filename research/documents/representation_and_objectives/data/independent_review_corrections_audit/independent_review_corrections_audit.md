# corrected multiseed gauge affine and next raw binding independent_review correction audits

## No-anchor filter
Arm state rows removed by causal gauge experimental design no-bridge filter: 128 / arm rows 320; all removed rows direct changed h0/h2 = False. Relation counts: {'h0_dax': 64, 'h2_norp': 64}.

## tied_seed29002_underfit_original
train_cmp=0.875 n_wrong=24/192 prob_wrong_range=[0.1250266432762146, 0.8750851154327393] wrong_near_clamp_low/high=0/0 margin_wrong_range=[-1.9466885745002886, -1.088798243356924] comparison_bce_mean=0.3024 wrong_mean=1.6266
Wrong relation pairs: [{"count": 24, "relation1": "h2_norp", "relation2": "h3_ziv"}]

## tied_seed29002_underfit_repair
train_cmp=0.875 n_wrong=24/192 prob_wrong_range=[0.12526027858257294, 0.8747266530990601] wrong_near_clamp_low/high=0/0 margin_wrong_range=[-1.9435325797406338, -1.0918195102406498] comparison_bce_mean=0.3024 wrong_mean=1.6254
Wrong relation pairs: [{"count": 24, "relation1": "h2_norp", "relation2": "h3_ziv"}]

## Paired d_minus = alpha + beta d_plus regressions
| run | category | n | beta | alpha | corr | corr(-) | rmse | opp_frac |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| shared_trunk|seed29000|primary_gauge | direct_changed_psc | 128 | -1.099 | 1.838 | -0.9989630977392403 | 0.9989630977392403 | 0.648 | 1.000 |
| shared_trunk|seed29000|primary_gauge | graph_changed_psc | 128 | -1.132 | 2.794 | -0.9945484208997318 | 0.9945484208997318 | 1.469 | 1.000 |
| shared_trunk|seed29000|primary_gauge | graph_changed_same_psc | 64 | -1.132 | 2.794 | -0.9945484208997353 | 0.9945484208997353 | 1.469 | 1.000 |
| shared_trunk|seed29000|primary_gauge | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| shared_trunk|seed29001|primary_gauge_seed29001 | direct_changed_psc | 128 | -1.002 | -0.648 | -0.9994087651308853 | 0.9994087651308853 | 0.497 | 1.000 |
| shared_trunk|seed29001|primary_gauge_seed29001 | graph_changed_psc | 128 | -0.977 | -1.405 | -0.999345324774284 | 0.999345324774284 | 0.503 | 1.000 |
| shared_trunk|seed29001|primary_gauge_seed29001 | graph_changed_same_psc | 64 | -0.977 | -1.405 | -0.9993453247742833 | 0.9993453247742833 | 0.503 | 1.000 |
| shared_trunk|seed29001|primary_gauge_seed29001 | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| shared_trunk|seed29002|primary_gauge_seed29002 | direct_changed_psc | 128 | -0.900 | -0.889 | -0.9985996284971208 | 0.9985996284971208 | 0.727 | 1.000 |
| shared_trunk|seed29002|primary_gauge_seed29002 | graph_changed_psc | 128 | -0.827 | -0.723 | -0.9994229456277837 | 0.9994229456277837 | 0.430 | 1.000 |
| shared_trunk|seed29002|primary_gauge_seed29002 | graph_changed_same_psc | 64 | -0.827 | -0.723 | -0.999422945627783 | 0.999422945627783 | 0.430 | 1.000 |
| shared_trunk|seed29002|primary_gauge_seed29002 | unchanged_psc | 256 | 1.000 | -0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| tied|seed29000|primary_gauge | direct_changed_psc | 128 | -0.867 | 0.367 | -0.9811649459123528 | 0.9811649459123528 | 2.878 | 1.000 |
| tied|seed29000|primary_gauge | graph_changed_psc | 128 | -0.817 | -0.484 | -0.9703438283533821 | 0.9703438283533821 | 3.321 | 1.000 |
| tied|seed29000|primary_gauge | graph_changed_same_psc | 64 | -0.817 | -0.484 | -0.9703438283533812 | 0.9703438283533812 | 3.321 | 1.000 |
| tied|seed29000|primary_gauge | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| tied|seed29001|primary_gauge_seed29001 | direct_changed_psc | 128 | -1.042 | -1.045 | -0.9994214737709533 | 0.9994214737709533 | 0.561 | 1.000 |
| tied|seed29001|primary_gauge_seed29001 | graph_changed_psc | 128 | -1.054 | -1.418 | -0.9992829633227862 | 0.9992829633227862 | 0.608 | 1.000 |
| tied|seed29001|primary_gauge_seed29001 | graph_changed_same_psc | 64 | -1.054 | -1.418 | -0.999282963322787 | 0.999282963322787 | 0.608 | 1.000 |
| tied|seed29001|primary_gauge_seed29001 | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| tied|seed29002|primary_gauge_seed29002 | direct_changed_psc | 128 | -0.430 | 0.269 | -0.8291470680727756 | 0.8291470680727756 | 5.183 | 1.000 |
| tied|seed29002|primary_gauge_seed29002 | graph_changed_psc | 128 | -0.338 | 6.385 | -0.5060489736517866 | 0.5060489736517866 | 10.052 | 0.750 |
| tied|seed29002|primary_gauge_seed29002 | graph_changed_same_psc | 64 | -0.338 | 6.385 | -0.5060489736517868 | 0.5060489736517868 | 10.052 | 0.750 |
| tied|seed29002|primary_gauge_seed29002 | unchanged_psc | 256 | 1.000 | -0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| untied|seed29000|primary_gauge | direct_changed_psc | 128 | -0.917 | 0.023 | -0.9935932633589014 | 0.9935932633589014 | 1.410 | 1.000 |
| untied|seed29000|primary_gauge | graph_changed_psc | 128 | 0.301 | 5.863 | 0.46775025044744517 | -0.46775025044744517 | 3.706 | 0.250 |
| untied|seed29000|primary_gauge | graph_changed_same_psc | 64 | 0.301 | 5.863 | 0.4677502504474442 | -0.4677502504474442 | 3.706 | 0.250 |
| untied|seed29000|primary_gauge | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| untied|seed29001|primary_gauge_seed29001 | direct_changed_psc | 128 | -0.983 | -0.656 | -0.9982571378523656 | 0.9982571378523656 | 0.799 | 1.000 |
| untied|seed29001|primary_gauge_seed29001 | graph_changed_psc | 128 | 0.729 | -2.230 | 0.7621632322962818 | -0.7621632322962818 | 3.228 | 0.250 |
| untied|seed29001|primary_gauge_seed29001 | graph_changed_same_psc | 64 | 0.729 | -2.230 | 0.7621632322962812 | -0.7621632322962812 | 3.228 | 0.250 |
| untied|seed29001|primary_gauge_seed29001 | unchanged_psc | 256 | 1.000 | 0.000 | 1.0 | -1.0 | 0.000 | 0.000 |
| untied|seed29002|primary_gauge_seed29002 | direct_changed_psc | 128 | -0.966 | 0.145 | -0.9978130444379005 | 0.9978130444379005 | 0.880 | 1.000 |
| untied|seed29002|primary_gauge_seed29002 | graph_changed_psc | 128 | -0.389 | 5.041 | -0.36555612917608044 | 0.36555612917608044 | 8.816 | 0.750 |
| untied|seed29002|primary_gauge_seed29002 | graph_changed_same_psc | 64 | -0.389 | 5.041 | -0.36555612917608105 | 0.36555612917608105 | 8.816 | 0.750 |
| untied|seed29002|primary_gauge_seed29002 | unchanged_psc | 256 | 1.000 | -0.000 | 1.0 | -1.0 | 0.000 | 0.000 |

## Frozen-affine relation split (shared factorization result synthesis heldheld-only)
### fit_h0_only: n=64 a=-0.0933 b=-0.0554 calib_acc=1.000 min_margin=1.000
- direct_h0_psc: n=64 acc=1.0 margin=0.9999999999999996
- direct_h2_psc: n=64 acc=1.0 margin=1.1057741547936404
- graph_h1_psc: n=64 acc=1.0 margin=1.1489330397883901
- graph_h3_psc: n=64 acc=1.0 margin=1.0345237657483786
- graph_all_psc: n=128 acc=1.0 margin=1.0917284027683842
- graph_all_xt: n=64 acc=1.0 margin=1.0917284027683842

### fit_h2_only: n=64 a=-0.0844 b=-0.0881 calib_acc=1.000 min_margin=1.000
- direct_h0_psc: n=64 acc=1.0 margin=0.9043437990161914
- direct_h2_psc: n=64 acc=1.0 margin=0.9999999999999993
- graph_h1_psc: n=64 acc=1.0 margin=1.0390304700174542
- graph_h3_psc: n=64 acc=1.0 margin=0.9355651524894255
- graph_all_psc: n=128 acc=1.0 margin=0.9872978112534397
- graph_all_xt: n=64 acc=1.0 margin=0.9872978112534397

### fit_h0_h2: n=128 a=-0.0884 b=-0.0724 calib_acc=1.000 min_margin=0.927
- direct_h0_psc: n=64 acc=1.0 margin=0.9470046428074734
- direct_h2_psc: n=64 acc=1.0 margin=1.0471732584860876
- graph_h1_psc: n=64 acc=1.0 margin=1.0880449229545093
- graph_h3_psc: n=64 acc=1.0 margin=0.9796988092583859
- graph_all_psc: n=128 acc=1.0 margin=1.0338718661064477
- graph_all_xt: n=64 acc=1.0 margin=1.0338718661064477

## Scientific reading
The no-anchor filter is exact for this substrate if all removed arm state rows are h0/h2 changed anchors. The tied seed29002 bs-1 underfit is not explained by rows stuck at the probability clamp; wrong train comparisons remain away from clamp and persist under the tested longer/lower-LR schedule, so it should be described as a persistent optimization/local-solution issue under tested schedules, not an intrinsic impossibility. Regression slopes near -1 with low residuals support approximate sign-gauge reversal; deviations and the tied underfit exception should remain visible. The h0-only/h2-only affine split tests whether scalar calibration generalizes across direct relations as well as to graph relations.
