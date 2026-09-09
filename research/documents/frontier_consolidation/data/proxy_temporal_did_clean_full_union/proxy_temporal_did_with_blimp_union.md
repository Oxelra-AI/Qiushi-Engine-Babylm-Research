# earlier analysis — proxy temporal DiD with unbiased BLiMP union

This provisional control compares delivered compact_view_reinvest sparse seed gaps against existing COMPACT_EXPERIENCE clean-Qwen full-report seed gaps at 10/40/100M. It is weaker than the exact managed clean sparse control, but it corrects the post-hoc BLiMP selected-group artifact by adding the 18-file union.

## Main proxy excess gaps (reinvest seed gap − clean seed gap)
### 10M
- blimp_union: reinvest_gap=+1.944, clean_proxy_gap=-5.281, proxy_excess=+7.226
- supplement_all: reinvest_gap=-3.200, clean_proxy_gap=+0.324, proxy_excess=-3.524
- ewok_all_fast: reinvest_gap=+0.000, clean_proxy_gap=+2.476, proxy_excess=-2.476
- entity_full: reinvest_gap=-1.334, clean_proxy_gap=+1.171, proxy_excess=-2.505
- blimp_worst_selected: reinvest_gap=+2.150, clean_proxy_gap=-7.045, proxy_excess=+9.195
- blimp_control_selected: reinvest_gap=+1.688, clean_proxy_gap=-3.076, proxy_excess=+4.764
### 40M
- blimp_union: reinvest_gap=-0.250, clean_proxy_gap=+2.353, proxy_excess=-2.603
- supplement_all: reinvest_gap=+2.400, clean_proxy_gap=-3.332, proxy_excess=+5.732
- ewok_all_fast: reinvest_gap=-2.455, clean_proxy_gap=-1.398, proxy_excess=-1.056
- entity_full: reinvest_gap=-2.903, clean_proxy_gap=-4.400, proxy_excess=+1.498
- blimp_worst_selected: reinvest_gap=-1.300, clean_proxy_gap=+1.758, proxy_excess=-3.058
- blimp_control_selected: reinvest_gap=+1.062, clean_proxy_gap=+3.097, proxy_excess=-2.035
### 100M
- blimp_union: reinvest_gap=-3.333, clean_proxy_gap=-1.519, proxy_excess=-1.814
- supplement_all: reinvest_gap=-3.200, clean_proxy_gap=-1.334, proxy_excess=-1.866
- ewok_all_fast: reinvest_gap=-3.727, clean_proxy_gap=+0.235, proxy_excess=-3.962
- entity_full: reinvest_gap=-1.460, clean_proxy_gap=-0.501, proxy_excess=-0.959
- blimp_worst_selected: reinvest_gap=-14.050, clean_proxy_gap=-5.479, proxy_excess=-8.571
- blimp_control_selected: reinvest_gap=+10.062, clean_proxy_gap=+3.431, proxy_excess=+6.631

## Group summary
- blimp_union: final_100M_excess=-1.814, mean_excess=+0.936, late_excess_change_40_to_100M=+0.789
- supplement_all: final_100M_excess=-1.866, mean_excess=+0.114, late_excess_change_40_to_100M=-7.598
- ewok_all_fast: final_100M_excess=-3.962, mean_excess=-2.498, late_excess_change_40_to_100M=-2.905
- entity_full: final_100M_excess=-0.959, mean_excess=-0.655, late_excess_change_40_to_100M=-2.456
- blimp_worst_selected: final_100M_excess=-8.571, mean_excess=-0.811, late_excess_change_40_to_100M=-5.513
- blimp_control_selected: final_100M_excess=+6.631, mean_excess=+3.120, late_excess_change_40_to_100M=+8.666

## Interpretive notes
- BLiMP: Use blimp_union for the main BLiMP read. The selected worst/control groups are shown only to expose the post-hoc redistribution artifact.
- EWoK: Uses fast temporal EWoK domain scores. Full official 7618-row EWoK at 100M has a smaller seed gap (-1.6449) and positive seed43122 treatment effect (+1.4617), so do not use this proxy for final EWoK arithmetic.
- replacement: Replace this proxy with the exact clean full trajectory seed control clean sparse DiD when the clean sparse control becomes available.

Machine-readable output: `experiments/archive/frontier_consolidation/data/proxy_temporal_did_clean_full_union/proxy_temporal_did_with_blimp_union.json`
