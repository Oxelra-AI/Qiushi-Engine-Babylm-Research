# earlier analysis evidence-availability readout

Status: `EVIDENCE_AVAILABILITY_READOUT_DONE`

This readout holds Qwen current-view target words fixed while changing the evidence state around the target. It is not an official BabyLM score; it is a mechanism/evidence instrument for the preservation route.

## Task construction
{
  "scope": "post_prefix",
  "total_segments_available_in_scope": 40794,
  "segments_sampled": 96,
  "max_updates": 80,
  "words_per_update": 39533,
  "prefix_word_limit": 3162640,
  "sample_seed": 90090,
  "row_idx_min": 21795,
  "row_idx_max": 89038,
  "pair_count_sampled": 96,
  "candidate_kind_counts": {
    "unchanged_inherited_current": 96
  },
  "view_kind_counts": {
    "current_inherited_control": 96
  }
}

{
  "conditions_required": [
    "pair_correct_source",
    "pair_wrong_source",
    "view_only",
    "full_row_original",
    "full_row_this_source_erased",
    "full_row_all_sources_erased"
  ],
  "candidate_tasks_before_complete_filter": 1146,
  "tasks_after_complete_filter": 1116,
  "complete_base_targets": 186,
  "condition_counts": {
    "pair_correct_source": 186,
    "pair_wrong_source": 186,
    "view_only": 186,
    "full_row_original": 186,
    "full_row_this_source_erased": 186,
    "full_row_all_sources_erased": 186
  },
  "skips_before_complete_filter": {
    "full_row_all_sources_erased_target_truncated_or_unmapped": 6
  },
  "base_targets_seen": 192,
  "dropped_base_targets_missing_condition_counts": {
    "full_row_all_sources_erased": 6
  },
  "target_string_in_context_excluding_target_fraction_by_condition": {
    "pair_correct_source": 0.5698924731182796,
    "pair_wrong_source": 0.03225806451612903,
    "view_only": 0.03225806451612903,
    "full_row_original": 0.5752688172043011,
    "full_row_this_source_erased": 0.04838709677419355,
    "full_row_all_sources_erased": 0.03225806451612903
  },
  "source_region_contains_target_string_fraction_by_condition": {
    "pair_correct_source": 0.5698924731182796,
    "pair_wrong_source": 0.0,
    "view_only": 0.0,
    "full_row_original": 0.5752688172043011,
    "full_row_this_source_erased": 0.016129032258064516,
    "full_row_all_sources_erased": 0.0
  },
  "mean_input_len_by_condition": {
    "pair_correct_source": 64.06451612903226,
    "pair_wrong_source": 62.74731182795699,
    "view_only": 33.12903225806452,
    "full_row_original": 190.93548387096774,
    "full_row_this_source_erased": 284.64516129032256,
    "full_row_all_sources_erased": 476.7096774193548
  }
}

## Model summaries
### coherent86
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.311389601374492`
- view_absence_penalty_nll_view_minus_correct: `3.448069200762898`
- this_source_erasure_cost_nll: `3.7786745516552758`
- all_sources_erasure_cost_nll: `4.0214825261896125`
- delta_vs_coherent86_specific_source_penalty_nll: `0.0`
- delta_vs_coherent86_view_absence_penalty_nll: `0.0`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.0`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.0`
  - pair_correct_source: NLL `2.377852131773661`, rank `103.2168458781362`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - pair_wrong_source: NLL `5.689241733148154`, rank `293.7428315412186`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - view_only: NLL `5.825921332536559`, rank `370.26433691756273`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_original: NLL `2.358725149291503`, rank `99.81810035842294`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_this_source_erased: NLL `6.137399700946779`, rank `549.3990143369175`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_all_sources_erased: NLL `6.380207675481114`, rank `650.9424731182796`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`

### sparse_focus_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.314281379501398`
- view_absence_penalty_nll_view_minus_correct: `3.4458827474135965`
- this_source_erasure_cost_nll: `3.7830554378615346`
- all_sources_erasure_cost_nll: `4.03509603068124`
- delta_vs_coherent86_specific_source_penalty_nll: `0.0028917781269055913`
- delta_vs_coherent86_view_absence_penalty_nll: `-0.002186453349301676`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.00438088620625859`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.013613504491628334`
  - pair_correct_source: NLL `2.2952105993486973`, rank `99.72670250896059`, ΔNLL vs coherent86 `-0.08264153242496336`, KL coherent86→model `0.010080612011869492`
  - pair_wrong_source: NLL `5.609491978850096`, rank `287.48100358422937`, ΔNLL vs coherent86 `-0.07974975429805775`, KL coherent86→model `0.009574113557497302`
  - view_only: NLL `5.741093346762295`, rank `360.0335125448029`, ΔNLL vs coherent86 `-0.08482798577426501`, KL coherent86→model `0.008486533814872299`
  - full_row_original: NLL `2.2791175207421084`, rank `95.33422939068102`, ΔNLL vs coherent86 `-0.07960762854939449`, KL coherent86→model `0.010944031079442765`
  - full_row_this_source_erased: NLL `6.062172958603642`, rank `533.2465949820788`, ΔNLL vs coherent86 `-0.0752267423431359`, KL coherent86→model `0.011691024402550468`
  - full_row_all_sources_erased: NLL `6.314213551423349`, rank `638.8960573476703`, ΔNLL vs coherent86 `-0.06599412405776614`, KL coherent86→model `0.007604942464934287`

### densemask_sparselabel_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.484638756813699`
- view_absence_penalty_nll_view_minus_correct: `3.5792792614265765`
- this_source_erasure_cost_nll: `3.996323608496324`
- all_sources_erasure_cost_nll: `4.053533821499844`
- delta_vs_coherent86_specific_source_penalty_nll: `0.17324915543920652`
- delta_vs_coherent86_view_absence_penalty_nll: `0.13121006066367855`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.21764905684104768`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.03205129531023133`
  - pair_correct_source: NLL `2.3537216657136653`, rank `108.67114695340501`, ΔNLL vs coherent86 `-0.024130466059995657`, KL coherent86→model `0.028799454606338246`
  - pair_wrong_source: NLL `5.838360422527365`, rank `342.85008960573475`, ΔNLL vs coherent86 `0.14911868937921086`, KL coherent86→model `0.118692473884775`
  - view_only: NLL `5.933000927140243`, rank `427.8165770609319`, ΔNLL vs coherent86 `0.10707959460368294`, KL coherent86→model `0.08033230725671935`
  - full_row_original: NLL `2.3514507096117976`, rank `106.3176523297491`, ΔNLL vs coherent86 `-0.007274439679704929`, KL coherent86→model `0.04171744815112921`
  - full_row_this_source_erased: NLL `6.347774318108121`, rank `614.6356630824373`, ΔNLL vs coherent86 `0.21037461716134273`, KL coherent86→model `0.12082206009432138`
  - full_row_all_sources_erased: NLL `6.404984531111642`, rank `671.1112007168459`, ΔNLL vs coherent86 `0.024776855630526406`, KL coherent86→model `0.03510144970281297`

### clean_pres_lambda1_eval_full80
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.431503121625108`
- view_absence_penalty_nll_view_minus_correct: `3.5459346507115046`
- this_source_erasure_cost_nll: `3.915726170447473`
- all_sources_erasure_cost_nll: `4.041627612691087`
- delta_vs_coherent86_specific_source_penalty_nll: `0.12011352025061524`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09786544994860644`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.13705161879219674`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.02014508650147467`
  - pair_correct_source: NLL `2.356299746593998`, rank `105.07437275985663`, ΔNLL vs coherent86 `-0.02155238517966266`, KL coherent86→model `0.010494125775916616`
  - pair_wrong_source: NLL `5.787802868219106`, rank `326.68172043010753`, ΔNLL vs coherent86 `0.09856113507095259`, KL coherent86→model `0.05981852545483632`
  - view_only: NLL `5.902234397305503`, rank `409.53315412186384`, ΔNLL vs coherent86 `0.0763130647689438`, KL coherent86→model `0.041647228556823436`
  - full_row_original: NLL `2.348665799867701`, rank `102.4323476702509`, ΔNLL vs coherent86 `-0.01005934942380182`, KL coherent86→model `0.017022741345963238`
  - full_row_this_source_erased: NLL `6.264391970315175`, rank `588.1870071684588`, ΔNLL vs coherent86 `0.12699226936839492`, KL coherent86→model `0.05980495506601628`
  - full_row_all_sources_erased: NLL `6.390293412558787`, rank `662.8284050179211`, ΔNLL vs coherent86 `0.010085737077672859`, KL coherent86→model `0.017013199556465377`

### ordinary_inherited_wwm_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.3077325835776756`
- view_absence_penalty_nll_view_minus_correct: `3.4421173908672102`
- this_source_erasure_cost_nll: `3.7754572382324048`
- all_sources_erasure_cost_nll: `4.018158897140196`
- delta_vs_coherent86_specific_source_penalty_nll: `-0.00365701779681651`
- delta_vs_coherent86_view_absence_penalty_nll: `-0.0059518098956876245`
- delta_vs_coherent86_this_source_erasure_cost_nll: `-0.003217313422871111`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `-0.0033236290494169413`
  - pair_correct_source: NLL `2.367788679894833`, rank `102.27867383512545`, ΔNLL vs coherent86 `-0.01006345187882787`, KL coherent86→model `0.0004601168846579395`
  - pair_wrong_source: NLL `5.6755212634725085`, rank `292.9652329749104`, ΔNLL vs coherent86 `-0.013720469675644375`, KL coherent86→model `0.0012100122392260226`
  - view_only: NLL `5.809906070762044`, rank `367.3705197132617`, ΔNLL vs coherent86 `-0.016015261774515482`, KL coherent86→model `0.0011551006052117026`
  - full_row_original: NLL `2.3481611144289585`, rank `98.85125448028673`, ΔNLL vs coherent86 `-0.010564034862544362`, KL coherent86→model `0.0005193757484024538`
  - full_row_this_source_erased: NLL `6.123618352661363`, rank `544.2307347670251`, ΔNLL vs coherent86 `-0.013781348285415473`, KL coherent86→model `0.001422447208202474`
  - full_row_all_sources_erased: NLL `6.3663200115691545`, rank `647.5783154121864`, ΔNLL vs coherent86 `-0.0138876639119613`, KL coherent86→model `0.0012703793371177307`

### clean_pres_lambda1_eval_seed62065
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4301825137001836`
- view_absence_penalty_nll_view_minus_correct: `3.545060218107796`
- this_source_erasure_cost_nll: `3.9136368119162857`
- all_sources_erasure_cost_nll: `4.041033931825591`
- delta_vs_coherent86_specific_source_penalty_nll: `0.11879291232569121`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09699101734489797`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.13496226026100983`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.019551405635978226`
  - pair_correct_source: NLL `2.3567944157398895`, rank `105.05913978494624`, ΔNLL vs coherent86 `-0.021057716033771524`, KL coherent86→model `0.01042239218060499`
  - pair_wrong_source: NLL `5.7869769294400735`, rank `326.681541218638`, ΔNLL vs coherent86 `0.0977351962919197`, KL coherent86→model `0.059244541792436545`
  - view_only: NLL `5.901854633847686`, rank `410.19722222222225`, ΔNLL vs coherent86 `0.07593330131112647`, KL coherent86→model `0.0413064757655726`
  - full_row_original: NLL `2.3482017402901016`, rank `102.35976702508961`, ΔNLL vs coherent86 `-0.010523409001401115`, KL coherent86→model `0.01714446258408469`
  - full_row_this_source_erased: NLL `6.261838552206387`, rank `588.2050179211469`, ΔNLL vs coherent86 `0.12443885125960874`, KL coherent86→model `0.05890880826291942`
  - full_row_all_sources_erased: NLL `6.389235672115692`, rank `662.9732974910395`, ΔNLL vs coherent86 `0.00902799663457712`, KL coherent86→model `0.01715599847563182`

## Dense-mask comparisons
{
  "densemask_sparselabel_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.17035737731230105,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.13339651401298003,
    "this_source_erasure_cost_nll_mean_delta": 0.2132681706347892,
    "all_sources_erasure_cost_nll_mean_delta": 0.018437790818603972,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 46.424641577060896,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 58.83862007168466,
    "this_source_erasure_cost_rank_mean_delta": 70.40564516129035,
    "all_sources_erasure_cost_rank_mean_delta": 21.231720430107657,
    "pair_correct_source_mean_nll_delta": 0.058511066364967945,
    "pair_correct_source_mean_rank_delta": 8.944444444444429,
    "pair_wrong_source_mean_nll_delta": 0.22886844367726855,
    "pair_wrong_source_mean_rank_delta": 55.36908602150538,
    "view_only_mean_nll_delta": 0.19190758037794797,
    "view_only_mean_rank_delta": 67.783064516129,
    "full_row_original_mean_nll_delta": 0.0723331888696892,
    "full_row_original_mean_rank_delta": 10.983422939068078,
    "full_row_this_source_erased_mean_nll_delta": 0.2856013595044793,
    "full_row_this_source_erased_mean_rank_delta": 81.38906810035849,
    "full_row_all_sources_erased_mean_nll_delta": 0.09077097968829229,
    "full_row_all_sources_erased_mean_rank_delta": 32.21514336917562
  },
  "densemask_sparselabel_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.0,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.0,
    "this_source_erasure_cost_nll_mean_delta": 0.0,
    "all_sources_erasure_cost_nll_mean_delta": 0.0,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 0.0,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 0.0,
    "this_source_erasure_cost_rank_mean_delta": 0.0,
    "all_sources_erasure_cost_rank_mean_delta": 0.0,
    "pair_correct_source_mean_nll_delta": 0.0,
    "pair_correct_source_mean_rank_delta": 0.0,
    "pair_wrong_source_mean_nll_delta": 0.0,
    "pair_wrong_source_mean_rank_delta": 0.0,
    "view_only_mean_nll_delta": 0.0,
    "view_only_mean_rank_delta": 0.0,
    "full_row_original_mean_nll_delta": 0.0,
    "full_row_original_mean_rank_delta": 0.0,
    "full_row_this_source_erased_mean_nll_delta": 0.0,
    "full_row_this_source_erased_mean_rank_delta": 0.0,
    "full_row_all_sources_erased_mean_nll_delta": 0.0,
    "full_row_all_sources_erased_mean_rank_delta": 0.0
  },
  "clean_pres_lambda1_eval_full80_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.11722174212370984,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.10005190329790814,
    "this_source_erasure_cost_nll_mean_delta": 0.1326707325859382,
    "all_sources_erasure_cost_nll_mean_delta": 0.00653158200984727,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.85304659498206,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 44.151971326164926,
    "this_source_erasure_cost_rank_mean_delta": 47.842293906810085,
    "all_sources_erasure_cost_rank_mean_delta": 16.834229390681116,
    "pair_correct_source_mean_nll_delta": 0.06108914724530079,
    "pair_correct_source_mean_rank_delta": 5.347670250896044,
    "pair_wrong_source_mean_nll_delta": 0.1783108893690093,
    "pair_wrong_source_mean_rank_delta": 39.20071684587816,
    "view_only_mean_nll_delta": 0.16114105054320849,
    "view_only_mean_rank_delta": 49.49964157706097,
    "full_row_original_mean_nll_delta": 0.06954827912559258,
    "full_row_original_mean_rank_delta": 7.0981182795698885,
    "full_row_this_source_erased_mean_nll_delta": 0.20221901171153256,
    "full_row_this_source_erased_mean_rank_delta": 54.940412186379945,
    "full_row_all_sources_erased_mean_nll_delta": 0.07607986113543763,
    "full_row_all_sources_erased_mean_rank_delta": 23.932347670250806
  },
  "clean_pres_lambda1_eval_full80_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.05313563518859121,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.03334461071507189,
    "this_source_erasure_cost_nll_mean_delta": -0.080597438048851,
    "all_sources_erasure_cost_nll_mean_delta": -0.011906208808756702,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -12.571594982078835,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -14.686648745519733,
    "this_source_erasure_cost_rank_mean_delta": -22.563351254480267,
    "all_sources_erasure_cost_rank_mean_delta": -4.397491039426541,
    "pair_correct_source_mean_nll_delta": 0.0025780808803328448,
    "pair_correct_source_mean_rank_delta": -3.5967741935483843,
    "pair_wrong_source_mean_nll_delta": -0.050557554308259256,
    "pair_wrong_source_mean_rank_delta": -16.16836917562722,
    "view_only_mean_nll_delta": -0.030766529834739487,
    "view_only_mean_rank_delta": -18.283422939068032,
    "full_row_original_mean_nll_delta": -0.0027849097440966197,
    "full_row_original_mean_rank_delta": -3.885304659498189,
    "full_row_this_source_erased_mean_nll_delta": -0.08338234779294673,
    "full_row_this_source_erased_mean_rank_delta": -26.44865591397854,
    "full_row_all_sources_erased_mean_nll_delta": -0.014691118552854654,
    "full_row_all_sources_erased_mean_rank_delta": -8.282795698924815
  },
  "ordinary_inherited_wwm_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.0065487959237224835,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.0037653565463862115,
    "this_source_erasure_cost_nll_mean_delta": -0.007598199629129887,
    "all_sources_erasure_cost_nll_mean_delta": -0.016937133541044425,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 2.932258064516077,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 4.785035842293951,
    "this_source_erasure_cost_rank_mean_delta": 7.467114695340513,
    "all_sources_erasure_cost_rank_mean_delta": 5.165232974910509,
    "pair_correct_source_mean_nll_delta": 0.07257808054613557,
    "pair_correct_source_mean_rank_delta": 2.5519713261648604,
    "pair_wrong_source_mean_nll_delta": 0.0660292846224122,
    "pair_wrong_source_mean_rank_delta": 5.484229390681037,
    "view_only_mean_nll_delta": 0.06881272399974936,
    "view_only_mean_rank_delta": 7.337007168458797,
    "full_row_original_mean_nll_delta": 0.06904359368685009,
    "full_row_original_mean_rank_delta": 3.51702508960571,
    "full_row_this_source_erased_mean_nll_delta": 0.06144539405772065,
    "full_row_this_source_erased_mean_rank_delta": 10.984139784946251,
    "full_row_all_sources_erased_mean_nll_delta": 0.05210646014580522,
    "full_row_all_sources_erased_mean_rank_delta": 8.682258064516077
  },
  "ordinary_inherited_wwm_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.17690617323602353,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.13716187055936624,
    "this_source_erasure_cost_nll_mean_delta": -0.2208663702639191,
    "all_sources_erasure_cost_nll_mean_delta": -0.035374924359648396,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -43.49238351254482,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -54.05358422939071,
    "this_source_erasure_cost_rank_mean_delta": -62.93853046594984,
    "all_sources_erasure_cost_rank_mean_delta": -16.066487455197148,
    "pair_correct_source_mean_nll_delta": 0.014067014181167625,
    "pair_correct_source_mean_rank_delta": -6.392473118279568,
    "pair_wrong_source_mean_nll_delta": -0.16283915905485635,
    "pair_wrong_source_mean_rank_delta": -49.884856630824345,
    "view_only_mean_nll_delta": -0.12309485637819861,
    "view_only_mean_rank_delta": -60.446057347670205,
    "full_row_original_mean_nll_delta": -0.0032895951828391112,
    "full_row_original_mean_rank_delta": -7.466397849462368,
    "full_row_this_source_erased_mean_nll_delta": -0.22415596544675864,
    "full_row_this_source_erased_mean_rank_delta": -70.40492831541223,
    "full_row_all_sources_erased_mean_nll_delta": -0.038664519542487064,
    "full_row_all_sources_erased_mean_rank_delta": -23.532885304659544
  },
  "clean_pres_lambda1_eval_seed62065_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.11590113419878545,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.09917747069419969,
    "this_source_erasure_cost_nll_mean_delta": 0.130581374054751,
    "all_sources_erasure_cost_nll_mean_delta": 0.005937901144350555,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.868100358422936,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 44.83127240143375,
    "this_source_erasure_cost_rank_mean_delta": 47.93288530465952,
    "all_sources_erasure_cost_rank_mean_delta": 17.051702508960602,
    "pair_correct_source_mean_nll_delta": 0.06158381639119215,
    "pair_correct_source_mean_rank_delta": 5.332437275985654,
    "pair_wrong_source_mean_nll_delta": 0.17748495058997715,
    "pair_wrong_source_mean_rank_delta": 39.20053763440865,
    "view_only_mean_nll_delta": 0.16076128708539095,
    "view_only_mean_rank_delta": 50.16370967741938,
    "full_row_original_mean_nll_delta": 0.06908421954799326,
    "full_row_original_mean_rank_delta": 7.025537634408593,
    "full_row_this_source_erased_mean_nll_delta": 0.19966559360274516,
    "full_row_this_source_erased_mean_rank_delta": 54.9584229390681,
    "full_row_all_sources_erased_mean_nll_delta": 0.07502212069234293,
    "full_row_all_sources_erased_mean_rank_delta": 24.07724014336918
  },
  "clean_pres_lambda1_eval_seed62065_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.0544562431135156,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.03421904331878034,
    "this_source_erasure_cost_nll_mean_delta": -0.08268679658003819,
    "all_sources_erasure_cost_nll_mean_delta": -0.012499889674253417,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -12.55654121863796,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -14.007347670250908,
    "this_source_erasure_cost_rank_mean_delta": -22.47275985663083,
    "all_sources_erasure_cost_rank_mean_delta": -4.180017921147055,
    "pair_correct_source_mean_nll_delta": 0.0030727500262242025,
    "pair_correct_source_mean_rank_delta": -3.6120071684587742,
    "pair_wrong_source_mean_nll_delta": -0.0513834930872914,
    "pair_wrong_source_mean_rank_delta": -16.168548387096735,
    "view_only_mean_nll_delta": -0.031146293292557026,
    "view_only_mean_rank_delta": -17.619354838709626,
    "full_row_original_mean_nll_delta": -0.003248969321695938,
    "full_row_original_mean_rank_delta": -3.9578853046594844,
    "full_row_this_source_erased_mean_nll_delta": -0.08593576590173413,
    "full_row_this_source_erased_mean_rank_delta": -26.430645161290386,
    "full_row_all_sources_erased_mean_nll_delta": -0.015748858995949355,
    "full_row_all_sources_erased_mean_rank_delta": -8.13790322580644
  }
}

## Interpretation scope
- **scope**: The readout is fixed-target Qwen pair evidence-state scoring. It is not a BabyLM leaderboard component and should be combined with broad scores before judging a candidate.
- **held_out_default**: Default scope is post_prefix, so targets are outside the 80-update Qwen prefix used by the dense-mask training arms; use scope=prefix when direct trained-material acquisition is the question.
- **source_erasure**: Source-erased full-row states blank character spans to preserve target offsets while removing lexical source evidence. Other current-view text may still contain target words, and leakage flags are recorded.
- **preservation_use**: A useful preservation endpoint should reduce view-only, wrong-source, source-erased, and CDI/broad-language damage without merely erasing correct-source/Entity movement.
