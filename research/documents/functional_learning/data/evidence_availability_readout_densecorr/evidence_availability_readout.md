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
- specific_source_penalty_nll_wrong_minus_correct: `3.3113877406747516`
- view_absence_penalty_nll_view_minus_correct: `3.4480675727786374`
- this_source_erasure_cost_nll: `3.7786730845685583`
- all_sources_erasure_cost_nll: `4.021480251246655`
- delta_vs_coherent86_specific_source_penalty_nll: `0.0`
- delta_vs_coherent86_view_absence_penalty_nll: `0.0`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.0`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.0`
  - pair_correct_source: NLL `2.3778548641841546`, rank `103.2168458781362`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - pair_wrong_source: NLL `5.689242604858906`, rank `293.7428315412186`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - view_only: NLL `5.8259224369627916`, rank `370.26433691756273`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_original: NLL `2.3587278908684617`, rank `99.81810035842294`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_this_source_erased: NLL `6.1374009754370205`, rank `549.3990143369175`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_all_sources_erased: NLL `6.3802081421151176`, rank `650.9424731182796`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`

### sparse_focus_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.3142795449556264`
- view_absence_penalty_nll_view_minus_correct: `3.445880949361984`
- this_source_erasure_cost_nll: `3.783053623624505`
- all_sources_erasure_cost_nll: `4.035093658383206`
- delta_vs_coherent86_specific_source_penalty_nll: `0.002891804280875081`
- delta_vs_coherent86_view_absence_penalty_nll: `-0.0021866234166534613`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.004380539055946217`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.013613407136551105`
  - pair_correct_source: NLL `2.2952131888005134`, rank `99.72670250896059`, ΔNLL vs coherent86 `-0.08264167538364096`, KL coherent86→model `0.010080508745958743`
  - pair_wrong_source: NLL `5.60949273375614`, rank `287.48100358422937`, ΔNLL vs coherent86 `-0.07974987110276589`, KL coherent86→model `0.009574070054228568`
  - view_only: NLL `5.741094138162498`, rank `360.0335125448029`, ΔNLL vs coherent86 `-0.08482829880029444`, KL coherent86→model `0.008486461888139037`
  - full_row_original: NLL `2.279120416163699`, rank `95.33422939068102`, ΔNLL vs coherent86 `-0.0796074747047629`, KL coherent86→model `0.010943972742012928`
  - full_row_this_source_erased: NLL `6.0621740397882045`, rank `533.2465949820788`, ΔNLL vs coherent86 `-0.07522693564881669`, KL coherent86→model `0.011690901628967482`
  - full_row_all_sources_erased: NLL `6.314214074546905`, rank `638.8960573476703`, ΔNLL vs coherent86 `-0.06599406756821179`, KL coherent86→model `0.007604908962672706`

### densemask_sparselabel_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4846370784202376`
- view_absence_penalty_nll_view_minus_correct: `3.579277473642102`
- this_source_erasure_cost_nll: `3.996321542081029`
- all_sources_erasure_cost_nll: `4.053531531544265`
- delta_vs_coherent86_specific_source_penalty_nll: `0.17324933774548576`
- delta_vs_coherent86_view_absence_penalty_nll: `0.13120990086346448`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.2176484575124702`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.032051280297609785`
  - pair_correct_source: NLL `2.353724371516321`, rank `108.67114695340501`, ΔNLL vs coherent86 `-0.02413049266783364`, KL coherent86→model `0.02879933990997094`
  - pair_wrong_source: NLL `5.838361449936558`, rank `342.85008960573475`, ΔNLL vs coherent86 `0.1491188450776521`, KL coherent86→model `0.11869238856132037`
  - view_only: NLL `5.933001845158423`, rank `427.8192652329749`, ΔNLL vs coherent86 `0.10707940819563086`, KL coherent86→model `0.08033225240962898`
  - full_row_original: NLL `2.3514535410289747`, rank `106.3176523297491`, ΔNLL vs coherent86 `-0.00727434983948707`, KL coherent86→model `0.04171747311114631`
  - full_row_this_source_erased: NLL `6.347775083110004`, rank `614.6356630824373`, ΔNLL vs coherent86 `0.2103741076729831`, KL coherent86→model `0.12082205326176428`
  - full_row_all_sources_erased: NLL `6.404985072573239`, rank `671.1112007168459`, ΔNLL vs coherent86 `0.024776930458122716`, KL coherent86→model `0.03510153154157416`

### ordinary_inherited_wwm_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.307730804124068`
- view_absence_penalty_nll_view_minus_correct: `3.442115456472904`
- this_source_erasure_cost_nll: `3.7754557649571807`
- all_sources_erasure_cost_nll: `4.018156971060775`
- delta_vs_coherent86_specific_source_penalty_nll: `-0.0036569365506833535`
- delta_vs_coherent86_view_absence_penalty_nll: `-0.005952116305733104`
- delta_vs_coherent86_this_source_erasure_cost_nll: `-0.0032173196113782034`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `-0.003323280185879977`
  - pair_correct_source: NLL `2.367791535740362`, rank `102.27867383512545`, ΔNLL vs coherent86 `-0.010063328443792537`, KL coherent86→model `0.00046005976440940813`
  - pair_wrong_source: NLL `5.675522339864431`, rank `292.9652329749104`, ΔNLL vs coherent86 `-0.013720264994475902`, KL coherent86→model `0.001210022835923846`
  - view_only: NLL `5.809906992213266`, rank `367.3705197132617`, ΔNLL vs coherent86 `-0.01601544474952564`, KL coherent86→model `0.001155083439356218`
  - full_row_original: NLL `2.348163875450082`, rank `98.85125448028673`, ΔNLL vs coherent86 `-0.010564015418379548`, KL coherent86→model `0.0005193541733011299`
  - full_row_this_source_erased: NLL `6.123619640407262`, rank `544.2307347670251`, ΔNLL vs coherent86 `-0.013781335029757745`, KL coherent86→model `0.0014224572791031555`
  - full_row_all_sources_erased: NLL `6.366320846510858`, rank `647.5783154121864`, ΔNLL vs coherent86 `-0.013887295604259515`, KL coherent86→model `0.0012703937951952726`

### clean_pres_lambda1_eval_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4315018450556525`
- view_absence_penalty_nll_view_minus_correct: `3.545933207144396`
- this_source_erasure_cost_nll: `3.9157242180763303`
- all_sources_erasure_cost_nll: `4.041625655819386`
- delta_vs_coherent86_specific_source_penalty_nll: `0.12011410438090121`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09786563436575836`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.1370511335077717`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.020145404572731326`
  - pair_correct_source: NLL `2.35630233882366`, rank `105.07437275985663`, ΔNLL vs coherent86 `-0.021552525360494398`, KL coherent86→model `0.010494012901637354`
  - pair_wrong_source: NLL `5.787804183879313`, rank `326.68172043010753`, ΔNLL vs coherent86 `0.09856157902040681`, KL coherent86→model `0.05981852872782805`
  - view_only: NLL `5.902235545968057`, rank `409.53315412186384`, ΔNLL vs coherent86 `0.07631310900526396`, KL coherent86→model `0.04164716328749718`
  - full_row_original: NLL `2.3486686836243407`, rank `102.4323476702509`, ΔNLL vs coherent86 `-0.010059207244121085`, KL coherent86→model `0.01702273354371739`
  - full_row_this_source_erased: NLL `6.264392901700671`, rank `588.1870071684588`, ΔNLL vs coherent86 `0.12699192626365063`, KL coherent86→model `0.059805014870752495`
  - full_row_all_sources_erased: NLL `6.390294339443727`, rank `662.8284050179211`, ΔNLL vs coherent86 `0.010086197328610253`, KL coherent86→model `0.01701330379718551`

### clean_pres_lambda1_eval_seed62065
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4301807612686637`
- view_absence_penalty_nll_view_minus_correct: `3.5450581214534442`
- this_source_erasure_cost_nll: `3.9136346923542047`
- all_sources_erasure_cost_nll: `4.041031478660925`
- delta_vs_coherent86_specific_source_penalty_nll: `0.11879302059391206`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09699054867480664`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.13496160778564595`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.01955122741427036`
  - pair_correct_source: NLL `2.356797339831455`, rank `105.05913978494624`, ΔNLL vs coherent86 `-0.021057524352699957`, KL coherent86→model `0.010422339915145394`
  - pair_wrong_source: NLL `5.786978101100119`, rank `326.681541218638`, ΔNLL vs coherent86 `0.09773549624121211`, KL coherent86→model `0.05924451601070923`
  - view_only: NLL `5.9018554612848995`, rank `410.19722222222225`, ΔNLL vs coherent86 `0.07593302432210669`, KL coherent86→model `0.04130644739540707`
  - full_row_original: NLL `2.348204811948425`, rank `102.35976702508961`, ΔNLL vs coherent86 `-0.010523078920036767`, KL coherent86→model `0.0171444353891643`
  - full_row_this_source_erased: NLL `6.26183950430263`, rank `588.1996415770609`, ΔNLL vs coherent86 `0.12443852886560919`, KL coherent86→model `0.05890874209425253`
  - full_row_all_sources_erased: NLL `6.38923629060935`, rank `662.9732974910395`, ΔNLL vs coherent86 `0.009028148494233598`, KL coherent86→model `0.017156118852725607`

### densecorr_pres_lambda1_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.3169012988801687`
- view_absence_penalty_nll_view_minus_correct: `3.448970451717461`
- this_source_erasure_cost_nll: `3.7836901292264424`
- all_sources_erasure_cost_nll: `4.02174124936351`
- delta_vs_coherent86_specific_source_penalty_nll: `0.005513558205417216`
- delta_vs_coherent86_view_absence_penalty_nll: `0.0009028789388234259`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.005017044657883457`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.0002609981168549717`
  - pair_correct_source: NLL `2.3552696451935993`, rank `102.7706093189964`, ΔNLL vs coherent86 `-0.02258521899055522`, KL coherent86→model `0.0014668183678057097`
  - pair_wrong_source: NLL `5.672170944073768`, rank `296.2123655913978`, ΔNLL vs coherent86 `-0.017071660785138026`, KL coherent86→model `0.002446631059615112`
  - view_only: NLL `5.80424009691106`, rank `370.78010752688175`, ΔNLL vs coherent86 `-0.021682340051731797`, KL coherent86→model `0.0022601407155390563`
  - full_row_original: NLL `2.335292938796315`, rank `99.0931899641577`, ΔNLL vs coherent86 `-0.02343495207214702`, KL coherent86→model `0.0015660165388852122`
  - full_row_this_source_erased: NLL `6.118983068022756`, rank `548.473476702509`, ΔNLL vs coherent86 `-0.01841790741426357`, KL coherent86→model `0.003302487965284662`
  - full_row_all_sources_erased: NLL `6.357034188159825`, rank `649.2709677419355`, ΔNLL vs coherent86 `-0.023173953955292047`, KL coherent86→model `0.0026726204078316128`

## Dense-mask comparisons
{
  "densemask_sparselabel_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.17035753346461124,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.13339652428011783,
    "this_source_erasure_cost_nll_mean_delta": 0.21326791845652382,
    "all_sources_erasure_cost_nll_mean_delta": 0.018437873161058604,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 46.424641577060896,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 58.84130824372767,
    "this_source_erasure_cost_rank_mean_delta": 70.40564516129035,
    "all_sources_erasure_cost_rank_mean_delta": 21.231720430107657,
    "pair_correct_source_mean_nll_delta": 0.058511182715807575,
    "pair_correct_source_mean_rank_delta": 8.944444444444429,
    "pair_wrong_source_mean_nll_delta": 0.22886871618041837,
    "pair_wrong_source_mean_rank_delta": 55.36908602150538,
    "view_only_mean_nll_delta": 0.1919077069959254,
    "view_only_mean_rank_delta": 67.78575268817201,
    "full_row_original_mean_nll_delta": 0.07233312486527588,
    "full_row_original_mean_rank_delta": 10.983422939068078,
    "full_row_this_source_erased_mean_nll_delta": 0.28560104332179925,
    "full_row_this_source_erased_mean_rank_delta": 81.38906810035849,
    "full_row_all_sources_erased_mean_nll_delta": 0.0907709980263336,
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
  "ordinary_inherited_wwm_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.0065487408315583195,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.003765492889079791,
    "this_source_erasure_cost_nll_mean_delta": -0.007597858667324431,
    "all_sources_erasure_cost_nll_mean_delta": -0.016936687322431254,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 2.932258064516077,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 4.785035842293951,
    "this_source_erasure_cost_rank_mean_delta": 7.467114695340513,
    "all_sources_erasure_cost_rank_mean_delta": 5.165232974910509,
    "pair_correct_source_mean_nll_delta": 0.07257834693984844,
    "pair_correct_source_mean_rank_delta": 2.5519713261648604,
    "pair_wrong_source_mean_nll_delta": 0.066029606108291,
    "pair_wrong_source_mean_rank_delta": 5.484229390681037,
    "view_only_mean_nll_delta": 0.0688128540507682,
    "view_only_mean_rank_delta": 7.337007168458797,
    "full_row_original_mean_nll_delta": 0.06904345928638334,
    "full_row_original_mean_rank_delta": 3.51702508960571,
    "full_row_this_source_erased_mean_nll_delta": 0.06144560061905757,
    "full_row_this_source_erased_mean_rank_delta": 10.984139784946251,
    "full_row_all_sources_erased_mean_nll_delta": 0.052106771963952525,
    "full_row_all_sources_erased_mean_rank_delta": 8.682258064516077
  },
  "ordinary_inherited_wwm_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.17690627429616956,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.13716201716919763,
    "this_source_erasure_cost_nll_mean_delta": -0.22086577712384825,
    "all_sources_erasure_cost_nll_mean_delta": -0.03537456048348986,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -43.49238351254482,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -54.056272401433716,
    "this_source_erasure_cost_rank_mean_delta": -62.93853046594984,
    "all_sources_erasure_cost_rank_mean_delta": -16.066487455197148,
    "pair_correct_source_mean_nll_delta": 0.014067164224040862,
    "pair_correct_source_mean_rank_delta": -6.392473118279568,
    "pair_wrong_source_mean_nll_delta": -0.16283911007212737,
    "pair_wrong_source_mean_rank_delta": -49.884856630824345,
    "view_only_mean_nll_delta": -0.12309485294515721,
    "view_only_mean_rank_delta": -60.44874551971321,
    "full_row_original_mean_nll_delta": -0.003289665578892542,
    "full_row_original_mean_rank_delta": -7.466397849462368,
    "full_row_this_source_erased_mean_nll_delta": -0.22415544270274168,
    "full_row_this_source_erased_mean_rank_delta": -70.40492831541223,
    "full_row_all_sources_erased_mean_nll_delta": -0.03866422606238107,
    "full_row_all_sources_erased_mean_rank_delta": -23.532885304659544
  },
  "clean_pres_lambda1_eval_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.1172223001000261,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.10005225778241211,
    "this_source_erasure_cost_nll_mean_delta": 0.13267059445182516,
    "all_sources_erasure_cost_nll_mean_delta": 0.006531997436179715,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.85304659498206,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 44.151971326164926,
    "this_source_erasure_cost_rank_mean_delta": 47.842293906810085,
    "all_sources_erasure_cost_rank_mean_delta": 16.834229390681116,
    "pair_correct_source_mean_nll_delta": 0.06108915002314674,
    "pair_correct_source_mean_rank_delta": 5.347670250896044,
    "pair_wrong_source_mean_nll_delta": 0.1783114501231733,
    "pair_wrong_source_mean_rank_delta": 39.20071684587816,
    "view_only_mean_nll_delta": 0.16114140780555886,
    "view_only_mean_rank_delta": 49.49964157706097,
    "full_row_original_mean_nll_delta": 0.0695482674606418,
    "full_row_original_mean_rank_delta": 7.0981182795698885,
    "full_row_this_source_erased_mean_nll_delta": 0.20221886191246696,
    "full_row_this_source_erased_mean_rank_delta": 54.940412186379945,
    "full_row_all_sources_erased_mean_nll_delta": 0.07608026489682196,
    "full_row_all_sources_erased_mean_rank_delta": 23.932347670250806
  },
  "clean_pres_lambda1_eval_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.053135233364585144,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.03334426649770572,
    "this_source_erasure_cost_nll_mean_delta": -0.08059732400469866,
    "all_sources_erasure_cost_nll_mean_delta": -0.01190587572487889,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -12.571594982078835,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -14.689336917562741,
    "this_source_erasure_cost_rank_mean_delta": -22.563351254480267,
    "all_sources_erasure_cost_rank_mean_delta": -4.397491039426541,
    "pair_correct_source_mean_nll_delta": 0.0025779673073391685,
    "pair_correct_source_mean_rank_delta": -3.5967741935483843,
    "pair_wrong_source_mean_nll_delta": -0.05055726605724509,
    "pair_wrong_source_mean_rank_delta": -16.16836917562722,
    "view_only_mean_nll_delta": -0.030766299190366553,
    "view_only_mean_rank_delta": -18.28611111111104,
    "full_row_original_mean_nll_delta": -0.002784857404634078,
    "full_row_original_mean_rank_delta": -3.885304659498189,
    "full_row_this_source_erased_mean_nll_delta": -0.08338218140933229,
    "full_row_this_source_erased_mean_rank_delta": -26.44865591397854,
    "full_row_all_sources_erased_mean_nll_delta": -0.014690733129511635,
    "full_row_all_sources_erased_mean_rank_delta": -8.282795698924815
  },
  "clean_pres_lambda1_eval_seed62065_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.11590121631303729,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.09917717209146026,
    "this_source_erasure_cost_nll_mean_delta": 0.13058106872969955,
    "all_sources_erasure_cost_nll_mean_delta": 0.0059378202777189415,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.868100358422936,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 44.83127240143375,
    "this_source_erasure_cost_rank_mean_delta": 47.92750896057345,
    "all_sources_erasure_cost_rank_mean_delta": 17.051702508960602,
    "pair_correct_source_mean_nll_delta": 0.06158415103094139,
    "pair_correct_source_mean_rank_delta": 5.332437275985654,
    "pair_wrong_source_mean_nll_delta": 0.17748536734397913,
    "pair_wrong_source_mean_rank_delta": 39.20053763440865,
    "view_only_mean_nll_delta": 0.16076132312240166,
    "view_only_mean_rank_delta": 50.16370967741938,
    "full_row_original_mean_nll_delta": 0.06908439578472603,
    "full_row_original_mean_rank_delta": 7.025537634408593,
    "full_row_this_source_erased_mean_nll_delta": 0.19966546451442557,
    "full_row_this_source_erased_mean_rank_delta": 54.953046594982084,
    "full_row_all_sources_erased_mean_nll_delta": 0.07502221606244497,
    "full_row_all_sources_erased_mean_rank_delta": 24.07724014336918
  },
  "clean_pres_lambda1_eval_seed62065_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.05445631715157395,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.03421935218865757,
    "this_source_erasure_cost_nll_mean_delta": -0.08268684972682427,
    "all_sources_erasure_cost_nll_mean_delta": -0.012500052883339663,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -12.55654121863796,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -14.010035842293917,
    "this_source_erasure_cost_rank_mean_delta": -22.478136200716904,
    "all_sources_erasure_cost_rank_mean_delta": -4.180017921147055,
    "pair_correct_source_mean_nll_delta": 0.003072968315133817,
    "pair_correct_source_mean_rank_delta": -3.6120071684587742,
    "pair_wrong_source_mean_nll_delta": -0.05138334883643925,
    "pair_wrong_source_mean_rank_delta": -16.168548387096735,
    "view_only_mean_nll_delta": -0.031146383873523753,
    "view_only_mean_rank_delta": -17.622043010752634,
    "full_row_original_mean_nll_delta": -0.0032487290805498503,
    "full_row_original_mean_rank_delta": -3.9578853046594844,
    "full_row_this_source_erased_mean_nll_delta": -0.08593557880737368,
    "full_row_this_source_erased_mean_rank_delta": -26.436021505376402,
    "full_row_all_sources_erased_mean_nll_delta": -0.015748781963888625,
    "full_row_all_sources_erased_mean_rank_delta": -8.13790322580644
  },
  "densecorr_pres_lambda1_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.002621753924542336,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.003089502355476892,
    "this_source_erasure_cost_nll_mean_delta": 0.0006365056019372872,
    "all_sources_erasure_cost_nll_mean_delta": -0.013352409019696587,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 5.687455197132579,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 7.702688172042997,
    "this_source_erasure_cost_rank_mean_delta": 11.467921146953472,
    "all_sources_erasure_cost_rank_mean_delta": 6.6159498207886145,
    "pair_correct_source_mean_nll_delta": 0.06005645639308588,
    "pair_correct_source_mean_rank_delta": 3.0439068100358213,
    "pair_wrong_source_mean_nll_delta": 0.06267821031762821,
    "pair_wrong_source_mean_rank_delta": 8.731362007168457,
    "view_only_mean_nll_delta": 0.06314595874856188,
    "view_only_mean_rank_delta": 10.746594982078875,
    "full_row_original_mean_nll_delta": 0.05617252263261596,
    "full_row_original_mean_rank_delta": 3.758960573476685,
    "full_row_this_source_erased_mean_nll_delta": 0.05680902823455192,
    "full_row_this_source_erased_mean_rank_delta": 15.226881720430129,
    "full_row_all_sources_erased_mean_nll_delta": 0.04282011361291982,
    "full_row_all_sources_erased_mean_rank_delta": 10.374910394265157
  },
  "densecorr_pres_lambda1_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.1677357795400689,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.13030702192464094,
    "this_source_erasure_cost_nll_mean_delta": -0.21263141285458653,
    "all_sources_erasure_cost_nll_mean_delta": -0.03179028218075519,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -40.73718637992832,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -51.13862007168467,
    "this_source_erasure_cost_rank_mean_delta": -58.93772401433688,
    "all_sources_erasure_cost_rank_mean_delta": -14.615770609319043,
    "pair_correct_source_mean_nll_delta": 0.0015452736772783027,
    "pair_correct_source_mean_rank_delta": -5.900537634408607,
    "pair_wrong_source_mean_nll_delta": -0.16619050586279016,
    "pair_wrong_source_mean_rank_delta": -46.637724014336925,
    "view_only_mean_nll_delta": -0.12876174824736353,
    "view_only_mean_rank_delta": -57.039157706093135,
    "full_row_original_mean_nll_delta": -0.016160602232659915,
    "full_row_original_mean_rank_delta": -7.224462365591393,
    "full_row_this_source_erased_mean_nll_delta": -0.22879201508724734,
    "full_row_this_source_erased_mean_rank_delta": -66.16218637992836,
    "full_row_all_sources_erased_mean_nll_delta": -0.047950884413413775,
    "full_row_all_sources_erased_mean_rank_delta": -21.840232974910464
  }
}

## Interpretation scope
- **scope**: The readout is fixed-target Qwen pair evidence-state scoring. It is not a BabyLM leaderboard component and should be combined with broad scores before judging a candidate.
- **held_out_default**: Default scope is post_prefix, so targets are outside the 80-update Qwen prefix used by the dense-mask training arms; use scope=prefix when direct trained-material acquisition is the question.
- **source_erasure**: Source-erased full-row states blank character spans to preserve target offsets while removing lexical source evidence. Other current-view text may still contain target words, and leakage flags are recorded.
- **preservation_use**: A useful preservation endpoint should reduce view-only, wrong-source, source-erased, and CDI/broad-language damage without merely erasing correct-source/Entity movement.
