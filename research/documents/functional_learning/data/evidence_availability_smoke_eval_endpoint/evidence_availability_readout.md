# earlier analysis evidence-availability readout

Status: `EVIDENCE_AVAILABILITY_READOUT_DONE`

This readout holds Qwen current-view target words fixed while changing the evidence state around the target. It is not an official BabyLM score; it is a mechanism/evidence instrument for the preservation route.

## Task construction
{
  "scope": "post_prefix",
  "total_segments_available_in_scope": 40794,
  "segments_sampled": 4,
  "max_updates": 80,
  "words_per_update": 39533,
  "prefix_word_limit": 3162640,
  "sample_seed": 90090,
  "row_idx_min": 39857,
  "row_idx_max": 72231,
  "pair_count_sampled": 4,
  "candidate_kind_counts": {
    "unchanged_inherited_current": 4
  },
  "view_kind_counts": {
    "current_inherited_control": 4
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
  "candidate_tasks_before_complete_filter": 23,
  "tasks_after_complete_filter": 18,
  "complete_base_targets": 3,
  "condition_counts": {
    "pair_correct_source": 3,
    "pair_wrong_source": 3,
    "view_only": 3,
    "full_row_original": 3,
    "full_row_this_source_erased": 3,
    "full_row_all_sources_erased": 3
  },
  "skips_before_complete_filter": {
    "full_row_all_sources_erased_target_truncated_or_unmapped": 1
  },
  "base_targets_seen": 4,
  "dropped_base_targets_missing_condition_counts": {
    "full_row_all_sources_erased": 1
  },
  "target_string_in_context_excluding_target_fraction_by_condition": {
    "pair_correct_source": 0.3333333333333333,
    "pair_wrong_source": 0.0,
    "view_only": 0.0,
    "full_row_original": 0.3333333333333333,
    "full_row_this_source_erased": 0.0,
    "full_row_all_sources_erased": 0.0
  },
  "source_region_contains_target_string_fraction_by_condition": {
    "pair_correct_source": 0.3333333333333333,
    "pair_wrong_source": 0.0,
    "view_only": 0.0,
    "full_row_original": 0.3333333333333333,
    "full_row_this_source_erased": 0.0,
    "full_row_all_sources_erased": 0.0
  },
  "mean_input_len_by_condition": {
    "pair_correct_source": 44.333333333333336,
    "pair_wrong_source": 41.666666666666664,
    "view_only": 22.666666666666668,
    "full_row_original": 203.0,
    "full_row_this_source_erased": 268.3333333333333,
    "full_row_all_sources_erased": 506.0
  }
}

## Model summaries
### coherent86
- complete base targets: `3`
- specific_source_penalty_nll_wrong_minus_correct: `-0.5322136481602987`
- view_absence_penalty_nll_view_minus_correct: `0.3794608910878499`
- this_source_erasure_cost_nll: `-0.03253300984700521`
- all_sources_erasure_cost_nll: `0.7173546552658081`
- delta_vs_coherent86_specific_source_penalty_nll: `0.0`
- delta_vs_coherent86_view_absence_penalty_nll: `0.0`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.0`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.0`
  - pair_correct_source: NLL `6.786475261052449`, rank `98.5`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - pair_wrong_source: NLL `6.254261612892151`, rank `91.16666666666667`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - view_only: NLL `7.1659361521403`, rank `135.0`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_original: NLL `6.554517229398091`, rank `77.83333333333333`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_this_source_erased: NLL `6.521984219551086`, rank `199.83333333333334`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`
  - full_row_all_sources_erased: NLL `7.271871884663899`, rank `275.3333333333333`, ΔNLL vs coherent86 `0.0`, KL coherent86→model `0.0`

### sparse_focus_seed62064
- complete base targets: `3`
- specific_source_penalty_nll_wrong_minus_correct: `-0.38864612579345703`
- view_absence_penalty_nll_view_minus_correct: `0.477338711420695`
- this_source_erasure_cost_nll: `0.09401082992553711`
- all_sources_erasure_cost_nll: `0.8839366833368937`
- delta_vs_coherent86_specific_source_penalty_nll: `0.14356752236684164`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09787782033284505`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.12654383977254233`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.1665820280710856`
  - pair_correct_source: NLL `6.4895962079366045`, rank `88.5`, ΔNLL vs coherent86 `-0.2968790531158447`, KL coherent86→model `0.0169550112914294`
  - pair_wrong_source: NLL `6.1009500821431475`, rank `88.33333333333333`, ΔNLL vs coherent86 `-0.15331153074900308`, KL coherent86→model `0.013094308708483974`
  - view_only: NLL `6.9669349193573`, rank `126.33333333333333`, ΔNLL vs coherent86 `-0.19900123278299967`, KL coherent86→model `0.014166647102683783`
  - full_row_original: NLL `6.290013035138448`, rank `66.83333333333333`, ΔNLL vs coherent86 `-0.26450419425964355`, KL coherent86→model `0.020053265305856865`
  - full_row_this_source_erased: NLL `6.384023865063985`, rank `193.33333333333334`, ΔNLL vs coherent86 `-0.13796035448710123`, KL coherent86→model `0.018009774076441925`
  - full_row_all_sources_erased: NLL `7.173949718475342`, rank `264.0`, ΔNLL vs coherent86 `-0.09792216618855794`, KL coherent86→model `0.007779498759191483`

### densemask_sparselabel_seed62064
- complete base targets: `3`
- specific_source_penalty_nll_wrong_minus_correct: `-0.4942588011423747`
- view_absence_penalty_nll_view_minus_correct: `0.7332101662953695`
- this_source_erasure_cost_nll: `-0.22975373268127441`
- all_sources_erasure_cost_nll: `-0.011260430018107096`
- delta_vs_coherent86_specific_source_penalty_nll: `0.03795484701792399`
- delta_vs_coherent86_view_absence_penalty_nll: `0.35374927520751953`
- delta_vs_coherent86_this_source_erasure_cost_nll: `-0.1972207228342692`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `-0.7286150852839152`
  - pair_correct_source: NLL `6.991335948308309`, rank `78.0`, ΔNLL vs coherent86 `0.20486068725585938`, KL coherent86→model `0.10271311504766345`
  - pair_wrong_source: NLL `6.4970771471659345`, rank `83.83333333333333`, ΔNLL vs coherent86 `0.24281553427378336`, KL coherent86→model `0.16215131804347038`
  - view_only: NLL `7.724546114603679`, rank `201.16666666666666`, ΔNLL vs coherent86 `0.5586099624633789`, KL coherent86→model `0.09130802859241764`
  - full_row_original: NLL `7.419488906860352`, rank `83.83333333333333`, ΔNLL vs coherent86 `0.8649716774622599`, KL coherent86→model `0.3422515763280292`
  - full_row_this_source_erased: NLL `7.189735174179077`, rank `279.8333333333333`, ΔNLL vs coherent86 `0.6677509546279907`, KL coherent86→model `0.396647682103018`
  - full_row_all_sources_erased: NLL `7.408228476842244`, rank `300.0`, ΔNLL vs coherent86 `0.13635659217834473`, KL coherent86→model `0.0548490925381581`

### clean_pres_lambda1_eval_full80
- complete base targets: `3`
- specific_source_penalty_nll_wrong_minus_correct: `-0.48821882406870526`
- view_absence_penalty_nll_view_minus_correct: `0.7099647919336954`
- this_source_erasure_cost_nll: `-0.20305530230204263`
- all_sources_erasure_cost_nll: `0.2392592430114746`
- delta_vs_coherent86_specific_source_penalty_nll: `0.04399482409159342`
- delta_vs_coherent86_view_absence_penalty_nll: `0.3305039008458455`
- delta_vs_coherent86_this_source_erasure_cost_nll: `-0.17052229245503744`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `-0.4780954122543335`
  - pair_correct_source: NLL `6.886247277259827`, rank `79.33333333333333`, ΔNLL vs coherent86 `0.09977201620737712`, KL coherent86→model `0.04955436126329005`
  - pair_wrong_source: NLL `6.398028453191121`, rank `86.16666666666667`, ΔNLL vs coherent86 `0.14376684029897055`, KL coherent86→model `0.08657244375596444`
  - view_only: NLL `7.596212069193522`, rank `185.0`, ΔNLL vs coherent86 `0.43027591705322266`, KL coherent86→model `0.047044709247226514`
  - full_row_original: NLL `7.115961074829102`, rank `76.66666666666667`, ΔNLL vs coherent86 `0.5614438454310099`, KL coherent86→model `0.17726893754055104`
  - full_row_this_source_erased: NLL `6.912905772527059`, rank `232.16666666666666`, ΔNLL vs coherent86 `0.3909215529759725`, KL coherent86→model `0.23514117129767934`
  - full_row_all_sources_erased: NLL `7.355220317840576`, rank `285.8333333333333`, ΔNLL vs coherent86 `0.08334843317667644`, KL coherent86→model `0.029900159842024248`

## Dense-mask comparisons
{
  "densemask_sparselabel_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.10561267534891766,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.2558714548746745,
    "this_source_erasure_cost_nll_mean_delta": -0.3237645626068115,
    "all_sources_erasure_cost_nll_mean_delta": -0.8951971133550007,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 6.0,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 85.33333333333334,
    "this_source_erasure_cost_rank_mean_delta": 69.5,
    "all_sources_erasure_cost_rank_mean_delta": 19.0,
    "pair_correct_source_mean_nll_delta": 0.5017397403717041,
    "pair_correct_source_mean_rank_delta": -10.5,
    "pair_wrong_source_mean_nll_delta": 0.39612706502278705,
    "pair_wrong_source_mean_rank_delta": -4.5,
    "view_only_mean_nll_delta": 0.7576111952463789,
    "view_only_mean_rank_delta": 74.83333333333333,
    "full_row_original_mean_nll_delta": 1.1294758717219038,
    "full_row_original_mean_rank_delta": 17.0,
    "full_row_this_source_erased_mean_nll_delta": 0.8057113091150923,
    "full_row_this_source_erased_mean_rank_delta": 86.49999999999997,
    "full_row_all_sources_erased_mean_nll_delta": 0.23427875836690237,
    "full_row_all_sources_erased_mean_rank_delta": 36.0
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
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.09957269827524823,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.23262608051300043,
    "this_source_erasure_cost_nll_mean_delta": -0.2970661322275797,
    "all_sources_erasure_cost_nll_mean_delta": -0.6446774403254191,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 7.0,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 67.83333333333334,
    "this_source_erasure_cost_rank_mean_delta": 29.0,
    "all_sources_erasure_cost_rank_mean_delta": 12.0,
    "pair_correct_source_mean_nll_delta": 0.39665106932322214,
    "pair_correct_source_mean_rank_delta": -9.166666666666671,
    "pair_wrong_source_mean_nll_delta": 0.29707837104797363,
    "pair_wrong_source_mean_rank_delta": -2.166666666666657,
    "view_only_mean_nll_delta": 0.6292771498362226,
    "view_only_mean_rank_delta": 58.66666666666667,
    "full_row_original_mean_nll_delta": 0.8259480396906538,
    "full_row_original_mean_rank_delta": 9.833333333333343,
    "full_row_this_source_erased_mean_nll_delta": 0.5288819074630737,
    "full_row_this_source_erased_mean_rank_delta": 38.833333333333314,
    "full_row_all_sources_erased_mean_nll_delta": 0.18127059936523438,
    "full_row_all_sources_erased_mean_rank_delta": 21.833333333333314
  },
  "clean_pres_lambda1_eval_full80_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.006039977073669434,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.023245374361674065,
    "this_source_erasure_cost_nll_mean_delta": 0.02669843037923178,
    "all_sources_erasure_cost_nll_mean_delta": 0.2505196730295817,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 1.0,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -17.5,
    "this_source_erasure_cost_rank_mean_delta": -40.5,
    "all_sources_erasure_cost_rank_mean_delta": -7.0,
    "pair_correct_source_mean_nll_delta": -0.10508867104848196,
    "pair_correct_source_mean_rank_delta": 1.3333333333333286,
    "pair_wrong_source_mean_nll_delta": -0.09904869397481342,
    "pair_wrong_source_mean_rank_delta": 2.333333333333343,
    "view_only_mean_nll_delta": -0.12833404541015625,
    "view_only_mean_rank_delta": -16.166666666666657,
    "full_row_original_mean_nll_delta": -0.30352783203125,
    "full_row_original_mean_rank_delta": -7.166666666666657,
    "full_row_this_source_erased_mean_nll_delta": -0.2768294016520185,
    "full_row_this_source_erased_mean_rank_delta": -47.66666666666666,
    "full_row_all_sources_erased_mean_nll_delta": -0.053008159001668,
    "full_row_all_sources_erased_mean_rank_delta": -14.166666666666686
  }
}

## Interpretation scope
- **scope**: The readout is fixed-target Qwen pair evidence-state scoring. It is not a BabyLM leaderboard component and should be combined with broad scores before judging a candidate.
- **held_out_default**: Default scope is post_prefix, so targets are outside the 80-update Qwen prefix used by the dense-mask training arms; use scope=prefix when direct trained-material acquisition is the question.
- **source_erasure**: Source-erased full-row states blank character spans to preserve target offsets while removing lexical source evidence. Other current-view text may still contain target words, and leakage flags are recorded.
- **preservation_use**: A useful preservation endpoint should reduce view-only, wrong-source, source-erased, and CDI/broad-language damage without merely erasing correct-source/Entity movement.
