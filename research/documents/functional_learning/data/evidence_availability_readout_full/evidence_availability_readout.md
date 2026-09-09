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

### dense_focus_seed62064
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.48746370047875`
- view_absence_penalty_nll_view_minus_correct: `3.580548806704821`
- this_source_erasure_cost_nll: `3.9972278591597683`
- all_sources_erasure_cost_nll: `4.051990838583542`
- delta_vs_coherent86_specific_source_penalty_nll: `0.17607595980399804`
- delta_vs_coherent86_view_absence_penalty_nll: `0.1324812339261837`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.2185547745912098`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.030510587336887315`
  - pair_correct_source: NLL `2.347770635035944`, rank `108.62679211469533`, ΔNLL vs coherent86 `-0.03008422914821084`, KL coherent86→model `0.0321284002809086`
  - pair_wrong_source: NLL `5.835234335514694`, rank `345.52571684587815`, ΔNLL vs coherent86 `0.1459917306557872`, KL coherent86→model `0.1255613113194318`
  - view_only: NLL `5.928319441740765`, rank `430.57356630824376`, ΔNLL vs coherent86 `0.10239700477797282`, KL coherent86→model `0.08596705867044438`
  - full_row_original: NLL `2.3453989163832136`, rank `106.07123655913979`, ΔNLL vs coherent86 `-0.01332897448524837`, KL coherent86→model `0.04554496969272921`
  - full_row_this_source_erased: NLL `6.342626775542982`, rank `615.7106630824372`, ΔNLL vs coherent86 `0.20522580010596142`, KL coherent86→model `0.128296996609612`
  - full_row_all_sources_erased: NLL `6.397389754966756`, rank `670.9041218637993`, ΔNLL vs coherent86 `0.017181612851638963`, KL coherent86→model `0.03873757196671253`

### dense_focus_seed62065
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.491186897186617`
- view_absence_penalty_nll_view_minus_correct: `3.584312296189609`
- this_source_erasure_cost_nll: `4.001482137346742`
- all_sources_erasure_cost_nll: `4.053330023858845`
- delta_vs_coherent86_specific_source_penalty_nll: `0.1797991565118651`
- delta_vs_coherent86_view_absence_penalty_nll: `0.13624472341097152`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.22280905277818344`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.031849772612189554`
  - pair_correct_source: NLL `2.3485546712251004`, rank `108.9323476702509`, ΔNLL vs coherent86 `-0.029300192959054196`, KL coherent86→model `0.03263969314507457`
  - pair_wrong_source: NLL `5.839741568411718`, rank `346.665770609319`, ΔNLL vs coherent86 `0.15049896355281087`, KL coherent86→model `0.1291181120740746`
  - view_only: NLL `5.932866967414709`, rank `432.35053763440857`, ΔNLL vs coherent86 `0.10694453045191732`, KL coherent86→model `0.08853195252773587`
  - full_row_original: NLL `2.3460844086953374`, rank `106.33915770609319`, ΔNLL vs coherent86 `-0.01264348217312456`, KL coherent86→model `0.0460648756470612`
  - full_row_this_source_erased: NLL `6.347566546042079`, rank `616.9237455197133`, ΔNLL vs coherent86 `0.2101655706050589`, KL coherent86→model `0.13160299336409204`
  - full_row_all_sources_erased: NLL `6.399414432554182`, rank `672.004211469534`, ΔNLL vs coherent86 `0.019206290439064997`, KL coherent86→model `0.04032121508204509`

### clean_pres_lambda1_eval_full80
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

### clean_pres_lambda1_train_full80
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4333721059441635`
- view_absence_penalty_nll_view_minus_correct: `3.5451210062737575`
- this_source_erasure_cost_nll: `3.944626133452194`
- all_sources_erasure_cost_nll: `4.047568624400585`
- delta_vs_coherent86_specific_source_penalty_nll: `0.12198436526941217`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09705343349512033`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.16595304888363518`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.02608837315392945`
  - pair_correct_source: NLL `2.3727317992095074`, rank `107.49283154121865`, ΔNLL vs coherent86 `-0.0051230649746473325`, KL coherent86→model `0.014684194532689477`
  - pair_wrong_source: NLL `5.806103905153671`, rank `328.84112903225804`, ΔNLL vs coherent86 `0.11686130029476484`, KL coherent86→model `0.06712597580259941`
  - view_only: NLL `5.917852805483266`, rank `411.53853046594986`, ΔNLL vs coherent86 `0.091930368520473`, KL coherent86→model `0.04663742750085199`
  - full_row_original: NLL `2.3661691777875244`, rank `105.27105734767025`, ΔNLL vs coherent86 `0.007441286919062699`, KL coherent86→model `0.025703905142218385`
  - full_row_this_source_erased: NLL `6.310795311239718`, rank `601.8188172043011`, ΔNLL vs coherent86 `0.1733943358026979`, KL coherent86→model `0.07234428664505621`
  - full_row_all_sources_erased: NLL `6.41373780218811`, rank `669.7612903225806`, ΔNLL vs coherent86 `0.03352966007299216`, KL coherent86→model `0.019811873802248435`

### pres_lambda1_trainmode_confounded
- complete base targets: `186`
- specific_source_penalty_nll_wrong_minus_correct: `3.4321196847728537`
- view_absence_penalty_nll_view_minus_correct: `3.5435341195203103`
- this_source_erasure_cost_nll: `3.942641785226622`
- all_sources_erasure_cost_nll: `4.0470877241062855`
- delta_vs_coherent86_specific_source_penalty_nll: `0.12073194409810205`
- delta_vs_coherent86_view_absence_penalty_nll: `0.09546654674167285`
- delta_vs_coherent86_this_source_erasure_cost_nll: `0.16396870065806346`
- delta_vs_coherent86_all_sources_erasure_cost_nll: `0.02560747285963093`
  - pair_correct_source: NLL `2.373380726699553`, rank `107.66308243727597`, ΔNLL vs coherent86 `-0.004474137484601494`, KL coherent86→model `0.014782686526881512`
  - pair_wrong_source: NLL `5.805500411472407`, rank `328.5368279569892`, ΔNLL vs coherent86 `0.11625780661350055`, KL coherent86→model `0.0665213708906056`
  - view_only: NLL `5.916914846219863`, rank `411.694623655914`, ΔNLL vs coherent86 `0.09099240925707137`, KL coherent86→model `0.04630219985518248`
  - full_row_original: NLL `2.3670675240997823`, rank `105.46729390681004`, ΔNLL vs coherent86 `0.008339633231320828`, KL coherent86→model `0.025731228460184587`
  - full_row_this_source_erased: NLL `6.309709309326405`, rank `601.5723118279569`, ΔNLL vs coherent86 `0.17230833388938432`, KL coherent86→model `0.07206873710686174`
  - full_row_all_sources_erased: NLL `6.414155248206069`, rank `670.2655913978494`, ΔNLL vs coherent86 `0.033947106090951785`, KL coherent86→model `0.01969544619369182`

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
  "dense_focus_seed62064_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.17318415552312372,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.13466785734283704,
    "this_source_erasure_cost_nll_mean_delta": 0.21417423553526316,
    "all_sources_erasure_cost_nll_mean_delta": 0.016897180200335704,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 49.14462365591393,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 61.63996415770612,
    "this_source_erasure_cost_rank_mean_delta": 71.7270609318997,
    "all_sources_erasure_cost_rank_mean_delta": 21.271057347670308,
    "pair_correct_source_mean_nll_delta": 0.05255744623543057,
    "pair_correct_source_mean_rank_delta": 8.900089605734749,
    "pair_wrong_source_mean_nll_delta": 0.2257416017585543,
    "pair_wrong_source_mean_rank_delta": 58.04471326164878,
    "view_only_mean_nll_delta": 0.18722530357826717,
    "view_only_mean_rank_delta": 70.54005376344088,
    "full_row_original_mean_nll_delta": 0.06627850021951476,
    "full_row_original_mean_rank_delta": 10.737007168458774,
    "full_row_this_source_erased_mean_nll_delta": 0.28045273575477747,
    "full_row_this_source_erased_mean_rank_delta": 82.46406810035842,
    "full_row_all_sources_erased_mean_nll_delta": 0.0831756804198509,
    "full_row_all_sources_erased_mean_rank_delta": 32.008064516129025
  },
  "dense_focus_seed62064_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.0028266220585124735,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.0012713330627192043,
    "this_source_erasure_cost_nll_mean_delta": 0.0009063170787393382,
    "all_sources_erasure_cost_nll_mean_delta": -0.0015406929607229003,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 2.719982078853036,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 2.7986559139784504,
    "this_source_erasure_cost_rank_mean_delta": 1.3214157706093488,
    "all_sources_erasure_cost_rank_mean_delta": 0.03933691756265034,
    "pair_correct_source_mean_nll_delta": -0.005953736480377003,
    "pair_correct_source_mean_rank_delta": -0.04435483870967971,
    "pair_wrong_source_mean_nll_delta": -0.0031271144218640856,
    "pair_wrong_source_mean_rank_delta": 2.675627240143399,
    "view_only_mean_nll_delta": -0.004682403417658243,
    "view_only_mean_rank_delta": 2.75430107526887,
    "full_row_original_mean_nll_delta": -0.00605462464576112,
    "full_row_original_mean_rank_delta": -0.2464157706093033,
    "full_row_this_source_erased_mean_nll_delta": -0.0051483075670217815,
    "full_row_this_source_erased_mean_rank_delta": 1.0749999999999318,
    "full_row_all_sources_erased_mean_nll_delta": -0.007595317606482688,
    "full_row_all_sources_erased_mean_rank_delta": -0.20707885304659612
  },
  "dense_focus_seed62065_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.17690735223099052,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.13843134682762503,
    "this_source_erasure_cost_nll_mean_delta": 0.21842851372223704,
    "all_sources_erasure_cost_nll_mean_delta": 0.018236365475639005,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 49.979121863799236,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 63.11137992831544,
    "this_source_erasure_cost_rank_mean_delta": 72.67222222222227,
    "all_sources_erasure_cost_rank_mean_delta": 22.103225806451633,
    "pair_correct_source_mean_nll_delta": 0.05334148242458703,
    "pair_correct_source_mean_rank_delta": 9.20564516129032,
    "pair_wrong_source_mean_nll_delta": 0.230248834655578,
    "pair_wrong_source_mean_rank_delta": 59.18476702508963,
    "view_only_mean_nll_delta": 0.19177282925221117,
    "view_only_mean_rank_delta": 72.31702508960569,
    "full_row_original_mean_nll_delta": 0.06696399253163854,
    "full_row_original_mean_rank_delta": 11.004928315412172,
    "full_row_this_source_erased_mean_nll_delta": 0.28539250625387425,
    "full_row_this_source_erased_mean_rank_delta": 83.67715053763447,
    "full_row_all_sources_erased_mean_nll_delta": 0.08520035800727666,
    "full_row_all_sources_erased_mean_rank_delta": 33.10815412186366
  },
  "dense_focus_seed62065_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.006549818766379278,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.005034822547507201,
    "this_source_erasure_cost_nll_mean_delta": 0.005160595265713219,
    "all_sources_erasure_cost_nll_mean_delta": -0.00020150768541959962,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 3.5544802867383396,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 4.270071684587776,
    "this_source_erasure_cost_rank_mean_delta": 2.2665770609319225,
    "all_sources_erasure_cost_rank_mean_delta": 0.8715053763439755,
    "pair_correct_source_mean_nll_delta": -0.005169700291220547,
    "pair_correct_source_mean_rank_delta": 0.26120071684589163,
    "pair_wrong_source_mean_nll_delta": 0.0013801184751596196,
    "pair_wrong_source_mean_rank_delta": 3.8156810035842454,
    "view_only_mean_nll_delta": -0.00013487774371423455,
    "view_only_mean_rank_delta": 4.531272401433682,
    "full_row_original_mean_nll_delta": -0.005369132333637339,
    "full_row_original_mean_rank_delta": 0.021505376344094884,
    "full_row_this_source_erased_mean_nll_delta": -0.0002085370679250076,
    "full_row_this_source_erased_mean_rank_delta": 2.288082437275989,
    "full_row_all_sources_erased_mean_nll_delta": -0.005570640019056938,
    "full_row_all_sources_erased_mean_rank_delta": 0.893010752688042
  },
  "clean_pres_lambda1_eval_full80_minus_sparse": {
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
  "clean_pres_lambda1_eval_full80_minus_densemask": {
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
  "clean_pres_lambda1_train_full80_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.11909256098853715,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.09924005691177351,
    "this_source_erasure_cost_nll_mean_delta": 0.161572509827689,
    "all_sources_erasure_cost_nll_mean_delta": 0.012474966017378186,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.59399641577059,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 43.73888888888894,
    "this_source_erasure_cost_rank_mean_delta": 58.635394265233,
    "all_sources_erasure_cost_rank_mean_delta": 20.92840501792125,
    "pair_correct_source_mean_nll_delta": 0.077518610408994,
    "pair_correct_source_mean_rank_delta": 7.766129032258064,
    "pair_wrong_source_mean_nll_delta": 0.1966111713975316,
    "pair_wrong_source_mean_rank_delta": 41.36012544802867,
    "view_only_mean_nll_delta": 0.17675866732076795,
    "view_only_mean_rank_delta": 51.50501792114699,
    "full_row_original_mean_nll_delta": 0.08704876162382558,
    "full_row_original_mean_rank_delta": 9.936827956989234,
    "full_row_this_source_erased_mean_nll_delta": 0.2486212714515137,
    "full_row_this_source_erased_mean_rank_delta": 68.57222222222231,
    "full_row_all_sources_erased_mean_nll_delta": 0.09952372764120465,
    "full_row_all_sources_erased_mean_rank_delta": 30.865232974910327
  },
  "clean_pres_lambda1_train_full80_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.05126497247607409,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.03415646736834432,
    "this_source_erasure_cost_nll_mean_delta": -0.051695408628834816,
    "all_sources_erasure_cost_nll_mean_delta": -0.005962907143680418,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -12.830645161290306,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -15.10241935483873,
    "this_source_erasure_cost_rank_mean_delta": -11.770250896057348,
    "all_sources_erasure_cost_rank_mean_delta": -0.30331541218640723,
    "pair_correct_source_mean_nll_delta": 0.01900742769318642,
    "pair_correct_source_mean_rank_delta": -1.1783154121863646,
    "pair_wrong_source_mean_nll_delta": -0.03225754478288678,
    "pair_wrong_source_mean_rank_delta": -14.008960573476713,
    "view_only_mean_nll_delta": -0.01514903967515746,
    "view_only_mean_rank_delta": -16.280734767025024,
    "full_row_original_mean_nll_delta": 0.0147156367585497,
    "full_row_original_mean_rank_delta": -1.0465949820788438,
    "full_row_this_source_erased_mean_nll_delta": -0.03697977187028556,
    "full_row_this_source_erased_mean_rank_delta": -12.816845878136178,
    "full_row_all_sources_erased_mean_nll_delta": 0.008752729614871058,
    "full_row_all_sources_erased_mean_rank_delta": -1.3499103942652937
  },
  "pres_lambda1_trainmode_confounded_minus_sparse": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": 0.1178401398172273,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": 0.09765317015832631,
    "this_source_erasure_cost_nll_mean_delta": 0.15958816160211686,
    "all_sources_erasure_cost_nll_mean_delta": 0.01199406572307904,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": 33.11944444444444,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": 43.72473118279572,
    "this_source_erasure_cost_rank_mean_delta": 58.192652329749194,
    "all_sources_erasure_cost_rank_mean_delta": 21.23646953405023,
    "pair_correct_source_mean_nll_delta": 0.07816753789903963,
    "pair_correct_source_mean_rank_delta": 7.9363799283153895,
    "pair_wrong_source_mean_nll_delta": 0.19600767771626693,
    "pair_wrong_source_mean_rank_delta": 41.055824372759844,
    "view_only_mean_nll_delta": 0.1758207080573655,
    "view_only_mean_rank_delta": 51.6611111111111,
    "full_row_original_mean_nll_delta": 0.0879471079360834,
    "full_row_original_mean_rank_delta": 10.133064516129025,
    "full_row_this_source_erased_mean_nll_delta": 0.2475352695382007,
    "full_row_this_source_erased_mean_rank_delta": 68.3257168458781,
    "full_row_all_sources_erased_mean_nll_delta": 0.09994117365916377,
    "full_row_all_sources_erased_mean_rank_delta": 31.36953405017914
  },
  "pres_lambda1_trainmode_confounded_minus_densemask": {
    "specific_source_penalty_nll_wrong_minus_correct_mean_delta": -0.05251739364738395,
    "view_absence_penalty_nll_view_minus_correct_mean_delta": -0.035743354121791526,
    "this_source_erasure_cost_nll_mean_delta": -0.05367975685440696,
    "all_sources_erasure_cost_nll_mean_delta": -0.006443807437979565,
    "specific_source_penalty_rank_wrong_minus_correct_mean_delta": -13.305197132616456,
    "view_absence_penalty_rank_view_minus_correct_mean_delta": -15.116577060931945,
    "this_source_erasure_cost_rank_mean_delta": -12.212992831541158,
    "all_sources_erasure_cost_rank_mean_delta": 0.004749103942572219,
    "pair_correct_source_mean_nll_delta": 0.019656355183232055,
    "pair_correct_source_mean_rank_delta": -1.0080645161290391,
    "pair_wrong_source_mean_nll_delta": -0.03286103846415145,
    "pair_wrong_source_mean_rank_delta": -14.313261648745538,
    "view_only_mean_nll_delta": -0.016086998938559915,
    "view_only_mean_rank_delta": -16.124641577060913,
    "full_row_original_mean_nll_delta": 0.015613983070807524,
    "full_row_original_mean_rank_delta": -0.8503584229390526,
    "full_row_this_source_erased_mean_nll_delta": -0.038065773783598544,
    "full_row_this_source_erased_mean_rank_delta": -13.06335125448038,
    "full_row_all_sources_erased_mean_nll_delta": 0.00917017563283018,
    "full_row_all_sources_erased_mean_rank_delta": -0.8456093189964804
  }
}

## Interpretation scope
- **scope**: The readout is fixed-target Qwen pair evidence-state scoring. It is not a BabyLM leaderboard component and should be combined with broad scores before judging a candidate.
- **held_out_default**: Default scope is post_prefix, so targets are outside the 80-update Qwen prefix used by the dense-mask training arms; use scope=prefix when direct trained-material acquisition is the question.
- **source_erasure**: Source-erased full-row states blank character spans to preserve target offsets while removing lexical source evidence. Other current-view text may still contain target words, and leakage flags are recorded.
- **preservation_use**: A useful preservation endpoint should reduce view-only, wrong-source, source-erased, and CDI/broad-language damage without merely erasing correct-source/Entity movement.
