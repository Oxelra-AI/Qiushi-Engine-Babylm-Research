# earlier analysis rollback control

This readout compares clean eval-mode preservation with a matched rollback along the acquisition-only `(M,S)` private-adapter update. It is an explanatory computation, not a candidate checkpoint.

## Matched ordinary-text drift

- Clean eval KL(parent||model) on ordinary non-Qwen calibration positions: `0.013327214029512199` over `270` positions.
- Dense-mask alpha=1 KL on the same positions: `0.028201220503593225`.
- Selected rollback alpha: `0.75` with KL `0.01244245159711252` and absolute error `0.0008847624323996793`.

## Evidence availability: selected deltas vs coherent86

### `coherent86`
- pair_correct_source: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- pair_wrong_source: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- view_only: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- full_row_original: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- full_row_this_source_erased: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- full_row_all_sources_erased: ΔNLL `0.0`, Δrank `0.0`, KL `0.0`
- source penalty ΔNLL: `0.0`; view absence ΔNLL: `0.0`; this-source erasure ΔNLL: `0.0`; all-source erasure ΔNLL: `0.0`

### `sparse_focus_seed62064`
- pair_correct_source: ΔNLL `-0.08264153242496336`, Δrank `-3.490143369175628`, KL `0.010080612011869492`
- pair_wrong_source: ΔNLL `-0.07974975429805775`, Δrank `-6.261827956989249`, KL `0.009574113557497302`
- view_only: ΔNLL `-0.08482798577426501`, Δrank `-10.230824372759859`, KL `0.008486533814872299`
- full_row_original: ΔNLL `-0.07960762854939449`, Δrank `-4.483870967741936`, KL `0.010944031079442765`
- full_row_this_source_erased: ΔNLL `-0.0752267423431359`, Δrank `-16.152419354838713`, KL `0.011691024402550468`
- full_row_all_sources_erased: ΔNLL `-0.06599412405776614`, Δrank `-12.04641577060932`, KL `0.007604942464934287`
- source penalty ΔNLL: `0.0028917781269055913`; view absence ΔNLL: `-0.002186453349301676`; this-source erasure ΔNLL: `0.00438088620625859`; all-source erasure ΔNLL: `0.013613504491628334`

### `rollback_matched_densemask_update`
- pair_correct_source: ΔNLL `-0.0225477120915157`, Δrank `3.5241935483870956`, KL `0.013860873637349216`
- pair_wrong_source: ΔNLL `0.07857702492959931`, Δrank `28.3747311827957`, KL `0.05195938362698254`
- view_only: ΔNLL `0.05823826232748139`, Δrank `34.15331541218638`, KL `0.03643376870976515`
- full_row_original: ΔNLL `-0.010752884175742756`, Δrank `4.11962365591398`, KL `0.023093958490289387`
- full_row_this_source_erased: ΔNLL `0.12050263344737497`, Δrank `40.41460573476702`, KL `0.0585866550863829`
- full_row_all_sources_erased: ΔNLL `0.006107997764769849`, Δrank `11.81021505376344`, KL `0.015794547731065155`
- source penalty ΔNLL: `0.101124737021115`; view absence ΔNLL: `0.08078597441899707`; this-source erasure ΔNLL: `0.13125551762311774`; all-source erasure ΔNLL: `0.0168608819405126`

### `clean_pres_lambda1_eval_full80_step090`
- pair_correct_source: ΔNLL `-0.0215497929500007`, Δrank `1.85752688172043`, KL `0.010494012901637354`
- pair_wrong_source: ΔNLL `0.09856245073115981`, Δrank `32.93888888888889`, KL `0.05981852872782805`
- view_only: ΔNLL `0.07631421343149704`, Δrank `39.26881720430107`, KL `0.04164716328749718`
- full_row_original: ΔNLL `-0.010056465667162152`, Δrank `2.614247311827956`, KL `0.01702273354371739`
- full_row_this_source_erased: ΔNLL `0.12699320075389225`, Δrank `38.787992831541224`, KL `0.059805014870752495`
- full_row_all_sources_erased: ΔNLL `0.010086663962612239`, Δrank `11.885931899641577`, KL `0.01701330379718551`
- source penalty ΔNLL: `0.1201122436811605`; view absence ΔNLL: `0.09786400638149774`; this-source erasure ΔNLL: `0.1370496664210544`; all-source erasure ΔNLL: `0.02014312962977438`

### `densemask_sparselabel_seed62064_step090`
- pair_correct_source: ΔNLL `-0.02412776025733994`, Δrank `5.454301075268817`, KL `0.02879933990997094`
- pair_wrong_source: ΔNLL `0.1491197167884051`, Δrank `49.107258064516124`, KL `0.11869238856132037`
- view_only: ΔNLL `0.10708051262186394`, Δrank `57.55492831541219`, KL `0.08033225240962898`
- full_row_original: ΔNLL `-0.007271608262528139`, Δrank `6.4995519713261665`, KL `0.04171747311114631`
- full_row_this_source_erased: ΔNLL `0.2103753821632247`, Δrank `65.23664874551972`, KL `0.12082205326176428`
- full_row_all_sources_erased: ΔNLL `0.0247773970921247`, Δrank `20.168727598566306`, KL `0.03510153154157416`
- source penalty ΔNLL: `0.17324747704574503`; view absence ΔNLL: `0.13120827287920386`; this-source erasure ΔNLL: `0.2176469904257529`; all-source erasure ΔNLL: `0.03204900535465284`

## Source response

- `rollback_matched_densemask_update`: Qwen Δspecific Tfit `-0.5032120496034622`, Δrank `-302.75`; common both `21/25`, Δswing Tfit `0.4116051115308488`, Δrank `84.77714285714288`.
- `densemask_sparselabel_seed62064_step090`: Qwen Δspecific Tfit `0.14678872493386735`, Δrank `66.97135416666667`; common both `21/25`, Δswing Tfit `0.5443552198083627`, Δrank `117.79809523809524`.
- `clean_pres_lambda1_eval_full80_step090`: Qwen Δspecific Tfit `0.10074627038440666`, Δrank `42.69649305555556`; common both `21/25`, Δswing Tfit `0.34587322203885923`, Δrank `74.87380952380953`.

## Interpretation

- evidence_comparison: Compare rollback_matched_densemask_update to clean_pres_lambda1_eval_full80_step090 at matched ordinary calibration KL. If rollback has materially weaker source response or larger wrong/view/source-erased costs, deterministic KL is doing more than simple update shrinkage; if they closely coincide, simple attenuation remains sufficient.
- source_response_comparison: Qwen and common-source summaries are non-official mechanism readouts; they test retained source-conditioned behavior after drift matching, not benchmark score.
- official_score_status: This script does not replace the official-sized zero-shot/Reading, repaired SuperGLUE, or measured AoA comparisons required for the candidate endpoint.
