# earlier analysis rollback control

This readout compares clean eval-mode preservation with a matched rollback along the acquisition-only `(M,S)` private-adapter update. It is an explanatory computation, not a candidate checkpoint.

## Matched ordinary-text drift

- Clean eval KL(parent||model) on ordinary non-Qwen calibration positions: `0.007061894240905531` over `33` positions.
- Dense-mask alpha=1 KL on the same positions: `0.015065510273747019`.
- Selected rollback alpha: `0.75` with KL `0.007379046892200484` and absolute error `0.0003171526512949531`.

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
- pair_correct_source: ΔNLL `-0.2256463515882691`, Δrank `-9.3`, KL `0.01568242542683341`
- pair_wrong_source: ΔNLL `-0.16638287703196203`, Δrank `-11.133333333333333`, KL `0.010873235622420907`
- view_only: ΔNLL `-0.16651397546132413`, Δrank `-46.13333333333334`, KL `0.009751429222524166`
- full_row_original: ΔNLL `-0.2396900046461572`, Δrank `-10.766666666666667`, KL `0.01725303648369542`
- full_row_this_source_erased: ΔNLL `-0.16953893105189`, Δrank `-74.78333333333332`, KL `0.01066391720281293`
- full_row_all_sources_erased: ΔNLL `-0.14857987562815342`, Δrank `-60.96666666666666`, KL `0.005723388415450851`
- source penalty ΔNLL: `0.059263474556307025`; view absence ΔNLL: `0.05913237612694493`; this-source erasure ΔNLL: `0.07015107359426712`; all-source erasure ΔNLL: `0.0911101290180036`

### `rollback_matched_densemask_update`
- pair_correct_source: ΔNLL `0.06912021196136867`, Δrank `6.55`, KL `0.03838969199447699`
- pair_wrong_source: ΔNLL `-0.1320780913035075`, Δrank `-15.38333333333333`, KL `0.048484759964048864`
- view_only: ΔNLL `-0.04380380312601728`, Δrank `-84.06666666666668`, KL `0.03393529629490028`
- full_row_original: ΔNLL `0.348663592307518`, Δrank `29.133333333333333`, KL `0.18062656171653846`
- full_row_this_source_erased: ΔNLL `0.23832210302352913`, Δrank `-7.2`, KL `0.11459045701582607`
- full_row_all_sources_erased: ΔNLL `0.0017932017644247012`, Δrank `-23.33333333333335`, KL `0.013743855257052929`
- source penalty ΔNLL: `-0.20119830326487625`; view absence ΔNLL: `-0.11292401508738603`; this-source erasure ΔNLL: `-0.11034148928398882`; all-source erasure ΔNLL: `-0.3468703905430933`

### `clean_pres_lambda1_eval_full80_step090`
- pair_correct_source: ΔNLL `0.0683995219723632`, Δrank `3.85`, KL `0.010494012901637354`
- pair_wrong_source: ΔNLL `0.8163452982902527`, Δrank `49.00000000000001`, KL `0.05981852872782805`
- view_only: ΔNLL `-0.03203425407409677`, Δrank `-80.26666666666668`, KL `0.04164716328749718`
- full_row_original: ΔNLL `0.3013239293824882`, Δrank `20.783333333333335`, KL `0.01702273354371739`
- full_row_this_source_erased: ΔNLL `0.23425736029942829`, Δrank `-8.96666666666663`, KL `0.059805014870752495`
- full_row_all_sources_erased: ΔNLL `0.0018578052520751953`, Δrank `-19.083333333333336`, KL `0.01701330379718551`
- source penalty ΔNLL: `0.7479457763178894`; view absence ΔNLL: `-0.10043377604646002`; this-source erasure ΔNLL: `-0.06706656908305994`; all-source erasure ΔNLL: `-0.2994661241304131`

### `densemask_sparselabel_seed62064_step090`
- pair_correct_source: ΔNLL `0.1341755797465642`, Δrank `12.75`, KL `0.02879933990997094`
- pair_wrong_source: ΔNLL `0.8373438199361164`, Δrank `49.55`, KL `0.11869238856132037`
- view_only: ΔNLL `-0.051616565386454295`, Δrank `-93.55`, KL `0.08033225240962898`
- full_row_original: ΔNLL `0.507252428525438`, Δrank `47.0`, KL `0.04171747311114631`
- full_row_this_source_erased: ΔNLL `0.39960375229517614`, Δrank `3.4666666666667028`, KL `0.12082205326176428`
- full_row_all_sources_erased: ΔNLL `0.01291809876759853`, Δrank `-31.666666666666696`, KL `0.03510153154157416`
- source penalty ΔNLL: `0.7031682401895523`; view absence ΔNLL: `-0.1857921451330186`; this-source erasure ΔNLL: `-0.10764867623026184`; all-source erasure ΔNLL: `-0.4943343297578394`

## Source response

- `rollback_matched_densemask_update`: Qwen Δspecific Tfit `None`, Δrank `None`; common both `21/25`, Δswing Tfit `0.41160587347689115`, Δrank `84.77714285714288`.
- `densemask_sparselabel_seed62064_step090`: Qwen Δspecific Tfit `0.14678872493386735`, Δrank `66.97135416666667`; common both `21/25`, Δswing Tfit `0.5443552198083627`, Δrank `117.79809523809524`.
- `clean_pres_lambda1_eval_full80_step090`: Qwen Δspecific Tfit `0.10074627038440666`, Δrank `42.69649305555556`; common both `21/25`, Δswing Tfit `0.34587322203885923`, Δrank `74.87380952380953`.

## Interpretation

- evidence_comparison: Compare rollback_matched_densemask_update to clean_pres_lambda1_eval_full80_step090 at matched ordinary calibration KL. If rollback has materially weaker source response or larger wrong/view/source-erased costs, deterministic KL is doing more than simple update shrinkage; if they closely coincide, simple attenuation remains sufficient.
- source_response_comparison: Qwen and common-source summaries are non-official mechanism readouts; they test retained source-conditioned behavior after drift matching, not benchmark score.
- official_score_status: This script does not replace the official-sized zero-shot/Reading, repaired SuperGLUE, or measured AoA comparisons required for the candidate endpoint.
