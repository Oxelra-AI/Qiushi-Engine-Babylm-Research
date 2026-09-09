# fast full seed43122 reconciliation — fast/full reconciliation for seed43122

Official seed43122 Overall is `41.24823958912208` with margin `-0.5517604108779182` over visible 41.8; it is `-0.7848952009527181` below seed43022.
Fast equal7 delta was `-1.355714285714285`; official full delta averaged over the same seven columns is `-0.8468886294750527`.
Pearson correlation over seven matched columns: `0.8273797106782358`.

Column reconciliation:
- BLiMP: fast `-0.8799999999999955`, official `-1.2079230106099175`, full-minus-fast `-0.327923010609922`, same_sign `True`
- Supplement: fast `-3.200000000000003`, official `-1.3234112442271524`, full-minus-fast `1.8765887557728504`, same_sign `True`
- EWoK: fast `-3.730000000000004`, official `-1.644857577603645`, full-minus-fast `2.085142422396359`, same_sign `True`
- Entity: fast `-1.3900000000000006`, official `-1.4604021386814061`, full-minus-fast `-0.07040213868140555`, same_sign `True`
- COMPS: fast `-0.4299999999999997`, official `-0.42963389043958244`, full-minus-fast `0.0003661095604172715`, same_sign `True`
- GlobalPIQA: fast `-0.48499999999999943`, official `-0.4854368932038824`, full-minus-fast `-0.0004368932038829598`, same_sign `True`
- Reading: fast `0.625`, official `0.6234443484402163`, full-minus-fast `-0.001555651559783655`, same_sign `True`

Most negative official families:
- EWoK: mean_delta `-1.6448575776036476`, worst [['material-dynamics', -11.558441558441558], ['spatial-relations', -4.285714285714292], ['physical-interactions', -3.237410071942449]]
- Entity: mean_delta `-1.4604021386814054`, worst [['ambiref_5_ops', -8.13008130081301], ['move_contents_3_ops', -8.128078817733993], ['regular_4_ops', -7.989690721649485]]
- Supplement: mean_delta `-1.3234112442271595`, worst [['qa_congruence_easy', -6.25], ['qa_congruence_tricky', -3.6363636363636402], ['hypernym', 0.4750593824227991]]
- BLiMP: mean_delta `-1.2079230106099135`, worst [['principle_A_reconstruction', -31.64426059979317], ['superlative_quantifiers_1', -23.799795709908068], ['wh_questions_object_gap', -16.06519208381839]]
- SuperGLUE: mean_delta `-1.1358364022491452`, worst [['wsc', -9.615384615384606], ['rte', -1.4388489208633075], ['mnli', -0.0407497962510206]]
- global_piqa_parallel: mean_delta `-0.9708737864077666`, worst [['global_piqa_parallel', -0.9708737864077666]]

Interpretation: the below-leader seed43122 result is not an AoA/EWoK coordinate artifact. Fast screening was directionally informative, but the official full vector shows a broad seed-dependent representation/optimization difference plus a SuperGLUE loss that fast evaluation did not observe.
JSON: `experiments/archive/representation_and_objectives/data/fast_full_seed43122_reconciliation/fast_full_seed43122_reconciliation.json`
