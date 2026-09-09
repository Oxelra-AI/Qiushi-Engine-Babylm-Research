# earlier analysis dense-focus score fragility and interpretation profile

Scope: already-written official-sized zero-shot/Reading payloads for dense seed62064 and seed62065, compared with the true coherent86/v4 reference. SuperGLUE and AoA are deliberately excluded here because the official evaluations are not yet complete at this stage and interim SuperGLUE fields may contain only partial task entries.

## Seven-column arithmetic versus scientific breadth
### seed62064 payload-column deltas
- BLiMP: delta `-0.45` (parent `68.51`, dense `68.06`).
- Supplement: delta `-0.6` (parent `63.64`, dense `63.04`).
- EWoK: delta `-0.1` (parent `50.02`, dense `49.92`).
- Entity: delta `1.05` (parent `28.32`, dense `29.37`).
- COMPS: delta `0.1` (parent `52.05`, dense `52.15`).
- GlobalPIQA: delta `1.485` (parent `38.565`, dense `40.05`).
- Reading: delta `0.055` (parent `8.165`, dense `8.22`).
- Seven-column sum delta `1.54`; mean delta `0.22`.
- GlobalPIQA delta `1.485`, fraction of seven-column sum `0.96428571`.
- Other six columns excluding GlobalPIQA: sum `0.055`, mean `0.0091666667`.
- Excluding both GlobalPIQA and Entity: five-column sum `-0.995`, mean `-0.199`.
- BLiMP+Supplement+EWoK sum `-1.15`, mean `-0.38333333`.

### seed62065 payload-column deltas
- BLiMP: delta `-0.49` (parent `68.51`, dense `68.02`).
- Supplement: delta `-0.55` (parent `63.64`, dense `63.09`).
- EWoK: delta `-0.25` (parent `50.02`, dense `49.77`).
- Entity: delta `1.1` (parent `28.32`, dense `29.42`).
- COMPS: delta `0.1` (parent `52.05`, dense `52.15`).
- GlobalPIQA: delta `1.485` (parent `38.565`, dense `40.05`).
- Reading: delta `0.045` (parent `8.165`, dense `8.21`).
- Seven-column sum delta `1.44`; mean delta `0.20571429`.
- GlobalPIQA delta `1.485`, fraction of seven-column sum `1.03125`.
- Other six columns excluding GlobalPIQA: sum `-0.045`, mean `-0.0075`.
- Excluding both GlobalPIQA and Entity: five-column sum `-1.145`, mean `-0.229`.
- BLiMP+Supplement+EWoK sum `-1.29`, mean `-0.43`.

Interpretation: the two-seed official-sized non-SuperGLUE surface preserves a real dense-induced redistribution. The positive seven-column mean is not broad in the scalar sense: GlobalPIQA carries nearly all of seed62064's positive sum and more than all of seed62065's positive sum, while the other six columns are near zero in aggregate and the five columns excluding Entity and GlobalPIQA are negative.

## GlobalPIQA item-level dependence
Across both GlobalPIQA slices there are `203` examples. Parent/seed62064/seed62065 correctness patterns are `{'000': 121, '011': 4, '100': 1, '111': 77}`.
Shared dense gains: `4`; shared dense losses: `1`; shared net item delta: `3`; dense-seed agreement fraction `1`.
- seed62064: parent correct `78/203`, dense correct `81/203`, gains `4`, losses `1`, net `3`, exact paired sign-test p `0.375`.
- seed62065: parent correct `78/203`, dense correct `81/203`, gains `4`, losses `1`, net `3`, exact paired sign-test p `0.375`.

### Shared GlobalPIQA gains
- `GlobalPIQA_parallel` `parallel_ex000000_eng_latn` [object_properties_interactions]: target="The amount of air in the bag stays the same"; parent="The amount of air in the bag increases"; dense64="The amount of air in the bag stays the same"; dense65="The amount of air in the bag stays the same"; prompt="A plastic bag is filled with air and then sealed. When an object is placed on the bag, what happens?"
- `GlobalPIQA_parallel` `parallel_ex000071_eng_latn` [object_properties_interactions, spatial]: target="The wick should be longer than the height of the candle wax"; parent="The wick should be shorter than the height of the candle wax"; dense64="The wick should be longer than the height of the candle wax"; dense65="The wick should be longer than the height of the candle wax"; prompt="How long is the wick of a candle, compared to the height of the candle wax?"
- `GlobalPIQA_nonparallel` `group0123_ex000032_eng_latn_0_v1` [None]: target="Place the clay sticks onto a baking sheet and bake at 350°F for one minute."; parent="Place the clay sticks onto a baking sheet and bake at 350°F for twenty minutes."; dense64="Place the clay sticks onto a baking sheet and bake at 350°F for one minute."; dense65="Place the clay sticks onto a baking sheet and bake at 350°F for one minute."; prompt="How do you bake small polymer clay sticks to make clay sprinkles?"
- `GlobalPIQA_nonparallel` `group0123_ex000084_eng_latn_0_v1` [None]: target="Place a dry towel on the counter first."; parent="Place a wet towel on the counter first."; dense64="Place a dry towel on the counter first."; dense65="Place a dry towel on the counter first."; prompt="How do I protect the counter from the heat of the iron ?"

### Shared GlobalPIQA losses
- `GlobalPIQA_parallel` `parallel_ex000094_eng_latn` [time]: target="Drying the laundry"; parent="Drying the laundry"; dense64="Folding the laundry"; dense65="Folding the laundry"; prompt="You put some cookies in the oven before doing some housework. What housework would not be possible to finish before the cookies need to be taken out of the oven?"

The identical GlobalPIQA score across seeds is therefore shared predictions on the same small example set, not independent support over new evaluation cases. It remains legitimate in the official aggregate, but it cannot carry the broader method claim by itself.

## Larger effect surfaces that do not rely on GlobalPIQA
Entity: seed62064 gains/losses `219/177` (net `42` over `6780`); seed62065 `222/179` (net `43`).
BLiMP: seed62064 gains/losses `776/1048` (net `-272` over `59875`); seed62065 `785/1080` (net `-295`).
Entity depth/family and BLiMP family localization remains in `research/documents/functional_learning/data/zero_reading_two_seed_profile/zero_reading_two_seed_profile.md`: dense helps Entity operation depths 2--5 and loses at 0_ops; BLiMP costs are spread across all coarse families rather than a single isolated subtask.
Trained unchanged-Qwen source-help also reproduces: seed62064 Δsource-help `0.10663173` and seed62065 `0.11037358` over 4,783 targets / 1,200 pairs.
Controlled common-target evidence response reproduces: seed62064 source-original/source-altered/held-source deltas `0.26365687`/`0.33972819`/`0.36622328`, seed62065 `0.25726718`/`0.33878437`/`0.36692036`; no-source movement stays near `0.02` in both seeds.

## Consequence for the dense-mask/sparse-label control
Sparse focus labels/masks `28590` Qwen target tokens; dense focus labels/masks `176607`; dense-mask/sparse-label would keep `28590` labels while masking `176607` content tokens, of which `148017` are mask-only. This directly separates input-side clue suppression from added supervised target coverage.
The control should be judged by whether it preserves the large Entity and source-responsive signals while reducing BLiMP/Supplement/EWoK costs. It is scientifically valuable even if the official aggregate narrowly wins, because the benchmark surplus is currently fragile whereas the gain/cost redistribution is stable and mechanistically important.

## Files
- coherent86_zero_reading_payload: `experiments/archive/frontier_consolidation/data/private_scale_sentinel_eval/coherent86_private_scale_0p75/per_target/coherent86_private_scale_0p75.json`
- dense_seed62064_zero_reading_payload: `experiments/archive/functional_learning/data/dense_focus_official_eval_seed62064/per_target/dense_focus_seed62064_u0080.json`
- dense_seed62065_zero_reading_payload: `experiments/archive/functional_learning/data/dense_focus_official_eval_seed62065/per_target/dense_focus_seed62065_u0080.json`
- input_label_profile: `experiments/archive/functional_learning/data/dense_input_label_profile/dense_input_label_profile.json`
- seed62064_common_target: `experiments/archive/functional_learning/data/unchanged_dense_focus_eval/common_target/summary_unchanged_correspondence_focus_weighted_u0080.json`
- seed62065_common_target: `experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/common_target/summary_unchanged_correspondence_focus_weighted_u0080.json`
- seed62064_qwen_view: `experiments/archive/functional_learning/data/unchanged_dense_focus_eval/qwen_view_surface/summary.json`
- seed62065_qwen_view: `experiments/archive/functional_learning/data/dense_focus_rep_seed62065_eval/qwen_view_surface/summary.json`
