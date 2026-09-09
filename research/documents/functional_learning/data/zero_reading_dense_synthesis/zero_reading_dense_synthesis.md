# dense focus profile and next uncertainty dense zero-shot/Reading synthesis

Scope: completed zero-shot/Reading columns currently present in both dense official-compatible payloads. SuperGLUE and AoA are not complete here, so this is not final v5 evidence.

## Score deltas vs coherent86

### seed62064
- BLiMP: parent `68.51596989215173`, dense `68.06551957317373`, delta `-0.4504503189779996`.
- Supplement: parent `63.63536879546028`, dense `63.037492760948034`, delta `-0.5978760345122467`.
- EWoK: parent `50.01961589868545`, dense `49.92247815652366`, delta `-0.09713774216179161`.
- Entity: parent `28.322219220191908`, dense `29.36695227588256`, delta `1.044733055690653`.
- COMPS: parent `52.0471751296737`, dense `52.153718082721866`, delta `0.10654295304816941`.
- GlobalPIQA_parallel: parent `29.12621359223301`, dense `30.097087378640776`, delta `0.9708737864077648`.
- GlobalPIQA_nonparallel: parent `48.0`, dense `50.0`, delta `2.0`.
- GlobalPIQA: parent `38.56310679611651`, dense `40.04854368932039`, delta `1.4854368932038824`.
- Reading: parent `8.165`, dense `8.219999999999999`, delta `0.054999999999999716`.
- cheap7_mean: parent `44.181207961754225`, dense `44.40210064836718`, delta `0.2208926866129559`.

### seed62065
- BLiMP: parent `68.51596989215173`, dense `68.0283001371816`, delta `-0.48766975497012766`.
- Supplement: parent `63.63536879546028`, dense `63.09340542857522`, delta `-0.5419633668850636`.
- EWoK: parent `50.01961589868545`, dense `49.76993067706473`, delta `-0.24968522162072304`.
- Entity: parent `28.322219220191908`, dense `29.415692306635133`, delta `1.0934730864432254`.
- COMPS: parent `52.0471751296737`, dense `52.14545705091899`, delta `0.09828192124529522`.
- GlobalPIQA_parallel: parent `29.12621359223301`, dense `30.097087378640776`, delta `0.9708737864077648`.
- GlobalPIQA_nonparallel: parent `48.0`, dense `50.0`, delta `2.0`.
- GlobalPIQA: parent `38.56310679611651`, dense `40.04854368932039`, delta `1.4854368932038824`.
- Reading: parent `8.165`, dense `8.21`, delta `0.045000000000001705`.
- cheap7_mean: parent `44.181207961754225`, dense `44.38733275567086`, delta `0.2061247939166364`.

## Seed concordance on full zero-shot/Reading items

Common items `170722`, seed agreement `0.9977858741111281`, shared gains `4761`, shared losses `4870`, shared net `-109`, gain/loss Jaccard `0.9629854368932039` / `0.9615004935834156`.

## By-column item concordance

- BLiMP: n `59875`, seed agreement `0.998580375782881`, shared gains/losses `765`/`1037`, shared net `-272`.
- Supplement: n `5218`, seed agreement `0.9988501341510158`, shared gains/losses `52`/`57`, shared net `-5`.
- EWoK: n `7618`, seed agreement `0.9964557626673668`, shared gains/losses `229`/`220`, shared net `9`.
- Entity: n `6780`, seed agreement `0.9986725663716814`, shared gains/losses `218`/`176`, shared net `42`.
- COMPS: n `91028`, seed agreement `0.997242606670475`, shared gains/losses `3493`/`3379`, shared net `114`.
- GlobalPIQA_parallel: n `103`, seed agreement `1.0`, shared gains/losses `2`/`1`, shared net `1`.
- GlobalPIQA_nonparallel: n `100`, seed agreement `1.0`, shared gains/losses `2`/`0`, shared net `2`.

## Scientific interpretation

The larger zero-shot/Reading surface preserves a reproducible dense gain in Entity, COMPS, GlobalPIQA, and Reading, with reproducible BLiMP/Supplement/EWoK costs. Whether this is net useful depends heavily on completed SuperGLUE and exact Overall arithmetic.
