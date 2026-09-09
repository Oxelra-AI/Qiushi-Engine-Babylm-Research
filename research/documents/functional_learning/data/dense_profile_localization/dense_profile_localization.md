# dense focus profile and next uncertainty dense-focus profile localization

This readout uses only dense focus official and mechanism state-known official seed62064 columns (BLiMP, Supplement, EWoK, Entity) plus frozen fast-screen payloads. It does not infer unfinished official columns.

## Completed official-column profile: dense seed62064 vs coherent86

- BLiMP: parent `68.515970`, dense `68.065520`, delta `-0.450450`, net items `-272` / n `59875`.
- Supplement: parent `63.635369`, dense `63.037493`, delta `-0.597876`, net items `-5` / n `5218`.
- EWoK: parent `50.019616`, dense `49.922478`, delta `-0.097138`, net items `13` / n `7618`.
- Entity: parent `28.322219`, dense `29.366952`, delta `1.044733`, net items `42` / n `6780`.

## Pending-official arithmetic

Known completed-column delta sum: `-0.100731`. The four unresolved non-AoA columns (COMPS, GlobalPIQA, Reading, SuperGLUE) need total delta `>0.100731` (average `>0.025183`) for the AoA0 projected Overall to exceed coherent86.
Fast seed62064 proxy for COMPS+GlobalPIQA+Reading is `1.646980`, which would tolerate SuperGLUE delta down to about `-1.546249` if the proxy were exact. This is only arithmetic context; terminal official payloads dominate.

## Entity localization in the completed official payload

- depth 4_ops: delta `3.1489`, parent `31.2340`, dense `34.3830`, net items `37`, n `1175`.
- depth 0_ops: delta `-3.1149`, parent `37.1836`, dense `34.0688`, net items `-48`, n `1541`.
- depth 5_ops: delta `2.7027`, parent `35.1351`, dense `37.8378`, net items `9`, n `333`.
- depth 2_ops: delta `1.8899`, parent `22.8431`, dense `24.7329`, net items `23`, n `1217`.
- depth 3_ops: delta `1.6129`, parent `28.1452`, dense `29.7581`, net items `20`, n `1240`.
- depth 1_ops: delta `0.0785`, parent `15.9341`, dense `16.0126`, net items `1`, n `1274`.

## BLiMP completed-payload family movement

- binding_anaphora: delta `-0.7680`, parent `65.8986`, dense `65.1306`, net `-50`, n `6510`.
- other_blimp: delta `-0.7431`, parent `79.8301`, dense `79.0870`, net `-21`, n `2826`.
- agreement_and_number: delta `-0.5834`, parent `78.9125`, dense `78.3291`, net `-81`, n `13885`.
- argument_structure_ellipsis: delta `-0.4277`, parent `65.4490`, dense `65.0214`, net `-48`, n `11224`.
- extraction_and_islands: delta `-0.3229`, parent `57.8337`, dense `57.5108`, net `-44`, n `13627`.
- quantifier_npi_scope: delta `-0.2372`, parent `69.7026`, dense `69.4654`, net `-28`, n `11803`.

## Fast-to-official subtask concordance

- BLiMP: shared subtasks `67`, Pearson(two-seed fast avg, official) `0.8392655634390194`, same-sign fraction `0.7878787878787878`.
- Supplement: shared subtasks `5`, Pearson(two-seed fast avg, official) `0.0983462290727785`, same-sign fraction `0.25`.
- EWoK: shared subtasks `11`, Pearson(two-seed fast avg, official) `0.5632997688193842`, same-sign fraction `0.45454545454545453`.
- Entity: shared subtasks `6`, Pearson(two-seed fast avg, official) `0.9995967998930199`, same-sign fraction `1.0`.

## Strongest two-seed-stable fast subtask movements

- COMPS/wugs_dist_before: shared net `151`, shared gains/losses `780`/`629`, seed agreement `0.9956822107081175`.
- COMPS/base: shared net `82`, shared gains/losses `1588`/`1506`, seed agreement `0.9976692338873125`.
- COMPS/wugs_dist_in_between: shared net `-68`, shared gains/losses `644`/`712`, seed agreement `0.9969775474956822`.
- COMPS/wugs: shared net `-51`, shared gains/losses `481`/`532`, seed agreement `0.9975532527345999`.
- Entity/regular_0_ops: shared net `-25`, shared gains/losses `7`/`32`, seed agreement `1.0`.
- Entity/regular_4_ops: shared net `11`, shared gains/losses `15`/`4`, seed agreement `0.9974226804123711`.
- BLiMP/wh_questions_object_gap: shared net `10`, shared gains/losses `11`/`1`, seed agreement `0.99`.
- Entity/regular_2_ops: shared net `10`, shared gains/losses `20`/`10`, seed agreement `0.9975308641975309`.
- BLiMP/existential_there_quantifiers_2: shared net `9`, shared gains/losses `10`/`1`, seed agreement `1.0`.
- BLiMP/superlative_quantifiers_2: shared net `-9`, shared gains/losses `1`/`10`, seed agreement `1.0`.
- BLiMP/left_branch_island_echo_question: shared net `-8`, shared gains/losses `0`/`8`, seed agreement `1.0`.
- BLiMP/only_npi_licensor_present: shared net `-8`, shared gains/losses `1`/`9`, seed agreement `1.0`.
- BLiMP/distractor_agreement_relational_noun: shared net `-7`, shared gains/losses `2`/`9`, seed agreement `1.0`.
- BLiMP/principle_A_domain_1: shared net `-7`, shared gains/losses `2`/`9`, seed agreement `0.995`.
- Entity/regular_1_ops: shared net `7`, shared gains/losses `14`/`7`, seed agreement `1.0`.
- BLiMP/transitive: shared net `-6`, shared gains/losses `0`/`6`, seed agreement `1.0`.
- BLiMP/animate_subject_trans: shared net `-5`, shared gains/losses `0`/`5`, seed agreement `1.0`.
- BLiMP/coordinate_structure_constraint_complex_left_branch: shared net `-5`, shared gains/losses `2`/`7`, seed agreement `0.995`.
- BLiMP/determiner_noun_agreement_1: shared net `-5`, shared gains/losses `0`/`5`, seed agreement `1.0`.
- BLiMP/matrix_question_npi_licensor_present: shared net `5`, shared gains/losses `9`/`4`, seed agreement `1.0`.

## Scientific use

If full official evaluation retains Entity/source-responsive gains but BLiMP/Supplement or SuperGLUE costs erase the aggregate, the prepared dense-mask/sparse-label contrast becomes a principled repair experiment: it separates input-side clue removal from dense supervised target coverage. If the full gains themselves disappear, dense should be treated as a fast/trained-material movement with weaker practical value.
