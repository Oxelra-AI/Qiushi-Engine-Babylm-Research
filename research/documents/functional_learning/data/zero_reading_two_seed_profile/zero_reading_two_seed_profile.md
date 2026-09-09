# dense focus profile and next uncertainty zero-shot/Reading two-seed profile

Scope: current completed zero-shot/Reading official-sized columns only; SuperGLUE and AoA are excluded.

## Payload score deltas vs coherent86

### seed62064
- BLiMP: `-0.45000000000000284`
- Supplement: `-0.6000000000000014`
- EWoK: `-0.10000000000000142`
- Entity: `1.0500000000000007`
- COMPS: `0.10000000000000142`
- GlobalPIQA: `1.4849999999999994`
- Reading: `0.054999999999999716`
- cheap7_mean_payload: `0.22000000000000597`

### seed62065
- BLiMP: `-0.4900000000000091`
- Supplement: `-0.5499999999999972`
- EWoK: `-0.25`
- Entity: `1.1000000000000014`
- COMPS: `0.10000000000000142`
- GlobalPIQA: `1.4849999999999994`
- Reading: `0.045000000000001705`
- cheap7_mean_payload: `0.2057142857142864`

## Entity by operation depth

- 4_ops: mean delta `3.1914893617021285`, seed62064 `3.1489361702127674`, seed62065 `3.2340425531914896`, same sign `True`, n `1175`.
- 0_ops: mean delta `-3.1148604802076605`, seed62064 `-3.1148604802076605`, seed62065 `-3.1148604802076605`, same sign `True`, n `1541`.
- 5_ops: mean delta `2.852852852852852`, seed62064 `2.7027027027027017`, seed62065 `3.003003003003002`, same sign `True`, n `333`.
- 2_ops: mean delta `1.8898931799507004`, seed62064 `1.8898931799507004`, seed62065 `1.8898931799507004`, same sign `True`, n `1217`.
- 3_ops: mean delta `1.57258064516129`, seed62064 `1.612903225806452`, seed62065 `1.5322580645161281`, same sign `True`, n `1240`.
- 1_ops: mean delta `0.07849293563579351`, seed62064 `0.07849293563579351`, seed62065 `0.07849293563579351`, same sign `True`, n `1274`.

## Entity by task family

- move_contents: mean delta `0.9654243376739995`, seed62064 `0.8980691513246519`, seed62065 `1.032779524023347`, same sign `True`, n `2227`.
- ambiref: mean delta `0.604751619870413`, seed62064 `0.6479481641468716`, seed62065 `0.5615550755939545`, same sign `True`, n `2315`.
- regular: mean delta `0.31277926720285976`, seed62064 `0.31277926720285976`, seed62065 `0.31277926720285976`, same sign `True`, n `2238`.

## BLiMP family movement

- binding_anaphora: mean delta `-0.7757296466973926`, seed62064 `-0.7680491551459312`, seed62065 `-0.7834101382488541`, same sign `True`, n `6510`.
- other_blimp: mean delta `-0.7430997876857788`, seed62064 `-0.7430997876857788`, seed62065 `-0.7430997876857788`, same sign `True`, n `2826`.
- agreement_and_number: mean delta `-0.5941663665826411`, seed62064 `-0.5833633417356907`, seed62065 `-0.6049693914295915`, same sign `True`, n `13885`.
- argument_structure_ellipsis: mean delta `-0.45883820384889873`, seed62064 `-0.42765502494654584`, seed62065 `-0.4900213827512516`, same sign `True`, n `11224`.
- extraction_and_islands: mean delta `-0.3595802451016361`, seed62064 `-0.32288838335657033`, seed62065 `-0.3962721068467019`, same sign `True`, n `13627`.
- quantifier_npi_scope: mean delta `-0.24570024570024884`, seed62064 `-0.23722782343472204`, seed62065 `-0.25417266796577564`, same sign `True`, n `11803`.

## Largest stable BLiMP subtask movements

- matrix_question_npi_licensor_present: mean delta `3.9289558665231397`, seed62064 `3.8751345532830967`, seed62065 `3.9827771797631826`, n `929`.
- principle_A_domain_1: mean delta `-3.3916849015317325`, seed62064 `-3.501094091903724`, seed62065 `-3.282275711159741`, n `914`.
- distractor_agreement_relational_noun: mean delta `-3.109137055837561`, seed62064 `-3.045685279187815`, seed62065 `-3.172588832487307`, n `788`.
- existential_there_quantifiers_2: mean delta `2.908891328210757`, seed62064 `3.073545554335894`, seed62065 `2.74423710208562`, n `911`.
- only_npi_licensor_present: mean delta `-2.8911564625850303`, seed62064 `-2.834467120181401`, seed62065 `-2.94784580498866`, n `882`.
- superlative_quantifiers_2: mean delta `-2.890466531440161`, seed62064 `-2.8397565922920904`, seed62065 `-2.941176470588232`, n `986`.
- determiner_noun_agreement_1: mean delta `-2.3143164693218594`, seed62064 `-2.2604951560818165`, seed62065 `-2.3681377825619023`, n `929`.
- wh_vs_that_no_gap: mean delta `-2.0905923344947865`, seed62064 `-2.0905923344947865`, seed62065 `-2.0905923344947865`, n `861`.
- left_branch_island_simple_question: mean delta `-1.9978969505783368`, seed62064 `-1.8927444794952635`, seed62065 `-2.10304942166141`, n `951`.
- left_branch_island_echo_question: mean delta `-1.9535374868004212`, seed62064 `-1.900739176346356`, seed62065 `-2.0063357972544864`, n `947`.
- sentential_negation_npi_licensor_present: mean delta `-1.8498367791077328`, seed62064 `-1.8498367791077328`, seed62065 `-1.8498367791077328`, n `919`.
- coordinate_structure_constraint_object_extraction: mean delta `-1.685985247629091`, seed62064 `-1.580611169652272`, seed62065 `-1.7913593256059102`, n `949`.
- sentential_subject_island: mean delta `1.6649323621227872`, seed62064 `1.5608740894901132`, seed62065 `1.7689906347554611`, n `961`.
- wh_questions_object_gap: mean delta `1.5715948777648379`, seed62064 `1.7462165308498214`, seed62065 `1.3969732246798543`, n `859`.
- ellipsis_n_bar_2: mean delta `1.5700483091787305`, seed62064 `1.5700483091787305`, seed62065 `1.5700483091787305`, n `828`.
- npi_present_1: mean delta `-1.540154015401539`, seed62064 `-1.6501650165016493`, seed62065 `-1.4301430143014287`, n `909`.
- animate_subject_trans: mean delta `-1.5167930660888373`, seed62064 `-1.5167930660888373`, seed62065 `-1.5167930660888373`, n `923`.
- drop_argument: mean delta `-1.4673913043478315`, seed62064 `-1.4130434782608745`, seed62065 `-1.5217391304347885`, n `920`.
- wh_vs_that_no_gap_long_distance: mean delta `-1.3714285714285808`, seed62064 `-1.3714285714285808`, seed62065 `-1.3714285714285808`, n `875`.
- determiner_noun_agreement_with_adj_irregular_2: mean delta `-1.3690476190476133`, seed62064 `-1.3095238095238102`, seed62065 `-1.4285714285714164`, n `840`.

## Other completed-column largest grouped movements

### Supplement_by_subtask
- qa_congruence_easy: mean delta `-1.5625`, seed62064 `-1.5625`, seed62065 `-1.5625`, same sign `True`, n `64`.
- hypernym: mean delta `-0.9501187648456053`, seed62064 `-0.9501187648456053`, seed62065 `-0.9501187648456053`, same sign `True`, n `842`.
- qa_congruence_tricky: mean delta `-0.606060606060602`, seed62064 `-0.606060606060602`, seed62065 `-0.606060606060602`, same sign `True`, n `165`.
- turn_taking: mean delta `0.1785714285714306`, seed62064 `0.0`, seed62065 `0.3571428571428612`, same sign `False`, n `280`.
- subject_aux_inversion: mean delta `0.09050943884147244`, seed62064 `0.12929919834496673`, seed62065 `0.051719679337978164`, same sign `True`, n `3867`.
### EWoK_by_subtask
- material-dynamics: mean delta `4.285714285714288`, seed62064 `4.155844155844157`, seed62065 `4.415584415584419`, same sign `True`, n `770`.
- social-properties: mean delta `-1.676829268292682`, seed62064 `-1.5243902439024382`, seed62065 `-1.8292682926829258`, same sign `True`, n `328`.
- physical-dynamics: mean delta `-1.6666666666666643`, seed62064 `-1.6666666666666643`, seed62065 `-1.6666666666666643`, same sign `True`, n `120`.
- physical-interactions: mean delta `-1.5287769784172696`, seed62064 `-1.6187050359712245`, seed62065 `-1.4388489208633146`, same sign `True`, n `556`.
- material-properties: mean delta `-0.588235294117645`, seed62064 `-0.588235294117645`, seed62065 `-0.588235294117645`, same sign `True`, n `170`.
- physical-relations: mean delta `-0.48899755501222586`, seed62064 `-0.48899755501222586`, seed62065 `-0.48899755501222586`, same sign `True`, n `818`.
- quantitative-properties: mean delta `-0.31847133757961643`, seed62064 `-0.31847133757961643`, seed62065 `-0.31847133757961643`, same sign `True`, n `314`.
- agent-properties: mean delta `-0.2262443438914019`, seed62064 `-0.2262443438914019`, seed62065 `-0.2262443438914019`, same sign `True`, n `2210`.
- spatial-relations: mean delta `0.20408163265306456`, seed62064 `0.20408163265306456`, seed62065 `0.20408163265306456`, same sign `True`, n `490`.
- social-relations: mean delta `0.09689922480620083`, seed62064 `0.32299741602066945`, seed62065 `-0.12919896640826778`, same sign `False`, n `1548`.
- social-interactions: mean delta `-3.552713678800501e-15`, seed62064 `0.680272108843532`, seed62065 `-0.6802721088435391`, same sign `False`, n `294`.
### COMPS_by_subtask
- wugs_dist_before: mean delta `1.0722510074841622`, seed62064 `1.05785837651122`, seed62065 `1.0866436384571045`, same sign `True`, n `13896`.
- wugs_dist_in_between: mean delta `-0.5037420840529734`, seed62064 `-0.4749568221070888`, seed62065 `-0.532527345998858`, same sign `True`, n `13896`.
- wugs: mean delta `-0.33822682786413694`, seed62064 `-0.3310305123776658`, seed62065 `-0.3454231433506081`, same sign `True`, n `13896`.
- base: mean delta `0.17936765301985957`, seed62064 `0.17430077016619094`, seed62065 `0.1844345358735282`, same sign `True`, n `49340`.
### GlobalPIQA_parallel_by_subtask
- global_piqa_parallel: mean delta `0.9708737864077648`, seed62064 `0.9708737864077648`, seed62065 `0.9708737864077648`, same sign `True`, n `103`.
### GlobalPIQA_nonparallel_by_subtask
- global_piqa_nonparallel: mean delta `2.0`, seed62064 `2.0`, seed62065 `2.0`, same sign `True`, n `100`.

## Scientific use

The completed official-sized non-SuperGLUE surface shows two-seed reproducible gains in Entity operation depths 2–5, COMPS base/wugs-dist-before, GlobalPIQA, and Reading, with stable costs in BLiMP families and EWoK. If terminal official results fail only through localized costs, these rows identify what the dense-mask/sparse-label control should try to preserve or reduce.
