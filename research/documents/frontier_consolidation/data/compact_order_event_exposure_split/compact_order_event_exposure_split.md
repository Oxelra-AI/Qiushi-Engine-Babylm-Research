# earlier analysis compact order event exposure split

Exact/no-model reconstruction of whether each fixed channel-probe event word was selected as a WWM target in the ordered and scrambled compact order experiment design 40M training streams. Use never-selected splits to separate category-level source-absent behavior from direct target exposure.

Events: 12288; checkpoints with losses: ['chck_20M', 'chck_40M']; debug max_batches=None.

## Counts
- retained_content: {'ordered_selected_1': 1523, 'scrambled_selected_1': 1521, 'either_selected': 2912, 'ordered_selected_0': 2156, 'scrambled_selected_0': 2126, 'neither_selected': 1184, 'ordered_selected_2': 374, 'scrambled_selected_2': 409, 'scrambled_selected_3': 38, 'ordered_selected_3': 40, 'scrambled_selected_4': 2, 'ordered_selected_4': 3}
- source_absent_content: {'ordered_selected_1': 1527, 'scrambled_selected_2': 398, 'either_selected': 2870, 'ordered_selected_0': 2135, 'scrambled_selected_0': 2132, 'neither_selected': 1226, 'scrambled_selected_1': 1518, 'ordered_selected_2': 390, 'ordered_selected_3': 43, 'scrambled_selected_3': 44, 'scrambled_selected_4': 4, 'ordered_selected_4': 1}
- function_other: {'ordered_selected_0': 2135, 'scrambled_selected_0': 2132, 'neither_selected': 1175, 'ordered_selected_1': 1538, 'either_selected': 2921, 'ordered_selected_2': 364, 'scrambled_selected_1': 1515, 'ordered_selected_4': 3, 'scrambled_selected_2': 390, 'ordered_selected_3': 56, 'scrambled_selected_3': 57, 'scrambled_selected_4': 2}

## Source-absent minus controls by exposure split
### chck_20M
- all: {'source_absent_delta': -0.053718, 'retained_content_delta': -0.080398, 'function_other_delta': -0.079314, 'source_absent_minus_controls_mean': 0.026138, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}
- neither_arm_selected: {'source_absent_delta': -0.047451, 'retained_content_delta': -0.07879, 'function_other_delta': -0.082193, 'source_absent_minus_controls_mean': 0.03304, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}
- ordered_never_selected: {'source_absent_delta': -0.033456, 'retained_content_delta': -0.071184, 'function_other_delta': -0.077818, 'source_absent_minus_controls_mean': 0.041045, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}
### chck_40M
- all: {'source_absent_delta': -0.25524, 'retained_content_delta': -0.010249, 'function_other_delta': -0.024953, 'source_absent_minus_controls_mean': -0.237639, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}
- neither_arm_selected: {'source_absent_delta': -0.239096, 'retained_content_delta': 0.077263, 'function_other_delta': -0.015026, 'source_absent_minus_controls_mean': -0.270215, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}
- ordered_never_selected: {'source_absent_delta': -0.178833, 'retained_content_delta': 0.06725, 'function_other_delta': -0.013275, 'source_absent_minus_controls_mean': -0.20582, 'meaning': 'negative source_absent_minus_controls_mean means source-absent has larger ordered NLL advantage than retained/function controls within this exposure split'}

JSON: `experiments/archive/frontier_consolidation/data/compact_order_event_exposure_split/compact_order_event_exposure_split.json`
