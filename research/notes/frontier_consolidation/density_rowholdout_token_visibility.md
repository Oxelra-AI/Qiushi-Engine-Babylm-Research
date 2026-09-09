# density cleanqwen overlay highprecision density overlay token visibility

## baseline16k
- near_core: over seq256 {'cleanqwen_lengthmatched_near_core': 114, 'cleanqwen_fineweb_repeat_near_core': 17, 'cleanqwen_fineweb_near_view_core': 16}; view-repeat mean token diff -0.027603513174404015; abs diff <=4 fraction 0.7936010037641155.
- compact_core_neutral: over seq256 {'cleanqwen_lengthmatched_compact_core_neutral': 166, 'cleanqwen_fineweb_repeat_compact_core_neutral': 83, 'cleanqwen_fineweb_compact_view_core_neutral': 115}; view-repeat mean token diff 7.644693473961767; abs diff <=4 fraction 0.3328938694792353.
- compact_reinvest: over seq256 {'cleanqwen_lengthmatched_compact_reinvest': 132, 'cleanqwen_fineweb_repeat_compact_reinvest': 19, 'cleanqwen_fineweb_compact_view_reinvest': 56}; view-repeat mean token diff 9.046601941747573; abs diff <=4 fraction 0.22071197411003235.

## leader40k
- near_core: over seq256 {'cleanqwen_lengthmatched_near_core': 99, 'cleanqwen_fineweb_repeat_near_core': 0, 'cleanqwen_fineweb_near_view_core': 0}; view-repeat mean token diff 0.038268506900878296; abs diff <=4 fraction 0.8462986198243413.
- compact_core_neutral: over seq256 {'cleanqwen_lengthmatched_compact_core_neutral': 149, 'cleanqwen_fineweb_repeat_compact_core_neutral': 64, 'cleanqwen_fineweb_compact_view_core_neutral': 66}; view-repeat mean token diff 6.167435728411339; abs diff <=4 fraction 0.42320369149637443.
- compact_reinvest: over seq256 {'cleanqwen_lengthmatched_compact_reinvest': 115, 'cleanqwen_fineweb_repeat_compact_reinvest': 2, 'cleanqwen_fineweb_compact_view_reinvest': 4}; view-repeat mean token diff 7.332686084142395; abs diff <=4 fraction 0.32103559870550163.

Full JSON: `experiments/archive/frontier_consolidation/data/density_rowholdout_token_visibility/density_overlay_token_visibility.json`
