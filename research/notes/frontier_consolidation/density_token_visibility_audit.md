# density cleanqwen overlay highprecision density token-visibility audit

## baseline16k
- near_core: changed rows 1,594; over seq256 {'official_lengthmatched_near_core': 16, 'fineweb_repeat_near_core': 17, 'fineweb_near_view_core': 16}; mean token diff fineweb_near_view_core minus fineweb_repeat_near_core: -0.027603513174404015; abs diff <=4 fraction: 0.7936010037641155.
- compact_core_neutral: changed rows 1,517; over seq256 {'official_lengthmatched_compact_core_neutral': 17, 'fineweb_repeat_compact_core_neutral': 15, 'fineweb_compact_view_core_neutral': 47}; mean token diff fineweb_compact_view_core_neutral minus fineweb_repeat_compact_core_neutral: 7.644693473961767; abs diff <=4 fraction: 0.3328938694792353.
- compact_reinvest: changed rows 1,545; over seq256 {'official_lengthmatched_compact_reinvest': 15, 'fineweb_repeat_compact_reinvest': 19, 'fineweb_compact_view_reinvest': 56}; mean token diff fineweb_compact_view_reinvest minus fineweb_repeat_compact_reinvest: 9.046601941747573; abs diff <=4 fraction: 0.22071197411003235.

## leader40k
- near_core: changed rows 1,594; over seq256 {'official_lengthmatched_near_core': 8, 'fineweb_repeat_near_core': 0, 'fineweb_near_view_core': 0}; mean token diff fineweb_near_view_core minus fineweb_repeat_near_core: 0.038268506900878296; abs diff <=4 fraction: 0.8462986198243413.
- compact_core_neutral: changed rows 1,517; over seq256 {'official_lengthmatched_compact_core_neutral': 9, 'fineweb_repeat_compact_core_neutral': 2, 'fineweb_compact_view_core_neutral': 4}; mean token diff fineweb_compact_view_core_neutral minus fineweb_repeat_compact_core_neutral: 6.167435728411339; abs diff <=4 fraction: 0.42320369149637443.
- compact_reinvest: changed rows 1,545; over seq256 {'official_lengthmatched_compact_reinvest': 12, 'fineweb_repeat_compact_reinvest': 2, 'fineweb_compact_view_reinvest': 4}; mean token diff fineweb_compact_view_reinvest minus fineweb_repeat_compact_reinvest: 7.332686084142395; abs diff <=4 fraction: 0.32103559870550163.

Full JSON: `experiments/archive/frontier_consolidation/data/density_token_visibility_audit/density_token_visibility_audit.json`
