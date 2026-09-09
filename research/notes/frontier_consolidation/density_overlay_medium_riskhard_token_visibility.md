# density cleanqwen overlay highprecision density overlay token visibility

## baseline16k
- near_core: over seq256 {'cleanqwen_lengthmatched_near_core': 243, 'cleanqwen_fineweb_repeat_near_core': 35, 'cleanqwen_fineweb_near_view_core': 29}; view-repeat mean token diff -0.37689577282994513; abs diff <=4 fraction 0.78702807357212.
- compact_core_neutral: over seq256 {'cleanqwen_lengthmatched_compact_core_neutral': 342, 'cleanqwen_fineweb_repeat_compact_core_neutral': 185, 'cleanqwen_fineweb_compact_view_core_neutral': 221}; view-repeat mean token diff 7.090539165818922; abs diff <=4 fraction 0.3574092912851814.
- compact_reinvest: over seq256 {'cleanqwen_lengthmatched_compact_reinvest': 258, 'cleanqwen_fineweb_repeat_compact_reinvest': 36, 'cleanqwen_fineweb_compact_view_reinvest': 78}; view-repeat mean token diff 8.419161676646707; abs diff <=4 fraction 0.24384564204923487.

## leader40k
- near_core: over seq256 {'cleanqwen_lengthmatched_near_core': 187, 'cleanqwen_fineweb_repeat_near_core': 3, 'cleanqwen_fineweb_near_view_core': 2}; view-repeat mean token diff -0.2665375927718619; abs diff <=4 fraction 0.8422071636011617.
- compact_core_neutral: over seq256 {'cleanqwen_lengthmatched_compact_core_neutral': 279, 'cleanqwen_fineweb_repeat_compact_core_neutral': 134, 'cleanqwen_fineweb_compact_view_core_neutral': 139}; view-repeat mean token diff 5.634452356731095; abs diff <=4 fraction 0.4530349270939302.
- compact_reinvest: over seq256 {'cleanqwen_lengthmatched_compact_reinvest': 213, 'cleanqwen_fineweb_repeat_compact_reinvest': 2, 'cleanqwen_fineweb_compact_view_reinvest': 7}; view-repeat mean token diff 6.7358616101131075; abs diff <=4 fraction 0.35395874916833003.

Full JSON: `experiments/archive/frontier_consolidation/data/density_overlay_medium_riskhard_token_visibility/density_overlay_token_visibility.json`
