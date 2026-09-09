# Step038d curated no-update neutral scorer

No-update context: raw source + same target use frame, with no update sentence. Positive neutral margin means coherent86 prefers the original source answer when no entity has been updated.

Neutral correct: 4/6 by mean-token score; mean neutral source-new=+0.7395.

After distractor update, mean RETAIN R=-4.0724; mean R-neutral drop=-4.8118.

| pair | neutral source-new | R after distractor update | drop | U target update | neutral correct |
|---|---:|---:|---:|---:|---|
| sgp_chart_netherlands_australia | -0.323 | -4.326 | -4.003 | +3.908 | False |
| sgp_chart_italy_billboard | -0.354 | -2.163 | -1.808 | +1.556 | False |
| sgp_pope_action_nicholas_photios | +1.071 | -3.684 | -4.756 | +2.136 | True |
| sgp_music_video_fatima_trainor | +1.871 | -2.849 | -4.720 | +3.320 | True |
| sgp_military_listing_iraqi_bombers | +0.110 | -6.364 | -6.474 | +1.360 | True |
| sgp_brand_label_mcafee_intel | +2.062 | -5.048 | -7.110 | +5.871 | True |
