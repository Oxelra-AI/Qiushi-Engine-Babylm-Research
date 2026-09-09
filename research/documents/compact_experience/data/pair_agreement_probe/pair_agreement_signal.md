# earlier analysis pair-agreement representation probe

Uses only clean qwen compliance and validity selected training original-Qwen rewrite pairs and submitted-model representations; no official AoA/CDI items, child curves, AoA scores, SuperGLUE labels, or downstream eval outputs are used.

Minimal falsifying probe for a possible pair-agreement objective: require aligned positive representations to separate from shuffled rewrites, and preferably stronger separation in clean-Qwen than in official/shuffled controls, before any weight-changing objective experiment.

## Results
- clean_qwen_seed43022_100M: margin_mean=0.3646, ci90=[0.3519325783709064, 0.3755700297188014], top1=0.9688, loss=1.3677
- official_lengthmatched_seed43022_100M: margin_mean=0.3627, ci90=[0.3493028233060613, 0.3739400644553825], top1=0.9414, loss=1.4182
- qwen_shuffled_seed43022_100M: margin_mean=0.3705, ci90=[0.35715698485728353, 0.3821088020922616], top1=0.9492, loss=1.3743

## Contrasts
- clean_qwen_seed43022_100M_minus_official_lengthmatched_seed43022_100M: delta_mean_margin=0.0019, delta_top1=0.0273
- clean_qwen_seed43022_100M_minus_qwen_shuffled_seed43022_100M: delta_mean_margin=-0.0060, delta_top1=0.0195
