# chck84 endpoint carrier validation late-weight-average scaffold

Created UTC: `2026-09-01T18:58:21Z`

## Scientific purpose

Same-trajectory late-iterate averaging is prepared as a possible competence-stabilization test, distinct from the closed prediction-vote/per-column-selection route and from the old two-checkpoint average in a different tokenizer coordinate. It must not be read as evidence until an averaged checkpoint is evaluated by the official-compatible selected tasks.

## Available candidates

### `reference_scale1p75_seed43022__prepeak_78_80_82_84_uniform`
- trajectory: `reference_scale1p75_seed43022`; adapter scale `1.75`; same seed/mask as reference: `True`
- endpoints: `chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.25, 0.25, 0.25, 0.25]`
- buildable: `True`
- reading: low-pass rising late iterates up to the 84M reference peak; excludes immediate 86M decline

### `reference_scale1p75_seed43022__center_80_82_84_uniform`
- trajectory: `reference_scale1p75_seed43022`; adapter scale `1.75`; same seed/mask as reference: `True`
- endpoints: `chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: short latest-window average ending at the 84M peak; closest LAWA-style candidate for reference trajectory

### `reference_scale1p75_seed43022__symmetric_82_84_86_uniform`
- trajectory: `reference_scale1p75_seed43022`; adapter scale `1.75`; same seed/mask as reference: `True`
- endpoints: `chck_82M, chck_84M, chck_86M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: tests whether the 84M peak sits in a local basin or whether 86M damage should be excluded

### `reference_scale1p75_seed43022__wider_76_78_80_82_84_uniform`
- trajectory: `reference_scale1p75_seed43022`; adapter scale `1.75`; same seed/mask as reference: `True`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.2, 0.2, 0.2, 0.2, 0.2]`
- buildable: `True`
- reading: broader rising-side low-pass; may preserve earlier BLiMP/COMPS while smoothing relation-state churn

### `reference_scale1p75_seed43022__ema_decay0p5_76_to_84`
- trajectory: `reference_scale1p75_seed43022`; adapter scale `1.75`; same seed/mask as reference: `True`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.032258, 0.064516, 0.129032, 0.258065, 0.516129]`
- buildable: `True`
- reading: post-hoc EMA-like rising-side average, most weight on 84M while retaining preceding iterate memory

### `scale1p25_seed43022__prepeak_78_80_82_84_uniform`
- trajectory: `scale1p25_seed43022`; adapter scale `1.25`; same seed/mask as reference: `True`
- endpoints: `chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.25, 0.25, 0.25, 0.25]`
- buildable: `True`
- reading: low-pass rising late iterates up to the 84M reference peak; excludes immediate 86M decline

### `scale1p25_seed43022__center_80_82_84_uniform`
- trajectory: `scale1p25_seed43022`; adapter scale `1.25`; same seed/mask as reference: `True`
- endpoints: `chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: short latest-window average ending at the 84M peak; closest LAWA-style candidate for reference trajectory

### `scale1p25_seed43022__symmetric_82_84_86_uniform`
- trajectory: `scale1p25_seed43022`; adapter scale `1.25`; same seed/mask as reference: `True`
- endpoints: `chck_82M, chck_84M, chck_86M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: tests whether the 84M peak sits in a local basin or whether 86M damage should be excluded

### `scale1p25_seed43022__wider_76_78_80_82_84_uniform`
- trajectory: `scale1p25_seed43022`; adapter scale `1.25`; same seed/mask as reference: `True`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.2, 0.2, 0.2, 0.2, 0.2]`
- buildable: `True`
- reading: broader rising-side low-pass; may preserve earlier BLiMP/COMPS while smoothing relation-state churn

### `scale1p25_seed43022__ema_decay0p5_76_to_84`
- trajectory: `scale1p25_seed43022`; adapter scale `1.25`; same seed/mask as reference: `True`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.032258, 0.064516, 0.129032, 0.258065, 0.516129]`
- buildable: `True`
- reading: post-hoc EMA-like rising-side average, most weight on 84M while retaining preceding iterate memory

### `scale1p75_seed43122__prepeak_78_80_82_84_uniform`
- trajectory: `scale1p75_seed43122`; adapter scale `1.75`; same seed/mask as reference: `False`
- endpoints: `chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.25, 0.25, 0.25, 0.25]`
- buildable: `True`
- reading: low-pass rising late iterates up to the 84M reference peak; excludes immediate 86M decline

### `scale1p75_seed43122__center_80_82_84_uniform`
- trajectory: `scale1p75_seed43122`; adapter scale `1.75`; same seed/mask as reference: `False`
- endpoints: `chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: short latest-window average ending at the 84M peak; closest LAWA-style candidate for reference trajectory

### `scale1p75_seed43122__symmetric_82_84_86_uniform`
- trajectory: `scale1p75_seed43122`; adapter scale `1.75`; same seed/mask as reference: `False`
- endpoints: `chck_82M, chck_84M, chck_86M`
- normalized weights: `[0.333333, 0.333333, 0.333333]`
- buildable: `True`
- reading: tests whether the 84M peak sits in a local basin or whether 86M damage should be excluded

### `scale1p75_seed43122__wider_76_78_80_82_84_uniform`
- trajectory: `scale1p75_seed43122`; adapter scale `1.75`; same seed/mask as reference: `False`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.2, 0.2, 0.2, 0.2, 0.2]`
- buildable: `True`
- reading: broader rising-side low-pass; may preserve earlier BLiMP/COMPS while smoothing relation-state churn

### `scale1p75_seed43122__ema_decay0p5_76_to_84`
- trajectory: `scale1p75_seed43122`; adapter scale `1.75`; same seed/mask as reference: `False`
- endpoints: `chck_76M, chck_78M, chck_80M, chck_82M, chck_84M`
- normalized weights: `[0.032258, 0.064516, 0.129032, 0.258065, 0.516129]`
- buildable: `True`
- reading: post-hoc EMA-like rising-side average, most weight on 84M while retaining preceding iterate memory

## Boundaries

- Do not average across different random initializations or independent seed trajectories with this script.
- Do not use these candidates for final endpoint claims without full selected cheap-task scoring and, if competitive, SuperGLUE/AoA completion.
- No model evaluation or leaderboard submission is performed by the dry-run manifest.

JSON: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/late_weight_average_scaffold_manifest.json`
