# context dependence ig probe result alpha0.75 artifact reconciliation

This is file/CPU-only endpoint identity work. No training, upload, or leaderboard submission was performed.

Status: `ALPHA075_ARTIFACT_RECONCILIATION`
alpha0.75 model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8` (length 64)
alpha0.75 config SHA256: `ca6792fe1842f0e89e7080e7709e0619fdcaf5ce48a65f69bf7137ac26ce9bb7`
alpha0.75 carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
Manifest matches recomputed digests: `{'model': True, 'config': True, 'carrier': True}`

## Weight identity
- weights bit-identical to coherent alpha1: `True`
- weights bit-identical to coherent replay: `True`
- weights bit-identical to alpha0.5: `True`
- interpretation: alpha0.75 is a config-scaled inference function over the same learned coherent86 private-path weights

## Config difference vs alpha1
- `private_adapter_scale`: alpha0.75 `0.75` vs alpha1 `1.0`

## CPU logits probe
`{'texts': ['The child put the toy in the box and then [MASK] smiled.', 'If the cup is full, water can [MASK] from it.'], 'status': 'ok', 'param_counts': {'alpha075': 36458592, 'alpha1': 36458592, 'alpha05': 36458592}, 'diffs': {'alpha075_vs_alpha1': {'max_abs': 0.27073097229003906, 'mean_abs': 0.0386270210146904}, 'alpha075_vs_alpha05': {'max_abs': 0.2740938663482666, 'mean_abs': 0.03873499110341072}}, 'private_off_vs_anchor': {'method': 'model.set_private_enabled(False)', 'max_abs': 0.0, 'mean_abs': 0.0}}`

## Endpoint score from existing truthful manifest
- Overall(AoA0): `42.1210247099666`
- cheap7: `44.18142857142857`
- SuperGLUE: `69.81922238969935`

JSON: `experiments/archive/frontier_consolidation/data/alpha075_artifact_reconciliation/alpha075_artifact_reconciliation.json`
