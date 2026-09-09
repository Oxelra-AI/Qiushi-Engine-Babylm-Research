# aoa shared measurement and densemask control trusted comparison with AoA evidence

Created: 2026-09-07T21:56:45.386950+00:00

Measured AoA is determined by aoa shared measurement and densemask control surprisal+official scoring evidence, not by aoa_score != 0. Missing-checkpoint placeholder zero is unmeasured; completed scorer zero is measured.

## coherent86

- SuperGLUE valid: `True` (None)
- SuperGLUE mean: `68.94571192183594`
- AoA status: `missing`, measured: `False`, score: `None`
- Projected Overall(AoA0): `42.023967991315104`
- Measured Overall: `None`

## dense_seed62064

- SuperGLUE valid: `True` (None)
- SuperGLUE mean: `68.5318200743064`
- AoA status: `missing`, measured: `False`, score: `None`
- Projected Overall(AoA0): `42.14909111936738`
- Measured Overall: `None`

## dense_seed62065

- SuperGLUE valid: `False` (payload_missing)
- AoA status: `missing`, measured: `False`, score: `None`
- Projected Overall(AoA0): `None`
- Measured Overall: `None`

## coherent86 vs dense_seed62064

- Status: `complete`
- State: `projected_positive_awaiting_measured_aoa`
- Projected AoA0 delta: `0.1251231280522731`

## coherent86 vs dense_seed62065

- Status: `incomplete_superglue`
- State: `None`

