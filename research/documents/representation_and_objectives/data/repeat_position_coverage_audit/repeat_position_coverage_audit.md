# earlier analysis repeat source-position/content coverage audit

JSON: `experiments/archive/representation_and_objectives/data/repeat_position_coverage_audit/repeat_position_coverage_audit.json`

## Mean content coverage
- compact: content 0.663275, tail 0.6991994007350335, content_fraction 0.692775
- prefix_repeat: content 0.600237, tail 0.0, content_fraction 0.532884
- hash_repeat: content 0.624365, tail 0.6201393902621392, content_fraction 0.553699
- best_bpe_repeat: content 0.651627, tail 0.8326372923013136, content_fraction 0.578620

## Mean compact-minus-repeat BPE deltas
- compact_minus_prefix: 2.574167 (median 2.0)
- compact_minus_hash: 2.007240 (median 2.0)
- compact_minus_best: 0.590292 (median 0.0)

## Decile coverage (content positions)
- compact: 0.667, 0.614, 0.609, 0.621, 0.624, 0.631, 0.632, 0.651, 0.701, 0.759
- prefix_repeat: 1.000, 1.000, 1.000, 0.997, 0.945, 0.695, 0.319, 0.074, 0.001, 0.000
- hash_repeat: 0.625, 0.627, 0.618, 0.621, 0.621, 0.615, 0.619, 0.613, 0.620, 0.620
- best_bpe_repeat: 0.682, 0.496, 0.387, 0.364, 0.461, 0.694, 0.777, 0.826, 0.855, 0.852
