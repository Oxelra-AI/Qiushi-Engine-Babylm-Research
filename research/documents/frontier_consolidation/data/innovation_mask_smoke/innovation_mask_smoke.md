# earlier analysis — Innovation-biased WWM mask smoke test

CPU-only; no model training, no GPU.

- Batches tested: 20 × 64
- p_innov=0.5, p_copy=0.0, mask_prob=0.15

## Mass comparison
- Standard WWM total selected tokens: 42326
- Innovation-biased total selected tokens: 42266
- Ratio (biased/standard): 0.9986
- Standard effective rate: 0.150178
- Biased effective rate: 0.149965

## Innovation group selection
- Available: 514
- Selected: 253
- Empirical rate: 0.4922 (target: 0.5)

## Copyable group selection
- Available: 2529
- Selected: 0
- Empirical rate: 0.0 (target: 0.0)

## Row counts (20 batches)
- Changed rows: 58
- Unchanged rows: 1222

Full JSON: `experiments/archive/frontier_consolidation/data/innovation_mask_smoke/innovation_mask_smoke.json`
