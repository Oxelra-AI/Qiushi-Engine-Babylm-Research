# compact directional internal result compact directional endpoint decision synthesis

Eval root: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`

## Endpoint interactions

| metric | I = mixed - one-way | FF | RR | FR | RF |
|---|---:|---:|---:|---:|---:|
| BLiMP | 0.125000 | 59.3700 | 58.3800 | 58.7900 | 59.2100 |
| Supplement | 0.075000 | 52.2700 | 52.8200 | 52.4100 | 52.8300 |
| EWoK | -0.425000 | 50.2300 | 50.4300 | 50.0300 | 49.7800 |
| Entity | 0.080000 | 17.2000 | 16.9100 | 17.0300 | 17.2400 |
| COMPS | -0.100000 | 49.7500 | 49.6600 | 49.6400 | 49.5700 |
| GlobalPIQA | -0.235000 | 33.1650 | 33.6650 | 31.6950 | 34.6650 |
| Reading | -0.060000 | 1.5050 | 0.9150 | 1.1100 | 1.1900 |
| cheap7 | -0.077143 | 37.6414 | 37.5400 | 37.2436 | 37.7836 |

## Full EWoK

- Overall full-EWoK I: `-0.21659228143870735`
- correctness transition analysis relational-domain group I: `0.4555808656036433`
- Adjacency-independent-domain group I: `-1.400862068965516`

## Scientific read

Decision status: `endpoint_has_positive_signal_needing_controls`

{
  "decision_status": "endpoint_has_positive_signal_needing_controls",
  "cheap7_I": -0.07714285714286007,
  "EWoK_column_I": -0.42499999999999716,
  "full_EWoK_relational_domain_I": 0.4555808656036433,
  "internal_forward_noncopied_I_loss": -0.010725251590203655,
  "internal_reverse_noncopied_I_loss": -0.028862876373970003,
  "relation_context_constraint": {
    "a02_step183_source_conditioned_ordering": "companion analysis found large source-conditioned ordering interactions in frozen DeBERTa for compact, prefix, and onegap families, with compact the smallest I_f because 21.7% of compact targets are source-absent. This strengthens the alternative explanation that compact-view gains come from general ordered source retrieval plus tail coverage, budget efficiency, and content density, not a compact-specific semantic recoding or reciprocal mechanism."
  },
  "interpretation": {
    "close_if": "If endpoint cheap7, EWoK column, and full relational-domain I are all absent or negative, the compact 20M reciprocal causal instantiation learned local pair prediction but did not transfer to the target surface.",
    "controls_if": "If endpoint transfer is positive, run semantic_extract and random_extract fork controls before attributing specificity to compact faithful rewrites, because companion analysis shows ordered source retrieval is not compact-specific.",
    "bounded_continuation_if": "If endpoint is weak/mixed but internal noncopied I is the only favorable clue, continuation must be bounded and cheaper than two full controls for distinguishing delayed transfer from null."
  }
}

JSON: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2/compact_endpoint_decision_synthesis.json`
