# equivariance control and state substrate repair — Equivariance truth-table, gradient, and composition diagnostic

## Why this run exists

equivariance protocol produced all-chance GRU/BiGRU numbers. This run treats that as an unmeasured experiment: it first checks the symbolic labels and gradient path, then only compares standard versus equivariant learning after every predicate has independent semantic support.

## Truth table and gradient path

- Legacy full-matrix debug rows: 5120 rows; true fraction 0.500; duplicate text conflicts 0.
- Memorization slice: 64 rows, labels {1: 32, 0: 32}; first-step grad norm 0.6441, first-step parameter delta 2.282.
- Initial acc/loss 0.500/0.7059; final acc/loss 1.000/0.0138; reached ≥0.999 acc epoch 30.

## Corrected composition experiment

- Train families 24, held families 12, held source-domain `football` families 12.
- Train pair cells 48; held pair cells 16; equivariance pairs 11520.
- Train rows 6912 = support 2304 + pair-composition 4608.

| Evaluation surface | Standard acc | Equivariant acc | Δ equiv-std |
|---|---:|---:|---:|
| semantic_support_held_family | 0.667±0.236 | 0.833±0.236 | +0.167 |
| train_pair_held_family | 0.664±0.238 | 0.836±0.231 | +0.172 |
| held_pair_seen_family | 0.669±0.234 | 0.822±0.228 | +0.153 |
| held_pair_held_family | 0.668±0.235 | 0.819±0.229 | +0.151 |
| held_pair_held_domain | 0.664±0.238 | 0.828±0.232 | +0.164 |

## Scientific reading

The equivariance protocol all-chance result should not be treated as evidence for a semantic-prior requirement: the corrected diagnostic has a live gradient path and can memorize a balanced slice.
The composition table is the first relevant ordinary-vs-equivariant comparison because predicate meanings are no longer zero-shot: each predicate appears in independent role-labeled support before held cp×hp combinations are tested.
These numbers remain a controlled small-model measurement. Natural-source work still needs independently attested event/state families, stronger domains, and connection to the EWoK/Entity bridge panel before any BabyLM-scale run is justified.

## Files

- Summary JSON: `experiments/archive/representation_and_objectives/data/equivariance_diagnostics/summary.json`
- Memorized slice rows: `experiments/archive/representation_and_objectives/data/equivariance_diagnostics/memorized_slice_rows.jsonl`
- Train/eval rows: `experiments/archive/representation_and_objectives/data/equivariance_diagnostics/semantic_support_train.jsonl`, `experiments/archive/representation_and_objectives/data/equivariance_diagnostics/composition_train_pairs.jsonl`, `experiments/archive/representation_and_objectives/data/equivariance_diagnostics/held_pair_held_family_eval.jsonl`
