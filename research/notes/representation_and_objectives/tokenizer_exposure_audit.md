# seqsafe96 interpretive foundation tokenizer-level exposure audit (seqsafe96 FineWeb source-breadth)

Tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model` (baseline16k, vocab 16384), seq_len=256.

## Aggregate per arm (10M single-epoch pool)

| arm | rows | words | tokens_seen | tok/word (seen) | trunc frac (tokens) | trunc frac (rows) |
|-----|------|-------|-------------|-----------------|---------------------|-------------------|
| treatment (FineWeb) | 75339 | 10000000 | 14524924 | 1.4525 | 0.02407 | 0.18499 |
| control (official) | 75339 | 10000000 | 14596271 | 1.4596 | 0.02402 | 0.18492 |

## Key deltas (treatment - control)

- tokens_seen delta: -71347 (-0.489% relative)
- tokens/word (seen) delta: -0.0071
- truncation frac (tokens) delta: 0.00005

## Interpretation

If tokens_seen_rel_pct is small (|.|<~1%) and tokens_per_word delta is small, the seqsafe96 arms are token-matched and the downstream contrast is a clean source-breadth contrast. A large positive treatment token surplus would mean FineWeb injection also increased the real gradient/prediction budget, confounding source breadth with token exposure.

JSON: `experiments/archive/representation_and_objectives/data/tokenizer_exposure_audit/tokenizer_exposure_audit.json`
