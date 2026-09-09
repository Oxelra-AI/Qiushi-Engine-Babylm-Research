# ideal binding dependency probe — Ideal binding-dependency probe

## Purpose
Before committing to any proposition-dense data route, prove that target-token
prediction genuinely depends on correct entity→state propositions. Tests the
UPPER BOUND of what proposition-dense data can provide under plain WWM.

## Design
- ~200 synthetic procedural passages with maximally clear entity→state bindings
- Bag-preserving entity-assignment swap controls (same words, same domain)
- Target: downstream state/location/property tokens
- Model: protected 100M DeBERTa (trained on official corpus only)

## Results
- Valid probes: 200
- Mean (correct - swapped): -0.6676
- Median: -0.0067
- Positive fraction: 0.45
- p10, p90: -0.9483, 0.2718

## Interpretation
If positive: the model uses entity-state binding for prediction in clean text →
  proposition-dense data provides learnable binding signal under WWM.
If null: binding cannot be exploited even in ideal conditions → bottleneck is
  objective/architecture, not data content.
