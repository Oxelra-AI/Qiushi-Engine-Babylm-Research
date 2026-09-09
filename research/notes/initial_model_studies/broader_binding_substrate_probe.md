# broader binding probe conclusion — broader binding-substrate probe

Model: protected 100M DeBERTa. Corpus: full official BabyLM.
Adjacent sentence pairs scanned: 16000
Binding pairs (≥2 entities, swappable predicates): 15
Bag-preserving swap examples: 5
Valid probe results: 4

## Probe design

Target: state/location/relation tokens (NOT entity names).
Both conditions have identical word multiset (bag preserved).
Only entity→predicate binding assignment is swapped.

## Results

Correct-minus-swapped target LL:
  mean = -0.7021
  median = -0.6695
  positive fraction = 0.0
  p10 = -0.8723, p90 = -0.5003

## Interpretation

If mean > 0 and positive fraction > 0.6: the protected model already uses
entity-state binding for prediction → training objective can amplify this.
If near zero: binding signal is real in corpus but model does not use it yet
→ training must teach binding from scratch (harder).
If negative or very sparse: official corpus does not support this mechanism.
