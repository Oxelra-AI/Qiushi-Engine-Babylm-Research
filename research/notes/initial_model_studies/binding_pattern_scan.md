# binding feasibility and route implications — binding-pattern scan of high_entity_state windows

Data: `experiments/archive/initial_model_studies/data/structure_density_revision_157/high_entity_state.jsonl` (unique epoch-0 windows: 31325)

| metric | count | fraction |
|---|---:|---:|
| ≥2 entities with swappable bindings | 1969 | 6.3% |
| + downstream entity reference (binding-dependent target) | 819 | 2.6% |

Mean bindings per qualifying window: 4.37
Mean swappable pairs per qualifying window: 6.52

## Feasibility assessment

If 819 windows × ~64 words/window ≈ 52k words of binding-dependent training examples exist, repeating to 10M needs 190× repetition. This is feasible if the fraction is >5% (>~1566 windows).

The scanner uses conservative heuristics; manual inspection of examples should verify quality.
