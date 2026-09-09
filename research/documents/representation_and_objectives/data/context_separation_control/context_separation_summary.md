# representational grounding boundary context separation control

| Format | Train | Before | After | Eval before | Eval after | BA cos |
|---|---:|---:|---:|---:|---:|---:|
| separated | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.8260 |
| ordinal | 0.463 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9832 |
| numbered | 0.438 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9908 |
| section | 0.925 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9954 |
| natural | 0.412 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9961 |

**separated** removes temporal selection (each query sees only its relevant context).
If separated works while others fail, the bottleneck is temporal selection,
not ranking learning capacity.
