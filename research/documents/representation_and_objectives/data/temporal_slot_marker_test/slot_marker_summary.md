# representational grounding boundary temporal slot marker test

| Format | Train acc | Train before | Train after | Eval before | Eval after | Before/After cos |
|---|---:|---:|---:|---:|---:|---:|
| natural | 0.491 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9962 |
| structural | 0.525 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9997 |
| hybrid | 0.494 | 0.500 | 0.500 | 0.500 | 0.500 | 0.9985 |

If structural/hybrid can fit changed rows while natural cannot, the bottleneck is
pretrained representational grounding of temporal cues, not architectural capacity.
