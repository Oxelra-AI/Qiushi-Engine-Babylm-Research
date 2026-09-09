# Realized WWM Matching for Packed Target Deletion

Status: completed selection audit and trajectory smoke; the two final 100M deletion outcomes were pending.

Local probes indicated a source-absent compact-content prediction channel. The next question was whether that channel mediated late downstream gains in the historical packed geometry. A copied-content control had to match realized masked whole-word events and BPE mass over all ten epochs, not only pool-level groups. The audit reproduced the training CUDA generator; CPU randomness was not an equivalent schedule.

| Deletion set | Masked word events | Masked BPE pieces | Unique positions |
|---|---:|---:|---:|
| Source-absent content | 30,830 | 48,105 | 16,540 |
| Pool-matched copied content | 30,882 | 48,387 | 16,469 |
| Copied minus absent | +52 | +282 | -71 |

Copied-minus-absent mean deltas were BPE length +0.0065, support-log mean -0.0725, support-log minimum -0.1180, token midpoint -0.0030, relative group position -0.0043 and row-order fraction +0.0010. All 64,740 pool rows appeared ten times; copied selections contained no noncontent events. Event-count mismatch was below 0.2% and BPE mismatch approximately 0.6%, so event-level rematching remained a fallback rather than the primary control.

The initial source-absent arm timed out after 3600 seconds at about 23M words without resumable optimizer/scheduler/random state. It was superseded, not a 100M result. A vectorized label-filtering implementation preserved the scientific schedule and reproduced a full-mode smoke: first loss 9.80984, 8,658 masked/kept tokens, 39,370 batch words, learning rate 6.622516556291391e-06 and 118,604 cumulative words at update 3.

The reference equal7 was 44.2886, with Supplement 66.4, EWoK 53.09, Entity 28.07, COMPS 51.97, GlobalPIQA 35.62 and Reading 8.24. Mediation required selective loss of the established Supplement/relational-EWoK advantage after source-absent deletion, not the same loss under copied deletion. The predeclared relational domains were social-properties, physical-dynamics, spatial-relations and physical-relations; material-properties and social-interactions formed the adjacency-independent comparison. Aggregate score alone was insufficient.
