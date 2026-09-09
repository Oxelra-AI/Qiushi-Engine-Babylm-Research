# route reopen conditional innovation — layerwise contextual-computation probe

CPU-only; no model update, no official evaluation, no corpus/tokenizer change, no H100 work.

Layerwise emergence of conditional source help on copyable versus innovation rewrite targets, used to compare targeted innovation-learning with deeper/iterative contextual-computation hypotheses.

- targets: `440`; changed rows loaded: `384`; counts: `{'copyable': 220, 'innovation': 220}`
- train SHA: `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- tokenizer SHA: `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Final-layer and late-layer summaries
### tokenmean_80M (encoder layers=8)
| group | final true loss | final source help | final true over decoy | true loss drop last2 | source-help gain last2 | true-over-decoy gain last2 | true loss drop last4 | source-help gain last4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 2.8651 | 2.5540 | 2.8403 | 2.8022 | 2.3711 | 2.5281 | 3.9577 | 2.4653 |
| innovation | 4.7178 | 0.9622 | 1.2665 | 1.1598 | 0.7848 | 0.8812 | 2.5181 | 0.9656 |
| copyable | 1.0125 | 4.1458 | 4.4140 | 4.4446 | 3.9575 | 4.1749 | 5.3973 | 3.9649 |
| innovation:relation_cue | 3.0576 | 0.3461 | 0.6375 | 0.3633 | 0.0777 | 0.2497 | 2.1657 | 0.3268 |
| innovation:content_or_other | 4.9599 | 1.0521 | 1.3582 | 1.2760 | 0.8879 | 0.9733 | 2.5695 | 1.0588 |
### clean_80M (encoder layers=8)
| group | final true loss | final source help | final true over decoy | true loss drop last2 | source-help gain last2 | true-over-decoy gain last2 | true loss drop last4 | source-help gain last4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 3.8783 | 1.9906 | 2.2364 | 2.3813 | 1.8938 | 2.0415 | 3.3306 | 1.9249 |
| innovation | 5.6695 | 0.4945 | 0.7551 | 0.9087 | 0.3823 | 0.4835 | 2.0439 | 0.4954 |
| copyable | 2.0872 | 3.4867 | 3.7178 | 3.8539 | 3.4053 | 3.5995 | 4.6174 | 3.3545 |
| innovation:relation_cue | 3.9414 | 0.0793 | 0.1157 | 0.4836 | -0.0778 | -0.1852 | 1.7725 | 0.0171 |
| innovation:content_or_other | 5.9215 | 0.5550 | 0.8483 | 0.9707 | 0.4494 | 0.5810 | 2.0835 | 0.5652 |

## Reading
- A large positive final-layer true_over_decoy for innovation groups confirms source-specific conditional information beyond generic row identity.
- If last-two-layer source-help and true-over-decoy gains are still large, more contextual computation/depth may be a high-leverage separate route; if they are small, the existing 8-layer model already computes the signal and the remaining high innovation loss is better attacked by training signal allocation on changed spans.
- Intermediate-layer MLM-head losses are not official scores; compare patterns, not absolute endpoint values.

Full JSON: `experiments/archive/frontier_consolidation/data/contextual_computation_probe/contextual_computation_probe.json`
