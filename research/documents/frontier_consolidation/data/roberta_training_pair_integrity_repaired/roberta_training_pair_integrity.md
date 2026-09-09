# earlier analysis RoBERTa training-pair integrity

Created: `2026-09-02T03:37:49Z`

This reads completed training logs only; it does not run evaluation or start any training.

## Mechanical pair status
- Compact log rows `2529`, repeat log rows `2529`.
- Exact batch-word match all steps: `True`; exact cumulative exposure match: `True`; LR match: `True`.
- Checkpoint names match: `True`; checkpoint exposure mismatches: `{}`.
- Compact exposure `100000000`, repeat exposure `100000000`; all checkpoints present compact/repeat `True`/`True`.
- Recipe mismatches excluding expected corpus identity: `[]`.

## Training surface (compact minus repeat)
- Endpoint losses: compact 3.9984281063079834, repeat 4.214375019073486, delta -0.215947.
- Total masked tokens: compact 21449599, repeat 21414321, delta +35278.
- Total candidate tokens: compact 142948930, repeat 142708930, delta +240000.

| window | n | mean loss Δ | mean masked Δ | mean candidate Δ | mean mask-rate Δ |
|---|---:|---:|---:|---:|---:|
| early_1_500 | 500 | 0.006350 | 13.950000 | 95.112000 | -0.000006 |
| mid_501_1500 | 1000 | -0.025231 | 14.175000 | 94.658000 | 0.000003 |
| late_1501_2529 | 1029 | -0.256153 | 13.729835 | 95.030126 | -0.000007 |
| last_250 | 250 | -0.213556 | 13.552000 | 94.548000 | -0.000007 |
| last_50 | 50 | -0.213104 | 14.040000 | 92.560000 | 0.000010 |

## Scientific reading
The two RoBERTa arms are mechanically matched on exposure, checkpoint cadence, LR, seeds, tokenizer, architecture, and batch words. Loss/masked-token differences are realized consequences of compact vs first-N-repeat text, not recipe drift; selected official-compatible scores and the late local bridge remain necessary before judging transfer.
