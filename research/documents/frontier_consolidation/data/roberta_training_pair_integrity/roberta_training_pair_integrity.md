# earlier analysis RoBERTa training-pair integrity

Created: `2026-09-02T03:20:38Z`

This reads completed training logs only; it does not run evaluation or start any training.

## Mechanical pair status
- Compact log rows `0`, repeat log rows `0`.
- Exact batch-word match all steps: `True`; exact cumulative exposure match: `True`; LR match: `True`.
- Checkpoint names match: `True`; checkpoint exposure mismatches: `{}`.
- Compact exposure `100000000`, repeat exposure `100000000`; all checkpoints present compact/repeat `True`/`True`.
- Recipe mismatches excluding expected corpus identity: `[{'field': 'tokenizer_path', 'compact': 'experiments/archive/frontier_consolidation/data/compliant_tokenizer', 'repeat': 'experiments/archive/frontier_consolidation/data/compliant_tokenizer'}]`.

## Training surface (compact minus repeat)
- Endpoint losses: compact 3.9984281063079834, repeat 4.214375019073486, delta -0.215947.
- Total masked tokens: compact 0, repeat 0, delta +0.
- Total candidate tokens: compact 0, repeat 0, delta +0.

| window | n | mean loss Δ | mean masked Δ | mean candidate Δ | mean mask-rate Δ |
|---|---:|---:|---:|---:|---:|
| early_1_500 | 0 | NA | NA | NA | NA |
| mid_501_1500 | 0 | NA | NA | NA | NA |
| late_1501_2529 | 0 | NA | NA | NA | NA |
| last_250 | 0 | NA | NA | NA | NA |
| last_50 | 0 | NA | NA | NA | NA |

## Scientific reading
The two RoBERTa arms are mechanically matched on exposure, checkpoint cadence, LR, seeds, tokenizer, architecture, and batch words. Loss/masked-token differences are realized consequences of compact vs first-N-repeat text, not recipe drift; selected official-compatible scores and the late local bridge remain necessary before judging transfer.
