# preservation controls and corrected interpretation zero-preservation replay compare

Status: `ZERO_PRESERVATION_REPLAY_DIFFERS`
Max updates: `1`
Original elapsed: `177.5` sec; clean elapsed: `192.7` sec
Checkpoint hash match: `True`
Original hash: `daabf480539502686c3227c5e1586a5011e2450685545593fb9795090db0a732`
Clean hash: `daabf480539502686c3227c5e1586a5011e2450685545593fb9795090db0a732`
Log diff count: `1`

This checks short-horizon equivalence of the clean lambda=0 trainer to the original (M,S) acquisition-only implementation.

## First diffs
- `update_1.pooled_loss`: 2.6042266726684193 vs None
