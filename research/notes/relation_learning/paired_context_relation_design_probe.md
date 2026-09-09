# compact_experience relation design prestate COMPACT_EXPERIENCE qwen relation design: probe readout

Created: 2026-09-06T12:19:04Z

This note scores the existing COMPACT_EXPERIENCE seed43022 relation-typed arms with the relation_learning mechanism probes. It deliberately ignores COMPACT_EXPERIENCE official-style Overall/Human-like aggregates for mechanism interpretation because AoA accounting differs across the five arms.

## Arms and records

- Arms: OFF, DUP, SHUF, SEP, ALN.
- Per-model records: copy 4000, compact T/U 11800, compact N 5900, Wikipedia 10800.
- `DUP` was verified from metadata and source script as local original+original duplication: 37,594 selected pairs, 12,550 packed duplicate rows, no truncation, `cur_segments.extend([p.original, p.original])`.
- `SHUF` preserves original and rewrite multisets and same-window rewrite adjacency while breaking pair correspondence for 37,594/37,594 pairs.
- `SEP` preserves original/rewrite coexistence but moves sides to separate rows; its row-count/row-length mismatch means ambiguous middle outcomes need N/ordinary-heldout follow-up.

## Compact FineWeb-register T/U/N, token-nonoverlap

Positive `delta_A_T` means the first arm gains more from the true compact source relative to neutral N. Positive `delta_G` means larger U−T source-conditioned gain.

| contrast | ΔA_T | ΔA_U | ΔG |
|---|---:|---:|---:|
| ALNminusOFF | +0.0592 | -0.0461 | +0.1053 |
| ALNminusSEP | +0.2599 | +0.0863 | +0.1735 |
| ALNminusSHUF | +0.1438 | +0.0683 | +0.0755 |
| SHUFminusOFF | -0.0845 | -0.1144 | +0.0298 |
| SEPminusOFF | -0.2006 | -0.1324 | -0.0682 |
| DUPminusOFF | -1.9263 | -0.1340 | -1.7923 |
| ALNminusDUP | +1.9855 | +0.0879 | +1.8976 |

## Wikipedia/Simple-English source use by target class

Positive values mean the first arm extracts more true-source benefit than the second arm. Overlap targets recur as tokenizer IDs in the English-Wikipedia source; nonoverlap targets are absent from it.

| contrast | overlap Δ(T vs N) | nonoverlap Δ(T vs N) |
|---|---:|---:|
| ALNminusOFF | +3.5975 ± 0.1236 | -0.5934 ± 0.0502 |
| ALNminusSEP | +6.1434 ± 0.1486 | -0.7701 ± 0.0502 |
| ALNminusSHUF | +4.5917 ± 0.1339 | -0.7698 ± 0.0507 |
| SHUFminusOFF | -0.9942 ± 0.0682 | +0.1765 ± 0.0298 |
| SEPminusOFF | -2.5459 ± 0.0915 | +0.1767 ± 0.0291 |
| DUPminusOFF | +3.3422 ± 0.1395 | -1.1840 ± 0.0659 |
| ALNminusDUP | +0.2553 ± 0.1252 | +0.5906 ± 0.0687 |

## Held-out natural-copy gain

| contrast | Δ copy gain |
|---|---:|
| DUPminusOFF | +1.7497 |
| DUPminusALN | +0.3980 |
| ALNminusOFF | +1.3517 |
| ALNminusSHUF | +1.7007 |
| ALNminusSEP | +2.7454 |

## Entity by relevant queried-state updates

These are computed from existing per-target official Entity predictions, not from aggregate Entity scores. Values are accuracy-point differences at the 100M checkpoint.

| contrast | rel_eq0 | rel_ge3 |
|---|---:|---:|
| ALNminusOFF | +12.98 | +0.53 |
| ALNminusSEP | +20.31 | +4.04 |
| ALNminusSHUF | +16.68 | +1.13 |
| DUPminusOFF | +15.51 | -2.26 |
| DUPminusALN | +2.53 | -2.79 |

## Scientific reading from this run

The compact readout is the cross-register test of whether the COMPACT_EXPERIENCE qwen relation practice installs the same FineWeb-compact source-use routine as the designed compact VIEW/REPEAT family. The Wikipedia readout is nearer to the qwen substrate because the inherited block includes many SimpleWiki pairs. The shuffled arm is the crucial new correspondence control: aligned-minus-shuffled isolates own-source correspondence while keeping same-window rewrite-register adjacency. If SHUF falls below OFF on `A_T=N-T` while `A_U=N-U` is flat or higher, that is not inert adjacency: it is a practiced non-correspondence/discounting routine, separated from general fit by N and ordinary held-out loss.

Use the CSV files for exact rows and additional overlap bins before making stronger statements.

## Output files

- `experiments/archive/relation_learning/data/paired_context_relation_design_probe/compact_TUN_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_relation_design_probe/wikipedia_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_relation_design_probe/copy_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_relation_design_probe/entity_contrasts_by_group.csv`
