# compact_experience relation design prestate COMPACT_EXPERIENCE qwen relation design: probe readout

Created: 2026-09-06T12:44:28Z

This note scores the existing COMPACT_EXPERIENCE seed43022 relation-typed arms with the relation_learning mechanism probes. It deliberately ignores COMPACT_EXPERIENCE official-style Overall/Human-like aggregates for mechanism interpretation because AoA accounting differs across the five arms.

## Arms and records

- Arms: OFF, ALN.
- Per-model records: copy 4000, compact T/U 11800, compact N 5900, Wikipedia 10800.
- `DUP` was verified from metadata and source script as local original+original duplication: 37,594 selected pairs, 12,550 packed duplicate rows, no truncation, `cur_segments.extend([p.original, p.original])`.
- `SHUF` preserves original and rewrite multisets and same-window rewrite adjacency while breaking pair correspondence for 37,594/37,594 pairs.
- `SEP` preserves original/rewrite coexistence but moves sides to separate rows; its row-count/row-length mismatch means ambiguous middle outcomes need N/ordinary-heldout follow-up.

## Compact FineWeb-register T/U/N, token-nonoverlap

Positive `delta_A_T` means the first arm gains more from the true compact source relative to neutral N. Positive `delta_G` means larger U−T source-conditioned gain.

| contrast | ΔA_T | ΔA_U | ΔG |
|---|---:|---:|---:|
| ALNminusOFF | -0.1931 | -0.1012 | -0.0919 |
| ALNminusSEP | NA | NA | NA |
| ALNminusSHUF | NA | NA | NA |
| SHUFminusOFF | NA | NA | NA |
| SEPminusOFF | NA | NA | NA |
| DUPminusOFF | NA | NA | NA |
| ALNminusDUP | NA | NA | NA |

## Wikipedia/Simple-English source use by target class

Positive values mean the first arm extracts more true-source benefit than the second arm. Overlap targets recur as tokenizer IDs in the English-Wikipedia source; nonoverlap targets are absent from it.

| contrast | overlap Δ(T vs N) | nonoverlap Δ(T vs N) |
|---|---:|---:|
| ALNminusOFF | +4.1998 ± 0.1379 | -0.8806 ± 0.0528 |
| ALNminusSEP | NA | NA |
| ALNminusSHUF | NA | NA |
| SHUFminusOFF | NA | NA |
| SEPminusOFF | NA | NA |
| DUPminusOFF | NA | NA |
| ALNminusDUP | NA | NA |

## Held-out natural-copy gain

| contrast | Δ copy gain |
|---|---:|
| DUPminusOFF | NA |
| DUPminusALN | NA |
| ALNminusOFF | +1.5171 |
| ALNminusSHUF | NA |
| ALNminusSEP | NA |

## Entity by relevant queried-state updates

These are computed from existing per-target official Entity predictions, not from aggregate Entity scores. Values are accuracy-point differences at the 100M checkpoint.

| contrast | rel_eq0 | rel_ge3 |
|---|---:|---:|
| ALNminusOFF | +21.03 | -0.79 |
| ALNminusSEP | NA | NA |
| ALNminusSHUF | NA | NA |
| DUPminusOFF | NA | NA |
| DUPminusALN | NA | NA |

## Scientific reading from this run

The compact readout is the cross-register test of whether the COMPACT_EXPERIENCE qwen relation practice installs the same FineWeb-compact source-use routine as the designed compact VIEW/REPEAT family. The Wikipedia readout is nearer to the qwen substrate because the inherited block includes many SimpleWiki pairs. The shuffled arm is the crucial new correspondence control: aligned-minus-shuffled isolates own-source correspondence while keeping same-window rewrite-register adjacency. If SHUF falls below OFF on `A_T=N-T` while `A_U=N-U` is flat or higher, that is not inert adjacency: it is a practiced non-correspondence/discounting routine, separated from general fit by N and ordinary held-out loss.

Use the CSV files for exact rows and additional overlap bins before making stronger statements.

## Output files

- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_probe/compact_TUN_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_probe/wikipedia_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_probe/copy_late_contrasts.csv`
- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_probe/entity_contrasts_by_group.csv`
