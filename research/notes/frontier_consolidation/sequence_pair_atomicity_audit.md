# sequence curriculum loop measurement sequence pair-atomicity audit
CPU-only audit of whether a short-sequence curriculum preserves source+rewrite same-window structure in the compact-view changed block. No model was trained and no official eval text was used.

## Inputs
- Pool SHA matched: `True`. Pair map rows: `3005`.
- Pair map source: `experiments/archive/frontier_consolidation/data/pair_span_map/pair_span_map.jsonl`; pool: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`.

## Pair token lengths
| tokenizer | pairs | source+rewrite token median/p95/max | source median | rewrite median |
|---|---:|---:|---:|---:|
| legal16k | 12155 | 46.0/86.0/157 | 27.0 | 19.0 |
| minfreq50_supportfloor | 12155 | 45.0/83.0/153 | 27.0 | 18.0 |

## Per-length preservation
Fractions are over the 12,155 compact source+rewrite pairs in the changed block. Prefix is the current schedule path; greedy is faithful word-boundary chunking without pair awareness; pair-atomic packs whole source+rewrite pairs when they fit.

| tokenizer | L | prefix full | prefix any cooccur | greedy full same chunk | greedy any cochunk | pair-atomic full/fittable | pair-atomic overlong |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal16k | 64 | 0.187 | 0.285 | 0.391 | 0.976 | 0.805 | 0.195 |
| legal16k | 128 | 0.502 | 0.596 | 0.758 | 0.991 | 0.998 | 0.002 |
| legal16k | 256 | 0.997 | 1.000 | 0.997 | 1.000 | 1.000 | 0.000 |
| minfreq50_supportfloor | 64 | 0.195 | 0.292 | 0.406 | 0.979 | 0.824 | 0.176 |
| minfreq50_supportfloor | 128 | 0.518 | 0.613 | 0.758 | 0.992 | 0.999 | 0.001 |
| minfreq50_supportfloor | 256 | 0.998 | 1.000 | 0.998 | 1.000 | 1.000 | 0.000 |

## Ten-epoch schedule pair preservation
| tokenizer | schedule | prefix full | greedy full | pair-atomic full/fittable | pair-atomic overlong |
|---|---|---:|---:|---:|---:|
| legal16k | 64x3_128x4_256x3 | 0.556 | 0.720 | 0.941 | 0.059 |
| legal16k | 64x7_256x3 | 0.430 | 0.573 | 0.864 | 0.136 |
| minfreq50_supportfloor | 64x3_128x4_256x3 | 0.565 | 0.725 | 0.947 | 0.053 |
| minfreq50_supportfloor | 64x7_256x3 | 0.436 | 0.583 | 0.877 | 0.123 |

## Scientific reading
- Existing prefix slicing hides most compact pairs during L64/L128 phases because many pairs occur after the row prefix, so it weakens the load-bearing same-window second-view signal while still charging words.
- Naive word-boundary chunking exposes suffix words but can split source and rewrite across chunks, especially at L64. A future sequence trainer should preserve pair atoms in changed rows when possible rather than merely stream whitespace chunks.
- Pair-atomic chunking is feasible for most pairs at L64 and almost all pairs at L128/L256; overlong pairs quantify the unavoidable context-length cost. This is a construction requirement, not a reason to launch sequence training before current word-mean/support-floor evidence is read.

Full JSON: `experiments/archive/frontier_consolidation/data/sequence_pair_atomicity_audit/sequence_pair_atomicity_audit.json`
