# transition structure match — compact-contained transition-structure contrast

## Purpose

transition containment and probe design made a legal warm-start relative-exposure probe feasible but also showed that the anchor matched controls 30k files are not a ready pair. This analysis re-derived treatment/control subsets only from exact-contained compact-pool sentences and matched source/style before any GPU work.

## Input subset

- Transition contained subset: 571 sentences / 25055 actual whitespace words.
- Anchor-control contained subset: 574 sentences / 24976 actual whitespace words.
- Both are still dominated by the COMPACT_EXPERIENCE-aligned narrative pool; this remains a transfer risk rather than a reason to train blindly.

## Matching results

| regime | fields | transition sentences | control sentences | words each arm | source L1 | origin L1 | length L1 | quote L1 | proper L1 | pronoun L1 | digit L1 | comma L1 | transition markers T/C |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `coarse_source_style` | `source_label;origin_source;length_bin;required_capability` | 430 | 431 | 19400 | 0.0000 | 0.0000 | 0.0000 | 0.1533 | 0.0441 | 0.1300 | 0.0872 | 0.1200 | 1.70/0.23 |
| `source_origin_len_cap_digit` | `source_label;origin_source;length_bin;required_capability;digit_bin` | 403 | 403 | 18360 | 0.0000 | 0.0000 | 0.0000 | 0.1284 | 0.0202 | 0.1293 | 0.0000 | 0.1090 | 1.68/0.22 |
| `source_origin_len_cap_quote` | `source_label;origin_source;length_bin;required_capability;quote_bin` | 357 | 354 | 16340 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0696 | 0.1486 | 0.1039 | 0.1468 | 1.67/0.27 |
| `source_origin_len_cap_proper` | `source_label;origin_source;length_bin;required_capability;proper_bin` | 369 | 367 | 16764 | 0.0000 | 0.0000 | 0.0000 | 0.1748 | 0.0000 | 0.1194 | 0.0676 | 0.1200 | 1.72/0.25 |
| `source_origin_len_cap_pronoun` | `source_label;origin_source;length_bin;required_capability;pronoun_bin` | 342 | 344 | 15972 | 0.0000 | 0.0000 | 0.0000 | 0.1983 | 0.0652 | 0.0000 | 0.0702 | 0.1271 | 1.70/0.22 |
| `source_origin_len_cap_comma` | `source_label;origin_source;length_bin;required_capability;comma_bin` | 362 | 361 | 16810 | 0.0000 | 0.0000 | 0.0000 | 0.2142 | 0.0358 | 0.1277 | 0.0619 | 0.0000 | 1.71/0.22 |
| `medium_style` | `source_label;origin_source;length_bin;required_capability;quote_bin;proper_bin;digit_bin` | 266 | 266 | 12418 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1681 | 0.0000 | 0.1052 | 1.69/0.27 |
| `surface_no_comma` | `source_label;origin_source;length_bin;required_capability;quote_bin;proper_bin;pronoun_bin;digit_bin` | 178 | 178 | 8357 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0833 | 1.63/0.28 |
| `surface_no_quote` | `source_label;origin_source;length_bin;required_capability;proper_bin;pronoun_bin;digit_bin;comma_bin` | 178 | 178 | 8536 | 0.0000 | 0.0000 | 0.0000 | 0.2474 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.66/0.21 |
| `strict_style` | `source_label;origin_source;length_bin;required_capability;quote_bin;proper_bin;pronoun_bin;digit_bin;comma_bin` | 101 | 101 | 4926 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.68/0.24 |

Preferred regime: `medium_style`. It gives exact word parity at **12418 words per arm** (266 transition sentences, 266 controls).

## Scientific reading

The compact-contained subset is sufficient only for a modest small mechanism probe: the preferred pair matches source/origin, length, required capability, quote bins, proper-name bins, and digit bins while preserving a large transition-marker separation. It can test whether extra exposure to transition structure moves EWoK conditional compatibility and GlobalPIQA hard-row ranks beyond an equally narrative-rich anchor control, but it is not strong enough to justify a full route by itself.

This remains a probe, not a SOTA data recipe. The material is narrative-heavy and mostly from the COMPACT_EXPERIENCE-aligned pool, so success must mean coupled movement in EWoK four-cell structure and GlobalPIQA hard ranks while preserving broad compact-arm strength. An EWoK-only shift or a broad-column drop would repeat the earlier relation-data failure mode.

## Continuation stream implication

A common whole-row filler removal of 12418 words is available exactly for one extra-subset insertion pass, excluding rows that contain preferred treatment/control sentences. This target is the per-arm extra subset, not treatment plus control combined; both arms can remove the same filler and add either the matched transition subset or matched anchor-control subset, keeping per-pass word exposure equal.
Existing trainers do not expose a true resume flag in their command-line interface; a later continuation trainer must load the same intermediate model weights for both arms and record that optimizer state is either recovered or deliberately reset symmetrically.

## Files

- summary_json: `experiments/archive/representation_and_objectives/data/transition_structure_match/transition_structure_matcher.json`
- match_regime_summary_csv: `experiments/archive/representation_and_objectives/data/transition_structure_match/match_regime_summary.csv`
- preferred_transition_jsonl: `experiments/archive/representation_and_objectives/data/transition_structure_match/preferred_transition_structure_extra.jsonl`
- preferred_anchor_control_jsonl: `experiments/archive/representation_and_objectives/data/transition_structure_match/preferred_anchor_control_extra.jsonl`
- preferred_filler_rows_manifest_csv: `experiments/archive/representation_and_objectives/data/transition_structure_match/preferred_filler_rows_manifest.csv`
- note: `research/notes/representation_and_objectives/transition_structure_match.md`
