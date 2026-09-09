# fw ewok interaction reader — Strict corpus-derived transition substrate

## Purpose

full ewok interaction synthesis EWoK and fw globalpiqa relevant substrate GlobalPIQA point to a common weakness in context-conditioned world relations. This miner asks whether allowed reservoirs contain a cleaner broad substrate for later small probes, without reading official evaluation item text or shaping examples from the 103-row parallel set.

## Main counts

- Deduplicated high-purity candidates: `22662` sentences / `747986` words.
- Very-high-purity subset: `6490` sentences / `258434` words.
- Balanced research slice: `4984` sentences / `200000` words at `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_balanced_200k_slice.jsonl`; this is not a committed training corpus.

## Route words by deduplicated candidates

| route | sentences | words |
|---|---:|---:|
| `physical_material_transition` | 8470 | 283780 |
| `spatial_transition` | 12649 | 433769 |
| `temporal_quantity_transition` | 13314 | 446281 |
| `affordance_procedure_transition` | 6348 | 209795 |
| `social_agent_transition` | 10752 | 382100 |
| `explicit_contrast_transition` | 1914 | 66790 |

## Source contribution

| source | candidates | words |
|---|---:|---:|
| `compact_experience_aligned_10m_rows` | 17611 | 610882 |
| `fw_frozen_sources_38167` | 3273 | 92816 |
| `fw_compact_rewrites` | 899 | 16686 |
| `fw_breadth_whole_sentence_companions` | 879 | 27602 |

## Scientific use

- This does not supersede the running FW compact-vs-breadth experiment. First read whether compact recurrence moves EWoK interaction failures and GlobalPIQA hard-row ranks/margins.
- If the FW arms do not move those relational errors, the next efficient route is a small shared-checkpoint factorial using a manually/automatically filtered 50k–200k word subset and/or a corpus-derived four-cell contrast objective; do not commit to 100M from this miner alone.
- Keep channels separate: physical/material, spatial, temporal/quantity, affordance/procedure, and social/agent transitions may support different EWoK/GlobalPIQA subsets.

## Files

- summary_json: `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_substrate_miner.json`
- summary_by_source_route_csv: `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_summary_by_source_route.csv`
- candidates_jsonl: `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_candidates.jsonl`
- very_high_candidates_jsonl: `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_candidates_very_high.jsonl`
- balanced_200k_slice_jsonl: `experiments/archive/representation_and_objectives/data/strict_transition_substrate/strict_transition_balanced_200k_slice.jsonl`
- note: `research/notes/representation_and_objectives/strict_transition_substrate.md`
