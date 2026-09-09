# anchor matched controls — transition substrate execution synthesis

## Active context

The FineWeb compact-versus-breadth comparison remained the pending experiment. No new H100 training was launched for this analysis, and no completed interleaved-breadth endpoint result was available.

full ewok interaction synthesis EWoK and fw globalpiqa relevant substrate GlobalPIQA still define the shared weak object: context-conditioned world relations. anchor matched controls therefore improved the future low-cost substrate for that object while keeping it subordinate to the FW endpoints.

## Core transition filter

Script: `experiments/archive/representation_and_objectives/scripts/core_transition_filter_and_controls.py`

Outputs under `experiments/archive/representation_and_objectives/data/core_transition_filter`:

- `core_transition_filter_and_controls.json`
- `core_transition_candidates.jsonl`
- `core_transition_very_strict.jsonl`
- `core_transition_treatment_50k.jsonl`, `core_transition_treatment_100k.jsonl`, `core_transition_treatment_200k.jsonl`
- `core_transition_neutral_lengthmatched_50k.jsonl`, `core_transition_neutral_lengthmatched_100k.jsonl`, `core_transition_neutral_lengthmatched_200k.jsonl`
- `core_slice_summary.csv`
- `length_source_match_summary.csv`

Main counts:

- fw ewok interaction reader input: 22,662 candidate sentences / 747,986 words.
- Core filtered: 5,353 sentences / 190,928 words.
- Core very-filtered: 1,007 sentences / 41,847 words.
- The first neutral controls matched source and length very closely: at 50k, 49,992 treatment words versus 49,985 neutral words; source L1 0.000045 and length-bin L1 0.000155.

Scientific reading: this was a stronger filter than fw ewok interaction reader, but sample reading still showed transcript artifacts, abstract-political cases, and broad narrative cases. It was not clean enough to use directly as a large route.

## Anchor-matched controls

The first neutral controls were too semantically distant in places, so I built anchored no-explicit-relation controls.

Scripts:

- `experiments/archive/representation_and_objectives/scripts/anchor_matched_control_builder.py`
- `experiments/archive/representation_and_objectives/scripts/anchor_matched_control_builder_v2.py`

Preferred v2 outputs under `experiments/archive/representation_and_objectives/data/anchor_matched_controls_v2`:

- `anchor_matched_controls_v2.json`
- `anchor_control_candidate_pool_v2.jsonl`
- `core_transition_anchor_control_v2_50k.jsonl`, `core_transition_anchor_control_v2_100k.jsonl`, `core_transition_anchor_control_v2_200k.jsonl`
- `anchor_control_v2_summary.csv`
- `anchor_control_v2_match_summary.csv`

V2 improved the required-capability match over v1: at the 50k slice, source L1 0.014936, length-bin L1 0.248840, required-capability L1 0.028708, and required-capability hit fraction 1.0. However sample reading still found enough residual corpus artifact and broad narrative material that a smaller, cleaner slice was preferable.

## Sample quality reading

Script: `experiments/archive/representation_and_objectives/scripts/transition_sample_quality_audit.py`  
Main output CSV: `experiments/archive/representation_and_objectives/data/transition_sample_quality/transition_sample_quality_summary.csv`

This bounded sample reading exposed why the larger core slices are risky as direct training material:

- Core treatment 50k: 144/1,156 rows with obvious surface-artifact markers and 84 rows with abstract extra terms.
- Core treatment 200k: 620/5,353 rows with obvious surface-artifact markers and 470 rows with abstract extra terms.
- Anchor-control v2 suppressed abstract terms by construction, but still had surface-artifact rates near 12% and occasional relation-like language not captured by the simple explicit-marker suppressor.

Scientific reading: these counts do not invalidate the whole substrate, but they show that volume alone would reintroduce the same problem as earlier relation routes: a measurable EWoK movement could come from artifacts, style, or broad narratives rather than reusable transition knowledge.

## Ultra-clean 30k probe asset

The first ultra-clean run revealed that the highest-purity pool was only about 35k–38k words and that a 50k target was too large. I repaired the subset fill and built a 30k treatment/control asset.

Scripts:

- `experiments/archive/representation_and_objectives/scripts/ultraclean_transition_probe_assets.py`
- `experiments/archive/representation_and_objectives/scripts/ultraclean_transition_probe_assets_v3.py`

Preferred outputs under `experiments/archive/representation_and_objectives/data/ultraclean_transition_probe_v3`:

- `ultraclean_transition_probe_assets_v3.json`
- `ultraclean_transition_candidate_pool_v3.jsonl`
- `ultraclean_transition_30k.jsonl`
- `ultraclean_anchor_control_30k.jsonl`
- `ultraclean_probe_v3_summary.csv`

Main counts:

- Ultra-clean candidate pool: 882 sentences / 35,572 words.
- Treatment slice: 711 sentences / exactly 30,000 words.
- Treatment bucket words: physical/material 22,659; spatial 6,000; temporal/quantity 1,341.
- Anchor control: 713 sentences / 29,958 words.
- Match: word difference −42; source L1 0.000322; length-bin L1 0.002912; required-capability L1 0.001955.

Scientific reading: this is the cleanest transition-substrate asset from this step. It deliberately sacrifices volume to reduce artifacts and abstract-topic confounds. It is not a new full route; it is a small later probe asset if the FW compact/breadth endpoints do not move relation-conditioned failures.

## Earlier small-route evidence absorbed

Script: `experiments/archive/representation_and_objectives/scripts/existing_smallroute_meta_synthesis.py`  
Note: `research/notes/representation_and_objectives/existing_smallroute_meta_synthesis.md`

The older INITIAL_MODEL_STUDIES evidence is important for not repeating a known mistake:

- INITIAL_MODEL_STUDIES earlier analysis legal BSM 1M: coherent-minus-swapped moved BLiMP +1.94, Entity +0.73, EWoK +0.55, but hurt Supplement −2.80, GlobalPIQA_parallel −3.89, GlobalPIQA_nonparallel −3.00, Reading −0.68; binding probes were 0.000 both-correct.
- INITIAL_MODEL_STUDIES earlier analysis BSM 4M: coherent-minus-swapped moved EWoK +3.09, but hurt GlobalPIQA_parallel −1.95 and nonparallel −2.00; continuous binding probes still stayed at 0.000 both-correct.
- INITIAL_MODEL_STUDIES pair-adjacency screens also showed EWoK/entity movement but broad-column fragility.

Scientific reading: relation-focused data can move EWoK, but previous forms often harmed GlobalPIQA or failed their own mechanism readout. Any future transition use must start with a small matched treatment/control probe and must include GlobalPIQA margins, not only EWoK or Overall.

## How to use this later

Do not launch a 100M transition route from anchor matched controls alone. First complete the active FW readout sequence:

1. official-compatible endpoint scores for compact, row-block breadth, and interleaved breadth when available;
2. EWoK four-cell interaction reader from fw ewok interaction reader;
3. GlobalPIQA hard-row margin wrapper from fw ewok interaction reader;
4. combined relational synthesis from fw ewok interaction reader.

Only if FW compact/breadth fails to move the shared relation weakness should the next low-cost experiment use the anchor matched controls v3 30k treatment and anchored control. The scientific question would be narrow and decision-changing: after a common legal shared checkpoint and equal remaining word exposure, does explicit corpus-derived transition material improve EWoK stable failures and GlobalPIQA hard-row ranks/margins relative to an anchored no-explicit-relation control, without damaging Supplement/Reading? If not, relation-substrate replacement should be closed rather than enlarged.
