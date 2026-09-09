# fw ewok interaction reader — relational substrate and FW readout synthesis

## Scientific Motivation

The current active expensive evidence is still the FineWeb compact-vs-breadth training family. full ewok interaction synthesis EWoK and fw globalpiqa relevant substrate GlobalPIQA show the same broad weakness: context-conditioned world relations. fw ewok interaction reader therefore prepared two things without spending GPU training compute:

1. post-endpoint readouts that will test whether FW compact recurrence moves both EWoK conditional reversals and GlobalPIQA hard-row ranks/margins;
2. a broad non-evaluation-derived transition substrate with matched neutral controls, only for a future small probe if FW corpus allocation does not move those relation errors.

The 103-row GlobalPIQA_parallel set is not treated as an isolated branch. It is only one window into the broader EWoK/GlobalPIQA relation problem.

## Completed CPU work

### Existing fw globalpiqa relevant substrate miner execution

Ran the previously written `scripts/general_consequence_substrate_miner.py` after correcting the writable output path. It produced:

- `data/general_consequence_substrate/general_consequence_substrate_miner.json`
- `data/general_consequence_substrate/general_consequence_substrate_summary.csv`
- `data/general_consequence_substrate/general_consequence_candidates.jsonl`

Main count: 16,818 deduplicated high-scoring candidate sentences / 522,132 words. Sampling showed useful material but also surface-noise risk, so fw ewok interaction reader built a stricter extractor.

### Strict transition substrate

Built and ran `scripts/strict_transition_substrate_miner.py` over existing allowed reservoirs, without reading official evaluation item text. Outputs:

- summary JSON: `data/strict_transition_substrate/strict_transition_substrate_miner.json`
- source/route CSV: `data/strict_transition_substrate/strict_transition_summary_by_source_route.csv`
- all candidates: `data/strict_transition_substrate/strict_transition_candidates.jsonl`
- very-high subset: `data/strict_transition_substrate/strict_transition_candidates_very_high.jsonl`
- balanced slice: `data/strict_transition_substrate/strict_transition_balanced_200k_slice.jsonl`
- note: `notes/strict_transition_substrate.md`

Main counts:

- 22,662 high-purity candidate sentences / 747,986 words.
- 6,490 very-high-purity candidate sentences / 258,434 words.
- Balanced 200k research slice: 4,984 sentences / exactly 200,000 words.
- Candidate route word totals: physical/material 283,780; spatial 433,769; temporal/quantity 446,281; affordance/procedure 209,795; social/agent 382,100; explicit contrast 66,790.

Scientific reading: this is a future substrate, not a training route. Sampling still shows some metaphor and broad narrative material. It needs semantic filtering before any full corpus materialization.

### Matched neutral controls

The first complete-control builder (`scripts/transition_matched_control_builder.py`) timed out after 900 s because it rescanned all reservoirs with the full strict feature extractor. I repaired the approach with `scripts/transition_matched_control_builder_fast.py`, which uses the existing treatment slices and a lightweight low-transition screen with early stopping.

Fast outputs:

- summary JSON: `data/transition_matched_controls_fast/transition_matched_control_fast_builder.json`
- summary CSV: `data/transition_matched_controls_fast/transition_matched_control_fast_summary.csv`
- treatment/control JSONL slices under `data/transition_matched_controls_fast/`
- note: `notes/transition_matched_controls_fast.md`

Matched word counts:

| slice | treatment words | neutral words | word difference |
|---:|---:|---:|---:|
| 50k | 49,999 | 49,999 | 0 |
| 100k | 99,992 | 99,992 | 0 |
| 200k | 200,000 | 199,996 | 4 |

Source-word distributions are closely matched, but neutral sentences are shorter on average than treatment sentences. A future small probe must decide whether exact word/source matching is enough or whether tighter length matching should be added.

### FW relation readout scripts

Prepared and compiled the post-endpoint readout scripts:

- `scripts/fw_globalpiqa_margin_wrapper.py`
- `scripts/fw_ewok_interaction_reader.py`
- `scripts/fw_relational_result_synthesis.py`

Dry runs correctly reported that FW checkpoints and readout summaries are not yet available:

- GlobalPIQA wrapper preflight: `data/fw_globalpiqa_margin_reader/preflight.json`
- EWoK interaction preflight: `data/fw_ewok_interaction_reader/preflight.json`
- combined synthesis: `data/fw_relational_result_synthesis/fw_relational_result_synthesis.json`
- combined note: `notes/fw_relational_result_synthesis.md`

The intended sequence after endpoints complete:

1. Run `scripts/fw_shared_anchor_posttrain_eval_controller.py` for completed targets.
2. Run `scripts/fw_ewok_interaction_reader.py` on completed targets.
3. Run `scripts/fw_globalpiqa_margin_wrapper.py` on completed targets.
4. Run `scripts/fw_relational_result_synthesis.py` to combine official scores, EWoK stable conditional-reversal failures, and GlobalPIQA hard-row margin/rank movement.

## Current scientific interpretation

- No new compliant BabyLM Strict-Small endpoint exists from this step.
- The FW compact/breadth experiment remains the main active data-mechanism test.
- A compact win should only be credited as strong if it improves the official vector and reduces relation failures in the EWoK and GlobalPIQA readouts.
- If compact and breadth do not move EWoK conditional reversals or GlobalPIQA hard-row ranks/margins, then corpus allocation alone is unlikely to cross 41.8–42.0, and a future route should test corpus-derived relation/transition objectives or filtered transition substrate slices at small shared-checkpoint scale before any full 100M run.
- The fw ewok interaction reader transition substrate and matched controls are ready as research assets for that later possibility, but they do not justify a new expensive run by themselves.

## Group communication
