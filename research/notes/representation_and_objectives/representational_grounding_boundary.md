# representational grounding boundary — Representational grounding boundary for temporal state selection

## Findings

Three convergent experiments identify the binding constraint on data-efficient temporal state updating:

### 1. Context temporal cues are representationally collapsed

Pretrained BabyLM DeBERTa CLS cosine between "In the earlier ATP ranking, A was above B" and "In the later ATP ranking, A was above B":

| Comparison | Cosine |
|---|---:|
| Hypothesis-only "Before" vs "After" | 0.854 |
| Context "earlier" vs "later" sentences | **0.992** |
| Full context+hypothesis before/after | **0.9976** (gap −0.0001) |

The hypothesis alone moderately distinguishes temporal reference (0.854), but within the full context the distinction vanishes (0.998). Entity-direction reversal in the hypothesis (A>B vs B>A, cos 0.980) is MORE similar than temporal reversal — yet direction reversal IS learnable from sparse data. The difference: direction appears in one context sentence, while temporal appears in BOTH context sentences with nearly identical representations.

### 2. Changed-only worlds cannot be fit in any temporal format

Training on 40 changed-only worlds (320 rows) for 12 epochs:

| Format | Train acc | Before | After | BA cos |
|---|---:|---:|---:|---:|
| natural | 0.491 | 0.500 | 0.500 | 0.996 |
| [TIME_1]/[TIME_2] structural | 0.525 | 0.500 | 0.500 | **1.000** |
| hybrid | 0.494 | 0.500 | 0.500 | 0.999 |

`[TIME_1]`/`[TIME_2]` markers are OOV tokens treated identically by the tokenizer — they make the cosine HIGHER, not lower. No format enables fitting.

### 3. Context separation reveals the precise bottleneck

| Format | Train | Per-query | Eval | BA cos |
|---|---:|---:|---:|---:|
| **separated** (each query gets only its temporal slice) | **1.000** | **1.000** | **1.000** | 0.826 |
| **section** ("Background:" / "Update:") | **0.925** | **1.000** | **1.000** | 0.995 |
| ordinal ("First" / "Second") | 0.463 | 0.500 | 0.500 | 0.983 |
| numbered ("one" / "two") | 0.438 | 0.500 | 0.500 | 0.991 |
| natural ("earlier" / "later") | 0.412 | 0.500 | 0.500 | 0.996 |

Three decisive contrasts:

**Separated vs natural**: The model learns rankings perfectly from 40 changed worlds when temporal selection is externally resolved. The bottleneck is not ranking learning, it is temporal slot selection.

**Section vs ordinal**: "Background"/"Update" enables temporal selection despite cosine 0.995 (nearly identical to natural's 0.996). "First"/"Second" fails despite LOWER cosine 0.983. CLS cosine is therefore NOT the binding metric. The operative distinction is **discourse-functional grounding**: "Background" carries the pretrained function "prior/established information," "Update" carries "new/changed information," and these connect through cross-attention to "Based on the background" / "Based on the update" in the hypothesis.

**Section vs [TIME_1/2]**: Novel structural markers carry no pretrained function and therefore cannot ground temporal selection, while existing discourse-function words can — even though their overall CLS representations are nearly identical.

## Connection to secondary retention and orientation interference full3 retention

The three-seed secondary-retention replication confirms:

- **Separate event and ranking coordinates**: Eflip_Rtrue inverts event (0.044±0.045) while preserving focal (0.987) and secondary (1.000) in all 3 seeds.
- **Shared focal-secondary ranking coordinate**: Rflip_only inverts secondary (0.000±0.000) in all 3 seeds despite no direct secondary labels.
- **Incomplete-anchoring interference**: Eflip_only produces high seed variance on focal (0.481±0.396) and secondary (0.408±0.421), showing staged updates without correct complementary anchoring destabilize the interface.

These are coordinate-sharing results, not conservation results: Rflip globally contradicts the ranking relation, so secondary inversion is the expected transfer of a predicate-wide remapping.

## Unified mechanism

Combining role coordinate anchor and state probe, the evidence supports a three-layer data-efficient relational learning mechanism:

**Layer 1: Reusable argument-slot interface** (identity orbit randomization and role coordinate)
Sparse data can orient a relation coordinate only when entity arguments are represented through reusable, in-distribution filler types. Fixed identity tokens create memorization shortcuts; family-stable or shared-pool aliases enable transfer. Token familiarity helps but does not explain the aligned/anti-aligned reversal.

**Layer 2: Sparse relation-coordinate orientation** (role coordinate anchor and state probe, 257–264)
A small amount of correctly aligned evidence can orient a relation-specific signed coordinate. Anti-aligned evidence reverses it. Held-held consistency alone cannot resolve the absolute coordinate (role coordinate anchor and state probe identifiability). Related readouts (focal/secondary ranking) share a coordinate and update together; unrelated readouts (event/ranking) are separate (secondary retention and orientation interference full3).

**Layer 3: Representational grounding boundary** (related experiments)
Sparse evidence can orient and update only those distinctions that are already representationally grounded in the pretrained model. The binding metric is not overall CLS cosine but **discourse-functional grounding**: whether the context markers carry pretrained functions that connect through attention to hypothesis queries.

- Event wording: moderate cosine (~0.85), grounded → transfer works
- "Background"/"Update" discourse labels: high cosine (0.995), discourse-functionally grounded → temporal selection works
- "Earlier"/"Later" temporal markers: high cosine (0.992), not discourse-functionally grounded → temporal selection fails
- "First"/"Second" ordinal markers: moderate cosine (0.983), not discourse-functionally grounded → temporal selection fails
- `[TIME_1]`/`[TIME_2]` novel markers: maximal cosine (1.000), no pretrained function → fails

## Three-seed replication

The section result is confirmed across 3 seeds (26600, 26601, 26602):

| Format | Train mean | Eval before mean | Eval after mean |
|---|---:|---:|---:|
| separated | 1.000 | 1.000 | 1.000 |
| section | 0.925 | 1.000 | 0.992 |
| natural | 0.465 | 0.500 | 0.500 |

Section works 3/3 seeds with eval_before=1.000 and eval_after=0.975-1.000. Natural fails 3/3 seeds at exactly chance. Separated is perfect 3/3 seeds. The discourse-functional grounding result is stable.

## What this does NOT yet establish
- The temporal bottleneck is measured on changed-only worlds without a stable base; the earlier analysis setting with mixed base adds prior-dominance confounds
- No evidence yet that the "Background"/"Update" interface survives held entities, held wording, or the full temporal-change bridge protocol
- This identifies the grounding boundary but does not yet show how to overcome it for genuine state updating (as opposed to temporal selection with semantically loaded markers)
- No natural EWoK or BabyLM-scale evidence from this mechanism route

## Connection to earlier analysis temporal bottleneck

earlier analysis showed: stable facts are learned and preserved (secondary_after 1.000) but changed focal-after is the bottleneck (0.200 in balanced_temporal, 0.625 in oracle_secondary). representational grounding boundary explains WHY: the pretrained model cannot perform temporal slot selection because context temporal cues are representationally collapsed. The model builds a strong stable-state prior from 5120 base rows and cannot use 256 sparse changed-focal rows to override it because the temporal query distinction (before vs after) is invisible in the combined context+hypothesis representation.

## Files

- Temporal bottleneck diagnostic: `data/temporal_bottleneck_diagnostic/`
- Slot marker test: `data/temporal_slot_marker_test/`
- Context separation control: `data/context_separation_control/`
- secondary retention and orientation interference full3 retention analysis: `data/secondary_retention_full3_analysis/`
- Scripts: `scripts/temporal_bottleneck_diagnostic.py`, `training/scripts/temporal_slot_marker_test.py`, `training/scripts/context_separation_control.py`
