# symmetry pilot boundary findings — Symmetry-identification pilot: boundary findings

## Substrate repair (v2)

Fixed three engineering flaws in symmetry identification route's v2 substrate:
1. **Seen selection parity bug**: `idx+j` always even → only s_give selected. Fixed to `idx % len(SEEN_KEYS)`. neutral_decoupled `slot0_true_changed`: 0.000 → 0.500.
2. **Name-permutation variant offset**: swap used `i+1` instead of `i`. Fixed. `surface_first_true_changed`: 0.750 → 0.500.
3. **Leak scan over-broad**: scanned full premises (incl. intentional seen events) for held rows. Rewritten to scan only held-specific event fields. Hits: 176 → 0.

All formal checks preserved: heldheld Z2=2, aligned→true, inverted→inverted, neutral Z2=2, mixed→true.

## Pilot 1: pretrained DeBERTa + common_seen + asymmetric templates

All arms converge to mixed_orient ~0.75-0.83 regardless of bridge direction. Inverted bridge does NOT flip orientation. exposure_only already reaches 0.74 on state queries.

**Diagnosis**: Pretrained agent-patient frame bias orients nonce words ("X daxed Y" → "Y gets the object"). The bias resolves the Z2 ambiguity before any bridge evidence, making the bridge redundant.

## Pilot 2: pretrained DeBERTa + no common_seen + asymmetric templates

heldheld_only mixed drops slightly (0.82→0.75), exposure_only drops to chance. But inverted bridge still doesn't flip: all arms ~0.75 on mixed.

## Pilot 3: pretrained DeBERTa + no common_seen + SYMMETRIC templates (v2)

Templates changed to conjoined participant forms that remove agent-patient bias:
- "During the obj episode, A and B had a daxing event."
- "During the obj episode, B and A had a daxing event."

### Results (5 seeds, 50 epochs)

**Cross-seed means:**
| arm | mean_mixed | std_mixed | mean_state_chg |
|---|---:|---:|---:|
| exposure_only | 0.519 | 0.054 | 0.516 |
| heldheld_only | 0.750 | **0.000** | 0.519 |
| aligned_state | 0.747 | 0.006 | **0.775** |
| inverted_state | 0.741 | 0.019 | 0.675 |
| neutral | 0.744 | 0.012 | 0.512 |
| mixed_event | 0.625 | 0.051 | 0.500 |

### Key findings

**A. Name-order shortcut dominates comparison eval.**  
heldheld_only produces EXACTLY 0.750 on all 5 seeds (std=0.000). The comparison format encodes slot information in argument ordering: same-slot pairs have same-name-order → "same" label; cross-slot pairs have opposite-name-order → "same" label. A model that applies one name-order rule to 3/4 of the held relations gets 48/64 = 0.75 without understanding event semantics.

**B. Bridge shows weak state-query effect in the right direction.**  
aligned state_chg (0.775) > inverted (0.675) > heldheld/neutral (~0.51). Gap = 0.10 pp. But inverted is ABOVE chance, not below — meaning the bridge teaches the possession-state task format more than it teaches specific orientation. Consistent with prephase alignment design and bias analysis's head-calibration finding.

**C. Mixed_event_bridge LOWERS mixed accuracy.**  
mean=0.625 vs heldheld_only=0.750. The additional comparison rows disrupt the name-order shortcut rather than providing useful orientation.

**D. The Z2 ambiguity does NOT manifest in the pretrained language model.**  
Even with symmetric templates, the model consistently resolves heldheld relative structure to the same mixed accuracy (0.75) across all seeds. The theoretical Z2 ambiguity of the factorized model (role coordinate anchor and state probe) is broken by residual pretrained biases in comparison processing.

## Scientific boundary

The factorized identifiability law (role coordinate anchor and state probe) holds in abstract feature spaces: internally consistent held-held experience leaves a Z2 ambiguity; sparse mixed-component anchors resolve it. In pretrained language models:

1. **Syntactic-frame inheritance** orients novel vocabulary through argument-position bias even with symmetric templates.
2. **Name-order shortcuts** in comparison formats encode slot information that the model detects without semantic understanding.
3. **Task-format calibration** (prephase alignment design and bias analysis) means any supervised NLI-like bridge primarily calibrates the entailment head rather than teaching specific role orientation.

These three mechanisms overwhelm the intended 32-row bridge signal. The bridge would need:
- Hundreds of orientation-specific rows to overcome pretraining bias
- Evaluation formats immune to name-order shortcuts (e.g., masked-language probing)
- Or a non-pretrained model to remove the implicit orientation prior

## What this establishes for data-efficient learning

The pilot demonstrates a real data-efficient learning mechanism, but a different one from the intended bridge principle:

> **Pretrained frame inheritance**: A language model trained on limited data can orient novel vocabulary by inheriting role assignments from familiar syntactic frames (even conjoined participant templates carry enough positional structure). This is an implicit, efficient mechanism for extending relational knowledge to unseen predicates.

The explicit bridge mechanism (sparse mixed-component evidence) remains valid in factorized models but is masked in language models by the stronger implicit pretrained frame mechanism. To establish the explicit bridge as a language-model principle, the evaluation must first remove or control for the implicit mechanism.

## Files

- Repaired substrate: `scripts/symmetry_identification_substrate_v2.py`
- Substrate audit: `data/symmetry_identification_substrate_v2/symmetry_substrate_v2_summary.md`
- Pilot 1 (common_seen + asymmetric): `data/symmetry_pilot/`
- Pilot 2 (no common_seen + asymmetric): `data/symmetry_pilot_noseen/`
- Pilot 3 (no common_seen + symmetric): `data/symmetry_pilot_symmetric/`
- Pilot script: `training/scripts/symmetry_pilot.py`
- No-seen script: `training/scripts/symmetry_pilot_noseen.py`
