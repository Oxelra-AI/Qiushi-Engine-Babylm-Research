# counterfactual propagation closed — Downstream Counterfactual State Propagation closed

## Verdict: mechanistic premise falsified; close the route (no BabyLM capability screen)

The within-v3 readout localization test at 200k is decisive. All three readout modes share the identical WWM stream (same `example_order_manifest.json` hash), identical auxiliary exposure (131,448 pair words), and clean HF artifacts, so differences are attributable only to *where* the compatibility head reads.

| readout mode | heldout pair acc | heldout random acc | train pair acc | final CF loss |
|---|---:|---:|---:|---:|
| dependent (s2 pronoun/deictic) | 0.578 | 0.625 | 0.494 | 1.0593 |
| s2_nondependent (last s2 word) | 0.609 | 0.644 | 0.516 | 1.0404 |
| s1_local (edited s1 span) | 0.656 | 0.641 | 0.516 | 1.0434 |

### What this proves
- **No localization to the dependent token.** Dependent readout is not better than a matched non-dependent s2 token; it is slightly *worse* (−0.031 heldout pair acc). The linked-definition design — engineered so s2's opening pronoun refers to the edited s1 subject — provides no dependency-specific advantage.
- **The signal is a local anomaly, broadcast.** s1-local readout is easiest (+0.078 over dependent). The perturbation is detectable everywhere in the sequence at similar ~0.58–0.66 accuracy, the signature of bidirectional attention diffusing a local edit oddity rather than genuine cross-sentence state propagation.
- This is the same failure class as procedural entity-state stories (not learned) and Entity Mention Consistency (surface-solvable, non-transferring): **an auxiliary minimizable without true cross-sentence binding.**

### Principle extracted (updated)
Constructing counterfactual data where s2 *should* depend on s1 does not force the model to *use* that dependency, because a bidirectional MLM can solve the perturbation-detection objective through local anomaly + attention diffusion. Readout-location ablation on matched data is the correct falsifier and should be applied to any future "cross-sentence" auxiliary BEFORE a capability screen. Perturbation-detection framing is intrinsically vulnerable to this shortcut; a genuinely binding objective must make the *label itself* computable only by integrating s1 state into an s2 prediction (e.g. the model must generate/rank s2 content that changes with s1), not merely detect that s1 was altered.

### Routes now closed (cross-sentence auxiliary family, tested forms)
- Repeated-form Entity Mention Consistency (entity consistency 20m decision)
- Arbitrary-swap counterfactual v2 (downstream counterfactual materialization v2, rejected pre-training)
- Linked-definition counterfactual propagation v3 (counterfactual propagation closed, readout-localization falsified)

### Next direction (do NOT scale counterfactual propagation)
The recurring dead-end is auxiliary-loss engineering on top of frozen WWM. Two candidate pivots with different mechanism class, to be compared before implementation:
1. **Objective/architecture change that makes cross-sentence integration load-bearing for the primary loss**, not an auxiliary — e.g. a genuine next/prev-sentence generative or infilling objective where s2 tokens are predicted and their likelihood truly depends on s1 content (measurable by s1-ablation drop), rather than a discriminative perturbation head.
2. **Move the bottleneck away from mechanism-injection entirely** and revisit the largest measured gaps (Entity −5.83, GlobalPIQA −4.04, EWoK −3.88) via data-distribution or curriculum interventions whose effect is directly on the primary MLM objective and measurable at the task level, with the same matched-control discipline.

Evidence: `data/cfprop_readout_200k_comparison.json`, `data/cfprop_readout_smoke_comparison.json`, `data/cfprop_control_screen_comparison.json`.
