# earlier analysis — Route reconstruction after 100M official AMLM failed to scale

## Where the research actually stands (evidence, not narrative)

Protected reference (only complete internal 9/9): DeBERTa-v2 8×480 WWM 100M — BLiMP 66.76, Supplement 59.88, EWoK 52.19, Entity 22.62, COMPS 52.19, SuperGLUE 68.02, GlobalPIQA 35.635, Reading 7.62, AoA -0.1745, Overall 40.527. Public visible leader Overall 41.80. Gap concentrated in Entity (-5.83), GlobalPIQA (-4.03), EWoK (-3.88).

Full-budget official AMLM (earlier analysis/261), the candidate the 10M 2×2 selected:
- seed42 (matched protected init/RNG): BLiMP 67.90, Supplement 59.77, Entity 23.57, COMPS 53.16, EWoK 50.43, GlobalPIQA 34.665, Reading 7.805.
- seed43: BLiMP 67.17, Supplement 61.06, Entity 21.56, COMPS 52.76, EWoK 50.91, GlobalPIQA 35.65, Reading 7.115.

## The decisive scientific finding of this reconstruction

The 10M official-AMLM signal did **not** scale. The only clean paired 100M comparison is AMLM-seed42 vs the protected WWM-seed42 (same init/RNG coordinates):
- BLiMP +1.14, Entity +0.95, COMPS +0.97, Reading +0.185 (modest gains)
- EWoK **-1.76**, GlobalPIQA **-0.97** (the two 10M-signal columns REVERSED)
- 7-column zero-shot/Reading mean +0.057 (essentially flat)

At 10M the AMLM−WWM effect was Entity +0.58, EWoK +0.785, GlobalPIQA +1.953, mean +0.413. At 100M seed42 the EWoK/GlobalPIQA gains inverted. This is the signature of a **transient early-training optimization effect**, not a better final representation. Official AMLM alone is therefore not a SOTA route.

## Two problems that must be fixed before any new mechanism is scaled

1. **Missing paired anchor.** There is no seed43 WWM 100M run. The AMLM Entity seed spread (23.57 vs 21.56 = 2.01) already exceeds the seed42 AMLM−WWM Entity gain (+0.95). A single-seed 100M comparison cannot support route decisions.

2. **No exposure trajectory.** 10M and 100M are single points. The real unresolved scientific object is the AMLM−WWM effect **as a function of exposure** Δ_m(e) for each metric m. The checkpoint ladders (chck_1M..chck_100M) already exist for all four 100M runs (2 WWM would be needed; 1 WWM + 2 AMLM exist), so a trajectory can be measured cheaply from saved checkpoints without retraining.

## Immediate highest-value work (cheap, decisive, uses existing artifacts)

A. **Train the missing seed43 WWM 100M** with identical geometry to the protected WWM reference so both objectives have 2 matched seeds at 100M. (One H100 long job.)

B. **Measure the exposure trajectory** by evaluating existing checkpoints (e.g. chck_5M, 10M, 20M, 40M, 60M, 80M, 100M) of protected WWM seed42, AMLM seed42, AMLM seed43 (and seed43 WWM once trained) on the 7-column zero-shot/Reading coordinate. This directly tests whether the 10M AMLM EWoK/GlobalPIQA advantage decays/reverses with exposure, which the single 100M point already suggests.

These two together convert the current ambiguous single-point comparison into a real characterization of the objective effect, and prevent choosing a new long route on a phase-transient artifact.

## Candidate mechanisms for the NEXT route (only after A/B clarify the objective effect)

The persistent gap is Entity + EWoK + GlobalPIQA together, with Supplement/Reading preserved. Prior evidence closes: WESS unlabeled transfer, R1 final-answer/order readout (frozen readout held-out both-correct = 0; unfrozen memorizes, test 0.12), custom-mix under WWM, SynCSE structured+AMLM (negative central interaction), and now official AMLM at 100M.

Serious mechanism candidates that are NOT repeats:

1. **Ordered persistent-state episodes + narrow causal event memory on the DeBERTa backbone.** Keep WWM (protects Supplement/Reading); add a strictly left-to-right event-memory channel with gated residual injection; ~10-15% exposure as multi-event episodes with counterfactual twins; paired consequence loss attributing outcome error to early transitions. Targets Entity (persistent identity), EWoK (relational/commonsense state), GlobalPIQA (action→physical consequence) jointly. First test: 4-arm 10M {WWM; shuffled episodes; ordered episodes no memory; ordered episodes + memory + paired loss}, with held-out unseen-entity/template/transition-composition probes AND the official columns; continue only if arm4 beats arm2/3 on compositional probes and moves official Entity/EWoK/GlobalPIQA together with ≤0.5 Supplement/Reading loss, and inference-time memory ablation selectively harms Entity/consequence.

2. **Intermediate-state supervision on official text + relational-progress masking.** Deterministically extract entity recency, subject-relation-object, location/possession change, coreference identity, negation/availability from official narrative; require intermediate layers to linearly recover current state at event boundaries (auxiliary loss), plus counterfactual consistency. Directly attacks the layer mixing learning response result frozen-readout=0 failure object. First test: 2×2 {WWM vs relational-progress masking} × {no state supervision vs intermediate state supervision}, matched official text/mask-count/params, evaluate at 5/10/20/30M; primary diagnostic is the layer mixing learning response result-style held-out counterfactual probe (target both-correct > 0.30) plus official columns holding or widening from 10M→30M. Any externally derived target must be checked against current official rules; safest version derives targets deterministically from budget-counted official text only.

3. **Dual-interface training on one backbone: bidirectional WWM + causal prefix/event scoring.** RecGPT shows causal topology gives GlobalPIQA 40.68 / BLiMP 73.11 but Entity 16.59, so causal alone is not a three-column fix. Keep WWM path for lexical/syntax/Supplement/Reading; add causal-mask path for ordered events/consequences; pre-register a fixed fusion rule (no post-hoc interface picking). First test: 10M 4-arm {WWM; deeper/wider param-matched DeBERTa; WWM+causal-mask no aux; WWM+causal-mask+ordered consequence loss}.

4. **Multi-view event-graph over the SAME natural passage (not re-pasted static pairs).** Fixes what structured-WWM got wrong: align entity/event representations across original + coreference-rewrite + equivalent-order paraphrase + state summary + minimal counterfactual of one passage (invariance loss + counterfactual margin), swapping views per epoch to keep unique-word/exposure budget fixed and preserve developmental text.

## Common continuation criterion for any new route

Require all three, not a single 10M official number:
1. Compositional state transfer: generalizes to unseen entities, templates, transition compositions, longer chains.
2. Representation/interface evidence: frozen readout recovers ordered state, or causal-memory ablation causes selective damage.
3. Learning-dynamics evidence: Entity/EWoK/GlobalPIQA advantage persists from 10M to at least 30M (does not evaporate like AMLM at 100M), while Supplement/Reading are not eroded.

## Route status updates
- Official-corpus AMLM at full budget: CLOSED as a standalone SOTA route (EWoK/GlobalPIQA reverse at 100M).
- SynCSE structured+AMLM: not to be scaled (negative central interaction).
- structured-WWM: EWoK/BLiMP gains real but Entity not improved and Supplement lost; only worth revisiting inside a redesigned event-graph mechanism (candidate 4), not as raw CHILDES replacement.

Evidence files: `data/official_amlm_seed42_100M_zeroshot_reading.json`, `data/official_amlm_seed43_100M_zeroshot_reading.json`, `data/debertav2_b256_true_9of9_coordinate.json`, `notes/complete_2x2_evidence_synthesis.md`, `notes/two_seed_zero_shot_summary.md`, `notes/layer_mixing_learning_response_result.md`.
