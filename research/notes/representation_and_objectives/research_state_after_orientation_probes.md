# complete orientation probe results — Research state after orientation probes

## Findings

Three probe conditions (frozen pretrained, frozen random, full fine-tuning) on the repaired
equivariant symmetry repair and macro context equivariant substrate produced a clear and consistent result:

**Mixed held-seen orientation = 0.50 across ALL conditions and ALL arms.** Bridge evidence 
(state or comparison format) does NOT propagate to relation comparisons.

**State inference generalizes to new entities.** Fine-tuned aligned arm: state_chg = 0.795 
(128 bridge rows → 80% accuracy on new names). Cross-template state: 0.78 (new templates too).

**Inverted evidence does NOT flip orientation.** On true-labeled eval, inverted arm 
state_chg = 0.738 > 0.5. The model reverts to the true agent-patient mapping on new 
entities rather than generalizing the inverted training.

## Mechanism identified: selective activation

Sparse fine-tuning evidence ACTIVATES a latent pretrained agent-patient frame rather than 
creating new abstract representations:
1. Frozen [CLS] carries no transferable role info (probe at chance)
2. Fine-tuning with state evidence activates a latent frame (state rises to 0.80)
3. Activation follows pretrained direction (inverted reverts to true on new names)
4. Activation is task-format-specific (state → comparison does not transfer)
5. Without state training (heldheld_only), the frame stays dormant (state = 0.50)

## What this changes in the research

**Symmetry identification**: The Z2 law holds formally but is pre-resolved by pretrained 
biases in neural models. The explicit sparse-bridge mechanism does not operate at language 
scale.

**Compact view**: The DeBERTa-specific compact advantage may be another instance of selective 
activation: compact rewrites that are consistent with DeBERTa's pretrained biases get 
amplified, while the same rewrites under RoBERTa (different biases) produce no benefit.

**Entity-state binding deficit**: The binding failure is now more precisely characterized: 
models learn entity-event associations (name-specific) not abstract role-entity bindings. 
State inference generalizes because it activates a pretrained frame; comparison fails because 
it requires entity abstraction that the pretrained model doesn't have.

## What the accumulated evidence establishes

Across compositional update test design, the experiments have characterized:
1. **Supplied coordinates solve binding** but this is trivial (tags, not semantics)
2. **Coordinate identifiability** is a formal structure that neural models don't implement
3. **Reusable argument slots** work in controlled settings but require identity support
4. **Pretrained frame activation** is the mechanism for first-order state generalization
5. **Second-order comparison** does not generalize across entities
6. **Inverted evidence** creates name-specific overrides, not generalizable representations

The integrating principle: **Data-efficient learning in pretrained LMs operates through 
selective activation of existing representational biases. Generalization scope is bounded by 
(a) the quality of pretrained biases, (b) task-format specificity of activation evidence, 
and (c) entity-specificity of learned overrides.**

## Open assessment questions

1. Is this "selective activation" principle strong enough to be THE data-efficient learning 
   contribution of this study?
2. Does it need additional strengthening experiments (cross-architecture, scaling, connection 
   to BabyLM evaluation)?
3. How does selective activation relate to the register-substitution comparison? Register-dependent effects of register removal would connect data composition to activation quality.
4. Should the next step consolidate the evidence into a principled synthesis, or continue 
   with more targeted experiments?

## Canonical file locations
- Frozen pretrained: `data/frozen_pretrained_probe/`
- Frozen random: `data/frozen_random_probe/`
- Fine-tuning: `data/finetune_orientation_probe/`
- equivariant symmetry repair and macro context substrate: `data/equivariant_symmetry_substrate/`
- equivariant symmetry repair and macro context controls: `data/surface_baselines/`, `bow_surface_baseline/`, `slot_orbit_solver/`
- Detailed analysis: `notes/complete_orientation_probe_results.md`
