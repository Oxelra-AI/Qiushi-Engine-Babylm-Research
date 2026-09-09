# complete orientation probe results: Complete frozen/fine-tuning orientation probe results

## Summary of findings

Three probe conditions tested whether bridge state evidence propagates role 
orientation through DeBERTa representations on the repaired equivariant symmetry repair and macro context equivariant 
symmetry-identification substrate.

### Condition 1: Frozen pretrained encoder + linear head (5 seeds × 80 epochs)
- All mixed readouts: **0.492–0.514** (no orientation, ±0.02 noise)
- heldheld_only comparison overfits training (0.960) but anti-transfers to eval (hh 0.338)
- Bridge arms don't even fit state_query well (train 0.77) with frozen reps
- **Finding**: Pretrained [CLS] does NOT carry transferable role-slot information

### Condition 2: Frozen random encoder + linear head (5 seeds × 80 epochs)  
- All mixed readouts: **0.449–0.553** (wider noise, no consistent direction)
- heldheld_only: train 0.865, hh 0.547 (chance, not anti-transfer like pretrained)
- **Finding**: Random reps are uninformative; pretrained anti-transfer confirms entity-specific encoding

### Condition 3: Full fine-tuning (5 seeds × 50 epochs)
| arm | train | hh | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.514 | 0.503 | 0.502 | 0.500 | 0.255 |
| heldheld_only | 1.000 | 0.372 | 0.495 | 0.498 | 0.510 | 0.255 |
| aligned_state | 0.909 | 0.423 | **0.492** | **0.795** | **0.758** | **0.559** |
| inverted_state | 0.914 | 0.414 | **0.499** | **0.738** | **0.842** | **0.592** |
| neutral | 0.995 | 0.436 | 0.503 | 0.566 | 0.654 | 0.366 |
| mixed_event | 1.000 | 0.405 | 0.493 | 0.517 | 0.473 | 0.236 |

## Three central findings

### Finding 1: State inference generalizes; comparison does not
Fine-tuned aligned arm: state_chg = 0.795 on new names (strong generalization)
Fine-tuned any arm: mixed = 0.49-0.50, hh = 0.37-0.44 (no comparison generalization)
Cross-template state: aligned xtempl_chg = 0.780, xtempl_unchg = 0.766 (generalizes across templates AND names)

The model learns first-order state consequences (who ends up with what) and generalizes 
them to new entities and new voice/template variations. But it does NOT learn second-order 
role comparisons (are these two events the same role structure) in a way that generalizes.

### Finding 2: Inverted evidence does NOT flip orientation on new entities
Under true labels (true = transfer-to-patient):
- aligned: state_chg = 0.795 (correct direction, consistent with training)
- inverted: state_chg = 0.738 (STILL true direction, inconsistent with inverted training)
- inverted implicit inverted accuracy: 1 - 0.738 = 0.262

The model learns the inverted mapping for TRAINING names (train_acc 0.914) but reverts 
to the true (agent-patient) mapping for EVAL names. This is name-specific override, 
not generalizable orientation flipping.

### Finding 3: Pretrained frame activation hierarchy
- heldheld_only (no state evidence): state_chg = 0.498 (frame NOT activated)
- neutral (seen-event state only): state_chg = 0.566 (partial activation)
- aligned (consistent bridge): state_chg = 0.795 (strong activation)
- inverted (contradictory bridge): state_chg = 0.738 (frame STILL activates in true direction)

State training ACTIVATES a latent pretrained agent-patient frame that was learned during 
BabyLM pretraining. The activation generalizes to new entities because the frame is 
entity-general. Aligned evidence reinforces the frame; inverted evidence creates name-
specific overrides that don't generalize.

## Mechanism: selective activation, not representation creation

The data-efficient learning mechanism revealed by this experiment is:

**Sparse evidence activates and amplifies existing pretrained representational biases 
rather than creating new generalizable representations.**

Evidence:
1. Frozen [CLS] carries NO transferable role information → the pretrained representations 
   don't have an accessible role-slot interface
2. Fine-tuning enables state inference to generalize → fine-tuning activates a latent 
   structural bias (agent-patient frame)
3. The activation follows the pretrained direction → aligned evidence strengthens the 
   default; inverted evidence creates name-specific overrides
4. heldheld_only (comparison-only, no state training) does NOT activate the frame → 
   activation requires format-matched evidence (state queries)
5. neutral (seen-event state) PARTIALLY activates → even indirect state evidence helps
6. The activation does NOT transfer across task formats (state → comparison fails)

## Connection to symmetry-identification principle

The role coordinate anchor and state probe Z2 identifiability law holds formally (slot-orbit positive control: 
aligned selects true, inverted selects inverted, neutral preserves ambiguity). 

But in a fine-tuned DeBERTa:
- The Z2 ambiguity is pre-resolved by the pretrained agent-patient frame
- Bridge evidence activates rather than orients the frame
- Inverted evidence cannot flip the frame for new entities (0.738 > 0.5 on true labels)
- The comparison readout (which tests orientation directly) shows no bridge effect

**The symmetry-identification law describes the formal structure of the learning problem,
but a neural language model with pretrained knowledge does not solve it through explicit
orientation. Instead, it leverages activation of existing biases.**

## Implications for data-efficient learning

1. **Activation over creation**: Limited data selects and amplifies existing representational 
   structure rather than building new structure from scratch. This is why pretraining is so 
   valuable — it provides the structure that sparse evidence can activate.

2. **Format-specificity**: Activation is task-format-specific. State-query evidence activates 
   state inference; it does NOT transfer to comparison inference. This limits the scope of 
   few-shot generalization.

3. **Direction-consistency**: Only evidence consistent with pretrained biases generalizes 
   to new entities. Contradictory evidence creates name-specific overrides that don't 
   transfer. This means curriculum/data design should align with, not fight, the model's 
   existing representations.

4. **First-order before second-order**: State inference (first-order: event → consequence) 
   generalizes before comparison inference (second-order: event ↔ event role matching). 
   Limited data teaches "what happens" before "how things relate."

## Files
- Frozen pretrained: `data/frozen_pretrained_probe/`
- Frozen random: `data/frozen_random_probe/`
- Fine-tuning: `data/finetune_orientation_probe/`
- equivariant symmetry repair and macro context substrate: `data/equivariant_symmetry_substrate/`
- equivariant symmetry repair and macro context baselines: `data/surface_baselines/`, `bow_surface_baseline/`
- equivariant symmetry repair and macro context positive control: `data/slot_orbit_solver/`
- Analysis note: `notes/frozen_encoder_and_finetune_analysis.md`
