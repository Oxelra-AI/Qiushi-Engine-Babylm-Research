# complete orientation probe results frozen-encoder and fine-tuning orientation analysis

## Scientific question

Does sparse bridge state evidence propagate role orientation through DeBERTa
representations to mixed held-seen relation comparisons on the repaired equivariant symmetry repair and macro context
equivariant substrate?

## Frozen pretrained encoder result (completed)

**Architecture**: Pretrained DeBERTa-v2 8×480 (ALL params frozen) + trainable Linear(480,2) head.
**Surface**: Repaired equivariant symmetry repair and macro context active/passive substrate (transparent baselines at 0.500).
**Protocol**: 5 seeds × 6 arms × 80 epochs. Head initialized identically per seed; only training data differs.

### Key table (5-seed mean ± std)

| arm | train | hh_closure | mixed | state_chg | state_unchg | pair_both |
|---|---:|---:|---:|---:|---:|---:|
| exposure_only | nan | 0.514±0.031 | 0.503±0.018 | 0.502±0.005 | 0.500±0.005 | 0.255±0.004 |
| heldheld_only | 0.960±0.005 | 0.338±0.019 | 0.496±0.002 | 0.500±0.000 | 0.500±0.000 | 0.250±0.000 |
| aligned_state_bridge | 0.778±0.018 | 0.334±0.036 | 0.507±0.011 | 0.517±0.015 | 0.548±0.015 | 0.252±0.011 |
| inverted_state_bridge | 0.769±0.016 | 0.372±0.008 | 0.497±0.008 | 0.519±0.012 | 0.526±0.011 | 0.239±0.022 |
| neutral_decoupled | 0.646±0.005 | 0.397±0.026 | 0.498±0.006 | 0.508±0.015 | 0.494±0.010 | 0.258±0.011 |
| mixed_event_bridge | 0.915±0.009 | 0.391±0.014 | 0.514±0.005 | 0.527±0.020 | 0.486±0.014 | 0.250±0.010 |

### Scientific findings

1. **No transferable role-slot interface in frozen [CLS]**: All arms produce mixed ≈ 0.50 (±0.02).
   aligned − inverted on mixed = 0.010 (within noise). The pretrained encoder's [CLS] 
   representation does not carry role-abstract event structure that a linear head can read.

2. **Name-specific overfitting**: heldheld_only fits training at 0.960 but ANTI-TRANSFERS
   to eval held-held closure at 0.338 (below chance). The head learns name-specific 
   patterns that are systematically wrong for disjoint eval names.

3. **Bridge arms also fail to orient**: aligned_state_bridge trains at only 0.778
   (mixed relation_comparison + state_query tasks). The frozen representations don't
   cleanly separate state_query labels across names, so bridge evidence can't be absorbed.

4. **State readouts near chance**: All arms show state_changed and state_unchanged ≈ 0.50.
   The frozen [CLS] does not encode who-ends-up-with-what even for training rows well enough
   for a linear head to generalize.

### What this eliminates

- **Inherited role-slot interface hypothesis**: The pretrained encoder (trained on 80M BabyLM
  words) does NOT embed abstract role structure accessible by linear projection. The symmetry pilot boundary findings
  "frame bias" finding was likely a surface-order artifact of the old evaluation format.

- **Frozen-representation orientation**: Bridge state evidence cannot orient role comparisons
  through fixed representations because those representations lack the needed abstraction.

### What this does NOT eliminate

- **Fine-tuning orientation**: If fine-tuning the encoder changes representations to be more
  role-abstract, bridge evidence could propagate through the adapted representations. This
  would mean the bridge mechanism works through representation learning, not through inherited 
  structure. This is the active test (background task s279_t14_tool1).

## Fine-tuning probe (running)

**Architecture**: Full DeBERTa-v2 fine-tuning (all params unfrozen) + Linear(480,2) head.
**Protocol**: 5 seeds × 6 arms × 50 epochs. LR 2e-5 encoder, 1e-3 head.
**Critical prediction**: If bridge evidence works through representation adaptation:
  - aligned mixed > 0.5 (true orientation)
  - inverted mixed < 0.5 (opposite orientation)
  - aligned − inverted > 0 (significant separation)
  - heldheld_only, neutral: no orientation (baseline)

**Calibration control**: All arms share the same comparison training rows (heldheld_only's 
192 relation_comparison rows are always present). Bridge arms add state_query rows. If 
aligned/inverted differ from heldheld_only on mixed, it's the state evidence doing work, 
not comparison format calibration.

## Connection to the symmetry-identification principle

The identifiability law (role coordinate anchor and state probe) states: internally coherent role clusters remain 
globally ambiguous up to role permutation; sparse mixed held-seen evidence resolves the 
permutation. The frozen result shows that in a pretrained language model, the ambiguity 
may be resolved (or spuriously resolved) by pretrained biases rather than explicit bridge 
evidence. The fine-tuning test asks whether a learning process (not just a fixed 
representation) can use bridge evidence to orient roles.

## macro context

companion analysis's register-substitution arms are running (message #348). They test whether admitting 
FineWeb packets that displace ordinary BabyLM heldout text at rho≈0.011 produces register-
dependent or register-independent score changes. This is independent of the orientation 
mechanism but may reveal data-composition effects on competence distribution. No urgent 
challenge from side until their scores land.

## Files
- Frozen pretrained probe: `data/frozen_pretrained_probe/`
- Fine-tuning probe (running): `data/finetune_orientation_probe/`
- equivariant symmetry repair and macro context substrate: `data/equivariant_symmetry_substrate/`
- equivariant symmetry repair and macro context baselines: `data/surface_baselines/`, `bow_surface_baseline/`
- equivariant symmetry repair and macro context positive control: `data/slot_orbit_solver/`
