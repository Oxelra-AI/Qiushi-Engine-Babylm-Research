# analysis framework for factorial probe — Analysis framework for factorial probe results

## What to read first
- Probe results: `data/factorial_probe/factorial_probe_summary.md`
- All raw results: `data/factorial_probe/all_results.json`

## Key transition readouts

### 1. same_changed_acc (paired_state_conservation)
**Underdetermined**: should be LOW (~0.0-0.3, anti-copy predicts wrong person)
**Disambiguated**: should be HIGH (~0.7-1.0, event-role learned)
**Δ > 0.3 is a strong transition signal**

### 2. mixed_true_stmt (mixed_held_seen_orientation)
**Underdetermined**: should be ~0.5 (no orientation transfer)
**Disambiguated aligned > inverted**: if orientation transfers through event-role
**Aligned - inverted > 0.05 is meaningful**

### 3. same_pair_both (paired_state_conservation)  
**Underdetermined**: should be ~0.0 (changed fails on initial=same)
**Disambiguated**: should be HIGH (changed + unchanged both correct)
**This is the single strongest conservation readout**

### 4. opposite_changed_acc comparison
**Both conditions**: should be HIGH (~0.7-1.0, both rules work on initial=opposite)
**If disambiguated opposite_changed drops substantially**: event-role isn't simply additive

## How to interpret outcomes

### A: Transition occurs (Δ same_changed > 0.3, orientation separates)
→ Disambiguation forces compositional event-role learning
→ General principle: compositional structure acquisition requires disambiguating simpler rules
→ Connect to role coordinate anchor and state probe identifiability
→ Proceed to articulate the principle and test robustness

### B: same_changed rises but no orientation
→ Event-role learned for state format but doesn't transfer to comparison format
→ Task-format-specific learning, not general compositional structure
→ Need to investigate why comparison format doesn't benefit

### C: No change (same_changed stays low in both)
→ Model can't learn event-role even when anti-copy is broken
→ The nonce-relation surface may be too hard for this DeBERTa
→ Try simpler substrates, more epochs, or different architecture

### D: Both conditions show high same_changed
→ Model already uses event-role information, not anti-copy
→ Check: is there another shortcut in the disambiguated condition?
→ Re-examine the formal equivalence derivation

### E: heldheld_only (no state training) shows high same_changed
→ The pretrained checkpoint already has this capability
→ State training is not the source of the competence
→ This would be evidence for an existing pretrained structural bias

## Files
- Substrate: `data/factorial_initial_ownership/`
- Formal derivation: `notes/formal_derivation_factorial_disambiguation.md`
- Verified confound: `notes/verified_initial_owner_shortcut_and_open_controls.md`
- equivariant symmetry repair and macro context base: `data/equivariant_symmetry_substrate/`
