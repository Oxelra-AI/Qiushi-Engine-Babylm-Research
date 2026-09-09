# evaluation repair synthesis Real AutoModel Interface Validation

Created: 2026-09-07T20:44:00.524398+00:00

## Purpose

Validates that `AutoModel.from_pretrained(path, trust_remote_code=True)` loads the correct `FrozenSlowPrivateDebertaV2Model` with both slow and private adapters, producing exact hidden-state agreement with the trusted MLM backbone. This is the real-interface test that superglue loader repair's direct instantiation established a reference for.

## coherent86_repaired
- **VALID** ✓
- Class: `FrozenSlowPrivateDebertaV2Model`
- Total params: 36210368
- Adapter params: 995584
- Private adapter params: 995584
- Adapter scales: [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
- Private scales: [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
- Hidden-state max diff: 0.0
- Exact match: True

## dense_seed62064_repaired
- **VALID** ✓
- Class: `FrozenSlowPrivateDebertaV2Model`
- Total params: 36210368
- Adapter params: 995584
- Private adapter params: 995584
- Adapter scales: [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
- Private scales: [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
- Hidden-state max diff: 0.0
- Exact match: True

## dense_seed62065_repaired
- **VALID** ✓
- Class: `FrozenSlowPrivateDebertaV2Model`
- Total params: 36210368
- Adapter params: 995584
- Private adapter params: 995584
- Adapter scales: [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75]
- Private scales: [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
- Hidden-state max diff: 0.0
- Exact match: True

## Negative Controls

### dense_seed62064_original
- Stock class: True
- Adapters missing: True
- Loaded class: DebertaV2Model

## Verdict

- All repaired valid: **True**
- All negatives stock: **True**
- Interface validated: **True**
