# closing evidence consolidation representative model packages

Created: `2026-09-08T08:11:15Z`

**All valid:** ✓

## v4: Qiushi-Engine-Frontier-Advancement
- Directory: `experiments/archive/functional_learning/data/representative_model_packages/Qiushi-Engine-Frontier-Advancement`
- Model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
- Overall: `42.023967991315104`
- Source: repaired coherent86 (model weights = original coherent86)
- All file hashes match source: ✓

## v5: Qiushi-Engine-Principle-Guided-Frontier-Advancement
- Directory: `experiments/archive/functional_learning/data/v5_candidate_export/Qiushi-Engine-Principle-Guided-Frontier-Advancement`
- Model SHA256: `1c6268a37a7774a78d725307f99bbaf1f5225b4315f9bfd9ddb2a5b47c680e4a`
- Overall: `42.246412332209445`
- Source: repaired clean seed62064
- All file hashes match source: ✓

## Cross-package checks
- `tokenizer.json`: DIFFER ✗
- `tokenizer_config.json`: identical ✓
- `special_tokens_map.json`: identical ✓
- Config differences: `identical`
- `modeling_code`: identical ✓
- Model weights differ (expected): ✓

## Loader tests

### v4
- `AutoModelForMaskedLM`: ✓ → `FrozenSlowPrivateDebertaV2ForMaskedLM`, 36458592 params, 995584 private params, 48 private tensors
- `AutoModel`: ✓ → `FrozenSlowPrivateDebertaV2Model`, 36210368 params, 995584 private params, 48 private tensors

### v5
- `AutoModelForMaskedLM`: ✓ → `FrozenSlowPrivateDebertaV2ForMaskedLM`, 36458592 params, 995584 private params, 48 private tensors
- `AutoModel`: ✓ → `FrozenSlowPrivateDebertaV2Model`, 36210368 params, 995584 private params, 48 private tensors
