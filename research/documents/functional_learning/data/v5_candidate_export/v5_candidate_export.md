# evidence decomposition and missing controls v5 candidate export package

Created: `2026-09-08T05:48:04Z`

**All valid:** `True`

## Candidate

- Name: `Qiushi-Engine-Principle-Guided-Frontier-Advancement`
- Overall: `42.246412332209445`
- Delta over coherent86: `+0.2224443408943415`
- Seed: `62064` (replication: `62065`)
- Export directory: `experiments/archive/functional_learning/data/v5_candidate_export/Qiushi-Engine-Principle-Guided-Frontier-Advancement`

## File copy verification

All files match evaluated source: `True`

| File | Size | SHA256 | Match |
|------|------|--------|-------|
| model.safetensors | 145,866,368 | `1c6268a37a7774a7...` | ✓ |
| config.json | 1,296 | `ca833c70856ad6a3...` | ✓ |
| tokenizer.json | 1,135,681 | `f38b6fd98f77dcb3...` | ✓ |
| tokenizer_config.json | 1,352 | `7d1c0433f64f6655...` | ✓ |
| special_tokens_map.json | 692 | `72e39f19b03b1f0c...` | ✓ |
| frozen82_private_modeling.py | 9,467 | `c96ea8f17065bf37...` | ✓ |
| bridge_metadata.json | 2,990 | `4ab174bfbe9bf90e...` | ✓ |

## Hash chain (training → repaired → export)

### model_safetensors
- training_source: `1c6268a37a7774a78d725307...`
- repaired_evaluated: `1c6268a37a7774a78d725307...`
- candidate_export: `1c6268a37a7774a78d725307...`
- All identical: `True`

### config.json
- training_source: `ca6792fe1842f0e89e7080e7...`
- repaired_evaluated: `ca833c70856ad6a3fddaa56d...`
- candidate_export: `ca833c70856ad6a3fddaa56d...`
- Note: AutoModel registration added during repair; model weights unchanged

### frozen82_private_modeling.py
- training_source: `88dc672ed9a3bcaa7bb80f85...`
- repaired_evaluated: `c96ea8f17065bf3713b07e8f...`
- candidate_export: `c96ea8f17065bf3713b07e8f...`
- Note: FrozenSlowPrivateDebertaV2Model class added during repair

### tokenizer.json
- training_source: `f38b6fd98f77dcb33568f77b...`
- repaired_evaluated: `f38b6fd98f77dcb33568f77b...`
- candidate_export: `f38b6fd98f77dcb33568f77b...`
- All identical: `True`

### tokenizer_config.json
- training_source: `7d1c0433f64f6655522bc461...`
- repaired_evaluated: `7d1c0433f64f6655522bc461...`
- candidate_export: `7d1c0433f64f6655522bc461...`
- All identical: `True`

### special_tokens_map.json
- training_source: `72e39f19b03b1f0c7b38a08f...`
- repaired_evaluated: `72e39f19b03b1f0c7b38a08f...`
- candidate_export: `72e39f19b03b1f0c7b38a08f...`
- All identical: `True`

## Loader test (trust_remote_code=True)

- AutoModelForMaskedLM: `FrozenSlowPrivateDebertaV2ForMaskedLM` (36,458,592 params) — valid: `True`
- AutoModel: `FrozenSlowPrivateDebertaV2Model` (36,210,368 params) — valid: `True`

## Six-endpoint map

| Alias | Role | Overall | Model SHA256 (prefix) |
|-------|------|---------|----------------------|
| coherent86 | parent / reference baseline | 42.023967991315104 | `e14d757ae51b41e3...` |
| dense_seed62064_MM | dense (M,M) acquisition control, seed 62064 | 42.14909111936738 | `0b21387bcc060a27...` |
| dense_seed62065_MM | dense (M,M) acquisition control, seed 62065 | 42.168422702303516 | `ef7f73c98e0449a8...` |
| exact_MS_seed62064 | acquisition-only (M,S), seed 62064 — same-seed direct compar | 42.20253795433653 | `c14ac2c82470748a...` |
| clean_seed62064 | v5 CANDIDATE — (M,S) acquisition + deterministic parent anch | 42.246412332209445 | `1c6268a37a7774a7...` |
| clean_seed62065 | full-policy REPLICATION — (M,S) acquisition + deterministic  | 42.23173113265801 | `b814dd54340a9290...` |

## Scientific scope

**Claim:** Highest among six complete endpoints on the repaired adapter-aware fixed coordinate

**Not claimed:**
- Unqualified global or historical platform SOTA
- Uniform capability improvement
- Universal preservation or monotonicity law
- Broad population-level superiority beyond fixed validation items

**Open:**
- Matched ftseed44 SuperGLUE comparison (in progress)
- Whether parent posterior matching is uniquely responsible vs. attenuation/rollback
- Whether preservation is most resource-efficient use of additional presentations
- Population-level generalization beyond fixed BabyLM evaluation items
