# superglue loader repair: SuperGLUE AutoModel Loader Repair

## Problem identified

The official SuperGLUE finetune pipeline in `evaluation_pipeline/finetune/classifier_model.py` line 77 calls:
```python
self.transformer = AutoModel.from_pretrained(config.model_name_or_path, trust_remote_code=True)
```

Our custom checkpoints (coherent86/v4 and dense candidates) only register `AutoModelForMaskedLM` in their `auto_map`:
```json
{"AutoModelForMaskedLM": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM"}
```

**Result:** `AutoModel` falls back to stock `DebertaV2Model` (model_type: "deberta-v2"), loading only 34,219,200 params and silently dropping:
- 995,584 slow adapter params (inherited chck_82M function at scale 1.75)
- 995,584 private adapter params (dense learned update at scale 0.75)
- **Total: 2,239,392 parameters lost**

**Forward confirmation:** Trusted MLM encoder vs stock AutoModel produces max hidden-state difference of **1.63** (mean 0.26). This is not a subtle difference — the entire adapter computation (the defining characteristic of this model) is absent.

**Consequence for old results:**
- Zero-shot/Reading evaluations: VALID (use `AutoModelForMaskedLM` path, which was correctly registered)
- SuperGLUE evaluations: ALL INVALID (both coherent86 and dense used the same stripped base model)
- Old SuperGLUE results for coherent86, dense seed62064, and dense seed62065 all evaluated the same stock encoder, explaining their nearly identical task scores
- The old Overall values depended on invalid SuperGLUE; they cannot be used for v5 comparison

## Repair applied

Created `FrozenSlowPrivateDebertaV2Model` — a base encoder variant with both adapter paths, registered under `AutoModel` in the config's auto_map.

### Repaired checkpoints
Located at `experiments/archive/functional_learning/data/automodel_repair`:
- `repaired_dense_seed62064_u0080/`
- `repaired_dense_seed62065_u0080/`
- `repaired_coherent86_alpha075/`

Each contains:
- Patched `frozen82_private_modeling.py` with `FrozenSlowPrivateDebertaV2Model` class
- Updated `config.json` with `"AutoModel": "frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model"`
- Symlinked `model.safetensors` (saves space, same weights)

### Validation results (two layers)

**superglue loader repair direct validation** (`direct_validation.json`): all three repaired checkpoints validated via
direct class instantiation with manual `deberta.` prefix stripping. `max_diff=0.0`, `mean_diff=0.0`,
36,210,368 base params, 995,584 slow + 995,584 private adapter params, scales 1.75/0.75.
This is the successful positive evidence. **Note:** `repair_validation.json`/`.md` record the
first attempt that failed due to read-only HF cache in the main shell; they should NOT be cited
as proving repair validity.

**evaluation repair synthesis real-interface validation** (`real_interface_validation/real_interface_validation.json`):
all three repaired checkpoints validated via the **actual** `AutoModel.from_pretrained(path, trust_remote_code=True)`
call with a writable `HF_HOME`, matching the evaluator environment. Same result:
`FrozenSlowPrivateDebertaV2Model`, 995,584+995,584 adapter params, scales 1.75/0.75, **max_diff=0.0**.
Negative control confirmed original unrepaired checkpoint loads as stock `DebertaV2Model` with 0 adapters.
This is the decisive test that the repaired model loads correctly through the actual interface.

### Evaluation status at the time
- Repaired coherent86 SuperGLUE: started.
- Repaired dense seed62064 SuperGLUE: started.
- Repaired dense seed62065 SuperGLUE: not yet started.

Both use the existing earlier analysis evaluation pipeline with no evaluator modifications — the fix is entirely in the model checkpoint files, as required for submission compatibility.

## AoA status (updated evaluation repair synthesis)

The BabyLM README (lines 209, 218) explicitly supports early stopping: "we require [...]
checkpoints [...] or up until the one you trained." Coherent86 (~86M words) and dense (~89M words)
both need checkpoints through chck_80M (the last 10M milestone ≤ their training endpoints).

- 17 early-stop ancestral checkpoints (chck_1M through chck_80M) verified available from earlier analysis ladder
- They are `DebertaV2ForMaskedLM` with slow adapter (35,463,008 params) — genuine ancestral states
- Final endpoints are `FrozenSlowPrivateDebertaV2ForMaskedLM` (36,458,592 params)
- Custom AoA runner (`aoa_runner.py`) subclasses official `StepSurprisalExtractor` for local paths
- Dry run validated. Ready to launch when GPU is available
- Both models share identical ancestral ladder; AoA difference will be very small

## Key files
- Diagnostic: `data/superglue_loader_validation/loader_diagnostic.json`
- Repair script: `scripts/repair_automodel.py`
- Direct validation (superglue loader repair): `data/automodel_repair/direct_validation.json` ✓
- Real-interface validation (evaluation repair synthesis): `data/real_interface_validation/real_interface_validation.json` ✓
- Stale first attempt (NOT evidence): `data/automodel_repair/repair_validation.json`
- AoA assessment: `data/aoa_staging_assessment/aoa_staging_assessment.json`
- AoA runner: `scripts/aoa_runner.py`
- Comparison script: `scripts/trusted_comparison.py`
