# evaluation repair synthesis Evaluation Repair and AoA Recovery Synthesis

## What This Step Established

### 1. Real AutoModel Interface Validation (Decisive)

The superglue loader repair repair added `FrozenSlowPrivateDebertaV2Model` to the modeling file and
registered `AutoModel` in `auto_map`. evaluation repair synthesis validated this through the **actual**
`AutoModel.from_pretrained` interface with a writable `HF_HOME`, not just direct import:

| Checkpoint | AutoModel class | Adapter params | Private params | Scales | Max hidden diff |
|---|---|---|---|---|---|
| coherent86_repaired | FrozenSlowPrivateDebertaV2Model | 995,584 | 995,584 | 1.75/0.75 | **0.0** |
| dense_seed62064_repaired | FrozenSlowPrivateDebertaV2Model | 995,584 | 995,584 | 1.75/0.75 | **0.0** |
| dense_seed62065_repaired | FrozenSlowPrivateDebertaV2Model | 995,584 | 995,584 | 1.75/0.75 | **0.0** |
| dense_original (negative control) | DebertaV2Model (stock) | 0 | 0 | — | 1.63 |

This is the **decisive** real-interface test. The running SuperGLUE evaluations are using
the correct adapter-equipped encoder.

### 2. Repaired Comparison Infrastructure

New `trusted_comparison.py` corrects all superglue loader repair logic issues:
- **Requires** all 7 SuperGLUE subtasks with primary metrics (not partial means)
- **Separates** three coordinates:
  - `projected_overall_aoa0`: shared AoA=0 placeholder for conditional arithmetic
  - `measured_overall`: uses real AoA from surprisal evaluation
  - `submission_ready`: all components complete and validated
- **Does not** set v5_established from projected delta alone
- **Preserves** historical platform-style record (42.1210...) as context, not baseline

### 3. AoA Recovery Plan (Actionable)

**Convention**: BabyLM Strict-Small early-stop (README lines 209, 218):
"checkpoints up until the one you trained"

**Checkpoints needed** (both coherent86 and dense):
- chck_1M through chck_9M + chck_10M through chck_80M = 17 ancestral
- + 1 final endpoint = 18 total

**All 17 ancestral checkpoints verified available** from earlier analysis ladder in frontier_consolidation.
They are `DebertaV2ForMaskedLM` with slow adapter (35,463,008 params).
Final endpoints are `FrozenSlowPrivateDebertaV2ForMaskedLM` (36,458,592 params).
Architecture change at 82M is genuine training history.

**Custom AoA runner** (`aoa_runner.py`) subclasses official `StepSurprisalExtractor`
to load from individual directories, ensuring exact numerical agreement with official code.
Dry run validated. Ready to launch when GPU is available.

**Key insight**: coherent86 and dense share the **same** ancestral ladder through 80M.
Their AoA difference will be minimal. AoA primarily affects absolute Overall level,
not the relative delta.

### 4. Early Repaired SuperGLUE Evidence

With adapters loaded, BoolQ already shows differentiation:
- coherent86 repaired: **67.278** (1100/1635)
- dense64 repaired: **67.951** (1111/1635) 
- old stock (no adapters): 68.563 (1121/1635)

The adapter computation **lowers** BoolQ but **differentiates** the models.
This is mechanistic evidence that the private adapters affect supervised finetune behavior.

### 5. Independent evaluation confirmation

Independent evaluation confirmed stock-encoder SuperGLUE range 0.053041 across all checkpoints.
Established lever rules for composed candidates requiring both private seeds to move
relevant columns consistently. The comparison target is faithful v4 (adapter-model SuperGLUE), not old
stripped 42.1210.

## Current Status

- coherent86 repaired SuperGLUE: BoolQ completed, ~6 subtasks remaining.
- dense64 repaired SuperGLUE: BoolQ completed, ~6 subtasks remaining.
- The AoA implementation was ready, but full evaluation had not started.
- Seed62065 repaired SuperGLUE had not started.

## Files Produced

- `data/real_interface_validation/real_interface_validation.json` — decisive interface test
- `data/trusted_comparison/trusted_comparison.json` — incomplete, waiting for SuperGLUE
- `data/aoa_staging_assessment/aoa_staging_assessment.json` — checkpoint availability
- `data/aoa_evaluation/coherent86/aoa_dry_run.json` — AoA runner validation
- `scripts/real_interface_validation.py`
- `scripts/trusted_comparison.py`
- `scripts/aoa_staging_assessment.py`
- `scripts/aoa_runner.py`

## Next Steps (Priority Order)

1. **Collect** repaired SuperGLUE results when tasks deliver
2. **Run** `trusted_comparison.py` with complete SuperGLUE
3. **Launch** AoA for coherent86 (and dense if projected positive)
4. **Launch** seed62065 repaired SuperGLUE
5. **Compute** measured Overall with real AoA scores
6. If v5 remains promising, launch dense-mask/sparse-label causal control
