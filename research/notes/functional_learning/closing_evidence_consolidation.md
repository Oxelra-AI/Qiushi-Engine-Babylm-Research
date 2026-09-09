# closing evidence consolidation Closing Evidence Consolidation

## Representative Model Packages — Verified

### v4: Qiushi-Engine-Frontier-Advancement
- **Model:** Repaired coherent86 (private scale 0.75)
- **SHA256:** `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`
- **Overall (repaired coordinate):** `42.023967991315104`
- **Parameters:** 36,458,592 total; 995,584 private-adapter (48 tensors)
- **Loader tests:** AutoModelForMaskedLM ✓, AutoModel ✓, fine-tuning smoke ✓
- **Export:** `experiments/archive/functional_learning/data/representative_model_packages/Qiushi-Engine-Frontier-Advancement`

### v5: Qiushi-Engine-Principle-Guided-Frontier-Advancement
- **Model:** Clean seed62064 ((M,S) acquisition + deterministic parent anchoring)
- **SHA256:** `1c6268a37a7774a78d725307f99bbaf1f5225b4315f9bfd9ddb2a5b47c680e4a`
- **Overall (repaired coordinate):** `42.246412332209445`
- **Parameters:** 36,458,592 total; 995,584 private-adapter (48 tensors)
- **Loader tests:** AutoModelForMaskedLM ✓, AutoModel ✓, fine-tuning smoke ✓
- **Export:** `experiments/archive/functional_learning/data/v5_candidate_export/Qiushi-Engine-Principle-Guided-Frontier-Advancement`
- **Exposure:** 89,685,369 words (within 100M budget)

### Cross-package
- Vocabulary and encoding: identical
- Config structure: identical
- Modeling code: identical
- Tokenizer padding/truncation default: benign difference (256 vs 512)
- Model weights: different (as expected — different endpoints)

## Complete Fixed-Coordinate Results (Repaired Coordinate)

| Endpoint | BLiMP | Suppl | EWoK | Entity | COMPS | SuperGLUE | GPIQA | Reading | AoA | Overall |
|----------|-------|-------|------|--------|-------|-----------|-------|---------|-----|---------|
| coherent86 | 68.51 | 63.64 | 50.02 | 28.32 | 52.05 | 68.946 | 38.565 | 8.165 | 0.0 | 42.024 |
| ordinary s64 | 68.47 | 63.63 | 49.83 | 28.16 | 52.00 | 68.988 | 39.535 | 8.22 | 0.0 | 42.093 |
| exact (M,S) s64 | 68.12 | 63.08 | 49.95 | 29.39 | 52.15 | 68.888 | 40.05 | 8.195 | 0.0 | 42.203 |
| clean s64 | 68.26 | 63.28 | 49.82 | 29.40 | 52.16 | 69.048 | 40.05 | 8.20 | 0.0 | **42.246** |
| clean s65 | 68.22 | 63.29 | 49.72 | 29.45 | 52.14 | 69.021 | 40.05 | 8.195 | 0.0 | 42.232 |
| dense (M,M) s64 | 68.06 | 63.04 | 49.92 | 29.37 | 52.15 | 68.532 | 40.05 | 8.22 | 0.0 | 42.149 |
| dense (M,M) s65 | 68.02 | 63.09 | 49.77 | 29.42 | 52.15 | 68.806 | 40.05 | 8.21 | 0.0 | 42.168 |

## Policy Contrast Chain (Seed62064)

1. **Parent → Ordinary:** `+0.06864` (mostly narrow GlobalPIQA item movement)
2. **Ordinary → Exact (M,S):** `+0.10993` (effective-input + target/credit change)
3. **Exact (M,S) → Clean:** `+0.04387` (anchoring-associated retention/transfer repair)
4. **Total clean over parent:** `+0.22244`

These are observed contrasts between specified policies, not three independently isolated causal effects.

## Mechanism Evidence Summary

### Dense-state acquisition vs preservation (common-support probe)
On 1,003 fixed common-support target tokens:

| Endpoint | Dense CE gain | Dense rank gain | Parent KL |
|----------|--------------|-----------------|-----------|
| Parent (coherent86) | 0.000 | 0.0 | 0.000 |
| Ordinary | −0.005 | 0.7 | 0.001 |
| Exact (M,S) | **0.757** | **132.8** | 0.503 |
| Clean s64 | **0.682** | **128.3** | 0.353 |
| Clean s65 | **0.679** | **128.0** | 0.345 |
| Dense-corr pres | 0.159 | 34.8 | 0.009 |

**Key finding:** Exact (M,S) improves dense-state ground-truth prediction substantially. Clean retains most of this fit while reducing KL. Dense-corruption preservation collapses KL but suppresses most acquired fit. Ordinary continuation acquires no dense-state fit.

### Evidence-availability trade (NLL delta vs coherent86)

| Endpoint | Correct src | Wrong src | View only | Source erased |
|----------|------------|-----------|-----------|---------------|
| Ordinary | −0.010 | −0.014 | −0.016 | −0.014 |
| Exact (M,S) | −0.032 | **+0.127** | **+0.070** | **+0.150** |
| Clean s64 | −0.022 | **+0.099** | **+0.076** | **+0.127** |
| Dense-corr | −0.023 | −0.017 | −0.022 | −0.018 |

**Key finding:** Exact (M,S) and clean create asymmetric source dependence (small correct-source benefit, larger wrong/absent-source costs). Dense-corruption preservation eliminates this asymmetry by suppressing acquired source dependence. Ordinary improves all conditions uniformly.

### Ordinary-function drift (KL from parent on clean WWM rendering)

| Endpoint | KL | ΔCE | Δrank |
|----------|------|------|-------|
| Ordinary | 0.00055 | −0.0023 | −0.74 |
| Exact (M,S) | 0.02449 | +0.0290 | +5.88 |
| Clean s64 | 0.00941 | +0.0100 | +2.43 |
| Dense-corr | 0.00137 | −0.00005 | −0.09 |

**Key finding:** Clean reduces ordinary-function drift from exact (M,S) levels while retaining most dense-state acquisition. Dense-corruption nearly eliminates drift but also eliminates acquisition.

## Scientific Interpretation

1. **Ordinary continuation is insufficient:** 80 extra private-adapter updates on the same rows under ordinary WWM do not reproduce the dense-mask evidence geometry. The additional-exposure hypothesis is rejected as a complete explanation.

2. **Effective-input clue suppression works:** Dense masking of second-view context makes source evidence more consequential for sparse prediction targets. This produces a distinctive source/evidence geometry not seen under ordinary training.

3. **Preservation must be selective:** Parent anchoring is useful at reliable ordinary states where acquisition causes broad drift. Applying parent anchoring to dense states where the parent is uncertain and acquisition has improved ground-truth prediction suppresses beneficial learning. Clean works better than dense-corruption because it bounds ordinary-function movement while leaving much of the dense effective-input acquisition intact.

4. **The result is a partial trade-off repair, not universal enhancement:** Clean retains acquired source-responsive behavior while recovering some grammar/Supplement competence. Wrong-source, view-only, and source-erased costs remain larger than correct-source benefits.

## Pending Items

- Paired-seed ordinary-continuation O62065 and acquisition-only (M,S)62065 results were still needed for training-seed attribution.
- Whether that attribution used Cheap7 or the full repaired coordinate remained unresolved.
- Historical platform values, repaired fixed-coordinate results and external SOTA comparisons must remain separate.

## Files

- Package verification: `representative_model_packages/representative_model_packages.json`
- Export smoke test: `export_path_smoke/export_path_smoke.json`
- Mechanism figure: `figures/mechanism_trade_figure.png`
- Common-support probe: `common_support_endpoint_probe_with_densecorr/`
- Evidence availability: `evidence_availability_readout_densecorr/`
- Drift profile: `preservation_drift_profile_densecorr/`
- Source readout: `temperature_source_readout_densecorr_full/`
- Complete coordinate: `same_coordinate_with_ordinary_complete/`
