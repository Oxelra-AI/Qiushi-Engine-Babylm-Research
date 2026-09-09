# preservation experiment design and evidence: Targeted Preservation Experiment Design

## Evidence basis (from temperature confidence scale and densemask interpretation-87)

### Causal localization
Dense second-view input masking with sparse labels `(M,S)` reproduces the full
dense `(M,M)` source-responsive shift. Against sparse `(S,S)`:
- Entity: +0.66, GlobalPIQA: +1.485, EWoK: +0.91
- Qwen source-help: +0.109
- Common-target movement: +0.219 (source-altered +0.404, held-source +0.385)
- Both-source decisions: 21/25 (same as all arms)

Against dense `(M,M)`, `(M,S)` differs by <0.04 on all readouts.
Dense target coverage is therefore unnecessary for the observed shift.

### The cost profile also transfers
- BLiMP: -0.48, Supplement: -0.40, EWoK: +0.91 (net positive but grammar costs)
- CDI endpoint: dense-mask ΔNLL +0.158 T=1, +0.133 fitted T, Δrank +111 vs coherent86
- Condition-wise: correct-source NLL improves slightly (-0.026), wrong-source worsens (+0.143), view-only worsens (+0.084)
- Calibration: all endpoints choose T=1.1 on legal-tail text; rank/order changes survive temperature adjustment

### Two-seed measured Overall (complete)
- coherent86: 42.024
- dense seed62064: 42.149 (Δ +0.125)
- dense seed62065: 42.168 (Δ +0.144)
- Two-seed mean delta: +0.135

## Preservation experiment

### Design
For each Qwen pair row in the training prefix:
- **Acquisition**: (M,S) dense second-view masks, sparse focus labels → focused CE (unchanged)
- **Preservation**: standard 15% WWM on same packed text → KL(student || frozen coherent86)

Non-Qwen rows: standard 15% WWM → CE (unchanged)

### Loss
```
L = λ_focus * mean(CE_focus) + (1−λ_focus) * mean(CE_ordinary) + λ_pres * mean(KL_pres)
```

### Arms in progress at the time
- λ_pres = 1.0, seed 62064
- λ_pres = 3.0, seed 62064

Both: 80 updates, checkpoints at 20/40/60/80, conservative exposure counting.

### Exposure accounting
Each Qwen row is presented twice. Total additional words ≈ 500K (3831 Qwen rows × avg tokens).
Conservative endpoint exposure ≈ 89.17M + 0.50M ≈ 89.67M (well under 100M).

### Success criteria
1. **Selective preservation (positive)**: Entity/source-help movement retained; view-only/CDI/BLiMP costs reduced. This shows the acquisition mechanism survives while drift is prevented.
2. **Uniform damping (neutral)**: Both gains and costs shrink proportionally. This means the preservation is too broad and must be more precisely targeted.
3. **Acquisition loss (negative)**: Source-responsive movement lost. The preservation pressure competes with acquisition on overlapping representations.

### Critical controls
- The (M,S) arm is the acquisition-only control at the same update horizon
- Both preservation arms use identical acquisition rendering (via earlier analysis monkeypatch)
- The preservation rendering uses independent seed family ("preservation-wwm-std")
- Teacher loads through the same bridge.load_model as student (verified: FrozenSlowPrivateDebertaV2ForMaskedLM, 995584 private adapter params, all frozen)

### Evaluation plan for delivered checkpoints
1. Fast profile: Cheap7 (BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, Reading)
2. Qwen source-help (earlier analysis probe)
3. Common-target movement (earlier analysis probe)
4. Temperature/rank source readout (temperature confidence scale and densemask interpretation/087 five-model readout)
5. Source margin decomposition
6. CDI endpoint temperature/rank profile
7. Compare against (S,S), (M,S), and (M,M) on all readouts
