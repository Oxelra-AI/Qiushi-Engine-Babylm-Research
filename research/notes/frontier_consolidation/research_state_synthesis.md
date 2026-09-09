# cross data binding comparison result: Research State Synthesis After Cross-Data Binding Comparison

## Findings

Ran counterbalanced entity-event-state binding evaluator on 7 exposure-matched stock DeBERTa models across 4 data treatments (compact, clean, extractive_balanced, extractive_wide). Result: EEBF varies by only ±0.008 (0.508-0.516 at 80M, 0.510-0.531 at 100M). Binding weakness is generic to DeBERTa at this scale, not compact-specific. This closes the binding-compact connection hypothesis and establishes orthogonality.

## Updated Scientific Interpretation

### Achieved
1. **Score target exceeded**: coherent86 α=0.75 at Overall 42.12 (local), chck82 at 41.94 (public submitted), chck84 at 42.02 (packaged). All above 41.8-42.0 target.
2. **Compact semantic second views validated**: +2.416 equal7 vs repeat, +1.346 mean7 vs clean at 80M. Strong, reproducible, seed-replicated DeBERTa data effect.
3. **Function-preserving residual capacity validated**: zero-output adapters redirect trajectory without spectral collapse, produce narrow late-peak phase.

### Scientifically Established
1. **DeBERTa-specific data effect**: compact views help DeBERTa bidirectional MLM but NOT GPT2 causal LM (+0.078 cheap7) or stock RoBERTa MLM (+0.042 cheap7).
2. **Not extractive keyword packing**: source-only selection (balanced or wide) fails on stable families (EWoK+Entity −1.6 to −2.8 at 100M).
3. **Not entity binding improvement**: EEBF varies <0.01 across data treatments.
4. **Not single-factor decomposable**: ordered/scrambled, cross-realization, deletion-triangle all show coupled mechanism. Source-absent targets contribute but copied/shared target supervision more consequential.
5. **Entity binding deficit is architectural**: generic to DeBERTa at this scale, unaffected by data treatment or adapter, requires hard role coordinates to solve (ceiling 0.82-0.85).
6. **Raw-token role induction fails at this scale**: 4 TinyMLM memory variants all at chance (related experiments).

### What Remains Scientifically Open
1. **WHY is the compact effect DeBERTa-specific?** Disentangled relative position attention? Content-position factorization? Enhanced mask decoder? Interaction with WWM? This is the deepest remaining mechanism question.
2. **Can the principle generalize?** Currently architecture-specific. Could modifications to other architectures (e.g., adding relative position to RoBERTa) recover the effect?
3. **Better late-trajectory control**: The narrow late peak (82-84M out of 100M) is structural-but-stochastic. No tested intervention reliably broadens or stabilizes it.

### Route Status
- The memory route is closed after 4 variants. Architecture specificity and distillation/interaction remain possible scientific questions.
- The bridge route is closed because construction yield was too low; the extractive route is also closed. Binding is orthogonal, and coherent88 improves only GlobalPIQA. The remaining question is why the compact effect is DeBERTa-specific.

## Scientific Recommendations

The research has produced substantial, well-evidenced scientific findings. The score target is exceeded. The main mechanism routes in both studies have been exhausted. The remaining open question (WHY DeBERTa-specific) is scientifically interesting but may require more architecture-controlled experiments than justified by the marginal score improvement it could yield.

**Three options**:
1. **One more targeted experiment**: Test whether adding relative position encoding to RoBERTa recovers the compact-view effect. This directly tests the "disentangled attention" hypothesis and would be a strong mechanistic result. But it requires implementing RoBERTa with relative position, which is non-trivial.
2. **Express**: Begin synthesizing the accumulated findings into a paper/report documenting the compact-view data efficiency principle, its DeBERTa specificity, the mechanism decomposition, and the entity-binding orthogonality.
3. **Score push**: Try genuinely new interventions (masking rate, different adapter topology, knowledge distillation from approved teachers) aimed at pushing above 42.12.

## Scientific Artifacts
- Script: `scripts/cross_data_binding_comparison.py`
- Results: `data/cross_data_binding_comparison/`
- Weight averaging script (unused): `scripts/cross_data_weight_average.py`  
- Synthesis: `notes/cross_data_binding_comparison_result.md`
- This note: `notes/research_state_synthesis.md`
