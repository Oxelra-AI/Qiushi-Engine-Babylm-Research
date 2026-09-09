# extractive result review and route — review of the source-only extractive result and route decision

## What I independently checked

Read the delivered extractive selected readout result evidence, not summaries:

- Integrity carrier `data/extractive_training_integrity_final/extractive_training_integrity.json`: both arms `ready_for_selected_eval=true`, `problems=[]`, 100,000,000 words, 2,529 steps, 34,467,424 params, DeBERTa 8x480/8h, WWM p=0.15, AdamW lr 0.001, batch256/seq256, legal tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, stream SHAs balanced `8a4e77af...5776c` / wide `b9e46668...58753`, all ten checkpoints present. Mechanically valid compliant tokenizer retrain status stock coordinate.
- Selected panel `.../extractive_selected_eval_stage1/extractive_selected_panel_summary.json`: `row_count=6`, `problems=[]`, six valid rows (balanced/wide/legal_compact × 80M/100M).
- Paired-interval reports (four files) with 170,722 common items each, zero loader warnings.

## The result stands, with its stated boundaries

Extractive minus legal_compact, late mean over 80M/100M:

| arm | cheap7 | cheap6 (no GlobalPIQA) | cheap5 | EWoK+Entity | Supplement | Entity | COMPS | BLiMP | GlobalPIQA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| balanced−compact | +0.246 | **−0.257** | **−0.287** | **−1.580** | +0.620 | −0.760 | −0.690 | +0.215 | +3.263 |
| wide−compact | −0.098 | **−0.662** | **−0.746** | **−2.820** | −2.065 | −2.355 | −0.035 | +1.190 | +3.283 |

Confirmed decisive facts:

1. Balanced's positive cheap7 is carried by GlobalPIQA (203 items, 2 clusters, all intervals cross zero) and partly BLiMP; the stable composite excluding GlobalPIQA is below compact at both checkpoints. Cheap7 does not establish parity.
2. EWoK+Entity item intervals are negative and exclude zero at 100M: balanced item [-1.852, -0.395], cluster [-1.870, -0.305]; wide item [-1.747, -0.221]. This is a real relation/state deficit, not interval-crossing noise.
3. Coverage ordering reverses the access explanation: wide has 0.982 source-content coverage / 1.000 tail coverage / 0.928 span, yet lags compact (0.657 / 0.702 / 0.807) on stable families. **Missing source/tail access alone cannot explain compact's advantage.**

Supported conclusion (construction-qualified): in the legal stock-DeBERTa coordinate, the tested balanced/wide original-word selectors do not reproduce natural compact's stable-family competence, and reduced source/tail coverage is ruled out as the explanation. The result does NOT isolate a single positive ingredient — compact and extractive differ jointly in source-absent content (compact 0.173 vs 0), continuity/fluency, skip fragmentation (compact 0.223 vs balanced 0.472 / wide 0.354), density, and BPE geometry.

## Route judgment

- Stop further 60M/70M/90M spending on these same balanced/wide streams. Both read checkpoints agree on the stable-family direction; intermediate points would describe nonmonotonicity but cannot resolve seed variance or aggregation semantics and give no independent replication. This closure is trajectory- and selector-specific only.
- Do NOT infer one missing ingredient from the deficit. "Generated re-expression alone" is not established; the extractive arms are telegraphic deletion surfaces, so a fluent source-attested construction has not been tested.
- Independent verifier hazards to preserve: wrapper vs raw-item are different estimands (wide's raw stable-five points are ~0 with intervals crossing zero despite negative wrapper composites); balanced Supplement shows a wrapper-positive / raw-item-negative sign reversal, so Supplement must not carry a mechanism claim; two checkpoints share a trajectory, so this is not seed replication.

## Next same-coordinate construction (motivated, not yet launched)

Both independent_review generator and verifier converge: the right next comparison occupies the missing quadrant — **compact-like fluency/continuity and geometry with zero source-unattested content vocabulary**. The verifier sharpens it into the cheaper, more decisive form:

- Start from `extractive_balanced` (already density-nearest to compact), keep only source-attested content in source order, then **restore intervening function words, connectors, and relation-bearing constituents** (or select complete dependency constituents/clauses) so copied-adjacency gap-1 moves toward compact's 0.778 while source-absent content stays exactly zero.
- Prefer a **graded gap-fill / continuity dose-response series** over a single "fluent" arm, holding source-absence fixed, so continuity is varied while content vocabulary is constant.
- Match compact on word count, content fraction (~0.647, not wide's 0.811), BPE tokens, active/maskable-token exposure under fixed WWM, source-span, and sentence lengths.
- No source-absent target mask/loss weighting, following the whole-word-control result. No benchmark-shaped objective.

Minimum evidence before any 100M run: corpus-level proof of (a) separation — zero unsupported content lemmas AND decisively lower skip fragmentation / higher continuity than both extractive arms; (b) compact-like geometry (word count, density, packed BPE/WWM mass, sequence lengths, repetition stats); (c) proposition preservation on a high-repair stratum by blinded human check. Then a reduced-budget same-seed pilot at a predeclared checkpoint, read on cheap6/cheap5/EWoK+Entity/Entity/Supplement/COMPS. A full run is justified only if movement is distributed toward compact and away from balanced across several stable families (especially EWoK+Entity/Entity plus Supplement or COMPS), not a GlobalPIQA/BLiMP/single-family gain.

## Cross-coordinate note
HS/LS/HD/LD RoBERTa factorial is not yet mature (HS/LS through chck_40M, HD/LD not started; only dry-run integrity/interaction summaries exist). It is a cross-coordinate survival test, not a numerical decomposition of the DeBERTa effect. Cross-read only when all four arms complete.

## Boundaries preserved
Protected endpoints (chck82, chck84, coherent86 α=0.75) remain frozen; no reopening of source-absent innovation masking, target-cooccurrence, ordered/scrambled continuation, or seed/scale scanning without new mature selected evidence.
