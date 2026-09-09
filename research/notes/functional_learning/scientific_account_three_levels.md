# scientific account three levels scientific account: three levels of evidence

This historical note distinguishes what the available evidence establishes at three scientific levels.

## Level 1: Verified higher score on the fixed official evaluation

Clean preservation seed62064 achieves Overall `42.246412332209445` on the fixed official BabyLM Strict-Small evaluation coordinate, exceeding the faithful repaired coherent86 reference `42.023967991315104` by `+0.2224443408943415`. Clean seed62065 replicates at `42.23173113265801`, `+0.20776314134290885` over coherent86.

This is a concrete factual measurement: the candidate model scores higher on the fixed evaluation items and tasks using the repaired adapter-aware coordinate (official-sized zero-shot/Reading, repaired AutoModel SuperGLUE with primary metrics, measured AoA). The historical platform-style `42.1210247099666` used stripped `AutoModel` SuperGLUE and remains a separate coordinate.

**What this establishes:** A verified higher score on the specific fixed evaluation, obtained by a lawful method within the BabyLM Strict-Small 100M-word budget (`89.685369M` conservative endpoint exposure). The improvement is not uniform across all components: Entity `+1.08`, COMPS `+0.11`, SuperGLUE `+0.102`, GlobalPIQA `+1.485`, Reading `+0.035` versus coherent86, while BLiMP `-0.25`, Supplement `-0.36`, EWoK `-0.20` remain costs.

**What this does not establish:** That the candidate is broadly superior to the parent on the underlying capabilities being measured. The benchmark items are fixed; the score is conditional on those specific items.

## Level 2: Replicated acquisition and partial retention effects

The clean method combines two components: (a) dense effective-input clue suppression with sparse-label acquisition `(M,S)`, and (b) deterministic eval-mode parent-function preservation via `KL(teacher || student)` on ordinary full-row WWM.

**Acquisition effect.** The exact acquisition-only `(M,S)` endpoint exceeds coherent86 by `+0.17856996302142392` Overall. This is the dominant effect. Its GlobalPIQA and Entity improvements are identical to clean preservation (same 5 changed items, net +3). Dense `(M,M)` controls also show Entity/GlobalPIQA improvement but at higher grammar/knowledge cost. The supported interpretation is that dense effective-input clue suppression installs contextual evidence dependence: the model learns to use present context more, but at the cost of context-independent competence.

**Preservation-specific effect.** Clean seed62064 exceeds exact `(M,S)` by `+0.04387437787291759` Overall. This is concentrated in BLiMP `+0.14`, Supplement `+0.20`, and SuperGLUE `+0.15987`, with EWoK `-0.13` as a remaining cost and GlobalPIQA/Entity unchanged. Paired conditional-item resampling (earlier analysis) shows this direct increment has median `+0.0447` with 2.5–97.5% interval from `-0.035` to `+0.129` and positive fraction `0.865`. The SuperGLUE portion (`+0.018` Overall units) is from downstream finetuning and subject to finetuning-seed variation not yet measured.

**Replication.** Two training seeds (62064, 62065) with different acquisition RNG produce near-identical behavior: item-level zero-shot agreement above 0.996, identical GlobalPIQA predictions, Reading correlation 0.99999, SuperGLUE within 0.027. This establishes repeatable functional displacement from the fixed policy, not just a one-seed accident.

**Mechanism readouts.** Both clean seeds reduce ordinary-function drift from coherent86 (KL ~0.0094 vs dense-mask ~0.0245), preserve correct-source NLL while reducing wrong-source/view-only/source-erased costs relative to dense-mask, and repair CDI endpoint temperature-rank costs (clean64 `+0.0622` vs dense-mask `+0.1329`). The repaired frozen-bank rollback control shows that bounded movement/attenuation at matched ordinary-text drift explains much of the benefit; no strong nonlinear trajectory claim is supported.

## Level 3: Uncertain broader superiority

The following remain uncertain and should not be claimed:

**Generalization beyond fixed items.** The evaluation uses fixed item pools. The conditional paired-item resampling measures sensitivity within those pools, not generalization to unseen items. Two training seeds on the same items confirm repeatability, not broad superiority. The complete-policy clean64 vs coherent86 paired interval also crosses zero (approximately -0.047 to +0.537) when resampled within fixed items.

**Downstream finetuning stability.** About `0.018` of the `+0.044` direct preservation increment comes from SuperGLUE, which depends on finetuning. No same-downstream-seed comparison exists yet for the preservation-specific claim. Matched ftseed44 runs for coherent86, exact (M,S), and clean64 are now running and will address this.

**Developmental harm absence.** Measured AoA is zero for all endpoints with legitimate_zero=True. Raw AoA refits from the earlier analysis show no near-significant harmful negative relation. But nonsignificant correlation is not evidence of absent developmental harm, and the known CDI costs (clean64 `+0.0622` NLL, `+46.0` rank vs coherent86) remain part of the result. The AoA metric measures one aspect of acquisition order; CDI costs are a separate observable.

**Universal acquisition–retention principle.** The method improves this specific model (frozen-trunk DeBERTa with private adapters, ~36M params) on this specific evaluation. Whether effective-input clue suppression with parent-function preservation generalizes across architectures, scales, data distributions, or evaluation regimes is not tested.

## Defensible scientific account

The strongest defensible account separates these levels:

1. **Score fact:** Clean preservation seed62064 achieves the highest measured Overall on the repaired BabyLM Strict-Small coordinate, replicated by seed62065.

2. **Method contribution:** Dense effective-input clue suppression supplies the dominant acquisition effect (Entity, GlobalPIQA, contextual evidence dependence). Deterministic parent-function preservation adds a modest but repeated retention/transfer repair, mainly in grammar/Supplement and supervised transfer. The two components together produce a higher aggregate than either alone or the parent.

3. **Trade shape:** The improvement is not uniform. Short-input and context-independent costs (BLiMP, Supplement, EWoK, CDI) persist relative to the parent. The method shifts the model toward contextual evidence use at the expense of some inherited context-independent competence, then partially repairs that cost through parent anchoring.

4. **Uncertainty:** The preservation-specific increment is small (Overall `+0.044`), its paired-item interval crosses zero, and its SuperGLUE portion depends on untested downstream-seed variation. The complete-policy improvement is larger and more stable but conditional on fixed evaluation items.

## Files

- All-six table: `experiments/archive/functional_learning/data/same_coordinate_all_candidates_final_all6`
- Paired uncertainty: `experiments/archive/functional_learning/data/direct_increment_paired_uncertainty`
- Provenance package: `experiments/archive/functional_learning/data/v5_candidate_provenance`
- Matched ftseed44: `experiments/archive/functional_learning/data/matched_ftseed44_superglue` (running)
- GlobalPIQA items: `experiments/archive/relation_learning/data/globalpiqa_item_table`
- Raw AoA refits: `experiments/archive/relation_learning/data/raw_aoa_refits_official`
