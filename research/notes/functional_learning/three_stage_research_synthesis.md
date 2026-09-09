# Three-Stage Research Synthesis: Frontier Advancement → Principle Discovery → Principle-Guided Frontier Advancement

This synthesis follows the mechanisms, evidence, and failures across the three scientific stages. Conclusions remain conditional where ftseed44 and external validation were still pending.

---

## Stage I — Frontier Advancement (initial_model_studies, compact_experience, representation_and_objectives, frontier_consolidation)

### Starting point and problem structure

BabyLM Strict-Small constrains a masked language model to ≤100M words from designated corpora, ≤36.5M parameters, and no external pretrained knowledge. The challenge evaluates nine official columns: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, SuperGLUE, Reading, and AoA; Overall is their unweighted mean. The visible Strict-Small leader at research start was approximately 41.8. The learning problem is severe: the model must acquire grammatical competence, commonsense reasoning, entity tracking, reading behavior, and developmental plausibility from experience equivalent to roughly one year of child-directed speech, with no access to large-scale pretrained representations.

### initial_model_studies — Initial infrastructure and baseline

Established the BabyLM training pipeline with DeBERTa-v2 masked LM, whole-word masking, the official data split, and the complete nine-column evaluation. Produced `chck_80M` at approximately 40.70 Overall. Key finding: checkpoint selection matters — `chck_80M` outperformed `chck_100M`, suggesting training dynamics are not monotone across the exposure trajectory.

### compact_experience — Low-dose aligned data and first frontier approach

Introduced Qwen-generated aligned text: for each official source segment, Qwen3.5 produced a compact restatement preserving the core content in different surface form. This created source/rewrite paired data within the legal budget. The key mechanism hypothesis was that paired exposure to the same content in different surface forms could force the learner to represent meaning invariants rather than surface co-occurrence patterns.

A 16k vocabulary, 25% Qwen-aligned mixture achieved Overall 41.48 (+0.78 over INITIAL_MODEL_STUDIES baseline). The remaining deficit versus the 41.8 leader concentrated in Entity (-5.75), EWoK (-3.81), COMPS (-1.59), and GlobalPIQA (-1.55), while exceeding the leader on Supplement (+6.27), SuperGLUE (+0.08), Reading (+2.99), and BLiMP (+0.51). The bottleneck was world/commonsense/entity knowledge, not syntax or supervised transfer.

### representation_and_objectives — Compact-view reinvestment achieves SOTA

The compact-view reinvestment strategy matured: Qwen-compacted versions of official text freed budget words that were reinvested in additional official-source training. This produced `compact_view_reinvest` at Overall 42.087, exceeding the 41.8 leader by +0.287. The 2×2 factorial (clean vs reinvest × seed43022 vs seed43122) showed the treatment was positive in both seeds: +0.686 and +0.480 respectively. SuperGLUE recovery (+1.07 over clean-Qwen) was attributed to the reinvested source diversity, not the compact view itself.

However, the EWoK treatment×seed interaction was large (-2.46), concentrated in material-dynamics, physical-dynamics, and spatial-relations domains. This showed that relation-sensitive knowledge formation was unstable across seeds — the compact-view mechanism did not reliably install commonsense relations.

### frontier_consolidation — Coherent86 and the private-adapter architecture

frontier_consolidation developed the coherent86 endpoint: a DeBERTa-v2 model with both slow adapters (scale 1.75) and private adapters (scale 0.75), trained through a coherent full-budget private phase. This architecture became the practical parent for all Stage III work. The `Qiushi-BabyLM-36M-Strict-Small-v4` model was prepared for submission with historical Overall reported as 42.1210247099666.

Key contribution: the fixed-budget substitution framework formalized what it means to add or replace experience:
$$\Delta_T(A,D;B,\mathcal{C},\tau,s) = Y_T(B-D+A;\mathcal{C},\tau,s) - Y_T(B;\mathcal{C},\tau,s)$$
conditioned on learner, curriculum, horizon, and seed. This made clear that useful support depends jointly on the displaced experience, learning stage, and the learner's ability to convert support into target competence — not on any single quality metric of the added data.

frontier_consolidation also established that DeBERTa late-value comparisons suggested view/breadth effects, but terminal loss and reversal evidence contradicted simple universal selectors. RoBERTa and DeBERTa no-disentangle controls did not reproduce compact/view results, tying them to the full DeBERTa positional-score coordinate. This required learner- and architecture-conditional claims.

### Stage I summary

Stage I achieved a new Overall SOTA through compact-view reinvestment and the private-adapter architecture. The key limitations that motivated Stage II: the treatment was seed-unstable on commonsense relations; the mechanism was described as "freed budget words" rather than identified at the learning-process level; and the evaluation coordinate mixed stripped-adapter SuperGLUE with adapter-aware zero-shot, creating a hidden confound discovered only in Stage III.

---

## Stage II — Principle Discovery (representation_and_objectives, frontier_consolidation, functional_learning, relation_learning)

### Controlled mechanism work (representation_and_objectives → functional_learning)

#### Gauge transport: how finite experience becomes reusable

The controlled PosAlign/dual-position experiments in representation_and_objectives established a precise mechanism for how finite experience can create transferable computation:

1. **State anchors** fix an absolute orientation on a subset of relations
2. **Relational comparison evidence** connects anchored and non-anchored relations through a graph
3. **Shared computational representation** allows anchor gradients to shape the coordinate that comparison evidence uses for non-anchored relations

Removing any one factor eliminated the transfer. This was not a metaphor: bridge-sign reversal (bs=-1) reversed the h1/h3 state coordinate on all 192 graph-transfer queries with correlation ≈0.97–0.99 in tied/shared-trunk models, but showed zero reversal in bridge-only or comparison-only conditions.

#### Effective input, not surface diversity

functional_learning tested the apparent "diversity" mechanism with a repaired cue experiment. The critical finding: surface variation of entity names was irrelevant because tokenizer canonicalization removed the variation before the learner saw it. Diversity matters only when it forces a new invariance or correspondence at the learner's effective input. This became the conceptual foundation for clue suppression.

#### Binding, credit allocation, and interface reach

Permutation-orbit binding tasks showed that query-before-context produced near-perfect counterfactual binding (NLL 0.008, top-4 0.998 by epoch 500), while blocking context access to the query reduced top-4 to 0.24–0.27. Activation patching identified a causally active L1 attribute-slot direction with 0.998–1.000 redirection.

The most important negative: held-symbol transfer remained fragile across multiple schedules and controls, establishing that the training trajectory determines which symbols can participate in acquired computation. Full-context prediction pressure and competing credit assignment could prevent self-formation of a selection interface even when the architecture supported it. The interpretation narrowed to **interface reach**: what matters is not total exposure or even target presence, but whether the learning dynamics allow the relevant computational interface to form under competing gradient signals.

#### Natural-language relation evidence

The relation-first specialist demonstrated contextual source retrieval and query-indexed selection across three seeds and multiple entity configurations, but lost substantial inherited competence (GlobalPIQA from 38.5 to ~34). This established that concentrated relation-aligned experience can create the target computation, but the practical challenge is cumulative learning that preserves broad language function — not isolated operation acquisition.

### Architecture-conditional limitations (frontier_consolidation → relation_learning)

relation_learning's controlled substitution experiments showed that useful support is conditional on the learner architecture and evaluation coordinate:

- DeBERTa late-value comparisons showed view/breadth effects not reproduced by RoBERTa
- No-disentangle controls did not reproduce compact/view results
- Descriptor-binding families transferred coarse base-rate priors rather than the intended relation operations
- The literal 480-map family transferred "operations don't update the queried source" rather than latest-assignment relations

These results strengthened the principle: the private branch can learn a relation when diverse clean instances make it the simplest useful predictor, but under transfer it carries the coarse statistical structure of the practiced family, not the intended semantic operation.

### Key principles established in Stage II

1. **Effective-input principle**: Learning responds to the structure available at the learner's actual input after preprocessing, not to the experimenter's description of the data. Surface diversity, length, quality scores, and register are irrelevant unless they change the effective input.

2. **Credit allocation under competition**: Whether finite experience installs reusable computation depends on how learning credit is distributed among competing predictive strategies. Broad context prediction can suppress the formation of targeted computational interfaces even when the data contains the relevant information.

3. **Architecture-conditional transfer**: The value of experience is not a property of the data alone. It depends jointly on the learner's architecture, the displaced experience, the learning stage, and the evaluation coordinate.

4. **Trajectory dependence**: The order and mixture of experience determines which computational interfaces form and which are suppressed. This is not simply curriculum optimization — it reflects the interaction between gradient-based learning dynamics and the availability of simpler competing predictors.

---

## Stage III — Principle-Guided Frontier Advancement (functional_learning, relation_learning)

### From principles to intervention design

The key insight bridging Stage II to Stage III was recognizing that the Qwen paired data already present in the legal budget contained both the target (content seen through a different surface form — the "source") and the cue (surface repetition clues that allowed the learner to predict without source understanding). If the cues were suppressed at the effective input while the source content remained available, the learner would be forced to rely on contextual evidence rather than surface shortcuts.

### Evaluation infrastructure repair

Before any training intervention could be scientifically interpreted, functional_learning discovered and repaired a critical evaluation confound: `AutoModel.from_pretrained` loaded stock `DebertaV2Model` with 34.2M parameters and zero adapters, instead of the intended `FrozenSlowPrivateDebertaV2Model` with 36.5M parameters and 995,584 private-adapter parameters. All prior SuperGLUE measurements used the stripped encoder. The repair introduced registered `AutoModel` with exact hidden-state equality to the trusted MLM encoder. This separated the historical platform coordinate from the repaired scientific coordinate:

- Historical v4 Overall: 42.1210247099666 (stripped-adapter SuperGLUE)
- Repaired coherent86 Overall: 42.023967991315104 (adapter-aware SuperGLUE)

AoA measurement was also repaired: earlier placeholder zeros were replaced with complete lawful ancestry through 17 shared checkpoints, measured by the official scorer at commit `6f825c29`.

### Dense clue suppression: the acquisition mechanism

**Dense unchanged-Qwen focus** applied aggregate focus weight λ=0.15 to all eligible second-view groups in the unchanged legal Qwen tail: 132,283 groups, 176,607 tokens, 20,475 rows, 3,162,742 words. The mechanism: by masking all eligible Qwen-pair positions (not just 15% random), the second-view content clues were removed from the learner's input, forcing prediction to rely on the available source evidence.

Two seeds reproduced a consistent functional displacement: Entity gains +1.04/+1.09, GlobalPIQA +1.49, but BLiMP/Supplement/EWoK costs and CDI (lexical prediction) degradation. The effect was not uniform improvement — it was a redistribution of competence toward contextual evidence dependence.

### (M,S) factorial: localizing the mechanism

The dense-mask/sparse-label factorial separated clue suppression from broad target coverage:

- **(S,S)**: sparse masks, sparse labels — the baseline
- **(M,S)**: dense masks (removing clues) but the exact same sparse labels — changes only effective input
- **(M,M)**: dense masks and dense labels — adds broad supervision

`(M,S)` nearly reproduced `(M,M)` in Entity, GlobalPIQA, source response, and aggregate movement. This established that **suppressing second-view clues at the effective input was sufficient for the observed displacement** — broad dense supervision was unnecessary. The retained sparse labels received approximately 6.19× the per-label focus weight, but the mask-only tokens still changed the effective input context even though they were not labeled.

### Temperature and source analysis: redistribution, not enhancement

Dense retained source effects after global temperature fitting (T=1.1 for all endpoints), excluding a pure confidence-scale explanation. The asymmetry is stark (relative to coherent86):

- Correct-source: ΔNLL **−0.019** (clean64), **−0.024** (exact M,S) — small improvement
- Wrong-source: ΔNLL **+0.082** (clean64), **+0.123** (exact M,S) — large worsening (4–6× the gain)
- View-only: ΔNLL **+0.051** (clean64), **+0.072** (exact M,S) — substantial worsening
- CDI (96-word developmental subset): ΔNLL **+0.062** (clean64), **+0.133** (exact M,S)

The correct interpretation is **predominantly degraded context-independent prediction, not enhanced contextual understanding**. The context-absent costs are 3–7× the context-present benefit. Entity/GlobalPIQA improve because their test format always provides structured context where increased sensitivity helps; the improvement is format-conditional, not evidence of general contextual learning capability.

### Clean preservation: bounding collateral drift

Because clue-suppressed acquisition produced both useful contextual behavior and collateral competence damage, **deterministic ordinary-full-row parent anchoring** was introduced. The preservation method:

- Uses the same Qwen-pair rows as acquisition, but under ordinary 15% WWM with source evidence visible
- Adds `KL(teacher || student)` gradients at preservation-masked positions before the same optimizer step
- Student runs in eval mode (deterministic) during preservation
- Acquisition RNG is restored after preservation forwards
- 517,332 words of counted student preservation presentations

Eval-mode deterministic anchoring reduced ordinary-text parent drift, wrong-source/view-only/source-erased damage, and CDI cost while retaining correct-source support and Entity/GlobalPIQA movement. A frozen-bank rollback control at matched ordinary-text KL showed that much of the preservation benefit resembles bounded movement or attenuation rather than a uniquely demonstrated directional mechanism.

### Complete six-endpoint result on the repaired coordinate

| Endpoint | Overall | Δ vs coherent86 |
|----------|---------|-----------------|
| coherent86 (parent) | 42.024 | — |
| dense (M,M) seed62064 | 42.149 | +0.125 |
| dense (M,M) seed62065 | 42.168 | +0.145 |
| exact (M,S) seed62064 | 42.203 | +0.179 |
| **clean seed62064** | **42.246** | **+0.222** |
| clean seed62065 | 42.232 | +0.208 |

Clean seed62064 is the highest among six complete endpoints on the repaired adapter-aware fixed coordinate. Clean seed62065 reproduces the full policy. The direct clean64-over-exact-`(M,S)` increment is only +0.044 Overall, with BLiMP +0.14, Supplement +0.20, EWoK −0.13, Entity +0.01, COMPS +0.01, SuperGLUE +0.160, GlobalPIQA 0, Reading +0.005, AoA 0. Most aggregate movement comes from acquisition (clue suppression), with a smaller preservation-associated retention/transfer repair.

### Behavioral stability

Clean seeds agree at item-level correctness >0.996 on BLiMP/Supplement/EWoK/Entity/COMPS, with identical GlobalPIQA predictions and Reading surprisal Pearson >0.99999. This establishes repeatable behavior on the reused item pool. However, paired item resampling intervals cross zero for both clean64 vs `(M,S)` and clean64 vs coherent86, so broader population-level superiority is not established by the current evidence.

---

## The Complete Intellectual Arc

### What the program discovered

1. **Effective-input clue suppression as a data-efficient learning principle**: The learner's behavior changes when the information it needs to predict must come from deeper contextual evidence rather than surface co-occurrence cues. This is not a claim about data quality, diversity, or quantity — it is about the computational requirements imposed by the effective input structure.

2. **Acquisition–retention as the central tradeoff**: Any intervention that increases contextual evidence dependence simultaneously risks degrading context-independent competence. The preservation term bounds this collateral drift without requiring that the full parent function be recovered.

3. **The principle is bounded, not universal**: It operates under the tested Qwen/DeBERTa policy, with specific architecture, data composition, and training dynamics. The reach of the principle follows its demonstrated mechanism and boundary conditions rather than its name.

### What the program corrected

- Surface diversity → effective-input structure
- Data quality scores → architecture-conditional value
- Broad target coverage → clue suppression at fixed labels
- Generic parent anchoring → corruption-geometry-specific retention
- Historical platform Overall → repaired adapter-aware coordinate
- Placeholder AoA → complete lawful ancestry measurement

### What remains open

1. **Matched downstream-seed SuperGLUE comparison** (ftseed44 in progress): determines whether the preservation-associated SuperGLUE transfer increment is stable across fine-tuning randomness
2. **Corruption-geometry ablation** (proposed): whether ordinary-mask vs. dense-mask preservation geometry causally determines Entity retention
3. **Population-level generalization**: the current evidence is on reused fixed BabyLM validation items; broader generality is undemonstrated
4. **External SOTA and submission validation**: "highest among six repaired endpoints" is accurate; conversion to a submitted official-platform result requires additional validation
5. **Whether parent posterior matching is uniquely responsible**: rollback/attenuation explains much of the preservation benefit; a no-teacher extra-presentation control or same-rendering ordinary-CE comparison would separate teacher-specific effects

---

## Key Files and Artifacts

### Stage I
- compact_experience full Overall: `experiments/archive/compact_experience/data/full_overall_eval`
- representation_and_objectives 2×2 decision: `experiments/archive/representation_and_objectives/data/consistent_official_2x2`
- Gauge transport: `experiments/archive/representation_and_objectives/data/primary_gauge`

### Stage II
- Orbit binding: `experiments/archive/functional_learning/data/query_first_binding`
- Causal intervention: `experiments/archive/functional_learning/data/causal_intervention`
- Relation-first specialist: `experiments/archive/functional_learning/data/saved_state_replicate_neutral_threeentity`
- Substitution framework: `research/notes/relation_learning/stage3_execution_synthesis.md`

### Stage III
- Dense-mask/sparse-label verification: `experiments/archive/functional_learning/data/densemask_sparselabel_fast_verify`
- Clean preservation trainer: `experiments/archive/functional_learning/scripts/clean_preservation_train.py`
- All-six repaired coordinate: `experiments/archive/functional_learning/data/same_coordinate_all_candidates_final_all6`
- Source-level verifier: `experiments/archive/functional_learning/data/v5_export_evidence_verifier`
- v5 candidate export: `experiments/archive/functional_learning/data/v5_candidate_export`
- Provenance package: `experiments/archive/functional_learning/data/v5_candidate_provenance`

### BabyLM compliance
- Track: Strict-Small, budget 100M words
- Parent exposure: 86,005,295 words
- Acquisition prefix: 3,162,742 words
- Preservation counted: 517,332 words
- Conservative endpoint exposure: 89,685,369 words (within budget)
- Architecture: DeBERTa-v2 with slow (1.75) and private (0.75) adapters, 36,458,592 parameters
