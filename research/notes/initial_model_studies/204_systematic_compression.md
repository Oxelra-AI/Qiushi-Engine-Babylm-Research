# Synthesis of related mechanism experiments

## Why this note exists

This note synthesizes the mechanism experiments following the protected DeBERTa-v2 8×480 WWM model. It records the available scores, supported findings, closed routes, recurring score patterns, checkpoint/evaluation limitations and unresolved scientific questions.

The comparison concerns BabyLM Strict-Small Overall under the recorded data constraints, end-to-end training implementations and official-compatible evaluation procedures.

## Current score state

### Only complete internal 9/9 coordinate

Evidence: `data/debertav2_b256_true_9of9_coordinate.json`

| column | score |
|---|---:|
| BLiMP | 66.76 |
| BLiMP Supplement | 59.88 |
| EWoK | 52.19 |
| Entity Tracking | 22.62 |
| COMPS | 52.19 |
| (Super)GLUE | 68.0218 |
| GlobalPIQA | 35.635 |
| Reading | 7.62 |
| AoA | -0.1745 |
| Overall | 40.5269 |

This protected model remains the only true complete internal coordinate. It is below the public Strict-Small leader `go76dof/wwm_curriculum_simplification_40k` at Overall about 41.80. The most important gaps are Entity, EWoK, GlobalPIQA, SuperGLUE, and COMPS; protected already has strong Supplement and Reading.

### S1/S2 100M closure status

Evidence:

- S1 available columns: `data/s1_100m_available_coordinate.json`
- S2 available columns: `data/true_s2_100m_available_coordinate.json`
- EWoK: `data/s1_s2_100m_ewok_scores.json`
- SuperGLUE: `data/s1_s2_100m_superglue_results.json`
- Current scoreboard: `data/current_scoreboard_status.json`

| model | BLiMP | Supp | EWoK | Entity | COMPS | SuperGLUE | GlobalPIQA | Reading | AoA | NLP mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| protected 8×480 | 66.76 | 59.88 | 52.19 | 22.62 | 52.19 | 68.02 | 35.635 | 7.62 | -0.1745 | 51.04 |
| S1 12×384 | 66.84 | 60.31 | 52.02 | 20.24 | 52.26 | 65.12 | 37.605 | 7.25 | — | 50.63 |
| S2 curriculum | 64.24 | 59.09 | 51.64 | 18.47 | 50.61 | 65.60 | 38.635 | 7.52 | — | 49.76 |

S1 and S2 are not hidden SOTA candidates. Their official-style NLP means are below protected after SuperGLUE is included. Both are behind protected on Entity and EWoK. S2 raises GlobalPIQA but loses grammar, Entity, EWoK, COMPS, and SuperGLUE relative to protected. S1 raises GlobalPIQA and slightly COMPS but loses Entity, EWoK, SuperGLUE, and Reading.

S1/S2 cannot become true official-style 9/9 coordinates from current files because their checkpoint roots contain `chck_10M` through `chck_100M` but not `chck_1M` through `chck_9M`. The trusted AoA procedure needs those early checkpoints. Future serious full-budget candidates must save the full strict-small checkpoint sequence from the start.

### Route B 10M surface adapter status

Evidence: `notes/route_b_surface_adapter_result.md`, `data/route_b_threearm_scores.json`

Surface adapter vs token-ID control:

- Entity +0.02
- COMPS +0.36
- EWoK -2.67
- GlobalPIQA +2.47
- Reading 0.00
- BLiMP -0.04
- Supplement +0.04

The surface/morphology-sharing adapter does not support the intended mechanism in this implementation. It should not be scaled to 100M.

## Reliable knowns from related experiments

### 1. The protected DeBERTa-v2 WWM backbone is strong but not enough

The protected 8×480 WWM coordinate remains the stable internal reference. It is broad and balanced, especially on Supplement and Reading. But it lacks the public leader’s Entity/EWoK/GlobalPIQA package. More ordinary exposure, shape changes, source-order changes, and small representation additions have not closed that gap.

### 2. Public leader model-side pieces do not explain the leader

S1 tested the public leader’s 12×384/intermediate1280 shape on legal official data. It improved GlobalPIQA relative to protected but lowered Entity and SuperGLUE and did not improve EWoK. S2 tested the leader-style length/masking curriculum; it improved GlobalPIQA further but damaged most other columns. Legal 40k tokenization and early LAMB evidence also failed to reproduce the target-column package. The public leader’s advantage is not explained by architecture shape, curriculum, 40k vocabulary, or LAMB alone under our official-data reconstructions.

### 3. Plain WWM sees structure but does not make correct relation assignment necessary

Several data-side experiments showed that recoverable structure can be present without transferring into Entity/EWoK ability:

- WikiAuto aligned simplification pairs did not beat shuffled controls on Entity/EWoK.
- High entity/state-density official windows did not improve Entity/EWoK under plain WWM.
- Entity-swap and bag-preserving probes showed surface repetition sensitivity but weak entity-to-state assignment.
- XSpan and counterfactual propagation routes found local familiarity/anomaly effects rather than robust cross-sentence binding.

The current evidence says the problem is not only lack of propositions per word. The loss and representation must make correct role/relation credit useful for predicting held-out content.

### 4. WESS found a real algorithm but not a deployable BabyLM route

Evidence: `notes/wess_route_closed.md`, `notes/wess_transfer_route_update.md`

Writable Entity-State Slots solved controlled synthetic and text-interface binding tasks when entity/state addresses were supplied. Causal interventions behaved correctly in that supplied-address setting.

The route fails for BabyLM because:

- Exported ordinary DeBERTa backbones did not inherit binding.
- Official-facing GlobalPIQA movement was reproduced by a no-address control and did not improve Entity/EWoK.
- Predicted-route WESS on held-out templates had query recall 0%, parse success 0%, pair accuracy 0%, and no slot-swap/write-removal effects.

WESS should not be repaired further unless a fundamentally new unlabeled-addressing mechanism exists. Its durable lesson is mechanistic: persistent entity-indexed state can solve role binding, but legal BabyLM text does not provide the addresses, and the current model does not learn them from the auxiliary route.

### 5. Local GlobalPIQA gains repeatedly decouple from the missing mechanism

GlobalPIQA improvements appeared under several routes that did not improve Entity/EWoK:

- S1 shape: GlobalPIQA +1.97 vs protected, Entity -2.38, EWoK -0.17, SuperGLUE -2.90.
- S2 curriculum: GlobalPIQA +3.00 vs protected, Entity -4.15, EWoK -0.55, BLiMP -2.52, COMPS -1.58.
- WESS/no-address official screens: GlobalPIQA movement did not correspond to address-specific binding or Entity/EWoK gains.
- Route B surface adapter: GlobalPIQA +2.47 vs token-ID while EWoK -2.67 and Entity +0.02.

GlobalPIQA-only movement is therefore not evidence of progress toward the hard part of the problem.

### 6. Small representation patches have not changed the learning dynamics enough

The surface adapter was constructed carefully: deterministic features, matched token-ID control, verified fused export, equal training exposure and official evaluation. The adapters’ learned scalar contributions stayed tiny and losses were indistinguishable. The result does not rule out morphology-aware representation in general, but it rules out this residual feature-adapter form as a scale-worthy route.

### 7. Complete evaluation requires the checkpoint sequence from the start of training

S1/S2 cannot produce true AoA after the fact because early checkpoints were not saved. Future serious candidates must save at least `chck_1M` through `chck_10M` and then every 10M through 100M, or an equivalent official sequence. Without that, a model may have many useful columns but still cannot become a trusted official-style coordinate.

## Closed routes and what each taught

| route family | status | main evidence | durable lesson |
|---|---|---|---|
| Entity Mention Consistency | stopped | replication collapsed; shuffled control not neutral | surface smoothing is not state tracking |
| Counterfactual propagation / XSpan | stopped | local likelihood/familiarity effects, no transferable binding | cross-sentence natural structure exists but WWM does not force binding |
| WikiAuto/ASSET aligned simplification | stopped | aligned lost to shuffled on Entity/EWoK/GlobalPIQA | lexical simplification redundancy is not dense entity-state training signal |
| Structure-density selection | stopped | high structure windows did not improve Entity/EWoK | more recoverable structure per word is insufficient under plain WWM |
| MNTP hybrid | stopped for tested form | 1:15 MNTP damaged BLiMP/Entity/COMPS/GlobalPIQA/Reading | fixed low-ratio causal auxiliary did not solve credit assignment |
| Naive contrastive binding | stopped | training success but held-out compositional failure | template contrast does not create transferable role variables |
| WESS | stopped | gold addresses work; predicted addresses and backbone transfer fail | explicit slots need legal unlabeled addressing and exportable transfer |
| S1 shape | not sufficient | NLP mean below protected; Entity lower | shape reallocates capacity toward GlobalPIQA, not Entity/EWoK package |
| S2 curriculum | not sufficient | GlobalPIQA up, broad column losses | hand length/masking curriculum produces score exchange, not general improvement |
| Route B surface adapter | not scale-worthy | surface loses to token-ID on EWoK | simple residual surface sharing is not enough |

## The recurring scientific pattern

The main repeated pattern is not random noise. Many interventions can move one or two columns, especially GlobalPIQA, without producing the combination needed for SOTA. Entity/EWoK remain difficult because they require reusable relational variables and world-knowledge contrasts that survive surface changes, long spans, and held-out compositions. Ordinary masked prediction often rewards local plausibility, lexical familiarity, and surface repetition rather than correct entity-role-state assignment.

The public leader’s strength is exactly in the missing package: Entity 28.45, EWoK 56.07, GlobalPIQA 39.665. The protected model’s advantage in Supplement/Reading should not be casually spent. A successful route must move the target package while preserving the protected strengths.

## The unresolved load-bearing unknown

The unresolved question is:

**How can a model trained on legal unlabeled BabyLM text be made to form reusable entity/relation/state and commonsense variables such that correct binding and contrastive world knowledge are useful for prediction, without gold addresses, leakage from evaluation data, or a mechanism that disappears when exported to an ordinary checkpoint?**

This question explains why many routes failed:

- Data density alone gives propositions but not the learning pressure to bind them.
- Gold-address mechanisms solve synthetic tasks but cannot read unlabeled official text.
- Frequency/tokenizer/adapter changes alter surface statistics but do not necessarily create role variables.
- GlobalPIQA gains can come from general regularization or answer-bias changes without Entity/EWoK movement.

The next main route should attack this unknown directly rather than selecting another local tweak.

## What the next route must preserve

Future serious candidates should:

1. Save the full official checkpoint sequence needed for AoA.
2. Enter a maintained score table early, with at least the target columns and a path to full 9/9.
3. State a mechanism hypothesis in terms of variables, learning signal, and predicted official-column movement.
4. Include a decisive short-budget experiment that distinguishes the intended mechanism from generic capacity, ordering, surface, or synthetic-regularization effects.
5. Treat Entity/EWoK plus GlobalPIQA as the core target package, not GlobalPIQA alone.
6. Preserve protected strengths: Supplement, Reading, and broad grammar cannot be spent without a larger compensating gain.

## Immediate next scientific work

The evidence motivates comparison of new mechanism-level routes, not necessarily restricted to the current DeBERTa/WWM recipe. A hypothesis may require changing representation, data organization, learning objective, architecture or training experience, but it must retain the data constraints, reproducibility requirements and official BabyLM measurements.

The proposed mechanisms should directly address unlabeled relation/state credit assignment or world-contrast learning, rather than another isolated architecture or tokenizer change. Candidate idea spaces include:

- self-supervised event/relation abstraction from naturally repeated entities without gold addresses;
- contrastive or denoising objectives where the target is impossible to solve from surface bags alone;
- training-experience construction that creates legal unlabeled contexts with verifiable relation dependencies and controlled leakage risk;
- architecture/objective changes that make persistent variables emerge but remain exportable to a standard checkpoint or official-compatible model.

The key standard is not whether the idea is easy to train. It is whether it can plausibly explain and overcome the repeated failure pattern documented above.
