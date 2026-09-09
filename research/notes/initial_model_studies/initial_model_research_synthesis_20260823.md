# Initial Model Studies: Baselines, Negative Results and the Binding Hypothesis

Historical scientific synthesis dated **2026-08-23**. The results and open questions below describe the initial-model studies at that date, not the latest results in other research collections. No experiment or evaluation was performed to prepare this record.

## Experimental Setting

These comparisons used a 10M-word training pool and a maximum budget of 100M cumulative word presentations. This describes the study's pool and exposure budget; ten passes is not a general official BabyLM rule. Evaluation text and hidden labels were not training inputs. The full assessment comprised BLiMP, Supplement, EWoK, Entity Tracking, COMPS, SuperGLUE, GlobalPIQA, Reading and AoA. Improvements on short screens or a subset of these columns were not equivalent to improvements in the complete nine-column Overall.

By this date, the study had established a strong baseline and a substantial negative-result record, but had not demonstrated an original mechanism that reliably improved complete Overall. The central question had become whether information encoded in a language model is correctly addressed, causally used for prediction, retained across updates and transferred across forms.

## Historical Score Reference

The strongest complete local result in this synthesis was the ordinary WWM baseline, seed 43, loaded from the exact `chck_80M` subdirectory:

| Measure | Recorded Value |
|---|---:|
| BLiMP | 66.54 |
| Supplement | 61.00 |
| EWoK | 50.44 |
| Entity Tracking | 22.20 |
| COMPS | 53.00 |
| SuperGLUE | 68.2551 |
| GlobalPIQA | 37.59 |
| Reading | 7.30 |
| AoA | 0.00 |
| Overall | **40.7028** |

The protected reference was DeBERTa-v2 8x480, a 16k tokenizer, WWM, official training text, seed 42 and 100M exposure, with complete Overall **40.5269**. The approximately **0.176 Overall** difference was a seed/checkpoint difference within the baseline family, not evidence of a new learning mechanism and not an official-server SOTA result.

The comparison below reproduces the historical synthesis's rounded differences against a leaderboard snapshot saved on **2026-08-14**, whose leading Overall was approximately **41.8011**. That snapshot was not rechecked on 2026-08-23. The reported remaining gap was approximately **1.098 Overall**, or **9.88** summed column points.

| Measure | Local Endpoint | Dated Leader Snapshot | Recorded Difference |
|---|---:|---:|---:|
| BLiMP | 66.54 | 67.20 | -0.66 |
| Supplement | 61.00 | 56.04 | +4.96 |
| EWoK | 50.44 | 56.07 | -5.63 |
| Entity | 22.20 | 28.45 | -6.25 |
| COMPS | 53.00 | 53.57 | -0.57 |
| SuperGLUE | 68.255 | 69.79 | -1.535 |
| GlobalPIQA | 37.59 | 39.665 | -2.075 |
| Reading | 7.30 | 5.425 | +1.875 |
| AoA | 0.00 | 0.00 | 0.00 |

Supplement and Reading were relative strengths. The main deficits lay in Entity, EWoK, GlobalPIQA and SuperGLUE. These differences motivate a joint-capability question; they do not establish that a particular training factor caused the gap.

The detailed baseline evidence is retained in [the complete DeBERTa reference](debertav2_b256_true_9of9_coordinate.md) and [the WWM endpoint verification](new_best_verification_and_assessment.md).

## Route Evolution

The following sequence preserves the chronological progression of the experiments, while distinguishing engineering feasibility, local mechanism evidence and downstream results.

| Successive Research Phase | Experiments and Resulting Boundary |
|---|---|
| End-to-end baseline | A 1M GPT-2 experiment established the corpus, tokenizer, training, checkpoint and evaluation chain. Its principal result was feasibility, not competitive capability. |
| Causal architectures and coverage | Sparse routing, prefix memory, data ordering, character and n-gram variants were compared. Several early positives were confounded by coverage, parameter count or evaluation differences; external memory was less consistently important than data coverage. |
| Masking objective | MLM/WWM comparisons supported retaining WWM. The tested length curriculum caused substantial damage in its original setting; this did not establish a universal result about every possible curriculum. |
| Re-expression and cross-view use | 40k tokenizers, simplified text, original/simplified pairing, cross-view masking and synthetic state stories did not produce stable joint gains. Adjacency and pairing did not by themselves force cross-view use of information. |
| Matched backbones | Parameter-matched BERT and DeBERTa-v2 comparisons favored DeBERTa-v2 8x480 and established the complete 40.5269 reference. Relation-biased masking introduced task trade-offs. |
| Relation objectives and recipe components | Entity Mention Consistency, counterfactual/XSpan objectives, 12x384 depth, 40k vocabulary, LAMB and curriculum did not reliably transfer local relation improvements. Transplanting leading-recipe components did not reconstruct their joint capability; greater importance of the data distribution remained a hypothesis. |
| Structured experience and binding objectives | Data reconstruction, proposition density, entity anchoring, MNTP and binding objectives did not yield sustained complete-Overall gains. These particular implementations were closed or downgraded. |
| Explicit state storage | WESS worked with supplied gold addresses and had causal storage/readout evidence. Predicted addresses failed, and gains did not survive export to the ordinary backbone. This was not an official-interface capability improvement. |
| Surface adapters | Morphology/surface adapter gates were nearly inactive, EWoK declined and GlobalPIQA changes were local fluctuations. The tested route was closed. |
| Ordered state experience | R1, SCMLM-R and noncommutative state/order tasks could teach local state, but composition order did not enter the ordinary prediction interface. Transfer was not established. |
| Recursive causal recipe | The public RecGPT learning system trained on legal official text reached complete Overall **34.2723**. Its public-model behavior could not be attributed to recursion alone; direct architecture transplantation was closed. The [RecGPT differential analysis](recgpt_differential_route_update.md) retains the full coordinate and data-distribution caveats. |
| Adaptive masking and long horizons | The official/structured x WWM/AMLM 2x2 showed an early learning-speed signal for AMLM, but no stable endpoint advantage across two 100M seeds. Structured data and AMLM interacted negatively. Raw AMLM did not qualify for promotion. |
| Trajectories and historical context | Checkpoint trajectories identified the WWM 80M endpoint at 40.7028. History and re-mention probes found decodable information, but not stable causal control of ordinary MLM content logits. |
| Relation filtering and restatement controls | Relation filtering, FineWeb paired restatements and hard negatives gave transient, unstable changes. Clean true pairs did not beat source-only or shuffled controls reliably. A checkpoint-loading error invalidated part of the short trajectory; exact-subdirectory reevaluation supported closure rather than persistent gain. |
| Credit-assignment diagnostics | Gradient-conflict analysis, CBCA, optional transport, cross-view credit and within-experience transfer-credit candidates were tested and several were rejected. An event-binding causal-use pattern survived as a diagnostic observation. |
| Identity-addressed retrieval | A failed induction-seeded state-subspace hypothesis produced an informative control: correct identity addressing could preserve retrieval despite displaced content, whereas a wrong identity address reduced performance. Stress tests and a parameter-neutral natural-text implementation supported a new candidate, but no natural-text H100 training or official evaluation was available at this date. |

## Corrections and Negative Evidence

### Incomplete Evaluation Is Not a Negative Model Result

On **2026-08-16**, the initial RecGPT scoring attempt could not establish complete nine-column Overall: the SuperGLUE/boolq branch had failed, while AoA used a low-parallelism serial path without intermediate shard outputs. Interrupting this incomplete evaluation was not a scientific failure of the candidate model.

The corrective requirements were to separate the AoA and SuperGLUE branches, repair and smoke-test the boolq model adapter before full fine-tuning, and make AoA scoring resumable over checkpoint x word shards with durable intermediate outputs. Complete metric collation was conditional on both repaired paths passing their checks. These were measurement-validity and throughput repairs, not changes to the learning method. The [existing sharded-evaluation record](sharded_eval_status.md) documents the implementation and its subsequent progress; the later completed RecGPT result is distinct evidence from the interrupted attempt.

### Checkpoint Identity

For a local model directory, passing `revision="chck_X"` did not select the intended checkpoint subdirectory. Some intermediate relation results therefore loaded the wrong model. The affected 1M/2M trajectory was withdrawn; reevaluation using exact subdirectories showed fluctuation rather than durable improvement. This correction changes which evidence can support the route, not merely how a path is displayed. See [the corrected relation-scaling record](relation_scaling_closure_and_eval_repair.md).

### Gradient Conflict

The hypothesis that simple negative gradient alignment predicts forgetting and explains the main plateau was not supported: the observed correlations were weak, permutation tests were not significant, and adding conflict features worsened held-out prediction. This rejects the tested explanation, not every possible form of interference or credit-assignment mechanism.

### Early Peaks and Proxy Success

The AMLM 40M peak was not evidence of a higher final capability. Nor did the experiments establish that FineWeb simplification alone caused the leader gap, that late representation narrowing was the unique cause, or that probe-decodable information could necessarily be converted into official-task gains. Those interpretations lacked joint support from seeds, checkpoints, causal interventions and full 100M evaluation.

The tested original/simplified adjacency, static relation-density filtering, raw-loss-adaptive masking, WESS predicted addressing, exported WESS backbone, R1/SCMLM-R state objectives, surface adapters, isolated 40k substitution and direct RecGPT transplantation did not justify repeated continuation under new labels. The negative claims are about these implementations and comparison settings, not blanket impossibility claims about their broader research families.

## Representation, Use and Retention

The synthesis separates six requirements for useful knowledge from limited experience:

**Coverage -> encoding -> binding/addressing -> causal prediction use -> retention across updates -> transfer across forms.**

Filtering can improve coverage while reducing diversity. A probe can establish encoding without establishing use. WESS with gold addresses demonstrates controlled routing while leaving natural address learning unresolved. AMLM can change learning speed without preserving endpoint gains. Synthetic-task success can establish a mechanism in a controlled setting without demonstrating natural-text transfer.

These distinctions explain why auxiliary loss reductions and locally learnable state tasks were insufficient. A model could exploit nearby token statistics without reading history or binding the correct entity. Even when an auxiliary module succeeded, the ordinary MLM or string-scoring interface did not necessarily inherit the behavior. A mediator connecting the intervention to that ordinary interface was required before interpreting an Entity/EWoK/GlobalPIQA change as mechanistic transfer.

The nine-column comparison also imposed a multi-objective constraint: relation masks, structured data and tokenizer changes often improved one or two columns while harming BLiMP, Supplement, Reading or other transfer measures. By the historical cutoff, no original intervention had produced a persistent net gain across two seeds, 100M exposure and all nine columns.

## Identity-Addressed State Retrieval: Candidate, Not Completed Result

The initial induction-seeded state-subspace hypothesis failed its predeclared test. The unexpected control pattern instead suggested that state retrieval might depend more on stable entity identity than on exact surface position or uninterrupted restatement:

- Disabling the mechanism reduced accuracy.
- Moving content while retaining the correct identity address preserved high accuracy.
- Using the wrong identity address reduced performance.

The pattern was independently reproduced with new names, objects and templates, variable entity counts, multiple overwrites and distractors. Across two seeds, exact-address performance was approximately **0.98-1.00**; displaced content retaining the correct identity address remained approximately **0.90-1.00**. The [stress-test record](identity_address_stress_replication.md) provides the split-level values.

**Later evidence boundary (R35 versus R36).** These figures belong to the supplied-address stress tests, whose paired examples query the target object's final location after swapping its location with another object's. They are distinct from the entity-memory miniscreen whose binding interpretation was later withdrawn under R36: its answers followed action outcomes regardless of the queried entity. High accuracy and write-permutation sensitivity did not establish query-conditioned binding in that miniscreen. The [later shortcut correction](../representation_and_objectives/entity_memory_miniscreen_synthesis.md) explicitly excludes the separate supplied-address experiments from that withdrawal. The figures above retain only their bounded historical interpretation under supplied addresses; they do not establish learned routing from raw text or successful natural-text training. A corrected miniscreen corpus and training script are not evidence of a completed successful corrected-model experiment.

A parameter-neutral natural DeBERTa-v2 implementation had identical initial weights for the off/true conditions, finite forward/backward checks and compatible Hugging Face loading. Those were implementation checks, not evidence of natural-training capability. See [the implementation verification](identity_address_smoke_verification.md).

At the historical cutoff, there were **no H100 natural-corpus training results, no two-seed 60M or 100M results, and no nine-column scores** for this candidate. Large-run memory use, speed, stability, bypass behavior and task trade-offs remained unknown. The controlled result therefore supported a candidate mechanism, not a generalizable learning principle or a breakthrough.

### Proposed Falsification Sequence

The proposed natural-text comparison was true identity addressing versus shuffled addressing, matched in initialization and budget. It was a plan, not a completed experiment:

1. Test seed 42 at **60M**, followed by the corresponding seed 43 comparison.
2. Extend to **100M** only if true addressing consistently exceeds shuffled addressing for both seeds, with concordant corpus-mechanism readouts and target-task behavior.
3. At 100M, require complete nine-column evaluation, mean **Delta Overall >= +0.25**, and no severe task trade-off.
4. If the comparison fails, close the natural-training candidate while retaining the narrower result that correct identity addresses support retrieval in the controlled tasks.

No natural-training outcome is inferred from implementation readiness. The historical conclusion remained: the study had narrowed a representation-versus-use problem, but had not yet converted that diagnosis into a reproducible improvement over the strong WWM baseline.
