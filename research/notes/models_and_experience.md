# Models, Objectives, and Compact Experience

## Scientific Questions

These studies asked which changes improve language learning under a small unique-text budget: model structure, prediction objective, tokenizer, organization of experience, or optimization. A second question was mechanistic: when a model's representations change after a contextual intervention, does it actually use the relevant relation to predict content?

The evidence separates a corpus's unique word budget from repeated training exposure. It also separates short-budget mechanism screens, checkpoint trajectories, full local benchmark evaluations, and historical public-model comparisons. A favorable screen is not a complete benchmark result, and a proposal is not an executed experiment.

## Establishing the Baseline

The original baseline pipeline includes official-corpus acquisition, source and word-count manifests, portable tokenizer export, model construction, optimizer schedules, checkpoint saving, and evaluation. Causal comparisons covered dense and untied-head controls, sparse routing, morphological features, character/ngram composition, prefix-average memory, and a no-persistence memory control. Shared-core initialization and separate training randomness were important controls: an identical nominal seed did not by itself establish an identical initial model.

Prefix memory, surface composition, and pure data ordering did not yield a stable broad advantage in the matched early screens. This motivated a move to masked encoders and whole-word masking, rather than an assumption that the causal side modules merely needed more training. BERT and DeBERTa comparisons then separated architecture, model size, update geometry, effective batch size, and gradient accumulation. These are recipe comparisons unless all other factors are held fixed.

Tokenizer construction is part of the learning method. The studies compare byte-level BPE and SentencePiece construction through vocabulary or merge definitions and tokenization-coupling measurements. A larger vocabulary changes segmentation, masked targets, embedding/output capacity, and memory cost together. It is not a parameter-neutral test of reduced fragmentation.

## Experience Construction and Compression

The rewriting studies progressed from existing simplification pairs to source-matched generated views. Early adjacency advantages did not establish a universal benefit from paraphrases: later controls compared aligned, shuffled, separated-window, original-only, and repeated-original material. Incorrectly paired text is not a neutral baseline; it can actively damage learning.

The clean generated-view construction preserved complete original/rewrite boundaries, rejected obvious lexical and entity errors, and matched training-row length sequences. It replaced an earlier procedure that could truncate pairs or pad rows with unrelated fragments. The realized clean pool contained 37,594 selected pairs and 1,656,800 paired words, or 16.568% of a 10-million-word pool. This realized dose differs from the intended maximum dose.

Correctly paired views improved some local endpoint comparisons, with notable Entity and SuperGLUE gains, but Reading, COMPS, and other columns imposed tradeoffs. Source-matched rechunking and selected-original repetition controls narrowed the interpretation without proving that semantic correspondence was the sole cause. Two seeds established a repeated direction in those comparisons, not a general variance estimate.

The compact-view route then distinguished near-paraphrases, approximately 65% and 50% length proposition capsules, and typed relation question-answer views. The generation templates constrain participants, negation, modality, quantities, and relations while limiting added information. Construction checks and evaluated generation panels establish properties of those transformations, not downstream superiority for every transformation family.

## Selection, Ordering, and Visible Exposure

The initial compositional-constraint and recoverable-signal scores were highly correlated: their row-level correlation was 0.835. Source, style, and tokenizer-compression features explained much of their variation. Residualized and matched selection was therefore necessary before interpreting a high-score subset as evidence for either proposed mechanism. The subsequent four-million-word screens did not establish broad gains from either residualized selector.

Curriculum experiments varied source order, readability, first-pass developmental order, sequence length, masking granularity, and adaptive mask priorities. These interventions change different quantities. A length curriculum can silently discard content unless packing and exposure accounting track the words actually visible to the model. Stagewise construction and token/source measurements determine whether the compared curricula expose the same material.

The same-content order comparison retained two random permutations as controls. Its curriculum improved some columns while reducing Supplement and GlobalPIQA; it did not establish an aggregate improvement. Similarly, adaptive masking's intermediate advantages did not persist uniformly to the final exposure budget. Two-seed masked-objective trajectories and the structured-data factorial comparison showed antagonistic interactions between individually promising data and objective changes.

## Relations, Memory, and Causal Use

The mechanism studies include coherent/corrupted state stories, relation-focused masking, entity consistency, counterfactual propagation, cross-span objectives, ordered state dynamics, address-based memory, and binding-switch controls. Their common distinction is between a relation being present in the text, being recoverable from a representation, and being used by the prediction mechanism.

Address-based state memory was effective when entity/state locations and routing were supplied. In the decisive unlabeled-address test, the learned router failed to recover query roles on held-out templates; predicted routing lost the intervention effects retained by gold routing. This is a meaningful negative control, not evidence that the supplied-address algorithm never worked.

Other probes distinguished changes caused by position or lexical familiarity from genuine content dependence. Corrected history probes, paired-gradient analyses, fixed-position prefix censuses, and direct checkpoint evaluations supersede stronger interpretations of their predecessors. Synthetic binding can be learnable without improving legal-corpus pretraining. Overwritten intermediate checkpoint bins prevent recovery of an exact exposure trajectory for the binding-switch comparison.

## Objective and Optimization Boundaries

Local gradient compatibility was insufficient to establish useful objective composition. Causal replacement, masked-next-token auxiliaries, and coordinated cross-view masking were evaluated separately from their motivating gradient or reconstruction probes. Full endpoint evaluation did not confirm broad gains for several objectives whose narrow task improvements had appeared promising without AoA.

The recursive autoregressive model, RecGPT, was studied with its packed loader, Aurora/auxiliary Adam optimization and NextLat auxiliary. A simplified local implementation was not equivalent: its optimizer equations, auxiliary initialization and exposure semantics differed materially. Unresolved Reading and GlobalPIQA differences in public-weight rescoring also prevent a claim of complete benchmark equivalence.

Optimizer-by-corpus experiments matched the learning-rate schedule within each comparison. A ten-million-word endpoint with a nearly exhausted cosine schedule is not dynamically equivalent to the ten-million-word checkpoint of a longer run. Faster loss reduction alone did not justify extrapolating an optimizer/data interaction to full exposure.

## Measurement Corrections

The corrected local scoring implementation averages nine benchmark columns after converting AoA raw correlation to leaderboard units by multiplying by 100. Older summaries that inserted raw correlation directly into Overall are superseded. Weighted fast proxies, equal-column screens without AoA or SuperGLUE, and complete nine-column Overall measure different aggregates.

A measured AoA of zero can be a valid significance-gated result. It must be distinguished from unavailable AoA represented as a leaderboard-convention zero. Checkpoint-ladder completeness, completed task measurements, raw correlation and scaled score distinguish these cases. Negative AoA also explains why some favorable no-AoA masking or objective screens failed at the full endpoint.

Evaluation implementation matters: EWoK filtering/tokenization, exact checkpoint-directory loading, Reading continuation tokenization, and classifier extraction of hidden states rather than vocabulary logits changed interpretation. Constant-label fine-tuning behavior is a diagnostic limitation rather than evidence of semantic transfer.

## Methods and Evidence

| Scientific comparison | Method | Evidence |
| --- | --- | --- |
| Whole-word versus token masking | [Paired training procedure](../../experiments/archive/initial_model_studies/scripts/run_masked_1m_grid_pos512.py) | [First-seed profile](../../experiments/archive/initial_model_studies/data/masked_1m_grid_pos512_profile.json), [second-seed profile](../../experiments/archive/initial_model_studies/data/masked_1m_seed43_profile.json) |
| Masking and learning stage | [Trajectory comparison](#selection-ordering-and-visible-exposure) | [Checkpoint trajectories](../../experiments/archive/compact_experience/data/curriculum_100M_eval/trajectory_compact_table.csv) |
| Causal auxiliary versus masked prediction | [Objective-combination analysis](#objective-and-optimization-boundaries) | [Complete endpoint](../../experiments/archive/compact_experience/data/mlm_mntp_full_eval/per_target/mlm_mntp_aux015_100M.json), [reference comparison](../../experiments/archive/compact_experience/data/mntp_eval_interpretation.json) |
| Supplied versus learned memory addresses | [Address-based state use](#relations-memory-and-causal-use) | [Unlabeled-address control](../../experiments/archive/initial_model_studies/data/unlabeled_address_decisive.json) |

These comparisons establish conditions on representation, experience and measurement. They do not by themselves identify the cause of gains in a later model.
