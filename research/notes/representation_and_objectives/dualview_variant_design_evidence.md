# dualview variant design evidence dual-view broad-preserving variant evidence

CPU/file-only synthesis while the matched `coupled_sparse20_shuffled_20M` control runs. No polling, no training, no model scoring, and the reproduced `chck_82M` endpoint is untouched.

## What changed in the mechanism picture

The broad collapse of `coupled_sparse20_aligned_20M` is not plausibly a simple exposure-volume effect. The coupled run charges only 183,484 aux words out of 19,995,185 charged words (0.918%) and has a main-word shortfall of -188,299 versus MLM-only. Separated aligned has essentially the same debit (183,484 aux words, -188,299 main-word shortfall) but gains cheap7 +0.5064; coupled aligned loses cheap7 -0.9929.

The distinguishing implementation fact is coupling: fullctx aux budget audit runs ordinary MLM with adapters enabled and steps one optimizer after main+aux gradients, freezing stock only during the auxiliary pass. counterfactual exchange clean subset separated runs main MLM with adapters off and stock-only updates, then trains the private adapter separately with aux plus KL neutrality to detached stock logits. Therefore the damage source to repair is coupled-path adapter/main co-adaptation, not just the existence of a source/free rewrite auxiliary object.

## Auxiliary object shape

Top20 aux object: 1,765 example ids, 2,431 pair records, selection `top_0.2_by_changed_source_absent_piece_per_charge`. Summary reports selected aux charge per full activation 109,051 and selected changed source-absent pieces 12,369.

Rewrite/source compression (words) mean 0.607, median 0.600; rewrite word overlap with source mean 0.712, median 0.706. Pairs where all rewrite words appear in source: 17/2,431; exact source=rewrite pairs: 0.

Rewrite marker counts include {'quantifier_number': 667, 'aux_modal': 1004, 'negation': 243, 'spatial': 688, 'causal_dynamic': 407, 'discourse_dialogue': 76}; pair records with markers include {'quantifier_number': 891, 'aux_modal': 2303, 'negation': 384, 'spatial': 1320, 'causal_dynamic': 565, 'discourse_dialogue': 88}. This confirms the selected object is a compact source-to-rewrite transformation set with some relation/function markers, but still mostly compression and lexical reuse rather than dense bidirectional operator orbits.

## Broad damage and hard repair to preserve

Coupled aligned vs MLM-only broad score deltas: BLiMP -2.52, Supplement -3.43, Reading -1.11, COMPS -0.98, Entity -0.46, EWoK aggregate +0.05, GlobalPIQA aggregate +1.50, cheap7 -0.9929.

Worst coupled report deltas (excluding per-example GlobalPIQA rows) include:

- BLiMP / UID ACCURACY / existential_there_quantifiers_2: -28.32
- BLiMP / UID ACCURACY / ellipsis_n_bar_2: -25.97
- BLiMP / UID ACCURACY / anaphor_number_agreement: -24.38
- BLiMP / LINGUISTICS_TERM ACCURACY / anaphor_agreement: -21.56
- BLiMP / UID ACCURACY / determiner_noun_agreement_2: -20.08
- BLiMP / UID ACCURACY / irregular_past_participle_adjectives: -19.05
- BLiMP / UID ACCURACY / anaphor_gender_agreement: -18.84
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_2: -18.39
- Supplement / UID ACCURACY / turn_taking: -15.71
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_irregular_1: -14.90
- BLiMP / UID ACCURACY / regular_plural_subject_verb_agreement_1: -14.83
- BLiMP / UID ACCURACY / determiner_noun_agreement_with_adj_irregular_2: -14.53

Best coupled report deltas include:

- BLiMP / UID ACCURACY / wh_island: +26.46
- BLiMP / UID ACCURACY / superlative_quantifiers_2: +25.25
- BLiMP / UID ACCURACY / left_branch_island_echo_question: +18.38
- BLiMP / UID ACCURACY / principle_A_domain_1: +16.74
- BLiMP / UID ACCURACY / distractor_agreement_relational_noun: +16.24
- BLiMP / UID ACCURACY / wh_questions_object_gap: +16.18
- BLiMP / UID ACCURACY / sentential_subject_island: +14.57
- BLiMP / UID ACCURACY / distractor_agreement_relative_clause: +11.94
- BLiMP / UID ACCURACY / only_npi_scope: +10.75
- EWoK / CONTEXT_CONTRAST ACCURACY / game: +10.00

coupled control interpretation infrastructure already localized hard EWoK repair mainly to agent-properties, physical-relations, physical-interactions/social/spatial domains and variable-swap ContextDiff rows. These are the surfaces the pending matched shuffled control must read before any variant is launched.

## Important implementation warning

The sparse20 fullctx aux budget audit/130 models use `adapter_modeling.py`, where `config.adapter_scale` is recorded but the adapter output is returned unmultiplied. In contrast, `chck_82M` scale1.75 uses `adapter_scaled_modeling.py`, where the update is explicitly multiplied by `self.scale`. Thus future sparse20 amplitude tests require a code-level scaled-modeling repair or explicit adapter toggling; changing `config.adapter_scale` alone is not evidence.

## How to use this after the pending control lands

If `coupled_sparse20_aligned` strongly beats matched `coupled_sparse20_shuffled` on the 1,471-row EWoK stable-reversal surface with coherent GlobalPIQA hard52 rank/margin movement, preserve coupled true-correspondence as the mechanism and design the smallest broad-preserving coupled variant. The first variants should target coupled-path interference: adapter influence on ordinary MLM, shared source-free/conditioned pressure, and explicit adapter scaling/neutrality. If shuffled repairs EWoK near aligned, the hard-row movement is not true-correspondence-specific and the proposed mechanism requires reconsideration.

Proposed minimal variants are recorded in the JSON under `future_minimal_variants_after_control_only`; none should be launched before the matched control readout.

JSON: `experiments/archive/representation_and_objectives/data/dualview_variant_design_evidence/dualview_variant_design_evidence.json`
