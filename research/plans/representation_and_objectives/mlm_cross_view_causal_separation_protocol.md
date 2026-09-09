# earlier analysis protocol: MLM cross-view causal separation for compact semantic views

## Research question

The compact-view triangle established a large DeBERTa masked-denoising gain from faithful compact semantic views over literal repetition, and correctness transition analysis showed that source-own adjacency carries Supplement and relational-EWoK movement while losing BLiMP. adjusted source use residual ruled against a stronger ordered copied-token retrieval explanation in the frozen source-use probe. What remains unproven is whether **source-absent semantic supervision from the correct partner** caused the downstream gain.

The next MLM study should answer one causal question:

> Holding each source and rewrite's own internal text, dose, positions, model, optimizer, tokenizer, WWM process, and total exposure fixed, does access to the **correct** source--rewrite partner, rather than a mismatched partner or no cross-boundary attention, recover the correctness transition analysis Supplement and relational-EWoK transition pattern, with evidence coming from source-absent compact targets rather than copied-token retrieval?

This is not a confirmation run. If the answer is negative, the compact semantic-abstraction route should stop as the next mechanism route and the research should return to coverage, marginal-density, or a different representation-level principle. If the answer is positive, only then does the route deserve independent architecture or data-source transfer.

## Evidence already available

- Triangle endpoint: `experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/triangle_noaoa_summary.json`
  - `compact_view_reinvest` 44.2886 equal7
  - `compact_repeat_reinvest` 41.8721 equal7
  - `adjbreak_reinvest` 43.0557 equal7
  - view-repeat +2.4164 exceeds the same-recipe fast equal7 seed-spread band 1.3557.
  - view-adjbreak +1.2329 is inside that band in aggregate, but correctness transition analysis shows opposing BLiMP and Supplement/EWoK movements.
- Correctness transitions: `experiments/archive/representation_and_objectives/data/correctness_transitions` and `research/notes/representation_and_objectives/correctness_transition_analysis.md`
  - view vs adjbreak: Supplement +9 net; EWoK +51 net, concentrated in social-properties, physical-dynamics, spatial-relations, physical-relations; BLiMP -217.
  - adjbreak vs repeat: BLiMP +209 but EWoK -27.
- Copied-token source-use residual: `experiments/archive/representation_and_objectives/data/adjusted_source_use_probe/adjusted_source_use_fast_bootstrap.json`
  - On matched copied targets, compact's source-conditioned ordering interaction is lower than extractive controls: compact minus mean extractives median -0.491 nats, 95% [-0.605,-0.384], p>0=0.000.
  - Therefore compact's downstream gain is not explained by enhanced ordered copied-token retrieval.
- DeBERTa pairwise attention-mask smoke: `experiments/archive/representation_and_objectives/data/deberta_pair_attention_mask_smoke/deberta_pair_attention_mask_smoke.json`
  - Custom path sending 2D padding mask to embeddings and 3D pairwise mask to the encoder is exact for fully visible attention: max logit delta 0.0, loss delta 0.0.
  - Blocking source<->rewrite attention changes logits: max delta 0.007668, masked-position delta 0.004486.
  - Backward pass has finite gradients through 50 tensors.
  - Important implementation fact: do **not** send a 3D/4D mask through stock `DebertaV2ForMaskedLM.forward`; embeddings misuse it. Use a custom MLM forward path.
- Wrong-partner preflight: `experiments/archive/representation_and_objectives/data/wrong_partner_matching_preflight/wrong_partner_matching_preflight.json`
  - Bijective derangement over all 12,155 pairs, 0 self-pairs.
  - 12,151/12,155 pairs preserve exact rewrite word length; total absolute rewrite-word delta 6.
  - Same primary-domain donor for 12,133/12,155 pairs.
  - Own source→rewrite recall mean 0.831; wrong source→rewrite recall mean 0.086; wrong never matches own recall.
  - Meaning: wrong-visible preserves view dose/internal context and subtracts generic cross-boundary context, but copied-target contrasts are intentionally not semantically matched.
- Target-origin profile over the exact 12,155-pair population: `experiments/archive/representation_and_objectives/data/compact_pair_target_origin_profile/compact_pair_target_origin_profile.json`
  - 162,507 profiled rewrite words.
  - Copied rewrite words: 135,920 (83.64%).
  - Source-absent rewrite words: 26,587 (16.36%).
  - Source-absent content words: 20,484 (12.60%).
  - 9,788/12,155 pairs (80.53%) contain at least one source-absent content word.
  - This source-absent content mass is large enough to support target-stratified MLM readout.

## Causal arms

The minimum causal separation is a 2×2 compact-pair design:

1. **Own-visible**: source_i + rewrite_i in one packed example, ordinary bidirectional attention across the source--rewrite boundary.
2. **Own-blocked**: identical tokens, positions, row order, WWM masks, optimizer updates, and dose as Own-visible, but source tokens cannot attend to rewrite tokens and rewrite tokens cannot attend to source tokens. Each view retains full internal bidirectional context.
3. **Wrong-visible**: source_i + rewrite_π(i), using the earlier analysis bijective derangement; full bidirectional cross-boundary attention is allowed.
4. **Wrong-blocked**: identical wrong-partner rows but source<->rewrite attention is blocked.

Primary estimand for any downstream metric Y or target-stratum loss L:

I_partner = (Own-visible − Own-blocked) − (Wrong-visible − Wrong-blocked)

This subtracts generic cross-boundary context and attention-mask effects. Because wrong partner intentionally destroys lexical alignment, copied-token losses are a mechanism-null/retrieval channel; the decisive compact-specific channel is source-absent content targets plus the downstream Supplement and relational-EWoK transitions.

## Data construction requirements

The proposed pair-aware MLM representation requires a JSONL schema or dataset class with more than plain `text` rows. Every row needs:

- pair_id and source_pair_id
- source_text
- rewrite_text (own or wrong donor)
- source_words, rewrite_words, total row words
- partner_type: `own` or `wrong`
- visibility: `visible` or `blocked`
- source segment token span and rewrite segment token span after tokenization
- word-level target-origin records for rewrite tokens: copied_unique, copied_multi, copied_tail, source_absent_content, source_absent_function_or_other
- deterministic row order shared across the four arms
- deterministic WWM mask choices paired across arms as far as possible; at minimum, the same random generator stream after tokenization, with exact masked-token count recorded by target stratum

Use the exact pair source:

`experiments/archive/frontier_consolidation/data/density_core_reinvestment_medium_riskhard/selected_compact_reinvest_pairs.jsonl`

Use the existing filler only if needed to restore the full 10M-word epoch:

`experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/filler_rows.jsonl`

The filler must be identical across all four arms. The pair block is only 423,511 words/pass, so most exposure remains filler; this is acceptable only if the four arms share filler, order, and dose exactly. The pair block should appear in the same stream locations across arms so the intervention is partner visibility rather than curriculum position.

## Implementation route

Start from the trusted COMPACT_EXPERIENCE `masking_curriculum_trainer.py` and the triangle oom repair and gc preflight activation-checkpointed wrapper, but construct an explicit pair-aware trainer rather than passing a 3D attention mask into the stock MLM forward.

Required custom forward behavior:

- embeddings receive the ordinary 2D padding mask;
- the DeBERTa encoder receives a 3D pairwise attention matrix of shape `[B,T,T]`;
- the MLM head and cross-entropy are unchanged;
- a fully visible pairwise mask is exactly equivalent to the stock public forward in eval mode;
- activation checkpointing remains available for H100 memory.

The earlier analysis smoke test verifies this forward path on a small model. The comparison still requires a trainer-level smoke test on the real tokenizer and a few compact rows.

## Low-cost sequence before full H100 commitment

### 1. CPU/data construction and single-batch real-tokenizer smoke

Build the four-arm dataset and inspect:

- exact one-to-one use of sources and rewrites;
- exact or recorded near-exact word counts;
- 0 self-pairs in wrong arms;
- same row count and same filler rows in all arms;
- pair segment boundaries after tokenization;
- target-origin stratum counts after tokenization;
- fully visible custom forward exactly matches stock DeBERTa for a tiny real-tokenizer batch;
- blocked attention changes only cross-boundary reachability, not padding or internal segment attention.

If these fail, repair before any H100 use.

### 2. Short H100 learning-signal screen: four arms, 10M--20M words

Purpose: determine whether the correct partner actually changes the learning signal in the target strata before paying for endpoint-level official readouts.

Run the four arms at 10M or 20M words, using the same DeBERTa 8×480 / baseline16k / WWM / AdamW coordinate and paired masks. Record per-step and per-checkpoint MLM losses by:

- copied_unique;
- copied_multi;
- copied_tail;
- source_absent_content;
- source_absent_function_or_other;
- source segment tokens;
- filler tokens.

The route remains alive only if the correct partner produces a stratum-specific signal not reproduced by wrong-visible. A pure copied-token win is not enough. The informative pattern is:

- Own-visible improves source_absent_content loss or retention relative to Own-blocked;
- Wrong-visible does not reproduce that source_absent_content benefit;
- copied-target benefits are present but do not dominate the only positive movement;
- training remains numerically stable and broad filler/source losses are not distorted by the mask implementation.

If this short screen shows no correct-partner source-absent signal, do not launch a full four-arm run. If it shows only copied-token retrieval, shift toward a coverage-matched source-extract principle rather than compact semantic abstraction.

### 3. Endpoint-level causal study: four arms to 100M no-AoA, only if the short screen is alive

This is the first endpoint-level study that can decide whether the mechanism deserves transfer. Use the same official-compatible no-AoA Fast readout used in the compact triangle. Full official AoA is unnecessary for mechanism selection.

Primary downstream readouts:

- Supplement item transitions, especially retained-correct plus newly-correct movement against Own-blocked and Wrong-visible.
- EWoK domains separately: social-properties, physical-dynamics, spatial-relations, physical-relations. Do not pool them into one number without showing each domain; earlier analysis showed pooled relational movement can be misleading.
- BLiMP movement, because correctness transition analysis found grammar/relation opposition.
- COMPS subtask fingerprint: wugs_dist_in_between versus wugs_dist_before.
- Aggregate equal7 only as secondary context.

Prospective support for correct-partner source-absent semantic supervision:

- Own-visible exceeds Own-blocked on Supplement with a retained-plus-new pattern resembling the correctness transition analysis view-vs-adjbreak movement.
- Own-visible improves the four specified EWoK domains individually, not merely through one or two domains.
- Wrong-visible does not reproduce those transitions.
- The partner-specific downstream movement is associated with source_absent_content target learning or exposure, not solely copied_unique/tail loss.
- BLiMP movement is either the expected relation/grammar tradeoff from correctness transition analysis or is improved by a future schedule; it must not be hidden by equal7 averaging.

Evidence against the route:

- Own-blocked keeps Supplement and relational-EWoK movement, meaning co-exposure/dose rather than cross-view access is enough.
- Wrong-visible reproduces Own-visible, meaning generic cross-boundary context or source-domain co-occurrence is enough.
- Only copied target losses move while source-absent content and downstream relation/Supplement do not.
- The four EWoK domains are inconsistent in the earlier analysis sense: a pooled average hides negative domains.

## Relation to coverage-matched extractive nulls

Do not launch source-extractive null arms in the first expensive round unless the compact four-arm study shows a real correct-partner source-absent signal. If the four-arm study is positive, the next null should be a copied-only source-extract pair family matched to compact on:

- view word count;
- tail coverage;
- unique/repeated source-match distribution;
- content-word fraction;
- row occupancy and source position;
- the same visible/blocked partner manipulation.

If copied-only extractive arms reproduce Supplement and relational-EWoK transitions, the transferable principle is reachable unique/tail content under denoising, not semantic rewriting. If compact retains a partner-visible residual on source-absent content targets, the stronger principle is relation-preserving noncopy semantic supervision anchored by a correct source partner.

## Exact outcome meanings for research direction

- **Positive compact four-arm result**: assess whether the evidence is strong enough for transfer to another bidirectional denoising architecture or a different data source. Do not immediately optimize BabyLM score.
- **Copy-only result**: abandon compact-specific semantic abstraction as the main mechanism and build a coverage-matched extractive/selection principle.
- **No correct-partner result**: treat the correctness transition analysis source-own movement as seed- or distribution-specific redistribution; return to broader compact marginal-density, coverage, or entity-event composition routes.
- **Implementation distortion**: repair the trainer; do not interpret endpoint scores.

## Practical endpoint boundary

The coherent86 alpha0.75 artifact is already complete at measured-AoA Overall 42.1210247099666 (`experiments/archive/representation_and_objectives/data/alpha075_collated_fast_aoa`). It is a practical endpoint artifact and not evidence for this mechanism. Do not spend this mechanism step on further alpha0.75 evaluation.
