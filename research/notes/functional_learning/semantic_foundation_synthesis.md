# curated source grounded probe semantic foundation synthesis

## Scientific change

The earlier analysis natural-row pilot is not merely a noisy but valid binding dataset. Its validator and prompt structure allow examples whose two intended answers are not clearly entailed by the rendered texts. This matters because the active BabyLM bridge is supposed to test acquisition of recipient-dependent selection, not ambiguity resolution or preference for a plausible recent phrase.

Manual inspection exposed two distinct semantic failures:

1. **Additive/non-replacement updates**: e.g. adopting tablets need not replace keeping journals. Both states may remain true, so the intended RETAIN/UPDATE opposition is not forced by the text.
2. **Use-frame leakage**: e.g. a use frame about high ceilings can favor a studio independently of which entity was updated.

The earlier analysis validator also had a concrete grounding bug: source-answer grounding used `source + generated_source_state`. A generated state description could therefore help certify its own unsupported answer. Lexical overlap with raw source is not entailment, but raw-source evidence must be separated from generator metadata.

## Audits run in this step

### curated source grounded probe semantic-foundation audit

Script: `experiments/archive/functional_learning/scripts/semantic_foundation_audit.py`.
Outputs: `experiments/archive/functional_learning/data/semantic_foundation_audit` and note `research/notes/functional_learning/semantic_foundation_for_natural_rows.md`.

Input: the earlier accepted state-update pairs `experiments/archive/relation_learning/data/contrastive_binding_validated_strict/accepted_contrastive_binding_pairs_strict_pilot512.jsonl` and the coherent86 multi-token scores `experiments/archive/functional_learning/data/multitoken_scorer_pilot512/pair_scores.jsonl`.

Quantitative findings over 55 accepted pairs:

- Both source answers are exact substrings of the raw source in 26/55 pairs (47.3%).
- Both source answers have `overlap_min >= 0.34` with the raw source in 44/55 pairs (80.0%).
- At least one source answer shows the self-certification pattern—weak raw-source overlap but acceptable overlap after adding the generated state description—in 11/55 pairs (20.0%).
- Both generated source-state descriptions have raw-source overlap >= 0.34 in 46/55 pairs (83.6%).
- Only 7/55 update frames (12.7%) contain a strong replacement/change/assignment-like cue; proxy event classes were 39 weak/unclear, 9 additive/descriptive, 7 replacement-like.
- Suspect entity surface forms appear in 13/55 pairs (23.6%).

Baseline coherent86 remains weak even after automatic subsets, but the subsets are not semantic truth judgments. All accepted rows: joint 3/55, mean beta=-0.009, mean |alpha|=3.324, mean gamma=-3.333. The automatic strict proxy retained only 3 rows and still had joint 0/3, mean gamma=-2.699.

### Step038b entity-answer association audit

Script: `experiments/archive/functional_learning/scripts/revision_038b_entity_answer_grounding_audit.py`.
Outputs: `experiments/archive/functional_learning/data/entity_answer_grounding_audit`.

Raw source overlap is still insufficient because a phrase can occur in the source while belonging to another entity/event. A cheap exact-span proximity probe found:

- Both source answers exact raw spans: 24/55 (43.6%).
- Both entities had their assigned source answer closer than the competing source answer under exact-span distance: 11/55 (20.0%).
- Rows with no entity-answer proximity flags: 10/55 (18.2%).
- Prominent flags include target source answer absent as exact raw span (20), distractor source answer absent as exact raw span (25), target answer closer to distractor than target (8), distractor answer closer to target than distractor (5), and cases where an entity was closer to the competing answer.

This does not replace entailment verification, but it confirms the validator needs relation-grounded spans or independent evidence for `(entity, relation, answer)`, not just answer/source lexical overlap.

## Contract produced for the next tranche

Path: `research/documents/functional_learning/data/source_grounded_contract/source_grounded_packet_contract.md`.

The next row object should be built in two stages:

1. **Independently grounded source relations**: raw-source/local-passage spans or an independent verifier for `(target_entity, relation_type, target_source_answer)` and `(distractor_entity, relation_type, distractor_source_answer)`. Generated `target_source_state` text must not certify source support.
2. **Explicit replacement/current-state update**: a single update event frame differing only by `{ENTITY}` substitution and making the current value/replacement relation explicit, plus one identical target use frame. Temporal/current-state wording such as `after`, `now`, `current`, or `changed to` is allowed if shared across variants and needed to make the state transition well-defined.

Copy support is allowed. With both candidate phrases visible, copying after selecting the correct entity is the desired computation; removing all copy support would turn the task into paraphrase generation rather than recipient selection.

## Tiny curated source-grounded probe

Script: `experiments/archive/functional_learning/scripts/revision_038c_curated_source_grounded_probe.py`.
Rows: `experiments/archive/functional_learning/data/curated_source_grounded_probe/curated_source_grounded_training_rows.jsonl`.
Pairs: `experiments/archive/functional_learning/data/curated_source_grounded_probe/curated_source_grounded_pairs.jsonl`.
Scorer output: `experiments/archive/functional_learning/data/curated_source_grounded_probe/scorer_coherent86_cpu`.
Neutral scorer: `experiments/archive/functional_learning/scripts/revision_038d_curated_neutral_scorer.py`, output `experiments/archive/functional_learning/data/curated_source_grounded_probe/neutral_scorer_coherent86_cpu`.

This is only a six-pair probe, not a training set or proof of natural-data quality. It uses raw earlier analysis source sentences but manually picks better grounded source relations and explicit current-state update frames.

Coherent86 baseline on the six curated pairs:

- UPDATE correct 6/6 by mean-token score.
- RETAIN correct 0/6.
- Joint correct 0/6.
- Mean U=+3.025, mean R=-4.072, mean U+R=-1.047.

No-update neutral scoring on the same source + same use frame but without an update sentence:

- Neutral source-answer preference correct in 4/6.
- Mean neutral source-minus-new margin: +0.739.
- Mean RETAIN margin after distractor update: -4.072.
- Mean drop from neutral to distractor-update RETAIN: -4.812.

This is the cleanest local evidence in the step that even when several source relations are grounded and the update event explicitly changes a current value, coherent86 often treats a distractor update as enough to shift the target use answer toward the new phrase. The evidence is small and manually curated, so it does not establish a general law; it supports the value of building a larger semantically grounded tranche and keeping the no-update neutral context as a required readout.

## Immediate implications

- Do not train a BabyLM bridge on the existing earlier analysis 55-pair pilot as if it were clean. Use it as scorer/pipeline evidence only.
- The next substantial natural-data investment should repair generation and validation around source-grounded relations and rendered-context verification, then preserve difficult semantically valid pairs where coherent86 fails.
- Ordinary answer CE remains a serious competitor; paired/beta shaping can only earn inclusion if pending decomposition and paired context results and later valid-row tests improve held gamma and joint correctness.
- The next ALN-preserving pilot should compare valid-packet experience support under fixed legal word exposure: narrow repeated support versus broader source-grounded support, with no-update neutral, UPDATE, RETAIN, gamma, and downstream Cheap7/Entity readouts.
