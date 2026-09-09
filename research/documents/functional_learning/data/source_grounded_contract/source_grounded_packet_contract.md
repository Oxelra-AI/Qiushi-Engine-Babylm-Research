# curated source grounded probe source-grounded recipient-contrast packet contract

## Scientific purpose

The natural BabyLM bridge should test whether limited natural experience can teach a reusable recipient-sensitive selection operation: after reading a source with two entity-state relations and an update that applies to only one entity, the learner should answer a target-entity use frame with the new state if the target was updated and with the original source state if only the distractor was updated.

The construction must make the two opposite answers true in the rendered text. It is not enough for rows to be entity-balanced, tokenizer-valid, or symmetric under candidate scoring. The source relation and update event must be semantically grounded so that a competent reader can determine both answers from the text.

## Unit of data

A source-grounded packet is built in two stages.

### Stage A: independently grounded source relations

For one raw source sentence or short local passage, record two entity-state facts with direct evidence in the raw text:

```json
{
  "source_id": "stable source row id",
  "raw_source_text": "...",
  "source_origin": "simple_wiki|gutenberg|bnc_spoken|childes|...",
  "supports": [
    {
      "role": "target",
      "entity_text": "Italy",
      "entity_span": [123, 128],
      "relation_type": "chart_rank",
      "relation_frame": "chart position for {ENTITY}",
      "source_answer_text": "number 41",
      "source_answer_span": [142, 151],
      "evidence_quote": "went to number 41 in Italy",
      "why_source_entails": "The quote explicitly gives the chart/rank value for Italy."
    },
    {
      "role": "distractor",
      "entity_text": "Billboard Adult Alternative Songs",
      "entity_span": [160, 194],
      "relation_type": "chart_rank",
      "relation_frame": "chart position for {ENTITY}",
      "source_answer_text": "number 18",
      "source_answer_span": [149, 158],
      "evidence_quote": "number 18 on the Billboard Adult Alternative Songs chart",
      "why_source_entails": "The quote explicitly gives the chart/rank value for the distractor chart."
    }
  ],
  "source_grounding_verifier": {
    "verifier_model_or_method": "independent from generator",
    "target_answer_from_source": "number 41",
    "distractor_answer_from_source": "number 18",
    "both_answers_supported": true,
    "notes": "No generator-supplied state description was used as source evidence."
  }
}
```

Hard requirements:

- `source_answer_span` and `entity_span` are character spans in the raw source text or raw local passage, not in a generated state description.
- If a paraphrase answer is useful, keep both `source_answer_text_raw_span` and `source_answer_text_for_scoring`; the raw span is what grounds the relation.
- The two source supports should use the same relation type whenever possible (rank, role, location, possession, assignment, status, diagnosis, file label, route, responsibility). This makes the later update operation symmetric.
- The validator may record proximity and lexical-overlap proxies, but final acceptance needs an independent reader/verifier judgment over the raw source relation.

### Stage B: explicit replacement/current-state update and identical target use frame

Given Stage A, construct one shared update event and one target use frame:

```json
{
  "packet_id": "...",
  "source_id": "...",
  "target_entity": "Italy",
  "distractor_entity": "Billboard Adult Alternative Songs",
  "target_source_answer": "number 41",
  "distractor_source_answer": "number 18",
  "shared_new_answer": "number 1",
  "update_event_frame": "After the reissue, the current chart position for {ENTITY} changed to number 1.",
  "update_event_relation_type": "chart_rank",
  "update_event_semantics": "explicit_current_replacement",
  "target_use_frame": "The current chart position for Italy is {STATE}.",
  "rendered": {
    "target_update_text": "<raw_source> After the reissue, the current chart position for Italy changed to number 1. The current chart position for Italy is number 1.",
    "distractor_update_text": "<raw_source> After the reissue, the current chart position for Billboard Adult Alternative Songs changed to number 1. The current chart position for Italy is number 41."
  },
  "rendered_verifier": {
    "target_update_answer": "number 1",
    "distractor_update_answer": "number 41",
    "opposite_answers_unsupported": true,
    "competent_reader_can_answer": true,
    "notes": "Both candidate phrases are visible; correctness requires choosing which entity the update applied to."
  }
}
```

Hard requirements:

- `update_event_frame` differs across variants only by `{ENTITY}` substitution.
- The update event must make a current value or replacement relation explicit. It may use shared temporal/current-state language such as `after`, `changed to`, `now listed as`, or `current`, because those make the transition well-defined. Such words are not shortcuts if they are identical across the two variants and do not identify the target.
- Avoid merely additive events when source and update can both remain true, e.g. `adopted cataloging tablets` does not necessarily replace `kept journals`; `moved into a studio with high ceilings` can leak answer preference if the use frame asks about high ceilings.
- `target_use_frame` mentions the target exactly once, does not mention the distractor, and does not contain lexical material that independently favors `shared_new_answer` over `target_source_answer` except the `{STATE}` slot itself.
- Copying the source answer or update answer is allowed: with both competing phrases visible, copying after selecting the correct entity is the desired operation. Removing all copy support would turn the task into paraphrase generation rather than recipient selection.

## Validator changes from earlier analysis

The earlier analysis validator checked source-answer overlap against `source + generated_source_state`, which allowed generated descriptions to help certify unsupported answers. The replacement validator must instead:

1. Check entity and source-answer spans against raw source/local passage only.
2. Preserve span offsets and evidence quotes in the accepted row.
3. Run an independent source-relation verifier that answers from the raw source without seeing generated state descriptions.
4. Run an independent rendered-context verifier over both fully rendered variants. It should answer the same target use question in each context and reject the row if either intended answer is not supported or the opposite answer remains plausible.
5. Validate tokenizer spans for answer and foil inside the final use sentence, but record rather than ban source/update occurrences of the same phrases.
6. Report UPDATE, RETAIN, pair-level beta, alpha, gamma=beta-|alpha|, and joint correctness for the inherited coherent86 parent before using the packet for training.

## Experimental comparison enabled by this contract

Once enough valid packets exist, the bounded BabyLM bridge should compare experience support and supervision under matched legal word exposure:

- ALN-preserving reference continuation: preserve `qwen_pair_packed` rows; replace only filler-like material.
- Narrow repeated support: repeated instances of a smaller valid packet set.
- Broader source-grounded support: more distinct valid packets at the same legal word budget.
- Ordinary answer CE and, only if decomposition and paired context/037/038 results justify it, an added beta-shaping paired term. Ordinary CE remains a serious competitor because balanced CE already rewards U>0 and R>0.

The readout is not mean beta alone. A useful natural intervention must improve held pair-level gamma and joint UPDATE+RETAIN behavior on semantically valid packets while preserving Cheap7/Entity and then official-compatible BabyLM transfer.
