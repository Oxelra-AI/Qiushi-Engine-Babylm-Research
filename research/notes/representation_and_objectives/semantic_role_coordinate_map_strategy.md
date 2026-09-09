# role coordinate collision and route — Semantic-role to coordinate map: corrected route and matched next experiment

## Corrected reading of temporal addressability bridge

The temporal addressability bridge record-address result should not be promoted to a key-uniqueness law from the failed indirect run. In that run three things changed together:

1. each random tag appeared both in a role-declaration sentence and in the entry header;
2. a declaration sentence introduced temporal-role words and extra syntax;
3. role-query rows required a new hop from a role phrase to the tag and then to the record.

The changed focal rows did not fit in the direct component once the declarations were present, and the role-only component stayed at chance. That is an informative construction boundary, but it does not isolate duplicated tags from compositional indirection. The safe scientific reading is narrower:

- Direct supplied record coordinates work: section words and arbitrary tags can let a small fine-tuned BabyLM DeBERTa select between contradictory state records.
- Natural temporal wording and held temporal-role paraphrases have not mapped reliably onto those coordinates.
- The unsolved operation is not more memory capacity, more relation readout, or more record keys; it is learning a sparse semantic-role-to-coordinate map from language under interference.

This connects the recent temporal results to the older sequence:

- argument identity becomes usable when an entity/slot coordinate is supplied or when aliases remove identity shortcuts while preserving a trained argument-slot interface;
- relation orientation becomes usable when sparse true anchors attach a new predicate cluster to an absolute signed coordinate;
- state-record selection becomes usable when the query is given the same address coordinate as the relevant record.

Across these layers, supplied coordinates reveal high ceilings. The scientific center is now whether a limited-data learner can learn the map from raw semantic roles to those coordinates, rather than merely exploiting coordinates when they are already shared by context and query.

## Immediate research question

Can we separate lexical collision from compositional indirection while holding the ATP changed-focal / stable-secondary worlds, labels, row counts, record order randomization, and direct relation readout fixed?

The result we need is not a higher small-probe number. We need to know which factor prevents changed-focal transition learning when role declarations are added:

- If duplicate tag mentions alone destroy changed-row fitting, the next object is an interference model of sparse address selection with repeated keys.
- If duplicate tag mentions are harmless but role declarations hurt direct tag readout, the next object is role-language interference with the record address.
- If direct tag readout survives declarations but role queries fail, the next object is compositional semantic-role-to-coordinate mapping.
- If role queries fit and reverse under role-to-tag swaps, the bridge has moved beyond supplied-coordinate retrieval toward a real time-indexed address principle.

## Matched construction

Use the earlier analysis/267 ATP world sampler and the non-oracle balanced-temporal training object: stable base rows plus sparse focal updates, no sparse secondary-state labels. Keep the same held changed focal and held stable secondary readouts, paired AB-vs-BA scoring, shuffled record order, and held query wording. Every arm should use the same selected worlds and the same positive/negative label balance.

### Arms

**A. `unique_tag_direct` — direct supplied-coordinate baseline**

Context has four shuffled tagged entries only:

```text
Entry amber: ...
Entry cobalt: ...
Entry elm: ...
Entry timber: ...
[SEP] According to entry cobalt, Nina was ranked higher than Wendy.
```

This recreates the successful arbitrary-tag coordinate. It must fit changed focal rows and recover focal before/after; otherwise the construction is broken.

**B. `neutral_dup_direct` — lexical duplication without role mapping**

Same as A, but add one neutral sentence per tag, shuffled separately from entries, with no temporal-role words and no link to a state:

```text
The catalog includes the label amber.
```

The query still names the tag directly. This isolates whether repeated tag tokens alone impair selection between contradictory records.

**C. `role_decl_direct` — role declarations present, but query remains direct-tag**

Same tagged entries, plus role declarations:

```text
The focal background snapshot is entry amber.
The focal update snapshot is entry cobalt.
The separate background snapshot is entry elm.
The separate update snapshot is entry timber.
```

The query still names the tag directly. This tests whether the declaration sentence and role words interfere with direct coordinate readout, without requiring role-to-tag composition. A role-declaration swap at evaluation should not change direct-tag answers if the model is using the entry tag rather than the role declaration.

**D. `role_decl_indirect` — compositional role-to-coordinate query**

Same context as C, but the query uses the role phrase and does not include the tag:

```text
According to the focal update snapshot, Nina was ranked higher than Wendy.
```

Success requires role phrase -> declared tag -> tagged record. Evaluation should include a role-to-tag swap: swapping only the focal background/update declarations while keeping entries fixed should reverse focal before/after answers, while direct-tag answers in C should remain tied to the entry.

### Optional add-on if the first four arms implicate lexical collision

If B already breaks changed-focal fitting, build a cleaner collision series before studying role composition:

- one neutral extra mention versus four neutral extra mentions;
- extra mention before versus after the entries;
- neutral sentence with the tag adjacent to `entry` versus a sentence where the tag appears as a bare word;
- direct query with the unique tag still present in the entry header.

This would identify whether the model needs a single-use key or whether a specific header-like phrase creates ambiguity.

### Optional add-on if C survives and D fails

If direct tag readout survives role declarations but role queries fail, build a role-map learning series rather than adding more tag variants:

- train several role families (`background/update`, `initial/latest`, `old/new`) and evaluate `original/current`;
- keep tag vocabularies disjoint between train and evaluation where possible;
- include role-swap evaluation, so success means the query follows the declared map rather than a global role polarity;
- include direct-tag rows only as a readout support/control, not as a substitute for role-query success.

## Fit-first interpretation

For each arm, first inspect sparse changed-focal training fit. Held readout only matters after the changed focal rows fit. A row family that never learns the contradictory changed records is a construction result, not evidence against semantic transfer.

Minimum useful run:

- one seed per arm first;
- 40--80 changed focal worlds and matched stable secondary worlds, using the already working temporal addressability bridge scale if possible;
- core train-context/train-hypothesis and train-context/held-hypothesis surfaces;
- same model path and fine-tuning schedule as temporal addressability bridge;
- run arms in parallel on both H100s only after dry construction verifies row counts, label balance, token lengths, and shared selected worlds.

## Decision table

| A unique direct | B neutral duplicate | C declaration direct | D role indirect | Scientific reading |
|---|---|---|---|---|
| fits | fails | not needed | not needed | duplicated key tokens alone can disrupt changed-state selection; study repeated-key interference before role composition |
| fits | fits | fails | likely fails | role-declaration language, not lexical duplication alone, disrupts direct coordinate use |
| fits | fits | fits | fails | direct coordinates survive; sparse semantic-role-to-coordinate mapping is the bottleneck |
| fits | fits | fits | fits and swaps reverse | evidence for compositional time-role addressing beyond supplied-coordinate retrieval |

## Relation to the long-term principle

The emerging general principle should be framed around coordinate formation and assignment, not around any single surface device:

> Data-efficient learners can generalize from sparse evidence when latent roles are attached to reusable, low-ambiguity coordinates shared by context, query, and prior knowledge. Supplied coordinates create high ceilings, but the hard scientific operation is learning the semantic-role-to-coordinate assignment from sparse language without shortcuts or collision-driven interference.

The next matched experiment decides where the current temporal bridge breaks: repeated lexical addresses, declaration/role-word interference, or true compositional role-to-coordinate mapping. Only the last outcome moves the route toward a general time-indexed state-learning principle; the first two keep it as a supplied-coordinate ceiling with a specific interference mechanism to model.
