# role coordinate collision and route — Coordinate collision, acquisition fragility, and the semantic-role map

## Purpose

temporal addressability bridge left a real but narrow result: a BabyLM-pretrained DeBERTa bridge can select between contradictory ATP ranking records when the query contains the same supplied coordinate as the relevant record. The strategist note for role coordinate collision and route correctly warned against reading the failed two-hop run as a simple key-uniqueness law, because duplicated tag tokens, declaration sentences, and role-to-tag indirection changed together.

The scientific question is revised as follows:

> Argument identity, relation orientation, and state-record selection become usable when a coordinate is supplied. The unsolved operation is learning the semantic-role-to-coordinate map from sparse language.

## Planning correction

I wrote `notes/semantic_role_coordinate_map_strategy.md` to preserve the corrected route. The intended comparison is not more arbitrary tag variants. It is to separate:

1. added context burden;
2. duplicated non-value tag mentions;
3. declaration-like tag binding syntax;
4. temporal-role language;
5. true role phrase -> coordinate -> record composition.

independent_review reviewed this plan in . It agreed that the four-arm ladder is useful but too coarse by itself. It proposed two stronger additions:

- an evaluation-time insertion matrix using an already fitted unique-tag model, to distinguish readout interference from training acquisition failure;
- a later map-retrieve-compose triangle measuring role->tag map, tag->record retrieval, and role->record composition separately.

The insertion matrix was the cheapest first action because it reused the same training condition that already worked in temporal addressability bridge.

## Construction and execution

Script created: `training/scripts/role_coordinate_collision_probe.py`.

Initial reduced 20/8 runs did not fit changed rows, so they were not used scientifically:

- `data/rolecoord_unique1/`: `sparse_changed_focal=0.508`
- `data/rolecoord_filler1/`: overall train about 0.506

The mismatch was traced to a support difference: temporal addressability bridge's successful arbitrary-tag run used `base_train_wording=true` and 5,120 base rows. I patched the role coordinate collision and route script to include `--base_train_wording` and reran the canonical scale:

- base stable worlds: 80, with two base wordings -> 5,120 base rows
- sparse changed worlds: 16 -> 256 changed focal rows
- sparse stable worlds: 16 -> 256 stable focal rows
- held changed and held stable worlds: 40 each
- one seed: 26700
- max length 200, no truncation

## Canonical direct-coordinate result

`data/rolecoord_unique_canonical2/`

The supplied-coordinate baseline fits:

- best train accuracy: 0.9998
- `sparse_changed_focal=1.000`
- `sparse_stable_focal=1.000`
- held changed/stable direct-tag readout: focal_before=1.000, focal_after=1.000, secondary_before=1.000, secondary_after=1.000
- focal_after margin: +25.18

This reproduces the temporal addressability bridge arbitrary-tag ceiling in the corrected script. It is the reference for the rest of the step.

## Insertion matrix: readout interference versus acquisition failure

Script created: `training/scripts/coordinate_insertion_matrix.py`.

This trains only the working unique-tag model above, then evaluates the same fitted model with extra context material inserted while the query still names the tag directly.

Output: `data/coordinate_insertion_matrix1/`.

The fitted model remains perfect when the inserted text contains no tag:

- `insert_filler`: focal_before=1.000, focal_after=1.000, secondary_before=1.000, secondary_after=1.000; focal_after margin +25.42

But inserting extra sentences that mention the tags perturbs changed focal selection while stable secondary remains perfect:

- `insert_labeldup`: focal_before=0.500, focal_after=0.500; secondary 1.000/1.000
- `insert_entrydup`: focal_before=0.750, focal_after=0.000; secondary 1.000/1.000; focal_after margin -6.68
- `insert_slotdecl`: focal_before=1.000, focal_after=0.000; secondary 1.000/1.000; focal_after margin -10.92
- `insert_roledecl`: focal_before=0.250, focal_after=0.500; secondary 1.000/1.000
- `insert_unique`: all four readouts 1.000

Interpretation: pure context length is not enough to disrupt a fitted address reader at inference time. Repeated coordinate-bearing tag mentions are the dangerous change, and the damage appears specifically on the contradictory changed focal record. Stable secondary rows remain high because before and after agree, so they do not by themselves prove record selection.

## Training on harder contexts

At the same canonical scale, I trained direct-tag models with the harder contexts. These runs ask whether the system can acquire robust direct coordinate readout when duplicate or declaration material is present during training.

- `data/rolecoord_filler_canonical2/`: training collapsed overall (`0.500`), including base rows. Because the same fitted unique model read filler contexts perfectly at evaluation, this is acquisition fragility, not inference-time readout damage.
- `data/rolecoord_labeldup_canonical1/`: base and stable rows fit (`1.000`), but `sparse_changed_focal=0.500`.
- `data/rolecoord_entrydup_canonical1/`: base and stable rows fit (`1.000`), but `sparse_changed_focal=0.500`.
- `data/rolecoord_slotdecl_canonical1/`: base and stable rows fit (`1.000`), but `sparse_changed_focal=0.543`.
- `data/rolecoord_roledecl_direct_canonical1/`: whole run stayed at chance (`0.500`), so it cannot test temporal-role composition.

These trained-context runs show a consistent pattern after excluding nonfit settings: stable/base facts can often be acquired despite duplicate coordinate mentions, but sparse contradictory changed-focal records fail. This matches the insertion matrix: collisions matter most when the learner must select between two conflicting states for the same focal pair.

## What is now supported

The new result is not a completed temporal-learning principle and not a bare key-uniqueness law. The supported scientific reading is:

> Supplied coordinates let the learner perform sparse state-record selection. Extra text without those coordinates is harmless at readout but can make acquisition fragile. Extra non-value mentions of the same coordinate create competing address evidence that selectively disrupts changed-state records, especially the focal after record, while stable rows can remain correct because either record gives the same answer.

This is a sharper version of the coordinate-supply pattern:

- Entity/argument identity: works when coreference is supplied by aliases or slot structure rather than memorized names.
- Relation orientation: works when sparse true anchors attach new predicates to an absolute signed coordinate.
- State-record selection: works when a query shares a clean supplied address with the relevant record; fails or becomes fragile when the coordinate is repeated in non-value roles or must be inferred from semantic role language.

## What remains open

The central missing operation remains semantic-role-to-coordinate learning. We still do not know whether a small learner can infer:

```text
"focal update snapshot"  ->  entry cobalt  ->  the focal later ranking record
```

or transfer:

```text
background/update  <->  prior/revised  <->  original/current
```

under sparse data while preserving changed focal and stable secondary facts.

The current role-declaration direct run collapsed entirely, so role-query composition must not be interpreted from it. The next construction must first restore a setting where the direct tag->record component fits in the presence of a role map.

## Next research action

Do not launch Strict-Small 100M training. The next useful bridge should implement the independent_review map-retrieve-compose triangle, but only after making the direct retrieval component fit in the same full context:

1. **Retrieve R(k, v):** given a tag, classify the relation in the tagged entry, with role declarations present.
2. **Map M(r, k):** given a role phrase, choose or verify its declared tag.
3. **Compose C(r, v):** given a role phrase, classify the relation in its bound entry.

The cleanest immediate repair is to reduce the role declaration context until `R` fits changed focal rows:

- try one declaration pair only for the focal records, omitting secondary declarations;
- or put declarations in a compact preamble with no repeated `entry TAG` phrase, e.g. `focal-background=amber; focal-update=cobalt;`;
- or use positional role binding (`focal update is the second record`) to remove repeated random tag tokens.

Once `R` fits in the full context, the map and compose rows can decide whether sparse language learns semantic-role-to-coordinate assignment. If `R` cannot be made to fit, the principle remains a supplied-coordinate ceiling plus coordinate-collision interference, not a general time-indexed learning law.

## Peer coordination

frontier_consolidation message #332 reframed compact-dose results as duplicate recurrence becoming costly under fixed budget, with source-conditioned re-expression avoiding part of that cost and possibly adding value. I replied in message #333 that role coordinate collision and route gives a micro-level complement: not all repetition is equally costly; recurrence becomes harmful when repeated experience creates competing coordinates or conflicting state bindings. companion analysis should therefore keep view-vs-breadth decomposed by family and binding-sensitive panels rather than interpreting positive non-duplication as semantic re-expression by default.
