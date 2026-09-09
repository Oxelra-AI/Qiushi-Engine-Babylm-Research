# deeper acquisition probe design Entity Overlap Analysis

## Critical finding: held/train entity overlap

The 120-pair relation-first dataset has massive entity overlap between train and held:
- 30/49 held entities also appear in train
- 25/30 held pairs share at least one entity with train
- Only 5/30 held pairs have BOTH entities unseen in train
- 12/30 have BOTH entities seen in train
- 13/30 have exactly one entity seen

Entity values are CONSISTENT: when an entity appears in both splits, it has
the same source value (same death_place, birthplace, etc.). This means the
model could memorize entity→value associations during training.

## Decomposition by entity familiarity

| Category                    | Count | Transfer meaning                          |
|-----------------------------|-------|------------------------------------------|
| Both entities unseen        | 5/30  | True generalization (no memorization possible) |
| Exactly one entity in train | 13/30 | Partial help from memorization            |
| Both entities in train      | 12/30 | Full memorization possible                |

## Implications for the 28/30 result

The 28/30 four-condition held success at e80 needs to be decomposed:
- For familiar-entity pairs: success could come from memorized entity→value
  associations rather than general entity-conditioned retrieval
- For both-unseen pairs: success requires a general computation

## What the three-way and reassignment probes reveal

### Three-way (cross-source discrimination):
- Familiar entities + high cross-source margin → memorization
- Unfamiliar entities + high cross-source margin → general retrieval
- Both + low cross-source margin → update-recipient gating (not retrieval)

### Source-reassignment:
- Familiar entities + follows reassignment → reads from context (not memorization)
- Familiar entities + ignores reassignment → memorization dominates
- Unfamiliar entities + follows reassignment → genuine context-based retrieval

The reassignment test is especially powerful for familiar entities because
memorization would predict the ORIGINAL value, not the swapped value.

## Both-entities-unseen held pairs

```
rf_death_place_13141: Klutznick(Chicago) vs Ehrenreich(Alexandria), shared_new=Lisbon
rf_death_place_8956:  Wagamese(Kamloops) vs Anselmo(Jundia), shared_new=Sydney
rf_death_place_8850:  Ghedini(Milan) vs Ehrenreich(Alexandria), shared_new=Vienna
rf_birthplace_13717:  Yeung(Hong Kong) vs Davidson(Riverside), shared_new=Tokyo
rf_birthplace_13725:  Yeung(Hong Kong) vs Afeez Oyetoro(Iseyin Local Government), shared_new=Athens
```

Note: Ehrenreich appears in two different held pairs (8850 and 13141), and Yeung
appears in two held pairs (13717 and 13725). So there are only 7 unique unseen
entities across the 5 both-unseen pairs.

## Next steps

1. Decompose three-way results by entity familiarity once the three-way probe is complete
2. Report entity overlap as a bounded caveat on all held success claims
3. For future data construction: enforce entity disjointness in the held split
