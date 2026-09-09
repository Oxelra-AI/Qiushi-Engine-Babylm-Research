# earlier analysis — Same-Entity Cross-Clause Hard Negative Design

## Why intra-simplification entity swaps failed

related experiments tried to create hard negatives by swapping entities WITHIN a single
simplified sentence. This failed because:

1. Most simplified sentences have 0-1 named entities (no swap target).
2. When 2+ entities exist, they rarely match on NER type + syntactic role + number + gender.
3. Strict compatible swaps yielded only 8% (422/5202 pairs), insufficient for 100k words.
4. Permissive swaps produced anomaly detection (cross-type/role/gender swaps).

The fundamental problem: intra-simplification entity swapping is too constrained
for FineWeb-Edu educational text where sentences typically focus on ONE entity's
attribute/action/role.

## New design: same-entity cross-clause hard negatives

### Core idea

Educational documents discuss the same entity across multiple sentences with
different predicates/attributes/events. Use THIS natural structure as the hard
negative rather than manufacturing artificial entity swaps.

### Construction

For a document containing entity E discussed in sentences S_A and S_B:

| arm | content |
|---|---|
| **true_pair** | original(S_A) + simplification(S_A) |
| **hard_negative** | original(S_A) + simplification(S_B) |

Both contain entity E in the original AND in the simplification.
Both are about the same topic and use the same vocabulary family.
Both are naturally readable and grammatically correct.
The ONLY difference: the simplification restates the SAME fact as the original
(true pair) versus a DIFFERENT fact about the same entity (hard negative).

### How the design satisfies the experimental constraints

| requirement | how satisfied |
|---|---|
| Same topic | Same document |
| Same entity set | Same entity E in both original and simplification |
| Same vocabulary/lexical overlap | Same document style, same entity names |
| Same entity type/role/number/gender | Literally the SAME entity |
| Broken entity-attribute/event correspondence | Different predicate/fact about E |
| Natural readability | Real text about real entity from real document |
| Not anomaly detection | Both sentences are real, grammatical, coherent |

### Example

Document about Stuart Brown:
- S_A: "Stuart Brown, president of the National Institute for Play, was speaking at the New York Public Library."
- S_B: "Brown has studied the role of play in animal behavior for more than twenty years."

True pair: original(S_A) + simplification(S_A) = 
  "Stuart Brown, president of the National Institute for Play, was speaking..." +
  "Stuart Brown is president of the National Institute for Play."

Hard negative: original(S_A) + simplification(S_B) =
  "Stuart Brown, president of the National Institute for Play, was speaking..." +
  "Brown studied the role of play in animal behavior."

Both contain Brown, both are about play/research, both are natural.
Only the specific fact/binding differs.

### Yield estimation

earlier analysis extracted 5,296 pairs from 1,007 basic-pass docs (~5.3 pairs/doc).
Many documents discuss the same entity across 3-10 sentences.
If even 30% of pairs share an entity with another pair from the same document,
that's ~1,600 hard-negative-eligible pairs from 5,000 pairs — well above 100k words.

For 1M words: likely 15,000-25,000 FineWeb docs needed. Feasible.

### Implementation plan

1. Stream FineWeb-Edu documents.
2. For each document, generate ALL simplifiable sentences (same rules as earlier analysis:
   appositive, relative clause, passive-to-active).
3. Group simplifications by shared entity: for each named entity E, collect all
   (original, simplification) pairs mentioning E.
4. For each entity group with ≥2 pairs from different sentences:
   - Each pair becomes a true-pair item.
   - Each pair also gets a hard-negative: original_A + simplification_B where B
     is from another sentence about the same entity E.
   - Record which entity is shared and what predicate/fact differs.
5. Pack into arms:
   - `true_pair_adjacent`: original + its own simplification
   - `hard_negative_same_entity`: original + different-fact simplification about same E
   - `orig_only`: originals only (source baseline)
   - `shuffled_pair_adjacent`: original + unrelated simplification (weak negative, preserved)

### Quality controls

- Both original and simplification must mention entity E by name (not just pronoun).
- The hard-negative simplification must come from a DIFFERENT sentence (different
  sentence index, not just a different simplification of the same sentence).
- Shared content words between original_A and simplification_B should be ≥2
  (ensuring topical overlap beyond just the entity name).
- Reject cases where simplification_B is nearly identical to simplification_A
  (deduplication check: Jaccard on content words < 0.7).

### What this tests scientifically

If `true_pair > hard_negative_same_entity` on Entity/EWoK:
→ Aligned restatement of the SAME fact/binding helps more than exposure to the
  same entity with different facts.
→ This is a transferable experience-organization mechanism: same-binding
  restatement creates better entity-attribute/event representations.

If `true_pair ≈ hard_negative_same_entity`:
→ What matters is multi-sentence entity exposure, not specific binding alignment.
→ Scale entity-rich FineWeb (fineweb relation vs random 1m profile EWoK lever) rather than pair organization.

If `hard_negative > true_pair`:
→ Diverse facts about an entity are more useful than redundant restatement.
→ The leader's simplification-pair effect may be from exposure diversity, not alignment.

### Files

- New materializer: `scripts/same_entity_cross_clause_materialize.py`
- Output: `data/same_entity_cross_clause_smoke/`
- This plan: `plans/same_entity_cross_clause_hard_negative.md`
