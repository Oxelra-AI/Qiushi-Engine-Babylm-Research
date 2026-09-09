# binding feasibility and route implications — binding mechanism route correction

## Interpretation limits of the source-swap antecedent probe

The +2.44 nats correct-antecedent advantage measured in source swap antecedent probe is most likely a **surface copy effect**: the model predicts a later occurrence of "Sire" better when "Sire" appears earlier in context, because it can literally copy from context. This is token-repetition sensitivity, not entity–state credit assignment.

To separate copy from genuine binding:
1. Both conditions must preserve **the same entity words and bag-of-words** — you cannot remove the answer word from context.
2. Only the **entity→property/location/action binding** should be disrupted.
3. The prediction target must be a **downstream state or relational content that depends on the binding**, not the repeated entity name itself.

## Example of the distinction

**Copy-based (invalid as entity mechanism):**
- Text: "King ordered ... King then ___" → mask second "King" → correct context has "King" earlier → copy.
- Swapping "King" for "Queen" earlier removes the answer from context, not just the binding.

**Binding-based (valid mechanism):**
- Text: "The red ball is in the box. The blue ball is on the shelf. John picks up the red ball from ___."
- Target: "box" (depends on red-ball→box binding).
- Control: "The red ball is on the shelf. The blue ball is in the box." → same words, same bag, only binding changed → now "box" is wrong.

## Consequence for the training design

Do NOT launch a 10M run with the anchored masking that targets later entity-name mentions. That route would amplify surface repetition prediction, not entity tracking.

Instead, build a **binding-dependent state prediction** objective:
1. Find windows with ≥2 entities and distinct attributed states/locations/actions.
2. Define targets as the state/location/action tokens that depend on correct entity binding.
3. Arm A (correct binding): text with intact bindings; mask downstream state tokens.
4. Arm B (binding swapped): same words and bag, but swap entity–state assignments; same masked positions.
5. Both arms predict the SAME target positions. Arm A gives the model correct binding context; Arm B gives wrong binding context for the same target.

If training from random init on Arm A improves Entity/EWoK over Arm B, the mechanism is verified as binding-dependent credit assignment.

## Feasibility question

Does the official BabyLM corpus contain enough natural multi-entity binding structures to materialize this objective at useful scale? The next work is a binding-pattern scanner on `high_entity_state.jsonl`.

## What the valid probe must show

A binding-swap probe (not name-swap) on the protected 100M model: mask a STATE token and compare its likelihood under correct-binding context vs. swapped-binding context (same words, same bag). If positive, the model already tracks bindings to some degree and the training objective can amplify this. If null, the starting model has no binding sensitivity and the training must teach it from scratch (harder, but still viable if the corpus contains enough binding examples).
