# route2 factorial causal review — Route 2 factorial/causal review

## Scientific question

route3 correspondence closure found a striking hidden-state fact: a linear readout trained on action-only `last_event` states transferred to contradictory-prior hidden states even when the MLM head scored the contradictory-prior crossed condition at 0.0. The route2 factorial causal review review tested whether that hidden signal is an entity-bound result-state update or only a family/template/action cue.

No model weights were changed. All tests used existing checkpoints and synthetic/factorial forward passes only.

## Files

- First factorial script: `experiments/archive/representation_and_objectives/scripts/factorial_entity_binding_probe.py`
- First factorial output: `experiments/archive/representation_and_objectives/data/factorial_entity_binding_probe/chck82_scale1p75/factorial_entity_binding_probe.json`
- Stricter paired-interaction script: `experiments/archive/representation_and_objectives/scripts/pair_interaction_causal_probe.py`
- Stricter paired outputs:
  - `experiments/archive/representation_and_objectives/data/pair_interaction_causal_probe/chck82_scale1p75/pair_interaction_causal_probe.json`
  - `experiments/archive/representation_and_objectives/data/pair_interaction_causal_probe/legal16k_base100/pair_interaction_causal_probe.json`
  - `experiments/archive/representation_and_objectives/data/pair_interaction_causal_probe/scale1p75_100M/pair_interaction_causal_probe.json`

## What the first factorial test established

The test held action/state tokens and sequence form largely fixed while swapping the affected entity and queried entity. On `chck82_scale1p75`:

- Baseline accuracy is exactly 0.5 because the model always follows the prior state:
  - unaffected query cases: accuracy 1.0, mean margin +5.782
  - affected query cases: accuracy 0.0, mean margin -5.640
- The hidden state contains a family-general code for whether the queried entity was the affected entity:
  - affected-query decoder, held-out state/action families: best 0.825 at layer 6
  - permutation nulls stay near 0.5
- But it does not contain a family-general final-state polarity readout:
  - final-pol decoder, held-out families: best 0.541
  - held-out action directions do not rise above chance in a useful way
- A causal patch/removal of the affected-query component at layers 4/5/6 did not recover output behavior. Accuracy stayed 0.5 for all modes; affected-query cases remained wrong and unaffected-query cases remained correct.

This means the model can represent “this is the acted-on entity” in the synthetic setup, but that is not enough to produce an entity-bound state update.

## What the stricter paired-interaction test established

The stricter test compared pairs with the same queried entity and the same action/state tokens, differing only in whether the action targeted that queried entity or the other entity. It then decoded the event/result polarity from the paired difference `h(query affected) - h(query other affected)` with state/action families held out.

Results across three checkpoints:

| model | same-query context-diff best held-out-family event-pol acc | same-context query-diff best held-out-family event-pol acc | causal held-out-family accuracy under patch/removal |
|---|---:|---:|---:|
| `chck82_scale1p75` | 0.546 (layer 8; nulls overlap) | 0.500 | 0.500 for all layers/modes |
| `legal16k_base100` | 0.500 | 0.500 | 0.500 for all layers/modes |
| `scale1p75_100M` | 0.517 | 0.506 | 0.500 for all layers/modes |

The paired-update component amplitudes are small and not output-aligned. For chck82 the train-family paired-direction amplitudes were approximately 0.003, 0.022, and 0.059 at layers 4, 5, and 6. Add/random/swapped patches did not move held-out-family accuracy; removals changed mean margins slightly at layers 5/6 but did not selectively repair affected cases.

## Interpretation

The route3 correspondence closure layer-4/6 to layer-7/8 reversal remains a real hidden-state phenomenon, but route2 factorial causal review shows it should not be read as an entity-bound result-state code. Renaming in route3 correspondence closure preserved action verbs, state family, template, and mask geometry. Once entity binding is factorized by holding tokens and geometry fixed and asking for paired interaction differences, the family-general event/state update signal disappears.

The behavior is sharper than a generic “readout failure”: these checkpoints strongly copy the prior state for both entities. They know, internally and family-generally, which queried entity is the acted-on entity, but they do not compose that affected-entity fact with the action-result polarity into an output-usable final state on held-out state/action families. A late patch or protected readout from this synthetic component is therefore not justified.

## Consequence for the route

This closes the current Route-2 instantiation as a late causal-readout or patch route. It does **not** close the broader scientific problem of entity/event/time-binding computation; instead it localizes the missing operation more precisely: the main path must learn a compositional state-update operation that binds `(entity, action, result state, time)` before the final MLM head, not merely preserve an affected-entity feature or an action-family feature.

No architecture training, endpoint mutation, synthetic-ladder training, or official-row patching is justified from the current component. The next scientific route should either rebuild the mechanism at the main-path training/architecture level with a non-synthetic official-relevant substrate, or shift near-term execution to complete the official score evidence for the strongest practical endpoint while a new mechanism route is selected.
