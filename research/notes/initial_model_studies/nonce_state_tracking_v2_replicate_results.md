# nonce state tracking v2 replicate results — corrected nonce-state v2 replication results

## Evidence files

- Corrected pilot/replication source: `scripts/nonce_state_tracking_v2.py`
- Five-seed runner: `scripts/nonce_state_tracking_v2_replicate.py`
- One-seed pilot JSON: `data/nonce_state_tracking/v2_pilot.json`
- Five-seed replication JSON: `data/nonce_state_tracking/v2_replicate_5seed.json`
- Prior failed fixed-label run note: `notes/nonce_state_tracking_failed_run.md`

## Why v2 is the first interpretable micro-world evidence

The nonce state tracking failed run first run was rejected because it had no actual query mask, collapsed unseen states to class 0, and produced WESS NaNs from all-masked padded events. The v2 repair changed the representation:

- The task is episode-local four-way candidate-state retrieval, so unseen nonce state strings are evaluable.
- Episodes are structured event streams: `[EV] entity state ... [Q] queried_entity`.
- Endpoint and step-state arms read from the Transformer query representation and score against the episode's four candidate state embeddings.
- WESS receives the same event stream as entity/state symbols, uses gold participant routing, updates persistent entity slots with a GRU, and reads only the queried entity slot.
- WESS interventions directly alter slots/events: final slot swap and last-write ablation.

This is a controlled structured-interface mechanism test, not yet BabyLM natural-language pretraining evidence.

## Five-seed aggregate results

Random baseline for four candidate states: 25%.

| split | endpoint | step-state | WESS |
|---|---:|---:|---:|
| iid | 0.790 ± 0.030 | 0.757 ± 0.010 | **1.000 ± 0.000** |
| new entity | 0.731 ± 0.027 | 0.705 ± 0.040 | **1.000 ± 0.000** |
| new state | 0.500 ± 0.046 | 0.378 ± 0.030 | **0.866 ± 0.063** |
| new entity + new state | 0.521 ± 0.028 | 0.360 ± 0.039 | **0.848 ± 0.044** |
| long sequences | 0.335 ± 0.016 | 0.318 ± 0.018 | **1.000 ± 0.000** |
| heavy overwrite | 0.361 ± 0.016 | 0.373 ± 0.056 | **1.000 ± 0.000** |

WESS causal interventions on the overwrite split:

| intervention | WESS result |
|---|---:|
| Slot swap transfer | **1.000 ± 0.000** |
| Last-write ablation reverts to previous state | **1.000 ± 0.000** |

Parameter counts: endpoint/step-state 167,616; WESS 107,808. WESS is not winning by parameter count.

## Scientific interpretation

### Main finding

A persistent entity-indexed update/readout mechanism solves the role-binding update problem in a structured nonce-symbol micro-world, including:

- unseen entity names;
- unseen state names through episode-local retrieval;
- unseen entity+state combinations;
- long sequence extrapolation;
- heavy overwrite;
- causal slot-swap transfer;
- last-write ablation reversion.

Endpoint-only and random intermediate step-state supervision do not solve the update-and-overwrite algorithm: they fit IID/new-entity cases reasonably, but collapse toward near-random on long and overwrite splits and only partially handle unseen states.

### What the result says about the mechanism

The difference between endpoint/step arms and WESS is not just supervision amount. In v2, the WESS route structurally matches the algorithm:

1. each event writes a state into one entity slot;
2. nonparticipating slots persist;
3. a later overwrite replaces the previous state;
4. query reads the indexed slot;
5. candidate retrieval allows the same rule to apply to unseen state tokens.

The perfect slot-swap and write-ablation interventions are the first causal evidence in these experiments that a model output can be made to depend on an entity-indexed state variable rather than on whole-passage co-occurrence.

### Scope limitation

This is **not** yet a BabyLM result and not yet a natural-language DeBERTa-v2 interface result.

Important simplifying assumptions in v2:

- Events are structured triples `[EV] entity state`, not free text.
- WESS has gold participant routing and direct access to the event's entity/state symbols.
- Candidate-state retrieval is episode-local; it tests variable binding and state update, but not unrestricted vocabulary MLM.
- The step-state supervision arm was not a full explicit table interface; it used additional intermediate Transformer queries and may be a weak credit-assignment baseline.
- Coreference, span detection, event parsing, and natural-language state extraction are not solved.

Therefore the correct conclusion is not that WESS is ready for BabyLM-scale training. The correct conclusion is that the role-binding hypothesis is alive and now has a concrete, causally validated micro-world mechanism: persistent entity slots with explicit write/read credit assignment can generalize where standard endpoint Transformer training fails.

## Next scientific work

Before BabyLM training, WESS must be moved one step closer to the target regime:

1. Keep episode-local candidate retrieval, but replace `[EV] entity state` triples with short natural-language events such as `dax moves to lup` and parse entity/state spans from tokens.
2. Remove gold route gradually: compare gold route, mention-detected route, and learned route.
3. Add stronger baselines:
   - explicit full state-table supervision with a proper entity-indexed table interface;
   - parameter-matched Transformer;
   - WESS without slot bottleneck;
   - WESS with shuffled routing.
4. Add more causal measurements:
   - persistence rate for nonparticipating slots;
   - selective write accuracy per event;
   - entity/state renaming equivariance;
   - state-swap and event-order controls.
5. If the natural-language interface works with causal interventions, then design a BabyLM-compatible auxiliary training route where all synthetic words count under the 10M budget and equal-word WWM/shuffled-route controls are trained.
