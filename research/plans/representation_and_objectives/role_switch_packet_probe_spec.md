# route portfolio context alternative binding plan — balanced role-switch packet probe (no H100 training yet)

## Purpose

Create an evaluation-independent, legally accountable text experience object where the same alternatives exchange roles under context changes. The object is not a training route until it passes no-training checks: useful WWM target density, context-dependence, target-prior cancellation, held-out family/entity transfer, and no official-text provenance leak.

This is designed to be different from corpus directional pair census: it is not a corpus-mined pair funnel, not an auxiliary perturbation detector, not local pivot-substitution, and not a saturated attested-fit objective. It changes the primary text experience distribution at legal word cost only if the object survives frozen tests.

## Inputs allowed

- Vocabulary lists derived only from the allowed training corpus, preferably common concrete nouns, containers, places, simple verbs, colors, and neutral names already present in the legal 10M pool.
- Hand-written grammar templates authored without consulting official evaluation rows or labels.
- No pretrained parser, POS tagger, external corpus, hidden teacher logits, or official evaluation text.

## Candidate family design

Each item is a short natural mini-discourse, not a symbolic `A above B -> A high` transform. The masked target in later tests should be a consequence phrase whose correct participant or state changes when the context is inverted.

### Family 1: vertical/support relation

Roles: two common objects or people, relation above/below/on top of/under/beneath.

Template pair examples:

- `The {obj_a} is on the shelf above the {obj_b}. The object in the higher place is the {obj_a}. The object in the lower place is the {obj_b}.`
- `The {obj_b} is on the shelf above the {obj_a}. The object in the higher place is the {obj_b}. The object in the lower place is the {obj_a}.`

Held-out paraphrase forms should use `over/under`, `top/bottom`, `upper/lower`, and different sentence order.

### Family 2: temporal before/after

Roles: two events or actions, relation before/after.

- `{person} {event_a} before {person} {event_b}. The earlier action was {event_a}. The later action was {event_b}.`
- `{person} {event_b} before {person} {event_a}. The earlier action was {event_b}. The later action was {event_a}.`

Use common simple actions from the training corpus (eat, wash, read, sleep, play, open, close, cook, walk). Held-out families reverse sentence order and use `after` rather than `before`.

### Family 3: transfer giver/receiver

Roles: two people, object transfer.

- `{giver} gave the {thing} to {receiver}. The person who had the {thing} afterward was {receiver}. The person who lost it was {giver}.`
- `{receiver} gave the {thing} to {giver}. The person who had the {thing} afterward was {giver}. The person who lost it was {receiver}.`

Held-out forms use `handed`, `passed`, `lent`, or passive constructions only if grammar remains simple.

### Family 4: open/closed state change

Roles: object and action/state.

- `{person} opened the {container}. After that, the {container} was open, not closed.`
- `{person} closed the {container}. After that, the {container} was closed, not open.`

The same alternatives (`open`, `closed`) exchange under action context. Held-out forms use `shut` or `left open` only if target alternatives remain balanced.

### Family 5: comparative more/less

Roles: two entities and property.

- `{a} had more {items} than {b}. The one with more {items} was {a}. The one with fewer {items} was {b}.`
- `{b} had more {items} than {a}. The one with more {items} was {b}. The one with fewer {items} was {a}.`

Avoid numbers matching official rows; use generic quantities and common nouns. Held-out paraphrases use `fewer than`, `not as many as`.

### Family 6: container in/out

Roles: object and container.

- `{person} put the {thing} inside the {container}. The {thing} was inside the {container}.`
- `{person} took the {thing} out of the {container}. The {thing} was outside the {container}.`

Use paired actions and state alternatives inside/outside.

## Balancing constraints

For each packet family:

- Equal count for each inverse direction.
- Equal first-mention position for role A and role B.
- Equal target-token frequency for each alternative in the generated packet.
- Equal sentence length bins as much as possible.
- No target word appears only in one direction without an inverse packet.
- Avoid template lexical hints that trivially map one pivot to one target without participant binding; include both pivot directions and alternative sentence orders.
- Reserve at least one family and one paraphrase style as held-out within the packet probe before any training.

## No-training tests to implement

### 1. Provenance and duplicate check

- Save packet manifest with generator code hash, vocabulary source files, random seed, family, template ID, entity IDs, relation direction, word count.
- Compare generated 7/8/10-token spans against official evaluation text already cached under `data/pristine_official_coordinate/.../evaluation_data`. Reject exact overlaps beyond trivial stopword spans; manually inspect any content overlap.

### 2. WWM useful-density simulation

Using the trusted baseline tokenizer/collator:

- Tokenize every packet sentence with the legal tokenizer of the candidate coordinate.
- Simulate the same WWM process over 10 passes for fixed seeds.
- Count all masked word groups and the subset whose prediction requires role-conditioned resolution: participant names in consequence sentences, state alternatives (`open/closed`, `inside/outside`, `higher/lower`, etc.), transfer receiver/giver words.
- Report `D_useful = useful_role_masked_events / total_packet_words_exposed`, per family and target class.
- Reject if useful density is too low for a small replacement block to matter.

### 3. Frozen checkpoint exchange scoring

On frozen existing checkpoints (at minimum legal40k fixed-256 100M, curriculum 100M, and if possible legal16k aoa mincontext discrepancy audit 100M or 20M):

- Score paired contexts and target alternatives with the same pseudo-log-likelihood backend as official evaluation.
- Compute target-main-effect-cancelled margin:
  `M = [s(C0,T0)-s(C0,T1)] + [s(C1,T1)-s(C1,T0)]`.
- Compare against target-pair permutation, context permutation, and context-erasure controls.
- Score target-only and local-window contexts. The full context must add information beyond target-only/local-window scores.
- Current endpoints must be unsaturated: if margins are huge positive and all contexts correct, training would likely repeat saturated attested-fit objectives.

### 4. Held-out transfer structure

- Split generated cases by family, paraphrase style, and entity vocabulary.
- Frozen scoring should be reported separately on train-style vs held-out-style packets. If one later trains, the held-out styles/entities are the primary packet readout.
- A training screen is meaningful only if held-out family packet accuracy improves and real EWoK variable-swap/stable-failure rows also improve.

## Legal replacement-pool concept, not yet to build

If the packet passes the tests, construct legal pools by replacing, not appending, equal word mass. The cleanest later training comparison has three arms:

1. original compact-view legal pool;
2. equal-mass role-switch replacement pool;
3. equal-mass natural-text replacement pool from similar source/length bins.

The third arm isolates whether effects come from role-switch structure rather than removing a particular source distribution. If only one H100 screen is possible, at least quantify the removed text's source labels, lengths, token frequencies, and existing-score sensitivity before replacing it.

## First Execute deliverables

Only no-training construction artifacts are proposed:

- `scripts/role_switch_packet_builder.py`
- `data/role_switch_packets/packets.jsonl`
- `data/role_switch_packets/packet_manifest.json`
- `scripts/role_switch_density_and_frozen_probe.py`
- `data/role_switch_probe/` with density and frozen scoring summaries

No H100 training should start from this plan until the object passes the tests above and paired-tail tasks have been read.
