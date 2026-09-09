# earlier analysis — Refined R1 route: state-dynamics experience for binding-variable learning

## Research purpose

The step100 204 systematic compression compression showed a repeated failure pattern: many interventions can move GlobalPIQA or small local columns, but Entity/EWoK do not improve together. The unresolved mechanism is how to make a legal BabyLM learner form reusable entity/relation/state variables from unlabeled text, without gold addresses and without a custom inference mechanism that disappears at export.

R1 is the next primary route because it changes the training experience itself. It tries to remove the shortcuts that plain WWM can exploit and make ordered entity-state composition useful for ordinary masked prediction.

## Mechanism statement

A protected DeBERTa-v2 WWM learner has not learned robust entity/state binding because official and natural text often allow prediction from lexical frequency, local plausibility, recent mention, or surface repetition. If part of the 10M-word training experience is replaced by legally generated state-dynamics passages where the masked target depends on composing ordered operations over entities, containers, and attributes, with entity identities relexicalized and frequency flattened, the ordinary backbone may learn a reusable position/binding variable. This should first improve official Entity Tracking; EWoK/GlobalPIQA improvement is a predicted transfer, not guaranteed.

## Why R1 is different from closed routes

- It is not WESS: there is no gold route at inference, no slot module, and no custom exported mechanism. The ordinary checkpoint must carry any benefit.
- It is not structure-density selection: instead of selecting natural windows with more propositions, it constructs distributions where local and frequency shortcuts are weak.
- It is not simplification-pair training: the learning pressure is ordered operation composition, not lexical redundancy.
- It is not the surface adapter: representation sharing is not patched into embeddings; the data distribution itself is changed.
- It does not accept GlobalPIQA-only movement as progress. The decisive target is Entity plus a binding probe, with EWoK monitored.

## Key interpretation risk

Generated procedural text alone is not enough. The masked targets must really require the final state, not just the nearest operation or the token bag.

A generated example is useful only when, after controlling for entity bag, local window, last operation, entity frequency, and recent mention, the target token still depends on the composed final state.

Good generated cases include:

- multiple moves of the same object;
- shared entities/containers where operations do not commute;
- final state different from initial state;
- same token multiset but different legal operation order and different final answer;
- distractor entities undergoing similar operations;
- target entity not mentioned immediately before the answer;
- operation chains that cannot be solved from the last relevant operation alone;
- several surface domains: boxes/containers, ownership, carrying, rooms/locations, locked/open states, hot/cold or clean/dirty attributes.

## Data legality and source isolation

Before any training, verify current BabyLM Strict-Small rules for self-constructed corpora and generated text. Generated words count toward the 10M-word corpus budget and exposure. The generated corpus should replace official words, not add to them.

The generator must not use official Entity Tracking or EWoK files, templates, object lists, operation-word lists, or generation scripts. It should use independently designed state domains and vocabulary. Before training, run source-isolation comparisons against local official evaluation data:

- exact sentence overlap;
- normalized n-gram overlap;
- delexicalized template overlap;
- object/container/state word overlap;
- operation-sequence form overlap;
- query wording overlap;
- graph/trajectory form overlap where easy to compute.

Save generator code, random seeds, word counts, source word lists, and all filtering results.

## First experiment design

Backbone: protected DeBERTa-v2 8×480 WWM, baseline16k tokenizer. Use this first because it is the only complete internal 9/9 reference and has better Entity/EWoK/SuperGLUE/Reading than S1.

Training scale: 10M word exposure first, with full checkpoint sequence retained (`chck_1M`–`chck_10M`) so AoA-compatible practice is restored. If later scaled to 100M, save `chck_1M`–`chck_10M` and every 10M to 100M.

Mixing: start with 10–15% generated words replacing official words. If generator target quality is very high, 20% may be tested; do not begin above 30% because grammar/Reading/Supplement may be damaged by synthetic style concentration.

Arms:

1. Fresh official-only protected baseline with full early checkpoints.
2. R1 ordered-uniform: official plus generated legal ordered state-dynamics passages, relexicalized and near-uniform entity frequency.
3. Frequency-control: same state programs, but entity surface distribution is Zipfian/fixed. This tests whether frequency and entity identity shortcuts block variable learning.
4. Legal order-control: same entity/operation/token statistics and fluent legal text, but cross-operation state dependence is removed or neutralized while preserving local coherence. Avoid invalid shuffled worlds.
5. Optional general synthetic-control: same amount of fluent procedural-style generated text without cross-sentence state dependence.

The stronger design is a 2×2 comparison: uniform vs Zipfian crossed with ordered-dependent vs legal dependency-removed, then inspect the interaction. If compute only permits three arms, prioritize official-only, ordered-uniform, and legal dependency-removed.

## Required pretraining measurements

Before model training, the generator itself must pass heuristic baselines:

- bag-of-entities baseline;
- last-operation baseline;
- nearest-mention baseline;
- entity-frequency baseline;
- local-window baseline;
- unigram/bigram surface baseline.

If these baselines solve the masked target, the generated data does not create the intended pressure and should be redesigned.

## Model measurements

Short-budget model readout:

- official Entity Tracking;
- full EWoK word-tokenize;
- GlobalPIQA parallel/nonparallel;
- BLiMP, Supplement, COMPS, Reading;
- ordinary MLM loss;
- frozen cross-vocabulary binding probe.

The frozen probe should use held-out generated worlds with disjoint entity words and surface forms. It should compare legal same-token-bag non-commuting operation pairs where operation order changes the final state. A low-capacity probe reads hidden states at query positions and predicts final state. Entity-name permutation should move predictions with the role, not the word identity; order swapping should move predictions with the computed final state.

Interpretation of outcomes:

- Entity and frozen binding probe both improve while Supplement/Reading/BLiMP stay stable: R1 mechanism is live.
- Only GlobalPIQA improves: this repeats the known generic-gain pattern and is not enough.
- Official Entity improves but frozen probe does not: likely surface adaptation to the official task, not the intended variable.
- Probe improves but official Entity does not: domain-internal state program learned but transfer failed; redesign surface/domain diversity before scaling.
- Ordered-uniform and dependency-removed arms both improve similarly: benefit is from general generated text or regularization, not ordered binding.

## Relation to EWoK

R1 has direct mechanism relevance to Entity Tracking. EWoK transfer is plausible only if the generated domains include attribute changes, affordances, causal preconditions, and contrastive world facts, not just object locations. The first R1 version should include a modest attribute/cause component but should not be judged successful by EWoK alone or GlobalPIQA alone.

## Next construction work

1. Verify current BabyLM rules on self-generated corpora and generated text.
2. Build a generator for legal state-dynamics passages with independent vocabulary and exact word accounting.
3. Build heuristic baseline tests on generated masked targets.
4. Build source-isolation comparisons against official Entity/EWoK data.
5. Build the frozen same-token-bag non-commuting-order probe.
6. Only then launch the 10M arms with full checkpoint sequence.

This route is high-risk but scientifically aligned with the compression: it attacks the training-experience distribution that lets the model avoid forming reusable entity/state variables.
