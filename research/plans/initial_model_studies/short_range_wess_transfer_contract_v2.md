# earlier analysis — revised short-range DeBERTa+WESS transfer experiment

## Purpose

wess v4 address challenge results showed that persistent entity-indexed addressing is a real algorithm in controlled text-rendered micro-worlds. The next experiment must test whether that algorithm can transfer into a DeBERTa-v2 MLM setting and alter masked natural word prediction in a way that depends on correct entity-to-state binding.

The experiment must be short and decisive. It must not be a 100M-word full run. It must also avoid the historical failure mode of these experiments: a model appearing to improve because the target is recoverable from local co-occurrence, recency, position, or direct copying rather than from entity-indexed state.

## Revised central question

Under matched DeBERTa-v2 MLM training on official BabyLM text plus a small amount of controlled, natural-language-rendered entity-state episodes, does WESS with persistent entity addressing outperform both plain MLM and no-address memory on counterfactual paired targets where the only varying fact is which entity owns which state?

A positive result must show all of the following:

1. lower loss / higher accuracy on paired binding targets;
2. stronger pair-level correctness than plain MLM and no-address memory;
3. direction-specific slot interventions: swapped slots move probability toward the other entity's state and away from the original state;
4. destroyed addressing removes the effect;
5. ordinary MLM loss on real official text does not degrade materially;
6. a small Entity-style probe moves in the expected direction or at least does not move against the mechanism.

## Data: controlled counterfactual pair suite, not ordinary natural text

The main transfer set must be generated natural-language episodes with strict balancing. Ordinary sampled natural text can be a supplementary transfer set, but it cannot carry the main evidence because shortcuts cannot be ruled out.

Each paired item contains two or more entities and two or more same-type states, with all candidate state words visible in both members. Members preserve the same templates, event positions, token multiset as closely as possible, distances, and query form. Only the entity-to-state assignment changes, and the answer for the same queried entity changes with that assignment.

Example family:

```text
A: dax moved to the garden. wug moved to the kitchen. dax later moved to the attic. wug moved to the garden. Where is dax? [MASK]
B: wug moved to the garden. dax moved to the kitchen. wug later moved to the attic. dax moved to the garden. Where is dax? [MASK]
```

The exact template should be repaired so that the same queried entity's answer differs between the paired members while all candidate states are present and same-type.

### Construction requirements

For every evaluation family:

- at least two entities and two same-type states;
- every candidate state appears in the visible context in both paired members;
- the queried entity has an earlier state and a later overwritten state;
- at least one other entity update occurs between the queried entity's last update and the query;
- the answer is not always the most recent state, last-mentioned state, first-mentioned state, nearest state, or fixed template position;
- entity names, state words, event order, template choice, distances, and query target are independently balanced;
- each entity appears with multiple states and each state appears with multiple entities across training;
- held-out splits include unseen entity-state combinations, unseen event orders, unseen state words, and unseen paraphrase families;
- paired evaluation reports pair accuracy: both members of a counterfactual pair must be answered correctly.

### Shortcut baselines to run before training

Before any DeBERTa+WESS training, run simple predictors on the constructed evaluation set:

- majority state;
- nearest state to the query;
- last state in the passage;
- first state in the passage;
- bag-of-words logistic or small MLP without entity order;
- local window around `[MASK]`;
- entity names collapsed to one token;
- entity mentions shuffled while other tokens remain.

The set is usable only if these predictors are near the four-choice baseline on the paired score. If a shortcut predictor is strong, rebuild the generator before training.

## Prevent target leakage through slots

Gold spans are allowed for this bridge experiment, but the target token at the query position must be masked before encoding and must never enter the span extractor, slot writer, or candidate representation for that example.

Two subsets must be reported separately:

1. **verbatim retrieval**: the correct answer word appeared earlier as the entity's state;
2. **inference/paraphrase**: the final answer requires a paraphrase or rule-derived state, if such examples are included.

A positive on verbatim retrieval supports entity-indexed retrieval, not general semantic inference. That is still valuable as a bridge result, but it must not be interpreted as full BabyLM Entity ability unless a richer subset also moves.

## Model arms

All arms use the same backbone shape, tokenizer, official text slice, episode mixture, update count, masking schedule, initialization seeds, and word accounting.

1. **plain_mlm** — DeBERTa-v2 MLM on the mixed corpus, no slot path.
2. **matched_no_address_memory** — receives the same gold spans, same state representations, same recurrent/projection parameter budget, and the same MLM fusion interface as WESS, but removes persistent entity addressing. Use a single shared memory, unordered set pool, or per-event reset memory. This is the key control for extra parameters, gold state access, and fusion path.
3. **wess_gold_address** — WESS with persistent entity slots and correct routing.
4. **wess_eventwise_random** — same WESS module but every event writes to a random slot independent of entity identity.
5. **wess_wrong_entity** — same WESS module but every event writes to a different entity's slot.
6. Optional **episode_only_plain_mlm** and **official_only_wess** if runtime permits, to separate episode text from module capacity.

Destroyed-address arms are necessary but not enough; the matched no-address memory arm is required because it controls for gold spans, content access, extra recurrence/projection, and the MLM fusion interface.

## Training scale

This is a short transfer run.

Record separately:

- unique corpus whitespace words;
- cumulative word exposure;
- epochs over each component;
- optimizer steps;
- masked-token count;
- synthetic episode word count.

Use a small but real short range such as 1M unique mixed words for 1–3 epochs, or a 5M–10M cumulative exposure slice, with every arm matched. Do not mix up unique words and cumulative exposure. Do not launch a 100M candidate until the measurements below are clear.

## Measurements

### Binding target measurements

For each counterfactual pair:

- per-example log probability of the correct state;
- pair accuracy: both paired members correct;
- log-odds of correct state versus counterfactual swapped state;
- performance by recency group: correct answer last, not last, far, near, after distractor update;
- performance by held-out group: new entity-state combinations, new states, new order, new paraphrase.

### Slot intervention measurements

Only use examples with two same-type states and a clear counterfactual answer. If entity `e_i` has state `v_i` and another entity `e_j` has state `v_j`, slot swap should be scored by the directional change:

```text
[log p(v_j) - log p(v_i)] after swap minus [log p(v_j) - log p(v_i)] before swap
```

A good intervention result should show:

- `p(v_j)` rises;
- `p(v_i)` falls;
- top-1 prediction transfers to `v_j` often above baseline;
- sham swaps and nonquery-slot swaps do not show a consistent direction;
- random-route and wrong-entity arms do not show coherent transfer.

Write-ablation is scored similarly: remove the last write for the queried entity and require probability to move from the new state toward the previous state.

### Official-like measurements

At the end of the short run:

- general MLM validation loss on official text;
- binding-critical held-out loss/accuracy;
- a small Entity Tracking slice or local proxy using the same scoring style as official evaluation;
- if a checkpoint looks promising, run the fast official columns available locally before any full expansion.

## Promotion to full candidate

Move to a full 100M nine-column candidate only if:

- `wess_gold_address` beats plain MLM and matched no-address memory on paired binding score across seeds;
- destroyed-address arms lose the advantage;
- slot swap and write-ablation move probabilities in the predicted direction;
- shortcut baselines are near chance on the test suite;
- official-text MLM loss is not materially harmed;
- the Entity-style probe is non-negative and preferably positive.

If only verbatim retrieval improves, the next work is sharper span/routing and richer state inference, not immediate SOTA scaling. If the matched no-address memory also improves as much as WESS, the effect is retrieval/fusion rather than entity-indexed state, and the WESS route must be redesigned before scale.

## Builder notes

Start from `babylm_masked_train_leadershape.py` for tokenizer, DeBERTa-v2 config, word accounting, masking, checkpoint conventions, and evaluator compatibility. Start from `wess_v4_address_challenge.py` for slot routing modes and interventions.

The first builder should produce:

1. episode generator with paired metadata and shortcut-baseline report;
2. short-range trainer with the six matched arms above;
3. binding-target evaluator with pair score and log-odds;
4. intervention evaluator on the same paired suite;
5. a compact JSON result and a route note stating whether the mechanism transfers into MLM.
