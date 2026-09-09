# earlier analysis — same-bag binding learning-response experiment contract

## Purpose

Test whether a loss that explicitly changes under same-word-bag entity→state swaps can induce binding behavior. This is a small method experiment before any BabyLM-scale corpus replacement.

The ideal binding dependency probe probe showed current protected WWM representations do not use entity→state assignment on the template family. That is a pretraining-inference observation. The needed next evidence is training response: after optimizing a binding-dependent loss, does held-out binding sensitivity increase relative to matched WWM and random-label comparisons?

## Best first objective: masked-target contrastive context loss

Use the MLM head so the experiment stays close to BabyLM MLM evaluation.

For each generated pair:

- `x+`: correct setup plus query with the target state/location token present.
- `x-`: same word bag, entity assignments swapped in setup, query unchanged, same target token present.
- The downstream target span in the query is masked in both contexts.

Compute:

\[
\ell^+ = \log p_\theta(t \mid \operatorname{mask}_t(x^+)),
\quad
\ell^- = \log p_\theta(t \mid \operatorname{mask}_t(x^-)).
\]

Training loss:

\[
L = L_{WWM}(x^+) + \lambda \max(0, m - \ell^+ + \ell^-).
\]

Use `m=0.5` and `lambda=0.5` as the first setting. The loss is low only when the same target is more predictable in the correct-binding context than in the swapped-binding context, despite both contexts containing the same words.

## Template design requirements

Avoid the source swap antecedent probe/173 copy failure mode:

- no entity-name targets;
- target must be a state/location/property word;
- query must not include an object word that locally determines the target independently of entity assignment;
- both correct and swapped contexts must have exactly the same word multiset;
- target surface and token positions in the query should match between correct and swapped examples;
- include distractor entities, overwritten states, and held-out names/locations.

Good pattern:

- Correct: `Alice placed a stone in the box. Bob placed a shell in the basket. Later Alice returned to the box.`
- Swapped: `Bob placed a stone in the box. Alice placed a shell in the basket. Later Alice returned to the box.`
- Mask target: `box` in the final query sentence.

Here `box` appears in both passages with the same frequency and local context. The only difference is whether `Alice→box` is true.

Bad pattern:

- `Later Alice got the stone from the box.`

because `stone→box` can solve the target without using Alice's state.

## Four matched learning arms

Use a small DeBERTa-v2 MLM for speed, e.g. 2 layers, hidden 128, 4 heads, seq 64 or 96. Same tokenizer, same initialization seed, same number of examples, same number of updates, same batch size.

1. `wwm_correct`: WWM on correct passages only.
2. `wwm_both`: WWM on correct + swapped passages, no binding labels.
3. `random_pair`: WWM + same contrastive loss but random positive/negative orientation per pair.
4. `true_binding`: WWM + true contrastive loss with correct context as positive.

The random-pair arm checks whether the ranking head simply regularizes or adds noise; the `wwm_both` arm checks whether seeing both word bags without a binding signal is enough.

## Pre/post measurements

Before training and after training, on held-out examples:

- `margin = log p(target | correct context) - log p(target | swapped context)`;
- binding-choice accuracy: fraction where margin > 0;
- mean margin by template family;
- mean margin on held-out entity names;
- mean margin on held-out locations/states;
- overwritten-state examples where an entity changes from one location to another;
- distractor examples with a third entity/state pair.

The desired learning response is:

- `true_binding` substantially increases held-out mean margin and binding-choice accuracy;
- `wwm_correct`, `wwm_both`, and `random_pair` remain near pretraining margin;
- improvement holds on held-out names/locations and at least one held-out template family.

This is the needed evidence that the objective, not memorized surface templates, induces binding sensitivity.

## If the small experiment works

Next, embed the same objective into the S1-shape BabyLM trainer:

- all episode text words counted under the 10M limit;
- include an equal-word WWM-only episode control and random-pair contrastive control;
- train seed 42 plus one replication seed;
- evaluate internal binding margin plus BabyLM BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, EWoK.

## If it fails

If true same-bag contrastive loss does not produce held-out binding margin, the next route should not be more data. It should be an architecture/state route, such as persistent entity/event slots or another mechanism that carries assignment state across the sequence.
