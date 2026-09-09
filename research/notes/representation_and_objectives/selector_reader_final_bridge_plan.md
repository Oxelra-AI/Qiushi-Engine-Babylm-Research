# selector reader bridge synthesis and macro challenge — Final high-information selector–reader bridge plan

## Why this experiment exists

The prephase alignment design and bias analysis prephase result corrected the address bootstrap and role query address-bootstrapping interpretation: aligned, permuted, and disjoint prephases all rescue inline direct-tag training, so that rescue is generic entailment-head calibration rather than reuse of a learned tag→record coordinate. The remaining address bootstrap and role query role-query failure should therefore **not** be elevated into an architectural impossibility. It only shows that one monolithic binary state-classification interface failed to learn role→address→state composition and damaged the direct reader.

The final useful bridge test is to separate the operations that the monolithic interface conflated:

- `R(k, v)`: a direct address-to-state reader. The context contains tagged records; the hypothesis names a tag and a state. This is established first and then held fixed.
- `M(r, k)`: a selector from role language to the tag/address that denotes the relevant record in the same context. This is trained with explicit positive/negative tag choices and does not see state labels.
- `M∘R`: composition. At evaluation, the selector chooses a tag for a role phrase, and the frozen reader is queried with that tag and the state hypothesis. The composed answer must not use an oracle or hand-routed tag.

This is still a small pretrained bridge test, not BabyLM-scale training. It decides whether this supplied-address route contains one more genuine, composable mechanism, or whether the loop should close pending frontier_consolidation's macro aligned-vs-permuted companion result.

## Minimum experiment

Use the existing ATP temporal-change construction and the same support geometry as Steps267–271: base stable support, sparse changed focal rows, sparse stable focal rows, held changed focal plus held stable secondary evaluation, shuffled record order, arbitrary per-context tags, and paired AB/BA contrastive state scoring.

### 1. Establish and freeze reader `R`

Train a direct-tag reader on inline-role contexts:

```text
Focal background entry amber: ...
Focal update entry marble: ...
Separate background entry kiwi: ...
Separate update entry river: ...
[SEP] According to entry marble, X was ranked higher than Y.
```

The reader must fit sparse changed focal rows and recover held direct-tag state readout. If it does not, the construction is invalid and no role-composition conclusion is allowed. After fitting, do not update this reader during selector learning.

### 2. Train selector `M` from role language to tag/address

Train a separate binary selector model on rows of the form:

```text
same context
[SEP] The focal update entry is marble.
```

For each role, include the true tag as a positive candidate and other in-context tags as negative candidates. The selector target is only role→tag identity; it never sees state truth labels. Use class weighting or balanced negatives so overall negative bias cannot masquerade as selector success. Measure selector fit by top-1 tag selection for each role, not just binary row accuracy.

### 3. Compose without hand routing

For each held query role and AB/BA state alternative:

1. score all candidate tags with `M(role, tag)`;
2. choose the top tag;
3. query frozen `R` with the chosen tag and the state hypothesis;
4. compute contrastive state accuracy and margins.

Report both selector top-1 accuracy and composed state accuracy. Also report the oracle-reader ceiling using the true tag, but do not treat oracle routing as the composition result.

## Required eval surfaces

- `held_exact_nsA`: held changed/stable worlds, new tag namespace, exact role wording (`focal background/update`, `separate background/update`).
- `held_exact_nsB`: same held worlds with a second independent tag namespace/permutation. This tests coordinate/tag permutation rather than memorized tags.
- `held_role_swap`: swap focal background/update role labels in the context while keeping record contents fixed; a role-following selector should swap chosen focal tags, and composed focal before/after labels should reverse accordingly.
- `held_para`: query-side role paraphrases (`prior/revised` or similar) against the same exact-role contexts. This separates exact surface role matching from semantic role transfer.

## Interpretation

- `R` fails: the run says nothing about composition; repair reader construction or stop.
- `M` fails on training/held exact selection: the selector interface did not learn role→tag mapping; do not infer model incapacity.
- `M` works and `M∘R` works on held exact and tag namespaces but fails on held paraphrases: decomposition solves exact role-address composition, but semantic temporal/discourse transfer is still absent. The supplied-address bridge remains bounded.
- `M` works on role swap and the composed answer follows the swapped roles while direct-tag reader remains tied to tags: this is genuine selector-reader composition, beyond direct supplied coordinates.
- `M` works but `M∘R` fails despite oracle `R` success: the bottleneck is interface mismatch between selected addresses and the reader query, not selector or reader alone.
- If exact and paraphrase composition both work under held tag permutations and role swaps, the bridge supports a factored-interface principle worth retaining: limited data can install a selector over an established reader when the operations are structurally separated.

## Cost and route decision before GPU use

The lowest-cost reliable action is one seed at the proven fit-capable seed 27000, using the same small bridge scale and one H100. It replaces another monolithic role-query run with a decomposed test that directly localizes the unresolved operation. The result will decide whether companion analysis continues the bridge through factored interfaces or closes the supplied-address loop and waits for macro correspondence evidence. No Strict-Small 100M training or official evaluation is justified from this test alone.
