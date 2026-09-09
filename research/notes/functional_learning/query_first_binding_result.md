# query first binding compact summary result: query-before-context order enables binding under answer-only training

## Why this test followed loss allocation preliminary result

Step015b repaired the answer-loss allocation test and showed that, in the original causal order, answer pressure alone does not rescue entity-attribute binding. Even answer-only training to 1000 epochs learns context-bag membership (`bag_mass≈0.986`) without query-conditioned selection (`top4≈0.273`, B-swap≈0, Q-swap≈0, corruption selectivity≈0).

This left a sharper possibility: the problem may be temporal/causal credit assignment rather than scalar answer-token allocation. In the original order,

```text
BOS  E0 HAS A0  E1 HAS A1  E2 HAS A2  E3 HAS A3  SEP  QueryEntity  IS  Answer
```

the context-token states cannot condition on the query while they are formed; the answer position must perform the entire entity match and attribute routing at the end. query first binding compact summary tested a query-first order,

```text
BOS  QueryEntity  SEP  E0 HAS A0  E1 HAS A1  E2 HAS A2  E3 HAS A3  IS  Answer
```

so each context triple can in principle be processed in the presence of the query. The architecture, vocabulary, K=4 orbit task, and direct binding probes remain otherwise matched.

## Script and data

- Script: `scripts/query_first_binding.py`
- Data: `data/query_first_binding/results.json`
- Figure: `figures/query_first_binding.png`
- Preceding repaired allocation result: `notes/loss_allocation_result.md`

## Direct result

| Arm | order/objective | final correct NLL | bag mass | top4 | query margin | B-swap | B-swap frac | Q-swap margin | Q-swap frac | Q-swap both | query-novel selectivity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| orig_ans_only_500 | original / answer-only | 1.487 | 0.981 | 0.267 | +0.013 | -0.003 | 0.479 | +0.000 | 0.493 | 0.003 | +0.001 |
| qfirst_full_500 | query-first / full | 1.017 | 0.982 | 0.518 | +3.279 | +3.284 | 0.706 | +3.295 | 0.730 | 0.345 | +0.344 |
| qfirst_ans_only_500 | query-first / answer-only | 0.008 | 1.000 | 0.998 | +11.385 | +11.371 | 1.000 | +11.459 | 1.000 | 0.999 | +0.993 |
| qfirst_w16_500 | query-first / weighted-full | 1.514 | 0.979 | 0.255 | +0.007 | +0.005 | 0.497 | +0.005 | 0.540 | 0.001 | +0.000 |
| qfirst_bag_ans_500 | query-first / BAG_INDEP answer-only | 1.463 | 0.981 | 0.254 | +0.011 | -0.007 | 0.504 | -0.005 | 0.441 | 0.004 | +0.001 |

The decisive positive control is `qfirst_ans_only_500`: all three seeds eventually acquire essentially perfect same-bag counterfactual binding. Per-seed transition epochs vary:

- seed 42: mid-level binding by epoch 225, strong by epoch 250;
- seed 43: mid-level by epoch 475, strong by epoch 500;
- seed 100: mid-level and strong by epoch 375.

At final epoch, top-among-four is about 1.0, B-swap and Q-swap margins are large and positive, Q-swap fraction is 1.0, Q-swap both-correct is about 0.999, and query-specific corruption selectivity is about 0.993. The BAG_INDEP query-first answer-only null remains at bag-level behavior, so the positive result is not caused by the format or probes alone.

The original-order answer-only arm remains bag-level at 500 epochs, confirming that answer-only pressure by itself is not sufficient.

## Full-objective and weighted-full asymmetry

The query-first full objective is heterogeneous across seeds:

- seed 43 undergoes a late transition around epoch 400 and reaches perfect binding by epoch 500;
- seed 100 shows a partial transition by epoch 500 (`top4=0.307`, Q-swap frac `0.711`, B-swap `0.130`);
- seed 42 remains bag-level by epoch 500.

Thus the full next-token objective can sometimes acquire binding in the query-first order, but it is slower and less reliable than answer-only training.

The query-first weighted-full arm with answer weight 16 remains bag-level through 500 epochs in all seeds. This is not yet understood. It may reflect an altered optimization path in which context-token objectives and a high but not exclusive answer objective stabilize context-bag prediction, or it may simply need longer training. It should not be interpreted as a monotone dose law.

## Scientific interpretation

The orbit binding design/015 negative was not caused by missing representational capacity in the baseline Transformer. The same 153K-parameter architecture can implement the required binding computation when the query appears before the context and the answer objective is isolated.

The current supported mechanism is narrower and more informative:

1. **Finite examples contain enough information for reusable binding.** The orbit data are sufficient; the failure in original order is not because the relation is statistically unrecoverable.
2. **Causal order and objective allocation determine whether the information becomes a reusable computation.** Query-first answer-only training crosses into a selector regime; original-order answer-only and original full/weighted objectives remain bag-level.
3. **The reusable computation is real under counterfactual probes.** It follows same-token-bag binding swaps, different query entities in the same context, and novel attribute insertion at the query slot.
4. **Ordinary full next-token training can obscure or delay acquisition.** In query-first order the full objective binds in one seed and partially in another by epoch 500; context-token losses do not make binding impossible, but they make the transition unreliable on this timescale.

This directly advances the general data-efficient learning question: limited experience becomes efficient when the sequence/objective makes the useful computation *reachable and reinforced*, not merely when the dataset contains examples of the relation. The learner otherwise uses the same experience to improve a lower-order statistical approximation: context-bag retrieval.

## Next research work

The next controlled step should not abandon this substrate. It should explain the query-first transition and test retention/consolidation:

1. Extend query-first full and query-first weighted-full to determine whether their apparent differences persist or are delayed transitions.
2. Branch from a demonstrably bound query-first answer-only checkpoint into: continued answer-only, ordinary full objective, and no-answer/context-only training. This asks whether the binding computation persists when the original objective resumes or when direct answer supervision is removed.
3. Add an auxiliary selector head in the original order: train the answer-position hidden state to identify the matching context slot while keeping answer prediction in the original format. This tests whether supplying an intermediate selector rescues original-order binding.
4. If original-order auxiliary selector succeeds, compare sample efficiency and use the result to formulate a principle of *selector access and consolidation*: data-efficient learning requires experience, causal format, objective allocation, and inductive bias to align around the reusable computation.

The natural-stream bridge remains valuable after this: natural language may supply query-before-evidence, repeated reference, or task structure that makes relation selection easier than the original synthetic order. But the controlled binding-rescue mechanism should be understood first.


## Added after independent_review verification and held-entity summary

Independent verification supported the narrow empirical reading: query-first answer-only training is a clean order intervention relative to original-order answer-only under this script, because both layouts have length 18, the same answer position, the same model class, the same generated latent rows, and the same minibatch orders. The sequence order result is therefore not explained by answer-position indexing or answer leakage.

Two boundaries matter for later use:

1. The result establishes acquisition on the standard entity distribution. Held-entity transfer is mixed. In the compact summary, `qfirst_ans_only_500` has standard top4 about `0.998`, but held top4 varies by seed (`0.504`, `0.809`, `0.965`) and held NLL varies widely (`8.202`, `0.550`, `0.094`). Thus the model can learn a strong selector over trained entity tokens, but transfer to unseen entity tokens is not yet uniformly established.
2. The full and weighted-full behavior should not be read as a monotone answer-weight law. Full training reaches perfect binding in one seed, partial binding in one seed, and bag-level behavior in one seed by 500 epochs. Weighted-full with answer weight 16 stays bag-level in all seeds. This points to path dependence and gradient interaction, not simply “more answer weight gives more binding.”

The next controlled work should begin from a demonstrably bound query-first answer-only checkpoint and branch it into continuation objectives and query-access interventions. This will distinguish active use of query-conditioned context states from slower retention or overwriting during subsequent training.

The natural restatement probe also narrows the natural-language bridge. REPEAT-side natural nonoverlap cost transfers to Wikipedia/Simple-English restatements (`R-C gain_T_vs_N = -0.217±0.029`, high-overlap nonoverlap `-0.349±0.084`), while VIEW's compact positive side does not stably transfer in that natural probe (`V-C +0.032±0.040`). Ordinary held-out same-window components are small; the corrected interpretation describes the mechanism as source-recognition/content pull rather than general precise identity copying. This matches the controlled binding finding that bag/source recognition and reusable selector-routing must be separated before generalizing to BabyLM.
