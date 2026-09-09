# loss allocation preliminary result plan: answer-loss allocation and reusable binding

## Why this is the next test

orbit binding design showed that the BOUND orbit-binding arm behaved like BAG_INDEP under the ordinary full next-token objective: same-bag swaps produced near-zero margins and query swaps did not follow the queried entity. The stronger interpretation that gradient descent is trapped in a bag-level local minimum is not yet warranted. In the implemented sequence, the binding answer is only one supervised token among sixteen non-PAD next-token targets. Perfect entity binding improves the ideal bag predictor by only `log(4)/16 = 0.0866` nats per non-PAD token, while most of the objective still trains grammar and randomly sampled context content. The indistinguishable aggregate training losses in orbit binding design therefore do not establish that entity-specific information is unavailable, wasted, or unrepresentable.

The next test keeps the orbit binding design task, data distribution, and baseline Transformer fixed, and changes only how much learning pressure is placed on the answer token. This asks whether the same finite experience can form a counterfactual binding computation when the training objective allocates enough pressure to it.

## Scientific question

When does the fixed small causal Transformer acquire reusable entity-attribute binding from the same orbit experience?

The distinction to resolve:

1. **Objective-allocation bottleneck:** the model can represent and learn binding, but the ordinary full next-token objective gives too little relative pressure to the answer token or lets high-entropy context-token prediction dominate early updates.
2. **Representation/optimization bottleneck beyond allocation:** even answer-focused training does not produce same-bag counterfactual binding, so the next step must first build a positive binding solution before attributing failure to a particular landscape mechanism.
3. **Retention bottleneck:** answer-focused training can acquire binding, but the computation decays when ordinary full next-token training resumes; finite budgets then require acquisition and consolidation conditions, not only useful examples.

## Arms

All arms use the same K=4, N_ENT=10, N_ATTR=12 orbit-binding substrate and the same d=64, nh=2, nl=3 Transformer as orbit binding design.

### Acquisition arms

- `full_w1_200`: orbit binding design-style full objective, 200 epochs, answer weight 1. Baseline.
- `ans_only_13`: answer-only objective for 13 epochs. This roughly matches the answer-token coefficient accumulated by `full_w1_200` (`200/16 = 12.5`). If this already binds, the issue is not simply answer-token update amount; context losses or allocation interfere.
- `w16_25`: full objective with answer weight 16 for 25 epochs. Normalized answer share is `16/(15+16)=0.516`; 25 epochs gives roughly the same answer coefficient as baseline but still trains surrounding tokens.
- `ans_only_200`: answer-only objective for 200 epochs. This tests representational capacity and direct learnability.
- `w16_200`: weighted full objective for 200 epochs. This tests whether strong answer allocation can coexist with grammar/context training.
- `bag_ans_only_200`: BAG_INDEP target with answer-only objective. This confirms that the probes do not produce false entity-binding positives merely from answer focus.

### Retention arms

Run only if the acquisition arms show binding:

- `ans50_full150`: 50 epochs answer-only, then 150 epochs ordinary full objective.
- `w16_50_full150`: 50 epochs weighted full objective, then 150 epochs ordinary full objective.
- `ans100_full100`: 100 epochs answer-only, then 100 epochs ordinary full objective.

These ask whether a learned binding computation persists under the original objective and whether it improves useful answer prediction while ordinary sequence modeling resumes.

## Measurements

Use the orbit binding design probes, plus more direct binding scores:

- `correct_nll`, RWT family mass, within-family NLL, MRR.
- `ctx_top1`: top-1 accuracy among the four context attributes.
- `query_margin`: logit(correct query attribute) minus mean logits of decoy context attributes.
- `B_swap`: same token bag, swapped entity-attribute binding; positive paired margin means prediction follows the current binding.
- `Q_swap`: same context, two different query entities; both predictions must follow their queried entity.
- `corrupt_novel_selectivity`: query-novel gain minus decoy-novel gain. This separates true query-bound update from bag membership response.

## Interpretation

If answer-focused or weighted training produces high B_swap, Q_swap, ctx_top1, and positive query-novel selectivity, then orbit binding design becomes evidence for objective allocation and retention conditions, not for absent capacity. The next work should return to the full objective and measure persistence and useful prediction.

If no acquisition arm binds, even with answer-only pressure, the immediate task is to construct a positive binding solution under this substrate or a minimal architectural variant before explaining the failure mechanistically.

If acquisition succeeds but retention fails, the emerging principle becomes about when limited experience not only creates a reusable computation but also protects it against later high-entropy or shortcut-compatible objectives.
