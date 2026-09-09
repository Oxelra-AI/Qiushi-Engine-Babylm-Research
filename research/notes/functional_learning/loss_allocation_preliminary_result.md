# loss allocation preliminary result preliminary result: answer pressure alone has not yet produced orbit binding

## Motivation

orbit binding design showed that the ordinary full next-token objective produced bag-level behavior on the orbit-binding substrate. The stronger interpretation that this reveals a bag-level local minimum was premature because the binding answer receives only one of sixteen supervised next-token losses. Perfect binding improves the ideal bag predictor by only `log(4)/16 = 0.0866` nats per token under that objective. The first loss allocation preliminary result intervention therefore kept the orbit binding design task and Transformer architecture fixed while changing answer-token learning pressure.

## First completed run

Script: `scripts/loss_allocation_binding.py`  
Data: `data/loss_allocation_binding/results.json`  
Figure: `figures/loss_allocation_binding.png`

Arms included ordinary full objective, answer-only objective, answer-weighted full objective, a BAG_INDEP answer-only null, and two phase schedules that switched from answer-focused training back to ordinary full next-token training.

Across three seeds, the 200-epoch answer-focused and weighted-full arms improved answer NLL to the same range as the ordinary full objective, but did not produce counterfactual binding:

| Arm | correct NLL | top among 4 context attrs | query margin | B-swap | Q-swap margin | query-specific novel follow |
|---|---:|---:|---:|---:|---:|---:|
| full_w1_200 | 1.539±0.011 | 0.254 | +0.025 | -0.0046 | +0.0033 | -0.0028 |
| ans_only_200 | 1.519±0.035 | 0.247 | -0.011 | +0.0082 | -0.0001 | -0.0075 |
| w16_200 | 1.507±0.039 | 0.250 | +0.017 | +0.0218 | +0.0003 | +0.0033 |
| bag_ans_only_200 | 1.483±0.017 | 0.254 | +0.033 | +0.0110 | +0.0004 | +0.0002 |

The top-among-four score remains near the bag-uniform value of 0.25; B-swap and Q-swap are near zero; query-specific corruption selectivity is near zero. The BAG_INDEP answer-only arm reaches the same answer NLL range as BOUND answer-only, confirming that lower NLL here can arise from context-bag retrieval without query-conditioned selection.

Short nominal answer-dose probes (`ans_only_13`, `w16_25`) reach NLL around 2.39–2.41 and also show no binding. These are only learning-speed probes, not a rigorous answer-gradient dose match, because AdamW step counts, moment histories, and gradient interactions differ across objectives.

## Corrections from independent verification

The preliminary run is useful but not enough for the strongest interpretation:

1. The answer index is correct: the target answer at original sequence position 16 is predicted by logits at input position 15, so `IS_POS=15` is the right training/evaluation index.
2. The first script used different realized finite streams across arms, so it tests the same distribution under different objectives, not the identical finite examples.
3. The `ctx_top1` metric was tie-biased because the correct attribute was placed first in the tested list. It should use the original context-slot order or tie-safe scoring.
4. The full-family RWT mass does not separate context-bag retrieval from selecting the queried item within the bag. A repaired metric should report mass on the four context attributes and within-bag quantities.
5. If 200-epoch answer-only training does not produce binding, an extended answer-only phase is needed before concluding that scalar answer pressure is insufficient within this architecture.
6. Retention schedules are interpretable only if a phase-boundary checkpoint has measurable binding.

## Repaired run now launched

Script: `scripts/revision_015b_paired_loss_allocation.py`  
Planned output: `data/revision_015b_paired_loss_allocation/results.json`, `figures/revision_015b_paired_loss_allocation.png`

Repairs:

- All BOUND objective arms consume the same generated rows and minibatch order for a given seed/epoch.
- Metrics include context-bag mass, within-bag NLL/entropy, tie-safe top-among-four, B-swap, Q-swap, and query-specific corruption selectivity.
- Added `w64_200` and `ans_only_1000` to test stronger/longer answer pressure.
- Added `bag_ans_only_1000` as a null for answer-focused training without query-conditioned targets.
- Added `ans500_full500` to test whether any binding after answer-only training persists under the ordinary full objective.

## Current scientific reading before Step015b returns

The first allocation run already weakens the idea that the orbit binding design failure was only caused by the answer token being one small part of the loss: even answer-only training for 200 epochs stays at bag-level behavior. But because the first run had finite-stream and metric weaknesses, the stable result should come from Step015b. If Step015b also shows no binding after answer-only 1000 epochs, the next research object should be a positive binding construction or a more direct circuit/architecture intervention, not another broad repetition/variation test and not a final theory of the plateau.
