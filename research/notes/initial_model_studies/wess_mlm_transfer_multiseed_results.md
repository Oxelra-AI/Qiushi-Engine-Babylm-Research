# wess mlm transfer multiseed results — Multi-seed WESS-MLM transfer bridge replication

## Evidence

- Replication script: `scripts/wess_mlm_transfer_multiseed.py`
- Source bridge script: `scripts/wess_mlm_transfer_pilot.py`
- Result JSON: `data/wess_mlm_transfer_multiseed.json`
- Single-seed pilot note: `notes/wess_mlm_transfer_positive_signal.md`

## Setup

Three-seed replication of the short DeBERTa-v2 WESS MLM transfer bridge. Same fixed evaluation pair suite and official-text validation slice across seeds; varied train episode seed, official training slice, model initialization, and training RNG.

Arms:
- `plain_mlm`
- `no_address`
- `wess_gold`
- `wess_eventwise_random`
- `wess_wrong_entity`

Each arm used the same 2-layer/128-hidden DeBERTa-v2 backbone, 180 training steps, batch 16, mixed counterfactual episodes + official BabyLM text. This is still a short transfer probe, not a BabyLM-scale candidate.

The fixed evaluation shortcut audit remains strong: first-state, last-state, nearest-state, majority-visible, and previous-query-state heuristics all have 0% pair accuracy on the counterfactual pair score.

## Aggregate results across 3 seeds

### Binding target signal

| arm | mean log-odds | example acc | pair acc | official MLM loss |
|---|---:|---:|---:|---:|
| plain_mlm | -0.0007 ± 0.0008 | 0.1542 | 0.0028 | 4.3854 |
| no_address | +0.0028 ± 0.0045 | 0.0861 | 0.0000 | 4.4801 |
| **wess_gold** | **+1.2056 ± 0.7074** | **0.2194** | **0.0472** | 4.4601 |
| wess_eventwise_random | +0.0002 ± 0.0023 | 0.0861 | 0.0000 | 4.4420 |
| wess_wrong_entity | +0.0006 ± 0.0019 | 0.0889 | 0.0000 | 4.4391 |

### Directed interventions

| arm | swap log-odds delta | ablation log-odds delta |
|---|---:|---:|
| no_address | 0.0000 | 0.0000 |
| **wess_gold** | **+2.0513** | **+1.6287** |
| wess_eventwise_random | -0.0011 | -0.0042 |
| wess_wrong_entity | -0.0050 | 0.0000 |

## Scientific interpretation

### 1. The wess mlm transfer positive signal positive signal is seed-stable

In all three seeds, `wess_gold` is the only arm with a substantial correct-vs-counterfactual log-odds advantage:

- seed 185: +0.283
- seed 286: +2.002
- seed 387: +1.332

All controls remain near zero across seeds. The positive signal is not a single-seed accident.

### 2. The effect is address-specific

The strongest control is `no_address`: it has the same gold spans, same state-token access, same recurrence/projection/fusion path, and same parameter count as WESS, but no persistent entity addressing. It remains at mean log-odds +0.0028 and zero pair accuracy.

Destroyed-address arms also remain at zero:
- eventwise random route: +0.0002 log-odds;
- wrong-entity route: +0.0006 log-odds.

Therefore the signal is not from extra parameters, visible state copying, gold span access, or the fusion path. It depends on persistent entity-indexed addressing.

### 3. The slot state causally changes MLM predictions

For `wess_gold`, swapping slots moves the logits toward the other entity's state by +2.05 nats on average; last-write ablation moves logits toward the previous state by +1.63 nats. The same interventions are zero in all controls. This is the central bridge result: WESS state is not merely correlated with the target; it causally steers DeBERTa-v2 MLM prediction in the predicted counterfactual direction.

### 4. Top-1 binding accuracy is still insufficient

`wess_gold` pair accuracy is only 0.0472, and example accuracy is 0.2194. This is a major improvement over zero controls but not yet strong enough for BabyLM-scale claims. The log-odds and intervention evidence justify scaling; they do not yet show a deployable recipe.

### 5. Official-text MLM is not catastrophically harmed but must be monitored

Official MLM loss for `wess_gold` (4.4601) is comparable to no-address/destroyed arms and only modestly worse than plain_mlm (4.3854) in this tiny training setting. This is acceptable for a bridge probe, but any scaled candidate must monitor ordinary MLM loss, BLiMP, Supplement, Reading, and GlobalPIQA to avoid trading away the protected model's strengths.

## Decision

The WESS transfer route has now met the core earlier analysis contract at the log-odds/intervention level across multiple seeds:

- shortcut audit passes;
- `wess_gold` beats plain/no-address/destroyed controls on binding log-odds;
- destroyed addressing removes the effect;
- slot swap and write-ablation have large directed effects only in `wess_gold`;
- official-text MLM loss is not catastrophically harmed.

The next step should scale the bridge, not repeat the micro-world or remain at the tiny 2-layer probe. The immediate target is a 5M–10M short-budget S1-shaped DeBERTa-v2 + WESS screen that tries to convert log-odds into top-1 binding and then into official Entity/EWoK/GlobalPIQA movement.

The route is not ready for a full 100M official candidate until the scaled short-budget screen shows top-1 binding improvement and non-negative official-like movement. But the evidence now justifies building that screen urgently.
