# counterfactual exchange clean subset — counterfactual exchange probe: the first unsaturated relation object

## What changed from full context pivot substitution probe

full context pivot substitution probe tightened lexical relation funnels (PVDM, visible-target auxiliary,
full-context pivot substitution, active pair pool, opposition/shared-slot,
intra-row two-argument joint-slot). Every one of those frozen no-update probes was
**saturated**: true margins were huge (6–13 nats), success >90%, and the failing
compact endpoint expressed the signal at least as strongly as the breadth arms.
They measured attested-context recovery / local slot assignment, which the models
already have, not the conditional alternative binding they lack.

counterfactual exchange clean subset changes the *representation*, not the funnel tightness. Each item is a
rule-certified **counterfactual exchange**: the same two attested arguments A/B stay
fixed; only the relation flips (above↔below, before↔after, higher↔lower,
caused↔caused-by, more/less ADJ). The correct alternative therefore *exchanges*
between the two contexts. The scored margin

`M = [s(C0,T0) - s(C0,T1)] + [s(C1,T1) - s(C1,T0)]`

cancels the target main effect, and a same-stratum target-pair permutation gives a
no-update null. Checkpoints: FW compact, rowblock breadth, interleaved breadth 100M.

## Result: this object is not saturated (v3 high-precision pilot, 900 cases)

Overall arm means (v3 grammar-cleaned pilot):

| metric | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---|---:|
| true_m | 0.0522 | -0.0207 | -0.3745 | compact > rowblock > interleaved | 0.427 |
| target_perm null | -0.0261 | -0.0022 | 0.0028 | — | 0.029 |

- True exchange margins are **near zero**, not 6–13 nats. Models do **not** reliably
  solve the counterfactual exchange; `both_contexts_correct` is only a few percent.
- The target-permutation null range (~0.03) is an order of magnitude smaller than the
  true-margin arm range (~0.43), so the arm separation is a real relation effect, not
  a permutation artifact — the opposite of every relation filtered pair pools/129 subpool where null
  ranges matched or exceeded true ranges.

Family structure (the load-bearing part):

| family | n | compact | rowblock | interleaved | rank | range |
|---|---:|---:|---:|---:|---|---:|
| spatial_vertical | 469 | -0.226 | **0.617** | -0.753 | rowblock > compact > interleaved | 1.37 |
| temporal_order | 275 | 0.329 | -1.109 | -0.021 | compact > interleaved > rowblock | 1.44 |
| causal_direction | 90 | 0.674 | -0.123 | 0.107 | compact > interleaved > rowblock | 0.80 |
| comparative_scalar | 56 | 0.018 | 0.135 | 0.207 | interleaved > rowblock > compact | 0.19 |

On spatial exchange, **rowblock** (the breadth arm that improved GlobalPIQA_parallel
24.27→29.13 and reduced the hard52 margin) is the arm that separates in the
relation-repair direction, while compact and interleaved are at/below zero. This is
the first probe whose direction follows the known EWoK/GlobalPIQA tradeoff rather
than compact's broad-capability ordering.

## The unresolved contamination

The pilot pool is still contaminated by degenerate templates where the attested
pivot is not a true relation:

- `over` acting as a quantity marker: "1.45 million barrels ... is over one quarter"
  → "barrels is above quarter" is meaningless spatially.
- Adjective/adverb arguments admitted by the surface grammar: "obvious is above
  past", "labour happened after higher".

These push the arguments away from concrete comparable entities and inject noise into
the spatial/temporal families. The signal survives anyway, which is encouraging, but
the object is not yet clean enough to certify or to train.

## Route judgment

This is a genuinely different and more promising relation object than the closed
full context pivot substitution probe–129 pools, and it is worth continuing — but it is not yet a training
objective. Before any exposure-debited H100 training:

1. **Enforce true-relation grammar.** Require concrete comparable entities on both
   sides (object/place nouns for spatial; event/dated nouns for temporal; scalar
   comparands for comparative), and reject pivots used as quantity markers
   (`over N`, `above N%`, `under N years`). No pretrained parser; hand-coded legal
   rules and corpus frequency only.
2. **Rebuild at higher precision and re-run frozen calibration** on the same three
   checkpoints with the target-permutation null and a new context-permutation null.
   Keep it unsaturated and confirm the spatial rowblock-repair direction persists on
   clean cases with a stable, non-degenerate sample.
3. **Only then** design an exposure-matched training objective, debiting every
   generated word-pass in `word_count_pair` against the Strict-Small budget, with a
   same-exposure WWM reference, and admit it only if it moves EWoK conditional
   reversals and GlobalPIQA hard ranks without damaging broad capability.

If a clean, dense, unsaturated exchange object at compliant density cannot be
obtained from the legal 10M substrate, that itself is a strong negative result about
the corpus, and the route should move to a genuinely different high-leverage
mechanism (e.g., whole-learning-system changes companion analysis is exploring: LAMB, masking-rate
schedules, cosine restarts, SuperGLUE fine-tuning optimization) rather than another
local-fit variant.

## Files

- Builder v3 (high-precision): `experiments/archive/representation_and_objectives/scripts/counterfactual_exchange_v3_builder.py`
- Probe + scorer: `experiments/archive/representation_and_objectives/scripts/counterfactual_exchange_probe.py`
- v3 pilot cases: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k`
- v3 scored: `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_v3_pilot20k_scored/counterfactual_exchange_score_summary.json`
- earlier v2 pilot (looser grammar, similar unsaturated signal): `experiments/archive/representation_and_objectives/data/counterfactual_exchange_probe_pilot20k_v2`

## Optimizer branch status

Pending `s128_t47_tool1` (50M Muon-switch broad columns) still running at counterfactual exchange clean subset
start. The 50M EWoK table already landed (intrarow calibration synthesis): continuous Muon improved EWoK
accuracy to 0.5175 / stable-failure 0.6757 vs AdamW 0.4924 / 0.6987, but abrupt
Muon→AdamW arms gave only 0.4997/0.4998 and muon50 gp and relation pair synthesis/128 GlobalPIQA showed no hard-rank
repair. Collect `s128_t47_tool1` before formally closing the abrupt empty-moment
switch artifacts; do not extend those exact runs.
