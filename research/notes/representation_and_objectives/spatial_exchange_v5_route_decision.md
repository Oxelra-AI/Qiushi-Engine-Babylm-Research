# curriculum parity probe — spatial counterfactual exchange v5: witness-grounded route decision

## What was tested

counterfactual exchange clean subset left one unresolved lead: spatial counterfactual exchange was the first
*unsaturated* frozen relation probe, and on the v3 clean subset (156 cases) it
tracked the known GlobalPIQA_parallel tradeoff (rowblock > compact > interleaved).
The planned witness-grounded factorial v5 test was deferred twice. curriculum parity probe built and
ran it as a low-cost, no-update probe, in parallel with the corrected 8×480
curriculum training run on GPU0.

`spatial_exchange_v5.py` enforces exact witnessed vertical relations:
locative/copular verbs + above/below/beneath, and "on top of". Each case expands
into a factorial of 8 views:
- counterfactual: `A is above B` vs `A is below B`, dual queries (higher, lower)
- inverse-equivalent: `B is below A` vs `B is above A` (same world, flipped surface)
- relation-erased control: `A is near B` (no vertical information)
Margins cancel additive target preference; a within-stratum target-pair permutation
gives a no-update null. No parser or external model; frequency-matched typed spans.

## Decisive corpus result

Over the entire legal 10M compact-view corpus (64,183 rows, 668,049 sentences):
- **only 58 regex matches; only 6 clean typed cases survive filtering**, all with
  attested relation `above` (the `below` branch is essentially absent, so orientation
  balancing collapses to 6). Rejects: 224,854 sentence-filter (quantity/threshold and
  dialogue), 45 bad_head, 4 freq_mismatch, 2 token_shape.
- The 6 surviving spans still carry grammar noise ("stars that is above the earth",
  "that this particular fort is above ..."), because a no-parser NP regex cannot cut
  clean argument heads at this density.

## Frozen-checkpoint score on the 6 clean cases (CPU, 3 arms)

- `cf_dual_m` mean ≈ **−0.078** (compact-arm), i.e. essentially zero; success_gt0 0.167.
- `all_four_cf_views_correct_frac` = **0.0**, `both_higher_worlds_correct_frac` = **0.0**.
- target-permutation null dual mean ≈ **−0.070** — same magnitude as the true signal,
  so with n=6 there is **no separation from the permutation control**.
- `erased_dual_abs` ≈ **3.36** with success_gt0 1.0: the models answer these cloze
  frames strongly from lexical/frequency priors even with the vertical relation erased.
- `equiv_above_gap_abs` ≈ 1.03, `equiv_below_gap_abs` ≈ 0.78: the model does not treat
  `A above B` and `B below A` as the same world, so the surface, not the relation,
  drives the score.

## Route judgment

The counterfactual exchange clean subset spatial-exchange "signal" was carried by the looser v3 subset whose
density depended on contaminated templates (quantity `over N`, adjective arguments,
first-mention). Under exact high-precision witnessing, the legal 10M substrate yields
too few clean, non-degenerate vertical relations to constitute either a reliable probe
or a debitable training object, and on the clean cases the relation-erased control
matches the counterfactual behavior. This realizes the negative branch anticipated in
`counterfactual_exchange_route_decision.md`:

> "If a clean, dense, unsaturated exchange object at compliant density cannot be
> obtained from the legal 10M substrate, that itself is a strong negative result about
> the corpus, and the route should move to a genuinely different high-leverage
> mechanism."

**Decision:** the counterfactual exchange route (spatial and by extension the other
relation families that were sparser still) is closed as a probe-derived training
object on the legal 10M corpus. The context-conditioned alternative-binding weakness
(EWoK stable reversals, GlobalPIQA_parallel hard ranks) remains the correct scientific
diagnosis, but it cannot be repaired by corpus-mined relation supervision at this
density. The load-bearing route now is the whole-learning-system change: the corrected
matched sequence-length curriculum (curriculum parity probe 8×480 AdamW run, `s134_t20_tool1`),
compared on matched cheap7 against baseline 43.1079, plus complementary
function-preserving/optimizer work.

## Artifacts
- Builder/scorer: `experiments/archive/representation_and_objectives/scripts/spatial_exchange_v5.py`
- Build summary: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/spatial_exchange_v5_build_summary.json`
- Samples: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/spatial_exchange_v5_samples.jsonl`
- Score summary: `experiments/archive/representation_and_objectives/data/spatial_exchange_v5/score_cpu/spatial_exchange_v5_score_summary.json`
