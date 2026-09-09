# 314 research route decision 20260822 research route decision (2026-08-22)

## Canonical status

- Latest verified internal Overall remains **40.7027874108** at WWM seed43 `chck_80M`.
- This is not a method gain and not SOTA. The gap to the 2026-08-14 local **Strict-Small** leaderboard snapshot (`41.8`) is about **1.0972**. Higher scores in the separate Strict track are not the comparison target.
- No model trained with the new mechanism has an official BabyLM score yet.
- The first mechanism-level candidate promoted by the current audit is **identity-addressed state retrieval**. Its present evidence level is: replicated controlled mechanism + official-compatible natural-text implementation, not natural-pretraining success.

## What was rejected

- H3 gradient conflict as the primary bottleneck: combined centered rho `+0.04167`, blocked permutation `p=0.31593`; conflict features worsened held-out MSE by `3.35%`.
- Counterfactual Binding Credit Assignment (counterfactual binding credit gate).
- Optional identity-edge residual transport (identity edge state transport gate); the trained model ignored the path.
- Cross-view transfer gain per word after independent replication (Steps308-309); apparent gain was explained by lexical overlap/order.
- Within-experience transfer credit (within experience transfer credit); aligned minus exact-word-bag control was null in both 60M seeds.

These routes must not be renamed and reopened without materially new evidence.

## Surviving causal observation

- Event-binding causal use first becomes positive at `60M` in both WWM seeds, so `10M` is an operational check, not a valid H2 mechanism decision point.
- At 100M, the protected WWM event-binding effect is `+0.08779`; the public-leader package is `+0.30185`; leader minus protected is `+0.21406` with a positive CI.
- This links output-usable binding to the leader phenotype, but does not establish what produced the leader's advantage.

## New mechanism family

### Mechanism

For a masked prediction position:

1. find the nearest eligible content cue in its local prefix that occurred previously;
2. use that identity match as an address;
3. retrieve a short contextual window at the prior occurrence;
4. hard-write the retrieved state into a fixed residual-stream subspace;
5. predict through the ordinary MLM vocabulary logits.

The mechanism adds zero trainable parameters and is equivariant to consistent cue renaming. It is not a WESS slot/router, an auxiliary classifier, a paraphrase objective, or a leader-data imitation.

### Controlled evidence

cpc profiles and prefix probe's exact-continuation candidate missed its preregistered `+0.15` advantage over the strongest active control, so cpc profiles and prefix probe remains formally rejected. However, its factorial controls generated a new hypothesis: the correct identity address, not exact continuation copying, is load-bearing.

identity address stress replication independently preregistered and confirmed that hypothesis on unseen names, objects, templates, variable entity counts, multiple overwrites, and long distractors:

| stress split | exact address seed42/43 | displaced content, correct address seed42/43 |
|---|---:|---:|
| variable entities | 1.000 / 0.987 | 0.903 / 0.947 |
| multiple overwrites | 1.000 / 0.980 | 0.997 / 1.000 |
| long distractors | 1.000 / 0.993 | 0.937 / 0.985 |

All same-model path-off and wrong-identity causal gates passed. No BabyLM evaluation data was used.

### Natural implementation evidence

- Vectorized device-side address construction; no Python token loops in training.
- Standard 8x480 parameter count preserved exactly: `34,467,424`.
- Off and true initial state dictionaries are exactly equal under matched RNG.
- Official-format smoke activation: `665/1209 = 55.0041%` of WWM targets.
- Matched parallel CPU elapsed ratios versus off: shuffled `1.004`, true `1.000`.
- Self-contained checkpoints load through both official interfaces:
  - `AutoModelForMaskedLM(..., trust_remote_code=True)`
  - `AutoModel(..., trust_remote_code=True)`
- Full 8x480 forward/backward completed with finite gradients.

This proves implementation readiness and natural-text coverage only. It does not prove a BabyLM task or Overall improvement.

## GPU execution contract

No H100 result is claimed for this design; GPU validation was pending. Exact two-process launch manifests are prepared under:

`data/identity_address_gpu_gate_launcher/`

Proposed comparison order on GPU hardware:

1. `smoke1m/true`: seed42 on GPU0, seed43 on GPU1.
2. `smoke1m/shuffled`: same allocation.
3. If memory, loading, finite-loss, activation, and throughput checks pass, run `full100m/true`.
4. Run `full100m/shuffled`.
5. Compare both arms at the saved 60M checkpoint for the mechanism gate and at 100M for retention, using the existing exact WWM models as the off references.

The launcher caps concurrency at two processes and one process per GPU. Formal settings match the protected backbone, data, tokenizer, batch 256, 100M learning-rate clock, and per-seed initialization/training RNG. `1M` is only a runtime gate. The first meaningful natural mechanism decision is `60M`, matching the observed H2 onset. Running through 100M in the same job avoids restarting training if the 60M checkpoint passes.

Historical H100 evidence for the matched 8x480 WWM model is `4028.6` seconds for 100M words. Allowing for the unmeasured GPU overhead of identity addressing, the preregistered wall-clock estimate from a GPU-visible start is: 10-20 minutes for both 1M runtime waves, 2.5-4 hours for both 100M arms plus the focused 60M/100M causal and target-cluster decision, and 6-8 hours for a complete two-seed nine-column verdict. These are planning ranges, not completed runtime measurements.

## Promotion boundary

At 60M, continue only if all hold:

- true addressing has acceptable H100 overhead and no memory instability;
- activation remains non-degenerate and exactly matched to shuffled addressing;
- true minus shuffled improves the causal event-binding measure in both seeds;
- the Entity/EWoK/GlobalPIQA cluster moves in the intended direction in both seeds;
- broad zero-shot/Reading behavior does not show a serious tradeoff.

At 100M, the project-wide rule remains unchanged: two seeds, complete nine-column evaluation, average `Delta Overall >= +0.25` versus a fully matched strong baseline, no major task exchange, then a third seed and public submission package. Only a real official server score above the current verified leader can be called post-deadline leaderboard SOTA.

## Novelty boundary

Induction heads and match-and-copy circuits are prior art. The candidate contribution is narrower and still provisional: pre-seeding identity-addressed state formation as a parameter-neutral, mandatory residual subspace for sample-efficient masked pretraining. A literature-complete novelty claim must wait for natural-training evidence and a dedicated related-work audit.

## Primary evidence

- `data/h3/two_seed_aggregate.json`
- `data/revision_302b_event_binding_positive_control.json`
- `data/event_binding_timescale.json`
- `data/within_experience_transfer_credit.json`
- `data/induction_seeded_state_subspace_gate.json`
- `data/identity_address_stress_replication.json`
- `data/identity_address_smoke_verification.json`
- `data/identity_address_gpu_gate_launcher/*.json`
