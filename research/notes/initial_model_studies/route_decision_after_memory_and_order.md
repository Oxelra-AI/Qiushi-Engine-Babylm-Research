# route decision after memory and order — Route after controlled memory and pure-order results

## Controlled facts now established

1. **Current prefix-average memory is not a main route.** Under paired shared-core initialization, fixed schedule, identical example order, two pools, and two seeds, the current `memory_causal` adapter has no stable positive effect. Mean memory-minus-dense deltas were near zero or negative for Entity, EWoK, COMPS, Reading-eye, and Supplement. This rejects this specific prefix-average design, not every possible identity-sensitive state mechanism.

2. **Training text condition is the strongest observed lever.** Dense pool10M vs pool1M at 1M exposure gave stable gains on Entity and Reading: Entity about +1.77, Reading-eye about +1.19, self-paced Reading about +0.55, with BLiMP and Supplement slightly lower. This is the most repeatable signal found so far.

3. **Pure single-pass order alone does not solve the tradeoff.** `source_stage` over exactly the same selected examples and source mix improved BLiMP (+0.43 mean) and EWoK (+2.09 mean), but reduced Entity (-1.54 mean) and Supplement (-1.60 mean). `readability_interleave` gave a small BLiMP gain (+0.40 mean), but Supplement, EWoK, Entity, COMPS, and Reading were mixed or lower. Thus order changes representations, but these two simple single-pass recipes do not preserve the pool10M Entity/Reading gains while recovering BLiMP/Supplement.

4. **The old route structure is now too weak.** We should not scale the current memory adapter, nor should we scale `source_stage` or `readability_interleave` directly. Their scientific value is that they reveal which families trade off under controlled data, not that they are candidate SOTA systems.

## Interpretation of the central bottleneck

The evidence points to a representation and interference problem, not merely an exposure problem.

- The full-pool dense condition sees richer and more diverse text and improves Entity/Reading, but syntax-like columns do not benefit.
- Source staging can strengthen BLiMP/EWoK while hurting Entity/Supplement, suggesting training time/order can rotate which structure dominates.
- Prefix averaging adds parameters but does not store identity or mention competition, so it cannot solve Entity.
- Training loss is not a reliable guide: source_stage has high final loss because the last segment is distributionally different, while readability_interleave has lower final loss but does not broadly improve the profile.

The next route must reduce competition between word-form/entity/discourse information and BPE-level syntactic generalization. A route that only reorders the same single pass is unlikely to be enough.

## Main next route: parameter-matched multi-granularity token representation

The strongest constructive route is to keep BPE for local syntax and add a learned token-surface composition path for low-frequency forms, names, and morphology-like patterns:

\[
 e_i = e^{\mathrm{BPE}}_{v_i} + \alpha_i W_c c(v_i),
\]

where `c(v_i)` is computed from the tokenizer string for each vocabulary item using a small character-level encoder or fixed character n-gram composition, and `alpha_i` is a learned gate. The surface-composition table can be precomputed from tokenizer vocabulary strings; it uses no extra language corpus. The hypothesis is not that characters replace BPE, but that BPE handles high-frequency syntax while the surface path shares information across names, inflections, subword variants, and rare forms, reducing the capacity competition seen in Strict-Small.

This route is more promising than the earlier `morph_side_causal` result because the earlier morph-side run was not paired with shared-core initialization, was not compared against a matched extra-capacity lookup adapter, and used a small fixed feature list rather than a learned composition path. The old morph-side signal was mixed: EWoK/Entity were locally higher, BLiMP/Supplement lower. That is precisely the kind of signal a controlled multi-granularity route should test properly.

### First controlled experiment

Run 1M full-pool random experiments using the same seeds and manifests as the earlier analysis dense pool10M baseline.

Variants:

1. **BPE-only dense-untied baseline**: already available from earlier analysis for seeds 42 and 43.
2. **BPE + learned char/surface adapter**: same GPT-2 core, same tokenizer, same order, same exposure; only add a gated token-surface composition path at the input embedding.
3. **BPE + parameter-matched token-ID adapter**: same number of additional parameters and similar fusion/gating shape, but the adapter is a learned per-token lookup with no character sharing. This separates surface-composition structure from added capacity.

Controls:

- same `example_pool_words=10000000`, `max_word_exposure=1000000`, `lr_total_steps=98`, seed 42/43;
- shared GPT-2 core initialization copied exactly into all three variants;
- extra adapter weights initialized from a separate seed;
- identical consumed example IDs and order to the dense baseline;
- root/`chck_1M`/revision loading;
- official fast/local profile: BLiMP, Supplement, EWoK, Entity, COMPS, Reading.

Useful result pattern:

- If char/surface adapter improves Entity and EWoK while preserving BLiMP/Supplement better than the token-ID adapter, the route has real structural value and should move to a 3M/5M trajectory.
- If char/surface and token-ID adapter perform the same, the effect is extra capacity rather than composition.
- If both adapters hurt Supplement/BLiMP or Entity, surface composition is not solving the current tradeoff at this scale and we should move to objective or identity-state mechanisms.

## Small companion route: source-recency and microcycle probe

The pure-order result should not be overread. `source_stage` may partly reflect source-recency or forgetting because one source family occupies the final training segment. A small follow-up can determine whether repeated interleaving is worth future work.

Minimal variants on the same selected full-pool sample set:

1. `reverse_source_stage`: same stages but reversed, to see whether task gains follow the final source.
2. `source_microcycle`: divide the selected examples by source stage and cycle through stage slices repeatedly, so no source monopolizes the end of the single pass.
3. optional `source_microcycle_short_rehearsal`: end each cycle with a short mixed rehearsal from early dialogue/child-directed examples.

These should use the same dense-untied control, same seeds, same selected examples, same source totals, same word exposure, and checkpoint(s) at 0.25M/0.5M/0.75M/1M if evaluation time permits. This is a fast route-sense experiment, not the main SOTA construction.

Useful result pattern:

- If reverse order flips the source_stage gains/losses, the earlier order result is mainly recency and should not be scaled.
- If microcycle lifts the lower envelope of BLiMP/Supplement and Entity/Reading together, repeated schedule becomes a real route.
- If microcycle just oscillates task families, scheduling is not the main breakthrough route.

## Other routes and why they are not first

### Objective route

Adaptive masking or sparse denoising is scientifically plausible because prior BabyLM systems show objective changes can strengthen NLP. But it changes supervised target density and evaluator compatibility, and previous AMLM-like systems can harm human-likeness. It should come after the multi-granularity representation test unless the representation route is negative. If used, it should be a low-weight pulsed auxiliary loss on top of CLM, with CLM-only and equal-supervision controls.

### Entity-state route

A better entity mechanism would need identity-sensitive slots with competition and persistence, not a prefix average. It should include dense, nonpersistent adapter, shuffled/allocation-broken persistent state, and proper entity perturbation measurements from the start. This is higher construction cost than the representation test and should wait unless the representation route fails or Entity remains the dominant obstacle after a better tokenizer/representation path.

### Full official evaluation and scaling

No current candidate justifies full 100M exposure or leaderboard-style submission. The next candidate must first show seed-stable fast/local promise across Entity/Reading and BLiMP/Supplement. AoA and fuller official evaluation should be added once a candidate clears the 1M and 3M/5M profile stage.

## Immediate next work

The next Execute step should implement the parameter-matched multi-granularity representation experiment with as little trainer disruption as possible:

1. Add a new custom model or embedding module for `char_surface_causal` and `lookup_adapter_causal`.
2. Extend shared-core initialization to copy the GPT-2 core into these models, as was done for memory.
3. Run direct save/reload and equality probes.
4. Run tiny smokes with identical example order.
5. Train the 1M full-pool seed42/43 grid and profile with the official fast/local suite.

The source-recency/microcycle probe can run after or in parallel if implementation is small, but it should not delay the representation route unless the trainer change becomes risky.
