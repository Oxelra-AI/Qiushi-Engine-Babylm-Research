# compact triangle implementation guard post-triangle mechanism analysis plan

## Scientific object

The active route is no longer endpoint search. The experiment is meant to decompose the replicated compact-view advantage into transferable mechanisms:

1. semantic information density plus reinvested breadth,
2. source-own compact-rewrite local consolidation,
3. repetition/generic breadth or seed/endpoint dynamics.

The triangle oom repair and gc preflight/203 triangle is useful only if implementation, seed, and broken-control confounds are kept explicit.

## Required carriers before mechanism interpretation

1. Both triangle oom repair and gc preflight control trainings complete with `scientific_metrics.json`, `training_log.jsonl`, `dynamics_traces.jsonl`, and `hf_model/chck_100M/model.safetensors`:
   - `experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2`
   - `experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2`
2. compact triangle implementation guard same-data implementation check exists:
   - `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json`
   - interpretation must be `trajectory_equivalent_to_checked_horizon` before using the historical compact-view arm causally.
3. Guarded readout exists:
   - `experiments/archive/representation_and_objectives/data/compact_triangle_noaoa_eval/triangle_noaoa_summary.json`
   - `causal_interpretation_allowed` must be true for mechanism interpretation against historical compact-view.
4. Guarded dynamics exists:
   - `experiments/archive/representation_and_objectives/data/compact_triangle_dynamics_guarded/triangle_dynamics_guarded_summary.json`

If the implementation guard is false, do not interpret view-control score gaps as data mechanisms; train/evaluate compact_view under the same GC implementation or restrict discussion to raw endpoint observations.

## Reading order after results arrive

1. Read task result JSONs and run-local `launcher_result.json` for `s202_t21_tool1`, `s202_t21_tool2`, `s203_t24_tool1`, and `s203_t28_tool1`.
2. Inspect each run's `scientific_metrics.json` for exact word exposure, steps, parameter count, tokenizer label/path, seeds, and checkpoint count.
3. Inspect `gc_reference_equivalence_summary.json` before reading triangle gaps.
4. Read `triangle_dynamics_guarded_summary.json` and `triangle_noaoa_summary.json`.
5. Only then produce an interpretation note.

## Interpretation logic

### Implementation factor

- If GC compact-view matches historical through 1M exactly or within negligible numerical noise: historical compact-view reference remains usable for the one-seed triangle.
- If it diverges in loss, masked-token counts, word exposure, or checkpoint metadata: the triangle is implementation-confounded. The next GPU action is a same-GC compact_view 100M run or a lower-cost same-GC score proxy before making mechanism claims.

### Training dynamics

Compare loss curves and milestone deltas at 1M, 2M, 5M, 10M, 20M, 40M, 60M, 80M, 100M.

- If repeat/adjbreak losses are much worse than compact_view from early training, endpoint gaps may reflect ordinary MLM difficulty rather than a subtle consolidation mechanism.
- If repeat and adjbreak have nearly identical losses but different task-family scores, the difference localizes to structure rather than gross optimization.
- If adjbreak loss stays near repeat and compact_view but scores collapse selectively in EWoK/Entity/GlobalPIQA_parallel, local source-own consolidation becomes plausible.

### Broad vs targeted columns

Use the protocol's broad/targeted distinction:

- Broad: BLiMP, Supplement, COMPS, GlobalPIQA_nonparallel, Reading.
- Targeted: EWoK, Entity, Entity_full, GlobalPIQA_parallel.

Adjbreak collapse across broad columns means the broken control is generically incoherent. It cannot by itself prove source-own rewrite consolidation. Redesign a softer distance-broken control instead of overreading.

### Seed band

Fast equal7 gaps below 1.3557 are not final one-seed mechanism evidence. Full-seven/Overall gaps below about 0.85/0.785 require a tie-breaking seed before a stable mechanism conclusion.

## Possible next actions, selected by evidence

- **View high, repeat low, adjbreak low, broad intact:** local source-own compact-rewrite consolidation is the leading mechanism. Next work: test the same adjacency relation on a different model/source coordinate with companion analysis; optionally add a softer adjbreak distance control if broad profile is ambiguous.
- **View high, repeat low, adjbreak near view:** compact rewrite marginals plus information density/breadth dominate; source-own adjacency is not load-bearing at this coordinate. Next work: derive and test a transferable data-selection/compression rule across source and model families.
- **Repeat near view:** semantic second views are not the central factor; reinvested breadth/repetition endpoint dynamics dominate. Next work: compare against existing source_breadth evidence and compress the principle around exposure diversity rather than rewrite pairing.
- **All arms close within seed band:** one seed is inconclusive. Queue only the tie-bearing second seed defined in the protocol, not broad random sweeps.
- **Implementation guard false:** run same-GC compact_view reference to the necessary horizon before mechanism interpretation.

## Additional non-GPU analysis worth doing before another expensive run

- Compare `example_order_manifest.json` between view/repeat/adjbreak to verify exposure counts and source-word bookkeeping.
- Compare training-log mask RNG observables (`masked_tokens`, `effective_mask_rate`) across arms; data-order changes should not be confused with masking implementation changes.
- If no-AoA score gaps are promising but ambiguous, sample prediction flips between arms on EWoK/GlobalPIQA_parallel and broad columns using existing per-target payloads before launching full evaluations.
