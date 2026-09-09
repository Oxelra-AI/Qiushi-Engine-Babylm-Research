# compact triangle implementation guard — implementation-equivalence guard for compact-view triangle

## Why this was needed

The triangle oom repair and gc preflight OOM repair made the two new triangle controls (`compact_repeat_reinvest` and `adjbreak_reinvest`) use `experiments/archive/representation_and_objectives/scripts/gradient_checkpointed_masking_curriculum_trainer.py`, while the compact-view reference was the historical live fineweb core fact filter run trained with the original COMPACT_EXPERIENCE trainer.

This creates a possible implementation factor in the triangle. Before any endpoint gap is read as a data mechanism, the repaired implementation must be checked on the **same compact-view reference data** at a meaningful early checkpoint. This follows the strategist note: if the checkpointed compact-view trajectory has already diverged from the historical trajectory, then the view arm must be put on the same implementation before interpreting the triangle causally.

## New execution artifact

Wrote `experiments/archive/representation_and_objectives/scripts/compact_view_gc_reference_equivalence.py`.

It launches or compares a compact-view reinvest run using the activation-checkpointed wrapper under the exact live fineweb core fact filter recipe:

- train file: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl`
- historical reference: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022`
- tokenizer: `experiments/archive/initial_model_studies/training/runs/fullcycle_official_wwm_debertav2_8x480_seed43_100M_b256/hf_model`
- fixed recipe: DeBERTa-v2 8x480, baseline16k, WWM 0.15, AdamW LR 0.001, batch 256, seq 256, seed 43 / extra_init_seed 43022 / train_rng_seed 43023
- default horizon: row-boundary `max_word_exposure=1,027,470`, which corresponds to 26 historical optimizer steps and the historical `chck_1M` save point. A literal 1,000,000-word cap would require a partial example and would fail under the trusted `example_jsonl` loader.
- default output: `experiments/archive/representation_and_objectives/training/runs/gc_compact_view_reinvest_1M_seed43022`
- comparison output: `experiments/archive/representation_and_objectives/data/gc_reference_equivalence/gc_reference_equivalence_summary.json`

The preflight succeeded with no missing required paths and identified the historical row reaching the 1M target: compact core joint visibility audit, loss `8.302618026733398`, cumulative words `1027470`, masked tokens `8489`, effective mask rate `0.1465`.

## Guard added to readout scripts

Patched `experiments/archive/representation_and_objectives/scripts/triangle_eval_and_dynamics.py` so the triangle summary now includes:

- `implementation_equivalence.summary_path`
- `implementation_equivalence.interpretation`
- `implementation_equivalence.allows_historical_reference_for_causal_triangle`
- `causal_interpretation_allowed`

The current summary-only run produced `causal_interpretation_allowed: false` because `gc_reference_equivalence_summary.json` does not exist yet. Thus the existing compact-view no-AoA payload can be used only as raw score context, not as causal mechanism evidence against GC-trained controls.

Also wrote `experiments/archive/representation_and_objectives/scripts/triangle_dynamics_with_equiv_guard.py`, which includes the same implementation guard and will include the GC compact-view 1M reference run once it exists. A first guarded dynamics run wrote `experiments/archive/representation_and_objectives/data/compact_triangle_dynamics_guarded/triangle_dynamics_guarded_summary.json`, currently with `compact_view_reinvest_gc_1M_reference` absent and the guard false.

## Expensive-work handling

The two triangle oom repair and gc preflight control trainings (`s202_t21_tool1`, `s202_t21_tool2`) are still managed asynchronously. I did not wait for or collect them in this note. The 1M compact-view GC reference should not be launched on top of those two H100 jobs if it would create GPU contention or OOM risk. Launch it as soon as one H100 slot is free, before reading endpoint gaps as mechanism evidence.

Decision-changing result:

- If the 1M GC compact-view reference is trajectory-equivalent to the historical compact-view run through the checked horizon (same losses / word exposures / masked-token counts up to tolerance), the historical compact-view reference may be used for causal triangle interpretation, while retaining the earlier analysis seed-spread bands.
- If it diverges materially, train/evaluate `compact_view_reinvest` under the same GC implementation before interpreting the endpoint triangle; otherwise the OOM repair is an unmeasured factor.
