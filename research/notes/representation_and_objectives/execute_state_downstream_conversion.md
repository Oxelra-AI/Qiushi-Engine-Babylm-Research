# execute state downstream conversion Execute state — converting repaired FineWeb breadth into downstream evidence

The active experimental question remains the repaired seqsafe96 interpretive foundation FineWeb seqsafe96 source-breadth contrast: does replacing 1,753,280 words of the 10M clean-Qwen/official pool with seq256-safe cached FineWeb source text move the deficient knowledge cluster (EWoK, Entity, COMPS, GlobalPIQA) enough to justify a larger live FineWeb source/source+view family?

Stricter source selection is not sufficient. V5 is not a sole substrate because it removes discourse, attribution, temporality, and anaphora that may be part of the relations tested by EWoK, COMPS, and GlobalPIQA. A proposed FineWeb comparison would use a stratified mixture of v5-like self-contained facts and v3-like relation-rich context, with faithful views testing an Entity/state sub-mechanism rather than explaining the entire result.

## What was built

To prevent the already-running H100 work from idling after no-AoA results appear, execute state downstream conversion built and checked a downstream-evidence converter:

- `experiments/archive/representation_and_objectives/training/scripts/select_fineweb_full_eval_endpoints.py`
  - reads `data/cached_fineweb_seqsafe96_noaoa_eval_repairseq/fineweb_seqsafe96_delta_summary.json`;
  - enriches each checkpoint with knowledge-cluster and protected-column sums, public/inherited comparisons, and SuperGLUE needed for 41.8 assuming AoA=0;
  - selects only one same-endpoint treatment/control pair: the best treatment equal7 checkpoint;
  - runs full completion only if the no-AoA pattern is scientifically consequential (knowledge-cluster sum ≥1.5, or treatment-control equal7 ≥0.20, or treatment equal7 ≥41.6).

- `experiments/archive/representation_and_objectives/training/scripts/prefill_fineweb_full_eval_from_noaoa.py`
  - writes full-eval per-target payloads by pre-filling BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading from the frozen repaired no-AoA trajectory, exactly as the earlier semantic-view full eval did.

- `experiments/archive/representation_and_objectives/training/scripts/fineweb_full_eval_runner.py`
  - injects the repaired FineWeb treatment/control run directories into the inherited COMPACT_EXPERIENCE full official-style evaluator and writes to `data/fineweb_seqsafe96_full_eval`.

- `experiments/archive/representation_and_objectives/training/scripts/summarize_fineweb_full_eval.py`
  - summarizes selected full-eval results and same-endpoint treatment-control deltas.

- `experiments/archive/representation_and_objectives/training/scripts/wait_select_full_eval_fineweb_seqsafe96.sh`
  - waits for a parseable repaired no-AoA delta summary;
  - runs the seqsafe96 interpretive foundation route interpreter and execute state downstream conversion endpoint selector;
  - exits without SuperGLUE/AoA if the no-AoA pattern is weak;
  - otherwise runs only the selected treatment/control pair sequentially on one safe GPU at a time, then summarizes.

- `experiments/archive/representation_and_objectives/scripts/interpret_fineweb_full_eval_and_route.py`
  - reads the selected full summary when it exists and combines it with the repaired no-AoA trajectory, public/inherited references, compact view evidence, and live fineweb core scale probe v3/v5 source-yield measurements;
  - its route logic explicitly avoids further selector churn and, if FineWeb continues, recommends a stratified natural mixture rather than v5-only neat facts.

## Checks actually performed

- Python AST checks passed for all four execute state downstream conversion full-eval scripts.
- `bash -n` passed for `wait_select_full_eval_fineweb_seqsafe96.sh`.
- The selector was validated on the completed semantic-view no-AoA summary only as a logic test; after pair-limiting, it chose exactly one endpoint (`chck_80M`) rather than the earlier two-endpoint behavior. This did not reinterpret the semantic-view result.
- `interpret_fineweb_full_eval_and_route.py` ran successfully before full results exist and wrote an `awaiting_full_eval` note.
- The first background submission failed schema validation and did not start. The corrected submission was accepted as `s19_t21_tool1`.

## Pending Experiments

- `s17_t37_tool1`: still the original repaired sequential training/no-AoA task for the FineWeb seqsafe96 contrast. Its result is not yet delivered.
- `s19_t21_tool1`: downstream converter waiting for the parseable no-AoA summary, then selecting at most one same-endpoint treatment/control pair for SuperGLUE+AoA if the no-AoA pattern warrants it.

No new training, Qwen generation, selector revision, BabyLM score, or final deliverable was produced.
