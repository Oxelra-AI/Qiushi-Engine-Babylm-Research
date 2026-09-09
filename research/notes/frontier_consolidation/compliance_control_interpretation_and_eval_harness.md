# compliance control interpretation and eval harness — compliant-tokenizer endpoint/control interpretation and evaluation readiness

## Scientific Motivation

The load-bearing experiment remains the compliant-tokenizer retrain of `compact_view_reinvest` seed43022.  The prior 42.0331347900748 endpoint remains strong scientific evidence for the density-reinvestment mechanism, but it cannot be a Strict-Small submission because its tokenizer was trained outside the 10M-word budget.  compliant tokenizer retrain status/37 fixed the submission-relevant tokenizer by training a structurally matched 16k BPE tokenizer on the `compact_view_reinvest` 10M pool only.

A tokenizer-exposure correction changed how the matched `clean_qwen` retrain should be interpreted.  The running clean-Qwen model uses the tokenizer trained on the reinvest 10M pool, while its own pretraining pool differs by the 423,520-word overlay.  Therefore the tokenizer-fitting text plus clean-Qwen pretraining text has a union larger than 10M.  This does **not** make the running clean model useless: it remains the cleanest fixed-tokenizer scientific control for isolating the pretraining-corpus effect under the same segmentation.  But it is not itself an independently valid Strict-Small submission artifact.

## Files produced or repaired

- Evaluation harness patched and revalidated: `experiments/archive/frontier_consolidation/scripts/evaluate_compliant_endpoint.py`
  - Added the missing call to `patch_superglue_primary_metric()` immediately after `base.eval_superglue(...)`, so the per-target JSON uses the current official-coordinate SuperGLUE convention: F1 for MRPC/QQP and accuracy for the other five tasks.
  - Updated arm descriptions and preflight records to distinguish the end-to-end submission-relevant reinvest endpoint from the fixed-tokenizer clean-Qwen scientific control.
  - AST parse passed after the patch.
  - Preflight rerun for both arms.  Both still correctly report `model_path_exists=false` while background retrains are unresolved, and no required evaluation support path is missing.
- Budget-union audit: `experiments/archive/frontier_consolidation/scripts/tokenizer_pretraining_budget_union_audit.py`
  - Outputs: `experiments/archive/frontier_consolidation/data/tokenizer_pretraining_budget_union_audit/tokenizer_pretraining_budget_union_audit.json` and `.md`.
- Clean-own tokenizer diagnostic: `experiments/archive/frontier_consolidation/scripts/clean_own_tokenizer_diagnostic.py`
  - Outputs: `experiments/archive/frontier_consolidation/data/clean_qwen_own_tokenizer_diagnostic/clean_qwen_own_tokenizer_diagnostic.json` and `.md`.
- Projection helper repaired: `experiments/archive/frontier_consolidation/scripts/project_compliant_eval_continuation.py`
  - Its outputs now carry `family`, `description`, and `budget_semantics`, and warn that clean-Qwen projections are fixed-tokenizer scientific comparisons only.

## Quantitative corpus-budget finding

From `tokenizer_pretraining_budget_union_audit.py`:

- Tokenizer fitting pool: `cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`, 64,740 rows, 10,000,000 words, SHA256 `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`.
- Reinvest pretraining 10M pool: exact same 64,740 rows and same 10,000,000 words; tokenizer+pretraining union = 10,000,000 words.
- Clean-Qwen pretraining 10M pool: 64,381 rows, 10,000,000 words, SHA256 `e0a3cdc20e39f2715fbdb0cfbc6c4aff61d51924c480a982049878272c5690b3`.
- Exact text-row overlap between tokenizer pool and clean-Qwen pool: 61,734 rows; construction metadata gives the more important designed decomposition:
  - common filler shared between clean-Qwen and reinvest: 9,576,480 words;
  - reinvest changed block: 423,520 words;
  - clean held-out rows replaced by that block: 423,520 words;
  - tokenizer+clean-Qwen pretraining designed union = 10,423,520 words, exceeding the 10M budget by 423,520 words.

Interpretation: the reinvest retrain is the only current end-to-end submission-relevant compliant-tokenizer endpoint.  The clean-Qwen retrain is a fixed-tokenizer causal contrast only.  If a `reinvest - clean_qwen` comparison becomes central, describe it as fixed-tokenizer causality and separate it from end-to-end rule-valid comparison.

## Tokenizer-corpus mismatch diagnostic

To quantify whether the fixed-tokenizer clean control is geometrically close to a clean-own-tokenizer system, I trained a diagnostic 16k BPE tokenizer on the clean-Qwen 10M pool only.  This did not launch or authorize a model retrain.

From `clean_own_tokenizer_diagnostic.py`:

- Clean-own tokenizer SHA256 `2fac71fd07fb67d2de800eceb1b1b1ba657a7bb434dffcf378ded60e6297b7d3`, vocab 16,384, special IDs `<unk>`=0, `<s>`=1, `</s>`=2, `<pad>`=3, `<mask>`=4.
- Vocab overlap between the reinvest-trained compliant tokenizer and the clean-own diagnostic tokenizer: 15,605/16,384 = 95.245% of each vocab.
- On the clean-Qwen 10M pool, total token ratio clean-own/reinvest-tokenizer = 0.999595; seq256-truncated rows 16,756 under reinvest-tokenizer vs 16,730 under clean-own tokenizer.
- On the reinvest 10M pool, total token ratio clean-own/reinvest-tokenizer = 1.000517; seq256-truncated rows 15,967 under reinvest-tokenizer vs 15,980 under clean-own tokenizer.

Interpretation: the fixed-tokenizer clean-Qwen control is not being grossly length-distorted by using the reinvest-trained tokenizer; length/truncation geometry is almost identical to a clean-own 10M tokenizer.  However, rule status remains different, and changed vocabulary identities can still affect model learning.  This diagnostic supports using the clean run as a scientifically meaningful control while preventing it from being mislabeled as a submission-valid system.

## Pending Training Comparisons

- Compliant-tokenizer `compact_view_reinvest` retraining remained unresolved; this is the submission-relevant endpoint.
- Clean-Qwen fixed-tokenizer control retraining remained unresolved; this is a scientific control only.

No result has been inferred from either task.  Do not inspect active task directories until terminal delivery unless the runtime explicitly delivers the result.

## Next execution once retrain delivery arrives

1. Obtain the completed reinvest and clean-Qwen training results.
2. Inspect completed run directories with `experiments/archive/frontier_consolidation/scripts/inspect_compliant_retrain.py`.
   - Require complete 100M exposure, all 19 AoA checkpoints, `hf_model/chck_100M`, and tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9` in the reinvest endpoint.
3. For the reinvest endpoint, run the official-compatible evaluation harness:
   - Start with lower-cost zero-shot columns plus Reading if GPU resources are tight.
   - Use `project_compliant_eval_continuation.py` after partial results to decide whether SuperGLUE+AoA are still scientifically warranted for beating the 41.8 visible leader.
   - Run SuperGLUE with the patched harness only when the partial surface can still support the submission-relevant result.
   - Run repaired AoA with `min_context=0` and 8,005 rows/checkpoint before any final scalar.
   - Use `experiments/archive/representation_and_objectives/scripts/stage_pristine_collate.py` for the authoritative official-coordinate collation.
4. Evaluate the clean-Qwen fixed-tokenizer control to measure the pretraining-corpus effect under fixed segmentation, but keep its status separate from the end-to-end valid reinvest endpoint.
5. If the reinvest endpoint lands below 41.8, use the tokenizer-shift evidence from compliant tokenizer shift and eval readiness to diagnose model-learning/vocabulary effects rather than attributing failure to crude length/truncation changes; use that diagnosis to choose a discriminating comparison before any new 100M model run.

## SuperGLUE metric patch regression

After writing the note above, I added and ran `experiments/archive/frontier_consolidation/scripts/superglue_metric_patch_regression.py`.  It uses known old-tokenizer `compact_view_reinvest` seed43022 SuperGLUE outputs and verifies that the patched compliant tokenizer shift and eval readiness harness reproduces the current official-coordinate SuperGLUE scalar exactly.

Output: `experiments/archive/frontier_consolidation/data/superglue_metric_patch_regression/superglue_metric_patch_regression.json` and `.md`.

- Expected SuperGLUE from pristine collate: 71.03604952825312.
- Patched harness SuperGLUE: 71.03604952825312.
- Inherited legacy accuracy-only mean: 71.38107147224343.
- Difference legacy accuracy-only minus primary metric: +0.34502194399031.
- Regression pass: true.

This validates the harness repair before any new expensive SuperGLUE finetuning run.  Final pristine collation remains the authoritative scalar, but the per-target JSON and partial projections will no longer use the wrong SuperGLUE coordinate.

## Path-root robustness repair

The repair to `evaluate_compliant_endpoint.py` and `inspect_compliant_retrain.py` derives `USER_ROOT` from `__file__` rather than from the process working directory. This prevents evaluation/inspection commands from failing or silently resolving recorded paths against the wrong base directory. AST parse passed for the updated evaluation, inspection, and projection scripts, and both evaluation preflights reran successfully after this repair.
