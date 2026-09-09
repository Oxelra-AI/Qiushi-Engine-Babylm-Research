# paired seed measurement closeout status paired-seed measurement close-out status

Created: 2026-09-08 UTC during paired seed measurement closeout status.

This note records the current measurement state for the paired seed62065 controls on the repaired adapter-aware BabyLM Strict-Small coordinate. It is a research-facing status note, not a final result.

## Scientific purpose

The remaining computations support the final Stage III attribution comparison:

- ordinary same-row continuation: tests whether lawful extra exposure alone reproduces the distinctive movement;
- dense-mask / sparse-label acquisition: tests the effective-input intervention that suppresses local second-view completion shortcuts while keeping sparse target credit;
- clean ordinary-state parent anchoring: compares the acquisition-only endpoint with the preservation-associated policy.

The comparisons only become meaningful after each endpoint is assembled from source-compatible component files on the same report-derived score surface.

## O62065 ownership and live computation

The seven O62065 SuperGLUE seed42 evaluations were in progress under `experiments/archive/functional_learning/data/o62065_superglue_restart`, using `experiments/archive/functional_learning/scripts/superglue_task_eval.py` and source checkpoint `experiments/archive/relation_learning/data/repair_ordinary62065_bundle/repaired_ordinary62065_u0080`.

Delivered and inspected:

- WSC: return code 0, primary accuracy `65.38461538461539`, result `.../o62065_wsc/superglue_task_result.json`.
- RTE: return code 0, primary accuracy `64.02877697841727`, result `.../o62065_rte/superglue_task_result.json`.
- MRPC: return code 0, primary F1 `87.58620689655172`, result `.../o62065_mrpc/superglue_task_result.json`.

The following evaluations were still in progress at this point in the record:

- BoolQ.
- MultiRC.
- QQP.
- MNLI.

O62065 BLiMP, COMPS and measured AoA were still outstanding at this point in the record. BLiMP and COMPS evaluation had started, but neither had produced a component payload.

Earlier cancelled or failed SuperGLUE attempts did not produce authoritative evaluation payloads and were not used as scientific evidence.

Current admitted O62065 fixed-coordinate components from `strict_split_eval_admission_after_o_sg3`:

- Supplement `63.62`, EWoK `49.81`, Entity `28.09`, GlobalPIQA_parallel `31.07`, GlobalPIQA_nonparallel `48.00`, GlobalPIQA mean `39.535`, Reading `8.195`.
- SuperGLUE partial primary metrics: RTE `64.02877697841727`, WSC `65.38461538461539`, MRPC F1 `87.58620689655172`.

O62065 remains missing BLiMP, COMPS, complete SuperGLUE, and measured AoA for Overall.

## MS62065 ownership and live computation

The only unresolved MS62065 metric at latest admission is SuperGLUE MultiRC (evaluation in progress at this point). Delivered admitted MS62065 SuperGLUE primary metrics are:

- BoolQ `67.95107033639144`
- RTE `63.30935251798561`
- WSC `63.46153846153846`
- MRPC F1 `87.88927335640139`
- QQP F1 `71.40917027743646`
- MNLI `60.28932355338223`

Known SuperGLUE partial sum excluding MultiRC is `414.3097285031356`, so after MultiRC score `m` arrives:

\[
\mathrm{SG}_{\mathrm{MS62065}}=(414.3097285031356+m)/7.
\]

All non-SuperGLUE MS62065 components are admitted:

- BLiMP `68.09`, Supplement `63.09`, EWoK `49.82`, Entity `29.29`, COMPS `52.14`, GlobalPIQA mean `40.05`, Reading `8.195`, measured AoA `0.0`.

The non-SuperGLUE sum is `310.675`, so after SuperGLUE mean `s` arrives:

\[
\mathrm{Overall}_{\mathrm{MS62065}}=(310.675+s)/9.
\]

This will allow the exact seed comparison against MS62064 Overall `42.20253795433653` and the clean-preservation residual comparison using clean62065 Overall `42.23173113265801` versus the already known seed62064 residual `+0.04387437787291759`.

## Latest strict admission output

The latest machine-readable admission pass is:

- `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_sg3/strict_split_eval_admission.json`
- `research/documents/functional_learning/data/strict_split_eval_admission_after_o_sg3/strict_split_eval_admission.md`
- `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_sg3/strict_split_eval_sources.csv`

It has no conflicts and intentionally withholds both O62065 and MS62065 Overalls until the remaining components land.

## Later paired seed measurement closeout status updates

Additional O62065 SuperGLUE deliveries were inspected after the first status note:

- MNLI: return code 0, primary accuracy `60.45232273838631`, result `experiments/archive/functional_learning/data/o62065_superglue_restart/o62065_mnli/superglue_task_result.json`.

The latest strict admission pass `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_mnli` now admits O62065 SuperGLUE partial metrics RTE `64.02877697841727`, WSC `65.38461538461539`, MRPC F1 `87.58620689655172`, and MNLI `60.45232273838631`; BoolQ, MultiRC, and QQP remain missing for the complete SuperGLUE mean at that pass. O62065 Overall is still intentionally withheld until BLiMP, COMPS, complete SuperGLUE, and measured AoA are present.

The package-facing export smoke was strengthened by `experiments/archive/functional_learning/scripts/export_path_smoke_mask_repaired.py`, which sets the HuggingFace dynamic-module cache before importing Transformers and uses the tokenizer-native mask token `<mask>` rather than the literal `[MASK]`. It ran CPU-only and passed for both representative packages. Output: `experiments/archive/functional_learning/data/export_path_smoke_mask_repaired/export_path_smoke_mask_repaired.json`. It confirms tokenizer load, `AutoModelForMaskedLM` load with `36,458,592` parameters and `995,584` private parameters, one real masked-token forward with one mask position, `AutoModel` encoder load with `36,210,368` parameters and `995,584` private parameters, encoder forward, and a classification-gradient smoke with gradients on all 48 private-adapter tensors for both v4 and v5. This is compatibility evidence for the package files, not an external-platform evaluation or a replacement for repaired-coordinate scores.

## paired seed measurement closeout status close-out update after AoA assembly and additional SuperGLUE deliveries

O62065 endpoint AoA extraction completed at `experiments/archive/relation_learning/data/o62065_aoa_endpoint_extract` with 8,005 finite endpoint rows. This raw endpoint output was assembled with the established earlier analysis shared ancestry using `experiments/archive/functional_learning/scripts/assemble_o62065_aoa_from_extract.py`, without re-running endpoint extraction. The measured manifest is `experiments/archive/functional_learning/data/o62065_aoa_measured_from_extract/o62065/full/aoa_manifest.json`; it reports `complete_measured_evidence=true`, 18 steps, 144,090 finite rows, AoA `0.0`, raw correlation `0.0`.

Additional O62065 SuperGLUE deliveries inspected:

- BoolQ: return code 0, primary accuracy `67.4006116207951`, result `.../o62065_boolq/superglue_task_result.json`.
- QQP: return code 0, primary F1 `71.6198210773428`, result `.../o62065_qqp/superglue_task_result.json`.

The latest strict admission pass `experiments/archive/functional_learning/data/strict_split_eval_admission_after_o_boolq` admits O62065 AoA and six of seven SuperGLUE tasks: BoolQ `67.4006116207951`, RTE `64.02877697841727`, WSC `65.38461538461539`, MRPC F1 `87.58620689655172`, QQP F1 `71.6198210773428`, MNLI `60.45232273838631`; MultiRC is still missing. The known six-task O62065 SuperGLUE partial sum is `416.4723546961086`, so after MultiRC value `m` arrives, `SuperGLUE=(416.4723546961086+m)/7`.

A bounded direct release-path compatibility check had started using `experiments/archive/functional_learning/scripts/official_export_finetune_entry_check.py`. It tests the unmodified strict fine-tuning entry `python -m evaluation_pipeline.finetune.run` with `--model_name_or_path` set directly to each public export package directory on CPU. At this point in the record, the v4 WSC test had started and its final result remained pending.


## earlier analysis update

- MS62065 MultiRC completed: primary accuracy `68.23432343234323`, return code 0, source `experiments/archive/relation_learning/data/repair_ms62065_bundle/repaired_ms62065_u0080`, predictions/results present.
- Strict admission rerun at `experiments/archive/functional_learning/data/strict_split_eval_admission_after_ms_multirc` makes MS62065 complete on the repaired coordinate: Overall `42.17887384024569`, MS65−MS64 `-0.023664114090834687`, clean65−MS65 `+0.05285729241231962`, compared with seed64 clean−MS `+0.04387437787291759`.
- Direct public export fine-tuning-entry check completed: `all_valid=true`; unmodified BabyLM strict `evaluation_pipeline.finetune.run` loaded v4/v5 public package directories directly for WSC and produced results/predictions. This verifies release-path entry compatibility locally but is not an external platform submission.
- The revised scientific figure is `experiments/archive/functional_learning/figures/mechanism_trade_revision.{png,pdf}`. O62065 remains incomplete because BLiMP, COMPS, and MultiRC are absent; no O65 Overall is inferred.
