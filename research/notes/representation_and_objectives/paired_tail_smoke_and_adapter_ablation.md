# paired tail smoke and adapter ablation — paired-tail execution smoke and adapter coupling readout

## Active run preserved

The corrected matched AdamW word-boundary curriculum run remains the primary experiment. I did not inspect or query its run state during this work. No new full training branch was launched.

## One-effective-batch smoke for the paired tail factor separation design paired-tail fallback

Scientific purpose: if corrected curriculum misses, the next held continuation should compare two 80M→100M tails from identical `chck_80M` weights rather than running the old opaque sleep package. Before spending H100 time on the full 505-step tails, I ran one actual-device effective batch on GPU1 for both arms to verify that loading, tail selection, masking, forward/backward, optimizer step, and LR assignment work.

Artifacts:

- script: `experiments/archive/representation_and_objectives/scripts/paired_tail_one_step_smoke.py`
- summary: `experiments/archive/representation_and_objectives/data/paired_tail_one_step_smoke/paired_tail_one_step_smoke_summary.json`

Result:

- source checkpoint: legal40k fixed-256 `chck_80M`, actual exposure `80,034,368` words, baseline step `2024`
- tail: row index `518144` onward, `129,256` examples / `19,965,632` words, `505` effective steps to the original 100M endpoint
- first effective batch: `39,780` words
- actual device: `cuda:1`
- parameter count: `45,826,720`
- both arms used identical actual CUDA WWM masks and labels on the first batch:
  - `same_actual_device_masked_inputs=true`
  - `same_actual_device_labels=true`
  - `same_actual_device_selected_positions=true`
- loss and gradient before the first optimizer step are identical across arms:
  - loss `2.4111779473432167`
  - grad norm before clipping `0.6364467740058899`
  - masked tokens `8278`
- LR geometry is as intended:
  - residual first update LR `1.0720866372880134e-4`, matching the baseline source-step logged LR
  - residual first-after-update LR `1.0680028468509506e-4`
  - reheat first update LR `0.0`
  - reheat first-after-update LR `1.0e-5`

Interpretation: the paired-tail fallback is executable and the two arms are genuinely paired in their first-batch stochastic path. If curriculum does not beat the fixed-256 baseline, full residual-vs-reheat tails can be launched without reworking the script. The comparison still should be read against the uninterrupted fixed-256 100M anchor, because both tails use fresh AdamW moments and a new mask/dropout seed.

## adapter128 live-branch disabled-at-inference ablation

Scientific purpose: matched-horizon adapter128 live 20M run lost cheap7 despite BLiMP/Supplement/COMPS gains. The open question was whether the loss came from the direct trained adapter residual output or from the altered stock-backbone trajectory during joint training. I created an inference-only proxy that changes only `adapter_enabled` from true to false in the live checkpoint config and symlinks all weights unchanged.

Artifacts:

- proxy script: `experiments/archive/representation_and_objectives/scripts/prepare_adapter_inference_disabled_proxy.py`
- proxy run: `experiments/archive/representation_and_objectives/training/runs/adapter128_live20M_inference_disabled_proxy`
- per-target evaluation: `experiments/archive/representation_and_objectives/data/adapter_inference_disabled_eval/per_target/adapter128_live20M_inference_disabled_proxy.json`
- synthesis: `experiments/archive/representation_and_objectives/data/adapter_inference_disabled_synthesis/adapter_inference_disabled_synthesis.json`

Scores at 20M:

| model/readout | BLiMP | Supp | EWoK | Entity | COMPS | GP | Reading | cheap7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aoa mincontext discrepancy audit / disabled-training anchor | 59.69 | 55.45 | 50.73 | 18.65 | 50.26 | 34.195 | 8.67 | 39.6636 |
| live adapter enabled | 60.38 | 56.23 | 49.49 | 18.65 | 50.44 | 32.225 | 8.31 | 39.3893 |
| live adapter disabled only at inference | 59.91 | 56.12 | 48.98 | 18.74 | 50.24 | 34.165 | 8.38 | 39.5050 |

Decomposition from the synthesis:

- direct adapter output effect (live enabled − live disabled): BLiMP `+0.47`, Supplement `+0.11`, EWoK `+0.51`, Entity `−0.09`, COMPS `+0.20`, GlobalPIQA `−1.94`, Reading `−0.07`, cheap7 `−0.116`
- stock-backbone drift effect (live disabled − aoa mincontext discrepancy audit anchor): BLiMP `+0.22`, Supplement `+0.67`, EWoK `−1.75`, Entity `+0.09`, COMPS `−0.02`, GlobalPIQA `−0.03`, Reading `−0.29`, cheap7 `−0.159`

Interpretation:

1. GlobalPIQA damage in the live adapter run is almost entirely the direct residual output: disabling the trained adapter at inference recovers GlobalPIQA from `32.225` to `34.165`, essentially the aoa mincontext discrepancy audit anchor `34.195`.
2. EWoK damage is mostly the changed stock trajectory: disabling the adapter at inference worsens EWoK to `48.98`, below both live-enabled `49.49` and the aoa mincontext discrepancy audit anchor `50.73`.
3. The trained adapter residual supplies some BLiMP/Supplement/COMPS/EWoK gains, but it disrupts GlobalPIQA enough that ordinary joint adapters are not a promoted route.
4. The useful next architecture idea is coupling control: protect the stock trajectory and regulate the residual output, rather than simply increasing adapter width or running the same ordinary joint adapter longer without a sharper reason.

This result complements earlier analysis. It does not create a SOTA endpoint and does not change the primary decision: read the corrected curriculum endpoint when delivered, then decide whether to evaluate it fully, close curriculum-alone, or run the paired-tail continuations.
