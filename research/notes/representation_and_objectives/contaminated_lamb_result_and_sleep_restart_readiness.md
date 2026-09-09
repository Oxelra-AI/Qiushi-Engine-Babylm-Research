# contaminated lamb result and sleep restart readiness — Contaminated LAMB cheap7 result and sleep-restart tail readiness

## 1. Contaminated LAMB 8×480 endpoint (joint-intervention only)

Completed cheap7 evaluation of the lamb curriculum route decision LAMB endpoint
(`lamb_curriculum_8x480_legal40k_seed43022/hf_model/chck_100M`).
Per-target file:
`data/lamb_contaminated_cheap7_eval/per_target/contaminated_lamb_curriculum_8x480_legal40k_seed43022.json`.

Cheap7 columns (BLiMP/Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading):

| column | contaminated LAMB | legal40k baseline (legal40k accum training completion) | delta |
|---|---:|---:|---:|
| BLiMP | 65.98 | 67.95 | -1.97 |
| Supplement | 56.24 | 60.97 | -4.73 |
| EWoK | 49.81 | 51.47 | -1.66 |
| Entity | 24.25 | 27.20 | -2.95 |
| COMPS | 52.07 | 51.67 | +0.40 |
| GlobalPIQA | 37.695 | 34.665 | +3.03 |
| Reading | 7.88 | 7.82 | +0.06 |
| **cheap7** | **41.99** | **43.108** | **-1.12** |

(GlobalPIQA baseline = mean(22.33 parallel, 47.00 nonparallel) = 34.665;
contaminated GlobalPIQA = mean(20.39, 55.00) = 37.695.)

The contaminated LAMB endpoint is **1.12 cheap7 below** the matched fixed-256
legal40k baseline. This is contaminated on multiple axes at once — LAMB lr0.007,
sequence-length curriculum, token-slicing chunking geometry, `pad_token_id=0`,
`bos/eos=null`, `pos_att_type=null`, `max_relative_positions=-1`, 42.13M params —
so it is not clean evidence about LAMB or curriculum. Its cheap7 loss is
consistent with independent finding that LAMB lr0.007 clamps trust-ratio in
most tensors and collapses hidden rank. The GlobalPIQA gain here is exactly the
known parallel/nonparallel redistribution (parallel down, nonparallel up), not a
hard-rank repair. Conclusion: this datapoint neither closes nor supports LAMB or
curriculum; it only confirms that the config-defect package is broadly harmful.

## 2. Corrected curriculum run remains the primary decision

The clean, matched, warmup-0.06, word-boundary AdamW curriculum
(64:20M→128:50M→256:100M) is still the single interpretable test of whether
sequence-length curriculum improves the legal40k coordinate. Compare its 100M
cheap7 against the fixed-256 baseline cheap7 43.108 and leader cheap7 ~43.77.
Do not read the contaminated LAMB result as its proxy.

## 3. Sleep-restart tail continuation: implemented and dry-run verified

Built `scripts/sleep_restart_tail_trainer.py` as a source-grounded
low-cost candidate if curriculum does not cross. It loads the recorded legal40k
`chck_80M` weights (80,011,326 words already consumed) and continues **only the
remaining stream words** with a fresh AdamW optimizer state and a fresh cosine
LR schedule (a "sleep"/state-reset motivated by FORGETTER's optimizer-state resets
and BabyLM smaller-batch/schedule evidence), reaching the same 100M total exposure
on the same legal compact-view stream, tokenizer, architecture, and WWM 0.15.

Dry-run manifest (`data/sleep_restart_tail_dryrun/sleep_restart_tail_manifest.json`):
- tail starts at row 518,144, tail = 129,256 examples / 19,965,632 words
- 505 effective steps at batch 256 to reach exactly 100,000,000 total words
- no partial-example splits; word accounting exact

This is a bounded, recoverable experiment (~505 steps, one GPU, well under an hour)
that tests whether a late optimizer/schedule sleep can lift the fixed-256
coordinate without a new full endpoint. It is held, not launched: both H100 lanes
are occupied and the curriculum comparison must be read first. It must never be
presented as the fixed-256 baseline itself; its comparison anchor is the same
legal40k baseline cheap7 43.108 and the standard 80M reference cheap7 42.9486.

## Decision logic after `s135_t23_tool1`
- cheap7 in leader region (≳43.77): full official eval (SuperGLUE + AoA).
- cheap7 improved but below leader: compare column movement, consider composition
  with matched-horizon adapter, then decide whether the sleep-restart tail
  or a curriculum+sleep combination is the next bounded spend.
- cheap7 not above 43.108: close curriculum-alone; the sleep-restart tail becomes
  the next cheapest source-grounded whole-learning-system probe from an existing
  checkpoint rather than another fresh 100M endpoint.
