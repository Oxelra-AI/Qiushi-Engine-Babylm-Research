# packed targetselect static load and readout patch — packed target-selective static label load and post-run reader repair

## Why this was done while the 100M arms were still running

The earlier analysis packed intervention compares two 100M arms in the exact historical compact-view packed stream:

- `drop_abs_content`: removes source-absent compact-content labels.
- `drop_copied_content_wholeword`: removes matched copied-content whole-word labels.

Those arms were still managed asynchronously when this note was written. I did not read their outputs. The useful independent work was to protect the later causal reading against a static training-interface alternative: even if the realized WWM-deleted BPE mass is matched, the two deletion populations could in principle differ in batch timing or mean-loss denominator enough to contaminate an endpoint comparison.

## New static load profile

Script:

- `experiments/archive/representation_and_objectives/scripts/packed_static_label_load_profile.py`
- `experiments/archive/representation_and_objectives/scripts/packed_static_label_load_profile_v2.py`

Primary output:

- `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2/packed_static_label_load_profile_v2.json`
- batch table: `experiments/archive/representation_and_objectives/data/packed_static_label_load_profile_v2/packed_static_label_load_batches.jsonl`

The v2 profile uses the same packed 100M stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, annotation SHA `3e1cf013450be48534011ff685bc00e0a42dfb9aff3087989f614d073baf9105`, copied-selection SHA `ff51b243043ae83e63e8f0299026efe9972a08fd98b4615334eb1206eaeb6ad2`, baseline16k tokenizer, 256-row batches, sequence length 256, mask probability 0.15, and the historical 2529-step warmup+cosine LR shape.

Key static totals over the full 100M stream:

- rows used: 647,400
- word exposure: 100,000,000
- total candidate BPE pieces: 143,915,480
- source-absent-content candidate pieces: 319,580
- selected copied-content candidate pieces: 322,370
- selected minus absent candidate pieces: +2,790
- expected WWM-realized selected-minus-abs deleted pieces at 0.15 mask rate: +418.5

This matches the earlier analysis realized CUDA-WWM measurement in scale and sign: realized source-absent deletion 48,105 BPE versus copied deletion 48,387 BPE, copied-minus-absent +282 BPE. The static expectation is slightly larger but still only about 0.29% of the deleted-label mass.

Batch-level timing and normalization:

- 2,529 optimizer batches.
- Selected-minus-abs candidate pieces per batch: mean +1.103, median +1, p05 -37, p95 +39, min -102, max +82.
- Source-absent and selected-copied batch loads are highly co-located: Pearson 0.826.
- Expected kept-label denominator if source-absent labels are deleted: mean 8,516.957.
- Expected kept-label denominator if copied labels are deleted: mean 8,516.792.
- Denominator weight ratio `drop_abs/drop_copied`: mean 0.99998071, median 0.99998221, p01 0.99902897, p99 1.00095741, max absolute deviation from 1 is 0.00182084.
- LR-weighted denominator ratio using the historical schedule is 0.99998037 (using the LR applied to the update) and 0.99998037 after scheduler step as well.

Step deciles all stay near the same total static excess, about +260 to +298 selected-minus-abs candidate pieces per decile. The LR-weighted expected selected-minus-abs deleted mass per batch is +0.1687 BPE. This is too small to explain a large Supplement or relational-EWoK endpoint separation by loss scaling alone.

Scientific reading: the copied whole-word control remains a close static and realized label-load match for the source-absent deletion at the level that matters for batch mean-loss normalization and training time. This does not say what the endpoint result will be; it only removes one alternative explanation for any later arm-to-arm movement.

## Post-run reader repair

Scripts touched:

- `experiments/archive/representation_and_objectives/scripts/eval_packed_targetselect_100M.py`
- `experiments/archive/representation_and_objectives/scripts/packed_targetselect_postrun_readout.py`

Change:

- `eval_packed_targetselect_100M.py` now accepts `--skip-signature` and can generate only endpoint predictions/per-target JSONs.
- `packed_targetselect_postrun_readout.py` now passes `--skip-signature` to the endpoint evaluator and then runs the strengthened earlier analysis/231 signature v2 exactly once.

Reason: the quick earlier analysis signature reader is superseded for mechanism reading. The v2 signature uses compound item keys, percentage-point historical vectors read from correctness transition analysis JSON, paired bootstrap summaries, EWoK domain vectors, and vector similarity to the historical compact-view pattern. Running both signatures would be redundant and could confuse later interpretation.

Syntax check passed for:

- `eval_packed_targetselect_100M.py`
- `packed_targetselect_postrun_readout.py`
- `packed_static_label_load_profile.py`
- `packed_static_label_load_profile_v2.py`

## Interpretation Criteria

When both managed 100M arms are delivered by the runtime, first inspect their terminal records and each `scientific_metrics.json`. If both have exact 100M exposure, 2529 updates, init SHA `f13f1f85923a6e04d033f755a180f8755be7247005dca327be01c7461493d509`, final `hf_model/chck_100M`, and realized deleted-label masses 48,105 versus 48,387 BPE, run:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/packed_targetselect_postrun_readout.py --force
```

The source-absent channel becomes load-bearing for the historical compact-view mechanism only if two things hold together:

1. The fixed-event readout shows `drop_abs_minus_drop_copied_word > 0` specifically on source-absent compact content, including source-disjoint held-out compact rewrites.
2. The endpoint contrast favors the arm retaining source-absent labels on Supplement and the correctness transition analysis relational EWoK domains, with a visible relation to the historical compact-view transition vector.

If only the local loss field moves, the result remains a real within-coordinate source-absent denoising channel, but not mediation of the historical downstream compact-view gain.
