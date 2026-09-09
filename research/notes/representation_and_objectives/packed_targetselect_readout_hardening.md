# packed targetselect readout hardening — hardened readout for packed target-selective 100M intervention

## Immediate scientific context

The two earlier analysis packed historical-geometry target-selective training arms remain managed asynchronously:

- `drop_abs_content`, output `training/runs/packed_drop_abs_content_100M_fast/`
- `drop_copied_content_wholeword`, output `training/runs/packed_drop_copied_content_wholeword_100M_fast/`

No endpoint result is inferred. The readout was prepared before the pending results became available.

## What was repaired / added

### 1. Historical-vector scale in endpoint signature v2

`scripts/packed_targetselect_endpoint_signature_v2.py` was repaired so that its comparison to correctness transition analysis historical compact-view vectors uses the saved correctness transition analysis JSON `accuracy_delta` values directly.

- Source: `data/correctness_transitions/correctness_transitions.json`
- Check artifact: `data/historical_vector_check/historical_vector_scale_check.json`

The repaired historical vectors are on a common percentage-point scale:

```json
{
  "view_vs_adjbreak": {
    "Supplement": 3.6,
    "social-properties": 14.0,
    "physical-dynamics": 12.0,
    "spatial-relations": 9.0,
    "physical-relations": 9.0,
    "material-properties": 3.0,
    "social-interactions": -1.0
  },
  "view_vs_repeat": {
    "Supplement": 4.4,
    "social-properties": 6.0,
    "physical-dynamics": 7.0,
    "spatial-relations": -1.0,
    "physical-relations": 0.0,
    "material-properties": 10.0,
    "social-interactions": 7.0
  }
}
```

The prior fallback path that mixed note-level EWoK net-item counts with Supplement percentage points was removed. The 20M rehearsal after repair is at `data/signature_v2_rehearsal_repaired_20M/` and is still only a rehearsal, not endpoint evidence.

### 2. One-command postrun reader

New script: `scripts/packed_targetselect_postrun_readout.py`

Purpose: after both earlier analysis training tasks have actually delivered, run all necessary checks and readouts in one reproducible chain:

1. verify each endpoint's `scientific_metrics.json` and final `hf_model/chck_100M`;
2. require exact integrity fields:
   - `word_exposure = 100000000`
   - `actual_training_steps = 2529`
   - init SHA `f13f1f85923a6e04d033f755a180f8755be7247005dca327be01c7461493d509`
   - final model files present
   - dropped-label mass matched to the actual CUDA-WWM realized schedule from earlier analysis:
     - source-absent: `dropped_abs_content = 48105`
     - copied whole-word: `dropped_copied_wholeword = 48387`
3. run `scripts/eval_packed_targetselect_100M.py` for official-compatible endpoint predictions;
4. run repaired `scripts/packed_targetselect_endpoint_signature_v2.py`;
5. run `scripts/packed_targetselect_denoising_probe_100M.py` for fixed-event local denoising on train-fixed and source-disjoint compact pairs;
6. write a compact mechanism summary to `data/packed_targetselect_postrun_readout/postrun_readout_summary.json` and `.md`.

The script syntax-checks, but it was not run in normal mode because the two training jobs were still unresolved.

## Interpretation Criteria

Primary causal comparison: `drop_abs_content` versus `drop_copied_content_wholeword`, because both arms are trained by the same earlier analysis fast implementation. The historical full compact endpoint remains a shape/scale reference for the original compact-view triangle.

Two signs must be kept straight:

- Local fixed-event loss: positive `drop_abs_minus_drop_copied_word` on `source_absent_content` means source-absent compact labels were more useful for that prediction category than matched copied labels.
- Endpoint accuracy: positive `drop_copied_word_minus_drop_abs` means the arm that retained source-absent labels did better than the arm that deleted them.

A strong mediation reading requires both:

1. local source-absent-content degradation under `drop_abs` on train-fixed and source-disjoint compact events; and
2. endpoint movement in the expected direction on Supplement plus the correctness transition analysis relational EWoK domains (`social-properties`, `physical-dynamics`, `spatial-relations`, `physical-relations`), not merely aggregate cheap7, GlobalPIQA, or Reading movement.

If local denoising survives but Supplement/relational-EWoK does not move coherently, the source-absent target-label channel remains a real within-coordinate learning channel but not an established mediator of the historical compact-view endpoint advantage.

## Related Experimental Evidence

frontier_consolidation message 265 corrected their compact/repeat BPE atlas: the individual pair-view difference is +31,289 active BPE over 12,155 compact-pair objects, while the exact packed training-stream audit remains +24,000 active/candidate BPE per 10M pass (+240,000 over 100M). This reinforces that the natural compact marginal is a coupled intervention: faithful compression, tail-content recovery, higher content density, source-absent content targets, BPE burden change, and reinvested source diversity are entangled unless a specific experiment separates them.
