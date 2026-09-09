# channel exposure reconstruction readiness — compact ordered-vs-scrambled channel-exposure reconstruction readiness

## Findings

The earlier analysis exposure-split analyzer and its dependent channel exposure reconstruction readiness wait-runner are now validated to
reconstruct the exact WWM target selection of the compact order experiment design ordered/scrambled 40M streams, but only
under **CUDA** RNG. This was verified against the real training logs, not assumed.

### Decisive RNG-device finding

`scripts/mask_reconstruction_debug.py` runs the trainer's own
`MaskedChunkDataset`, `collate`, `MaskingCurriculumState(wwm_fixed, p=0.15)`, and
`apply_masking_curriculum` over the first 20 batches of each arm's exact 40M pool and compares the
masked-token count per batch to the recorded `training_log.jsonl`.

- CPU RNG: masked-token counts diverge from the logs (e.g. ordered batch 1 diff +98, batch 3 +180),
  because the trainer used a CUDA `torch.Generator` and `torch.rand` draws differ by device.
- CUDA RNG (GPU0): masked-token counts match the logs **exactly** for both arms, all 20 batches,
  diffs all 0. Candidate-token and batch-word counts also match.

Exposure-split reconstruction is reproducible only on CUDA. The pending comparison uses `compact_order_event_exposure_split.py --device cuda`; `mask_count_validation.exact_prefix_match` must be True for both arms before interpreting the never-selected splits.

### Analyzer repairs made in earlier analysis/channel exposure reconstruction readiness

`compact_order_event_exposure_split.py` was fixed to:
- Merge compact order evaluation and channel readout plan `event_losses.jsonl` (which lacks `word_index`) with the parallel
  `probe_events.jsonl` by stable `event_index`, so ordered-side spans can be rebuilt.
- Use the exact lead route assessment after source use probe 10M pools (`compact_ordered_10M.jsonl` SHA
  `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, `compact_scrambled_10M.jsonl`
  SHA `07101d1391fbc1090f1318ead2c780a8a0071f98e2da06b6be30e9a19ca14b1d`) as four exact passes.
- Record per-batch `mask_count_validation` against the arm training logs so faithfulness is
  self-checking.

A CPU smoke over the preflight channel dir (60 events) mapped all 60 events to visible ordered
spans (`mapped_visible=60`), confirming the merge + span reconstruction works; its CPU
`mask_count_validation` correctly reported mismatch, matching the RNG-device finding.

## Why this matters for the compact-order route

The corrected scientific criterion requires that a source-absent channel advantage be:
1. larger than retained_content and function_other advantages (category interaction), and
2. persistent among probe words never WWM-selected in the relevant 40M stream.

Condition 2 depends entirely on a faithful selected-count reconstruction. channel exposure reconstruction readiness makes that
reconstruction trustworthy (CUDA) and self-validating. Without this, a lower source-absent NLL
could not be distinguished from direct target exposure.

## Pending Comparisons

- Selected cheap official scoring and the source-absent channel probe remain pending for 2 arms x chck_20M/chck_40M.
- Training-log/mask auditing remains pending.
- Scrambled 40M training was relaunched; the ordered arm was already complete and valid.
- The validated CUDA exposure split follows completion of the channel probe.

## Interpretation gate (unchanged from compact order evaluation and channel readout plan/205)

Only ordered-selective source-absent improvement (negative source_absent-minus-controls with
bootstrap support, ideally holding among never-selected events) **together with** stable official
family movement (cheap6-no-GlobalPIQA >= 0.1, EWoK+Entity >= 0) justifies a mature extension.
GlobalPIQA/Reading-only movement or an aggregate order effect without the source-absent interaction
should not promote the route.
