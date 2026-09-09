# earlier analysis pretrained ATP slot-orientation probe

Small pretrained-language bridge test for reusable filler interface plus sparse relation-coordinate orientation on independently attested ATP event/state and paired world pilot result paired-world material.

Model: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
Device: `cuda`, elapsed 376.89s

## Data construction

```json
{
  "atp_train_primary": 120,
  "atp_eval_primary": 120,
  "pair_train": 120,
  "pair_eval": 160,
  "atp_eval_conflict_worlds": 56
}
```

### Arm `exposure`

Train rows: 4032; label balance {1: 2016, 0: 2016}

Train row kinds:

```json
{
  "base_anchor_compound": 2880,
  "base_anchor_pair_event": 480,
  "sparse_exposure": 672
}
```

### Arm `aligned`

Train rows: 4032; label balance {1: 2016, 0: 2016}

Train row kinds:

```json
{
  "base_anchor_compound": 2880,
  "base_anchor_pair_event": 480,
  "sparse_aligned_compound": 576,
  "sparse_aligned_pair_event": 96
}
```

### Arm `flipped`

Train rows: 4032; label balance {1: 2016, 0: 2016}

Train row kinds:

```json
{
  "base_anchor_compound": 2880,
  "base_anchor_pair_event": 480,
  "sparse_flipped_compound": 576,
  "sparse_flipped_pair_event": 96
}
```

## Mean accuracy over seeds

### atp_probe_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained/full/aligned | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 0.999 | tennis_atp:1.000 |
| pretrained/full/exposure | 1.000 | 0.325 | 0.741 | 0.110 | 0.118 | 0.123 | 0.150 | tennis_atp:0.325 |
| pretrained/full/flipped | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tennis_atp:0.000 |

### atp_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained/full/aligned | 1.000 | 0.449 | 0.905 | 0.218 | 0.219 | 0.223 | 0.217 | tennis_atp:0.449 |
| pretrained/full/exposure | 1.000 | 0.395 | 0.877 | 0.152 | 0.125 | 0.156 | 0.170 | tennis_atp:0.395 |
| pretrained/full/flipped | 1.000 | 0.458 | 0.630 | 0.381 | 0.377 | 0.362 | 0.375 | tennis_atp:0.458 |

### pair_probe_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained/full/aligned | 1.000 | 1.000 | 1.000 |  |  |  |  | badminton:1.000, football:1.000, tennis:1.000 |
| pretrained/full/exposure | 1.000 | 0.737 | 0.737 |  |  |  |  | badminton:0.752, football:0.826, tennis:0.723 |
| pretrained/full/flipped | 1.000 | 0.000 | 0.000 |  |  |  |  | badminton:0.000, football:0.000, tennis:0.000 |

### pair_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| pretrained/full/aligned | 1.000 | 0.996 | 0.996 |  |  |  |  | badminton:0.998, football:1.000, tennis:0.995 |
| pretrained/full/exposure | 1.000 | 0.946 | 0.946 |  |  |  |  | badminton:0.965, football:0.932, tennis:0.941 |
| pretrained/full/flipped | 1.000 | 0.594 | 0.594 |  |  |  |  | badminton:0.581, football:0.614, tennis:0.596 |

## Orientation contrasts

### pretrained/full

| eval | metric | value |
|---|---|---:|
| atp_probe_heldfam | aligned_minus_exposure_event_role | 0.259 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state | 0.890 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state_conflict | 0.882 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state | 0.877 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state_conflict_context | 0.849 |
| atp_probe_heldfam | aligned_minus_flipped_event_role | 1.000 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state | 1.000 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state_conflict | 1.000 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state | 0.999 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state_conflict_context | 0.999 |
| atp_heldword_heldfam | aligned_minus_exposure_event_role | 0.027 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state | 0.066 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state_conflict | 0.094 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state | 0.067 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state_conflict_context | 0.046 |
| atp_heldword_heldfam | aligned_minus_flipped_event_role | 0.275 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state | -0.162 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state_conflict | -0.158 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state | -0.139 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state_conflict_context | -0.158 |
| pair_probe_heldfam | aligned_minus_exposure_event_role | 0.263 |
| pair_probe_heldfam | aligned_minus_flipped_event_role | 1.000 |
| pair_heldword_heldfam | aligned_minus_exposure_event_role | 0.050 |
| pair_heldword_heldfam | aligned_minus_flipped_event_role | 0.402 |

## Scientific reading

In `pretrained/full` on ATP probe held-family contexts, aligned-minus-flipped was +1.000 for event role, +1.000 for conflict focal-state rows, and +0.999 for untouched-state rows. Aligned-minus-exposure was +0.259 for event and +0.882 for conflict focal state.

The experiment is a small classifier/probe over an existing BabyLM DeBERTa checkpoint, not evidence that a Strict-Small LM trajectory will improve. A useful positive signal would be simultaneous aligned-over-flipped movement for event_role, conflict focal_state, and untouched_state under held wording/domain; failure or poor train fit means the present bridge is still too weak for a full corpus intervention.

JSON: `experiments/archive/representation_and_objectives/data/pretrained_atp_slot_orientation_full3/pretrained_atp_slot_orientation_summary.json`
