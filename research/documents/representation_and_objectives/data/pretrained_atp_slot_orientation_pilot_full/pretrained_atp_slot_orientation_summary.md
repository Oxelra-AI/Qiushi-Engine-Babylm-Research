# earlier analysis pretrained ATP slot-orientation probe

Small pretrained-language bridge test for reusable filler interface plus sparse relation-coordinate orientation on independently attested ATP event/state and paired world pilot result paired-world material.

Model: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
Device: `cuda`, elapsed 64.59s

## Data construction

```json
{
  "atp_train_primary": 40,
  "atp_eval_primary": 32,
  "pair_train": 32,
  "pair_eval": 32,
  "atp_eval_conflict_worlds": 14
}
```

### Arm `exposure`

Train rows: 1344; label balance {1: 704, 0: 640}

Train row kinds:

```json
{
  "base_anchor_compound": 960,
  "base_anchor_pair_event": 128,
  "sparse_exposure": 256
}
```

### Arm `aligned`

Train rows: 1312; label balance {1: 656, 0: 656}

Train row kinds:

```json
{
  "base_anchor_compound": 960,
  "base_anchor_pair_event": 128,
  "sparse_aligned_compound": 192,
  "sparse_aligned_pair_event": 32
}
```

### Arm `flipped`

Train rows: 1312; label balance {1: 656, 0: 656}

Train row kinds:

```json
{
  "base_anchor_compound": 960,
  "base_anchor_pair_event": 128,
  "sparse_flipped_compound": 192,
  "sparse_flipped_pair_event": 32
}
```

## Mean accuracy over seeds

### atp_probe_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full/aligned | 1.000 | 0.992 | 1.000 | 0.977 | 0.964 | 1.000 | 1.000 | tennis_atp:0.992 |
| full/exposure | 1.000 | 0.405 | 0.648 | 0.262 | 0.214 | 0.305 | 0.277 | tennis_atp:0.405 |
| full/flipped | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | tennis_atp:0.000 |

### atp_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full/aligned | 1.000 | 0.391 | 0.832 | 0.195 | 0.241 | 0.145 | 0.179 | tennis_atp:0.391 |
| full/exposure | 1.000 | 0.387 | 0.586 | 0.285 | 0.277 | 0.289 | 0.295 | tennis_atp:0.387 |
| full/flipped | 1.000 | 0.296 | 0.566 | 0.156 | 0.205 | 0.164 | 0.205 | tennis_atp:0.296 |

### pair_probe_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full/aligned | 1.000 | 1.000 | 1.000 |  |  |  |  | badminton:1.000, football:1.000, tennis:1.000 |
| full/exposure | 1.000 | 0.594 | 0.594 |  |  |  |  | badminton:0.607, football:0.500, tennis:0.594 |
| full/flipped | 1.000 | 0.000 | 0.000 |  |  |  |  | badminton:0.000, football:0.000, tennis:0.000 |

### pair_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full/aligned | 1.000 | 0.938 | 0.938 |  |  |  |  | badminton:0.964, football:1.000, tennis:0.927 |
| full/exposure | 1.000 | 0.812 | 0.812 |  |  |  |  | badminton:0.857, football:0.750, tennis:0.802 |
| full/flipped | 1.000 | 0.633 | 0.633 |  |  |  |  | badminton:0.643, football:0.750, tennis:0.625 |

## Orientation contrasts

### full

| eval | metric | value |
|---|---|---:|
| atp_probe_heldfam | aligned_minus_exposure_event_role | 0.352 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state | 0.715 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state_conflict | 0.750 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state | 0.695 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state_conflict_context | 0.723 |
| atp_probe_heldfam | aligned_minus_flipped_event_role | 1.000 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state | 0.977 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state_conflict | 0.964 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state | 1.000 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state_conflict_context | 1.000 |
| atp_heldword_heldfam | aligned_minus_exposure_event_role | 0.246 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state | -0.090 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state_conflict | -0.036 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state | -0.145 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state_conflict_context | -0.116 |
| atp_heldword_heldfam | aligned_minus_flipped_event_role | 0.266 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state | 0.039 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state_conflict | 0.036 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state | -0.020 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state_conflict_context | -0.027 |
| pair_probe_heldfam | aligned_minus_exposure_event_role | 0.406 |
| pair_probe_heldfam | aligned_minus_flipped_event_role | 1.000 |
| pair_heldword_heldfam | aligned_minus_exposure_event_role | 0.125 |
| pair_heldword_heldfam | aligned_minus_flipped_event_role | 0.305 |

## Scientific reading

In `full` on ATP probe held-family contexts, aligned-minus-flipped was +1.000 for event role, +0.964 for conflict focal-state rows, and +1.000 for untouched-state rows. Aligned-minus-exposure was +0.352 for event and +0.750 for conflict focal state.

The experiment is a small classifier/probe over an existing BabyLM DeBERTa checkpoint, not evidence that a Strict-Small LM trajectory will improve. A useful positive signal would be simultaneous aligned-over-flipped movement for event_role, conflict focal_state, and untouched_state under held wording/domain; failure or poor train fit means the present bridge is still too weak for a full corpus intervention.

JSON: `experiments/archive/representation_and_objectives/data/pretrained_atp_slot_orientation_pilot_full/pretrained_atp_slot_orientation_summary.json`
