# earlier analysis pretrained ATP slot-orientation probe

Small pretrained-language bridge test for reusable filler interface plus sparse relation-coordinate orientation on independently attested ATP event/state and paired world pilot result paired-world material.

Model: `experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
Device: `cuda`, elapsed 18.79s

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
| top2/aligned | 0.501 | 0.501 | 0.504 | 0.496 | 0.500 | 0.504 | 0.491 | tennis_atp:0.501 |
| top2/exposure | 0.586 | 0.503 | 0.492 | 0.516 | 0.545 | 0.500 | 0.500 | tennis_atp:0.503 |
| top2/flipped | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | tennis_atp:0.500 |

### atp_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| top2/aligned | 0.501 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | tennis_atp:0.500 |
| top2/exposure | 0.586 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | tennis_atp:0.500 |
| top2/flipped | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | 0.500 | tennis_atp:0.500 |

### pair_probe_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| top2/aligned | 0.501 | 0.500 | 0.500 |  |  |  |  | badminton:0.500, football:0.500, tennis:0.500 |
| top2/exposure | 0.586 | 0.492 | 0.492 |  |  |  |  | badminton:0.464, football:0.500, tennis:0.500 |
| top2/flipped | 0.500 | 0.500 | 0.500 |  |  |  |  | badminton:0.500, football:0.500, tennis:0.500 |

### pair_heldword_heldfam

| regime/arm | train acc | overall | event | focal state | conflict focal state | untouched state | untouched in conflict ctx | sport/domain |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| top2/aligned | 0.501 | 0.500 | 0.500 |  |  |  |  | badminton:0.500, football:0.500, tennis:0.500 |
| top2/exposure | 0.586 | 0.500 | 0.500 |  |  |  |  | badminton:0.500, football:0.500, tennis:0.500 |
| top2/flipped | 0.500 | 0.500 | 0.500 |  |  |  |  | badminton:0.500, football:0.500, tennis:0.500 |

## Orientation contrasts

### top2

| eval | metric | value |
|---|---|---:|
| atp_probe_heldfam | aligned_minus_exposure_event_role | 0.012 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state | -0.020 |
| atp_probe_heldfam | aligned_minus_exposure_focal_state_conflict | -0.045 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state | 0.004 |
| atp_probe_heldfam | aligned_minus_exposure_untouched_state_conflict_context | -0.009 |
| atp_probe_heldfam | aligned_minus_flipped_event_role | 0.004 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state | -0.004 |
| atp_probe_heldfam | aligned_minus_flipped_focal_state_conflict | 0.000 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state | 0.004 |
| atp_probe_heldfam | aligned_minus_flipped_untouched_state_conflict_context | -0.009 |
| atp_heldword_heldfam | aligned_minus_exposure_event_role | 0.000 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state | 0.000 |
| atp_heldword_heldfam | aligned_minus_exposure_focal_state_conflict | 0.000 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state | 0.000 |
| atp_heldword_heldfam | aligned_minus_exposure_untouched_state_conflict_context | 0.000 |
| atp_heldword_heldfam | aligned_minus_flipped_event_role | 0.000 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state | 0.000 |
| atp_heldword_heldfam | aligned_minus_flipped_focal_state_conflict | 0.000 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state | 0.000 |
| atp_heldword_heldfam | aligned_minus_flipped_untouched_state_conflict_context | 0.000 |
| pair_probe_heldfam | aligned_minus_exposure_event_role | 0.008 |
| pair_probe_heldfam | aligned_minus_flipped_event_role | 0.000 |
| pair_heldword_heldfam | aligned_minus_exposure_event_role | 0.000 |
| pair_heldword_heldfam | aligned_minus_flipped_event_role | 0.000 |

## Scientific reading

In `top2` on ATP probe held-family contexts, aligned-minus-flipped was +0.004 for event role, +0.000 for conflict focal-state rows, and +0.004 for untouched-state rows. Aligned-minus-exposure was +0.012 for event and -0.045 for conflict focal state.

The experiment is a small classifier/probe over an existing BabyLM DeBERTa checkpoint, not evidence that a Strict-Small LM trajectory will improve. A useful positive signal would be simultaneous aligned-over-flipped movement for event_role, conflict focal_state, and untouched_state under held wording/domain; failure or poor train fit means the present bridge is still too weak for a full corpus intervention.

JSON: `experiments/archive/representation_and_objectives/data/pretrained_atp_slot_orientation_pilot_top2/pretrained_atp_slot_orientation_summary.json`
