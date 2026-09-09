# earlier analysis corrected segmented replay schedule

Status: `ALPHA075_SEGMENTED_REPLAY_SCHEDULE`.
Known layerwise exchange readout and gradient anatomy prefix: 101 batches, 3,992,800 tail words, total 86,005,295, rows 530944-556790.
Prefix matches layerwise exchange readout and gradient anatomy log exactly: True.
Continuation starts at row 556791 and has 354 batches, ending row 647399; full segmented replay total batches 455 equals original LR schedule_total 455: True.
Official chck_90M should be saved after update 203 at actual total 90,037,558.
Official chck_100M should be saved after update 455 at actual total 100,000,000.

This repairs the naive full-suffix audit: a single DataLoader would make update 101 a full batch and fail to reproduce the known final86 state.
JSON: `experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_segmented_replay_schedule.json`
