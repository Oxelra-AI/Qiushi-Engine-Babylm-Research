# earlier analysis alpha0.75 exact tail replay schedule audit

Status: `ALPHA075_TAIL_SCHEDULE_AUDIT`.
From the verified 82,012,495-word anchor, the whole-row remaining suffix contains 17,987,505 words and ends at total 100,000,000 before row EOF would exceed 100M.
Full deterministic replay has 455 batches; original layerwise exchange readout and gradient anatomy schedule_total is 455; match = True.
The first 101 batches match the known layerwise exchange readout and gradient anatomy training log exactly on row ranges, batch words, tail words, and total words: False.
Known final86 is update 101, tail 3994234, total 86006729.
Genuine chck_90M crossing: {'target_tail_words': 7987505, 'crossing': {'update': 203, 'source_row_start': 582656, 'source_row_end': 582911, 'batch_words': 39567, 'tail_main_words': 8026476, 'total_consumed_words': 90038971}}.
Genuine chck_100M final/crossing: {'target_tail_words': 17987505, 'crossing': {'update': 455, 'source_row_start': 647168, 'source_row_end': 647399, 'batch_words': 35791, 'tail_main_words': 17987505, 'total_consumed_words': 100000000, 'partial_batch_rows': 232}}.

The next GPU action should be a guarded replay from the 82M anchor, not optimizer restart from alpha0.75 final. It must abort before extension if known 83-86 hashes are not reproduced.
JSON: `experiments/archive/representation_and_objectives/data/alpha075_exact_replay_plan/alpha075_tail_schedule_audit.json`
