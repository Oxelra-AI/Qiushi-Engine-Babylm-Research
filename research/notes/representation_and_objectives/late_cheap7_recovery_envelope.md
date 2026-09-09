# fw 70m execution synthesis — late cheap7 recovery envelope

This CPU-only mining step asks how much local completed trajectories usually move in no-AoA/equal7 from 70M or 80M to 100M. It is not a substitute for completed FW evidence; it calibrates the risk of spending full endpoint evaluation on a low 70M cheap7.

Extracted 23 checkpoint rows and 10 late-to-100M pairs from 7 summary files.
70M→100M cheap7 deltas: n=2, min=0.069, mean=0.215, max=0.361; values=0.069, 0.361.
80M→100M cheap7 deltas: n=3, min=-0.240, mean=-0.132, max=-0.008; values=-0.149, -0.240, -0.008.

## FW 70M requirement

| arm | 70M cheap7 | needed to leader cheap7 43.77 | historical 70M→100M max | count >= needed |
|---|---:|---:|---:|---:|
| `compact_view_70M` | 42.657 | 1.113 | 0.361 | 0/2 |
| `source_breadth_rowblock_70M` | 42.031 | 1.739 | 0.361 | 0/2 |

The extracted local trajectories make a late jump large enough to bring the 70M FW arms to leader cheap7 look unlikely: compact would need +1.113 and row-block breadth +1.739, while the observed 70M→100M range here is at most +0.361. This does not by itself close the 100M endpoints, because the FW data family has its own trajectory and cheap 80M/100M files are not complete yet. It does justify withholding new full official evaluations until a complete 100M cheap7 or other strong endpoint evidence appears.

## Files
- JSON: `experiments/archive/representation_and_objectives/data/late_cheap7_recovery_envelope/late_cheap7_recovery_envelope.json`
- checkpoint rows: `experiments/archive/representation_and_objectives/data/late_cheap7_recovery_envelope/late_cheap7_checkpoint_rows.csv`
- late pairs: `experiments/archive/representation_and_objectives/data/late_cheap7_recovery_envelope/late_cheap7_to_100M_pairs.csv`
- required recovery table: `experiments/archive/representation_and_objectives/data/late_cheap7_recovery_envelope/fw_70m_required_recovery_vs_history.csv`
