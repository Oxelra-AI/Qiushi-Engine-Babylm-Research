# fineweb rewrite faithfulness slice semantic-view training pair complete

This records the matched training artifacts only. The scientific effect is still pending the official-compatible no-AoA trajectory.

| target | word exposure | steps | checkpoints | last loss | example file |
|---|---:|---:|---:|---:|---|
| semantic_view_treatment | 100,000,000 | 2,665 | 100 | 2.547764 | `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/semantic_view_treatment_100M.jsonl` |
| original_packet_local | 100,000,000 | 2,665 | 100 | 2.412606 | `experiments/archive/representation_and_objectives/training/data/semantic_view/full_contrast_capped1/original_packet_local_100M.jsonl` |

Core training fields identical across arms: **True**.

Treatment last loss minus packet-local last loss: **0.135158**. This reflects the different input distribution/repetition structure and is not downstream competence evidence.

Changed block exposure: treatment `simplewiki_semantic_view` = 8,201,870; control `simplewiki_original_packet_local` = 8,201,870.

Downstream result awaited: `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json`.

JSON: `experiments/archive/representation_and_objectives/data/semantic_training_pair/semantic_training_pair_summary.json`
