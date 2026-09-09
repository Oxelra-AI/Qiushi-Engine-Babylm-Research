# fineweb high anchor slice semantic-view training snapshot while downstream evaluation is pending

This records the completed treatment/control training pair from representation_and_objectives without reading the locked downstream evaluation directory. It is not a BabyLM score.

| arm | exposure | steps | checkpoints | last loss | changed block |
|---|---:|---:|---:|---:|---|
| semantic_view_treatment | 100,000,000 | 2,665 | 100 | 2.547764 | 8,201,870 words over ten passes |
| original_packet_local | 100,000,000 | 2,665 | 100 | 2.412606 | 8,201,870 words over ten passes |

Core training fields identical: **True**. Treatment last loss minus packet-local last loss: **0.135158**; this is distribution/repetition evidence, not downstream competence.

The decisive result is `experiments/archive/representation_and_objectives/data/semantic_view_noaoa_eval/semantic_view_packet_local_delta_summary.json`. It remained unavailable at the time of this record.

JSON: `experiments/archive/frontier_consolidation/data/semantic_view_pending/semantic_view_training_snapshot.json`
