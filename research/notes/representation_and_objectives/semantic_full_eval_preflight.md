# fineweb rewrite faithfulness slice semantic-view selected full-eval preflight

All checked components ready: **True**. This did not launch evaluation.

## Checks

- Python script AST checks all pass: True.
- Shell launch syntax pass: True.
- Corrected BabyLM scoring self-test pass: True, AoA scale 100.0.
- COMPACT_EXPERIENCE repaired AoA helper available: True.
- `semantic_view_treatment`: AoA 19-step ladder complete=True, all 1M..100M checkpoints=True, exposure=100000000, last_loss=2.5477638244628906.
- `original_packet_local`: AoA 19-step ladder complete=True, all 1M..100M checkpoints=True, exposure=100000000, last_loss=2.4126060009002686.

## Use after no-AoA

If the semantic-view no-AoA trajectory shows a meaningful treatment gain, run selected full official-style evaluation with `TREAT_ENDPOINT=<ckpt> PACKET_ENDPOINT=<same_ckpt> bash experiments/archive/representation_and_objectives/training/scripts/launch_semantic_view_full_eval_selected.sh`. The zero-shot/Reading values will be prefilled from the frozen no-AoA run; SuperGLUE and AoA will be measured.

JSON: `experiments/archive/representation_and_objectives/data/semantic_full_eval_preflight/semantic_full_eval_preflight.json`
