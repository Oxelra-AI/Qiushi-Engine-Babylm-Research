# Compact anchor match — compute allocation correction and evaluation readiness

## What changed

The planned two-arm compact-versus-interleaved training comparison was cancelled before any model was launched. It produced preflight records only and provides no model evidence.

The compact arm was already in training and supplies the common anchor. The anchor-match audit found:

- compact 100M stream: `experiments/archive/representation_and_objectives/data/fw_source_breadth_arm/fw_preserved_compact_view_100M.jsonl`, SHA `c8d7f24b5edd2dad21f589c8cde72671d0b76038d9632fcd3d6c79178a24be68`
- shared legal 16k tokenizer: SHA `e70d167f620668130f88744998bd1bacbe05f6d0735012144fe501b271b04366`
- DeBERTa-v2 8x480, 8 heads, FFN x4
- fixed WWM 0.15, AdamW lr 0.001, warmup 0.06, weight decay 0.01
- seed 43, extra init seed 43022, train RNG seed 43023
- 100,000,000 words, 641,830 rows, checkpoints every 1M

Evidence: `experiments/archive/representation_and_objectives/data/compact_anchor_match/compact_anchor_match.json` and `research/notes/representation_and_objectives/compact_anchor_match.md`.

## Important coordinate correction

companion analysis compact is not a valid anchor for the original companion analysis queued microbatch-coordinate pair, because companion analysis had queued `accumulated_masking_curriculum_trainer.py` while companion analysis uses the COMPACT_EXPERIENCE full-batch trainer `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py` (SHA `d3cd4a46c0dd28b5dbc3443e564b72879c393d6c7e5212590764aa7786fbd5d4`). The accumulated trainer is a valid memory repair but is not bit-identical because dropout is executed over smaller forward-call shapes.

Therefore the repaired plan is: reuse compact as the shared compact anchor and train only interleaved whole-sentence breadth arm on the same COMPACT_EXPERIENCE full-batch coordinate. This preserves a valid comparison while saving one duplicate 100M H100 training run.

## Launched Comparison

The `fw_interleaved_breadth_fullbatch_train_anchor` comparison was launched; no completed result is established here.

Command:

`PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/archive/representation_and_objectives/scripts/guarded_interleaved_breadth_fullbatch_anchor.py --launch --full-word-count`

Write targets:

- `experiments/archive/representation_and_objectives/data/fw_interleaved_breadth_fullbatch_launch`
- `experiments/archive/representation_and_objectives/training/runs/fw_source_breadth_interleaved_wholesentence_fullbatch_shared16k_seed43022`

This script first waits for the compact anchor to finish successfully, then waits for one genuinely free H100 and trains only the interleaved breadth arm under the exact full-batch recipe.

## Evaluation assets prepared

Scripts:

- `experiments/archive/representation_and_objectives/training/scripts/full_eval_fw_shared_anchor.py`
- `experiments/archive/representation_and_objectives/scripts/fw_shared_anchor_posttrain_eval_controller.py`
- `experiments/archive/representation_and_objectives/scripts/fw_anchor_result_synthesis.py`

Prepared targets:

- `fw_compact_fullbatch_seed43022`: compact anchor
- `fw_breadth_rowblock_fullbatch_seed43022`: row-block whole-sentence breadth
- `fw_breadth_interleaved_fullbatch_seed43022`: interleaved whole-sentence breadth

Dry-run preflights currently show the models are still in training and the interleaved model has not started. When each training completes, run the posttrain controller on a free GPU for that target. It evaluates non-EWoK/non-AoA official columns, current-pristine 7,618-row EWoK, official min-context-zero AoA, and hardened pristine collation.

After at least compact and interleaved summaries exist, run:

`PYTHONDONTWRITEBYTECODE=1 python3 -B experiments/archive/representation_and_objectives/scripts/fw_anchor_result_synthesis.py`

This writes `experiments/archive/representation_and_objectives/data/fw_anchor_result_synthesis/fw_anchor_result_synthesis.json` and a score table.

## Scientific interpretation to preserve

- Compact beating both row-block and interleaved breadth supports same-proposition compact recurrence / aligned restatement as the stronger data substrate.
- Interleaved breadth beating compact means the current compact-pair design is not better than additional coherent FineWeb source coverage under this legal representation.
- Row-block vs interleaved disagreement would mean internal layout and local alternation are load-bearing, and the route should be read at the level of task-family vectors rather than only Overall.
- Similar compact/breadth movement makes the literal `source_repeat` arm the next attribution discriminator only if it can change the SOTA path.
