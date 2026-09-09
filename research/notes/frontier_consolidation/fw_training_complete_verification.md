# fw absolute progress decision discipline — FW compact-view vs source-breadth: training complete, verified

Both earlier analysis paired 100M trainings completed cleanly (returncode 0).

## Training completion facts

| Arm | run_dir | loss_first | loss_last | steps | word_exposure | ckpts |
|---|---|---:|---:|---:|---:|---:|
| compact_view | `training/runs/fw_compact_view_shared16k_seed43022` | 9.7905 | 2.4668 | 2508 | 100,000,000 | 100 (chck_1M..chck_100M) |
| source_breadth | `training/runs/fw_source_breadth_shared16k_seed43022` | 9.7884 | 2.4749 | 2508 | 100,000,000 | 100 (chck_1M..chck_100M) |

Both seed 43022, DeBERTa-v2 8×480, 34,467,424 params, WWM fixed 0.15, identical LR schedule (peak 0.001, warmup 0.06, cosine to 0), batch 256, seq 256.

## Compliance / matched-condition verification

- Actual `tokenizer.json` SHA in both saved checkpoints = `e70d167f62066813...` = the shared 16k tokenizer.
- That tokenizer was trained on `shared_tokenizer_pool_words = 9,681,149` (≤10M budget) per `experiments/archive/representation_and_objectives/data/fw_full_arms/fw_preservation_aware_arms_manifest.json`.
- The `tokenizer_label = baseline16k` string in scientific_metrics.json is a cosmetic trainer default and does **not** reflect the illegal 100M baseline tokenizer; the loaded tokenizer SHA is the compliant shared one.
- Both arms share identical tokenizer SHA, so the comparison is tokenizer-matched.

## Training-dynamics signal (from run logs)

compact_view has consistently **lower** MLM loss across the whole trajectory, e.g. at earlier analysis (~80M): compact 2.4351 vs breadth 2.6933; final 2.4668 vs 2.4749. This is expected: faithful restatements of the same source propositions are more predictable than independent FineWeb sentences. Lower MLM loss does not by itself predict better downstream transfer (density eval repair established downstream behavior, not MLM loss, is the governing judgment).

## Next

Cheap official-compatible eval at 70M/80M for both arms is running. After it completes:
1. Run `fw_absolute_decision.py` to separate within-pair mechanism delta from absolute progress vs the legal reference (70M ref cheap7 42.6086, 80M ref 42.9486).
2. Apply the discipline in `notes/fw_absolute_progress_decision_discipline.md`.
