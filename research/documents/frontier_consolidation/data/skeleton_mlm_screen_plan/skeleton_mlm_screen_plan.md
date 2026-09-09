# source wide skeleton recurrence integrated skeleton-vs-compact MLM screen dry plan

No command in this file was launched.

Question: Does source-wide exact-length skeleton recurrence reproduce the compact-view advantage over first-N repetition under the bidirectional MLM coordinate?

## Why This Was Not Launched
- DeBERTa common-grid results were still pending.
- Triangle official-compatible scores were not yet available.
- Source-only skeleton variants are telegraphic, so their scientific value requires assessment before training.

## Planned train/eval commands

### compact
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/train_density_arm.py --jsonl experiments/archive/frontier_consolidation/data/skeleton_reinvest_pool_scaffold/cleanqwen_fineweb_compact_skeleton_reinvest_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_compact_seed43022_40M --total-words 40000000 --checkpoint-interval 20000000 --seed 43022 --gpu 0
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_compact_seed43022_40M --out-dir experiments/archive/frontier_consolidation/data/skeleton_screen_eval_compact_40M --gpu 0 --endpoints chck_20M chck_40M
```

### prefix_repeat
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/train_density_arm.py --jsonl experiments/archive/frontier_consolidation/data/skeleton_reinvest_pool_scaffold/cleanqwen_fineweb_prefix_repeat_skeleton_reinvest_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_prefix_repeat_seed43022_40M --total-words 40000000 --checkpoint-interval 20000000 --seed 43022 --gpu 0
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_prefix_repeat_seed43022_40M --out-dir experiments/archive/frontier_consolidation/data/skeleton_screen_eval_prefix_repeat_40M --gpu 0 --endpoints chck_20M chck_40M
```

### content_spread
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/train_density_arm.py --jsonl experiments/archive/frontier_consolidation/data/skeleton_reinvest_pool_scaffold/cleanqwen_fineweb_content_spread_skeleton_reinvest_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_content_spread_seed43022_40M --total-words 40000000 --checkpoint-interval 20000000 --seed 43022 --gpu 0
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_content_spread_seed43022_40M --out-dir experiments/archive/frontier_consolidation/data/skeleton_screen_eval_content_spread_40M --gpu 0 --endpoints chck_20M chck_40M
```

### scored_source_skeleton
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/train_density_arm.py --jsonl experiments/archive/frontier_consolidation/data/skeleton_reinvest_pool_scaffold/cleanqwen_fineweb_scored_source_skeleton_skeleton_reinvest_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/compliant_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_scored_source_skeleton_seed43022_40M --total-words 40000000 --checkpoint-interval 20000000 --seed 43022 --gpu 0
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/selected_mlm_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/skeleton_screen_scored_source_skeleton_seed43022_40M --out-dir experiments/archive/frontier_consolidation/data/skeleton_screen_eval_scored_source_skeleton_40M --gpu 0 --endpoints chck_20M chck_40M
```

JSON: `experiments/archive/frontier_consolidation/data/skeleton_mlm_screen_plan/skeleton_mlm_screen_plan.json`
