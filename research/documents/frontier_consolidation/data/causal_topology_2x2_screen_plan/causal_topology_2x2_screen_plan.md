# topology 2x2 scaffold and deberta pending state causal topology 2×2 short-screen plan

This is a dry command plan only. No training or evaluation was launched.

Precondition: Do not run unless triangle or other direct evidence revives topology interaction despite reciprocal multiview mechanism and scaffold copied-token warning and after pending DeBERTa seed/scale grids are interpreted.

Settings: seed 43022, total_words 20,000,000, checkpoint_interval 10,000,000, batch_size 256, eval endpoints ['chck_10M', 'chck_20M'].

## Training commands

### compact_oneway

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python -B experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py --pool experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/causal_topology2x2_compact_oneway_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_compact_oneway_seed43022_20M --gpu 0 --seed 43022 --total-words 20000000 --checkpoint-interval 10000000 --batch-size 256
```

### repeat_oneway

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 python -B experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py --pool experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/causal_topology2x2_repeat_oneway_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_repeat_oneway_seed43022_20M --gpu 0 --seed 43022 --total-words 20000000 --checkpoint-interval 10000000 --batch-size 256
```

### compact_reciprocal

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python -B experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py --pool experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/causal_topology2x2_compact_reciprocal_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_compact_reciprocal_seed43022_20M --gpu 0 --seed 43022 --total-words 20000000 --checkpoint-interval 10000000 --batch-size 256
```

### repeat_reciprocal

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 python -B experiments/archive/frontier_consolidation/scripts/causal_gpt_trainer.py --pool experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/causal_topology2x2_repeat_reciprocal_10M.jsonl --tokenizer experiments/archive/frontier_consolidation/data/causal_transfer_scaffold/neutral_tokenizer --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_repeat_reciprocal_seed43022_20M --gpu 0 --seed 43022 --total-words 20000000 --checkpoint-interval 10000000 --batch-size 256
```

## Evaluation commands

### compact_oneway

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_compact_oneway_seed43022_20M --label topology2x2_compact_oneway --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_compact_oneway_20M --endpoints chck_10M chck_20M
```

### repeat_oneway

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_repeat_oneway_seed43022_20M --label topology2x2_repeat_oneway --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_repeat_oneway_20M --endpoints chck_10M chck_20M
```

### compact_reciprocal

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_compact_reciprocal_seed43022_20M --label topology2x2_compact_reciprocal --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_compact_reciprocal_20M --endpoints chck_10M chck_20M
```

### repeat_reciprocal

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=1 python -B experiments/archive/frontier_consolidation/scripts/selected_causal_checkpoint_eval.py --run-dir experiments/archive/frontier_consolidation/training/runs/topology2x2_repeat_reciprocal_seed43022_20M --label topology2x2_repeat_reciprocal --gpu 0 --out-dir experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_repeat_reciprocal_20M --endpoints chck_10M chck_20M
```

## Comparison command

```bash
PYTHONDONTWRITEBYTECODE=1 python -B experiments/archive/frontier_consolidation/scripts/compare_causal_topology_2x2.py --compact-oneway experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_compact_oneway_20M/selected_causal_trajectory.json --repeat-oneway experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_repeat_oneway_20M/selected_causal_trajectory.json --compact-reciprocal experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_compact_reciprocal_20M/selected_causal_trajectory.json --repeat-reciprocal experiments/archive/frontier_consolidation/data/topology2x2_selected_eval_repeat_reciprocal_20M/selected_causal_trajectory.json --scaffold-manifest experiments/archive/frontier_consolidation/data/causal_topology_2x2_scaffold/manifest.json --out-dir experiments/archive/frontier_consolidation/data/topology2x2_interaction_compare_20M
```

JSON: `experiments/archive/frontier_consolidation/data/causal_topology_2x2_screen_plan/causal_topology_2x2_screen_plan.json`
