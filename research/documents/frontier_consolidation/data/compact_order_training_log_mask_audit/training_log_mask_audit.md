# compact order evaluation and channel readout plan compact ordered-vs-scrambled training-log mask audit

Exact post-training file/log readout. It is meant to keep narrow official/channel differences grounded in finite training, exact exposure, checkpoint presence, and realized masked-token drift.

## ordered
- log rows: 1012; total words from logs: 40000000; total masked tokens: 8577444; finite losses: True
- checkpoint files: {'chck_20M': True, 'chck_40M': True}
- last log: {'step': 1012, 'loss': 3.0742650032043457, 'lr': 0.0007099445507801323, 'batch_words': 22068, 'cumulative_word_exposure': 40000000, 'seq_len': 256, 'masked_tokens': 4835, 'effective_mask_rate': 0.1522, 'mask_mode': 'wwm', 'mask_prob_nominal': 0.15, 'elapsed_sec': 1819.0}

## scrambled
- log rows: 1012; total words from logs: 40000000; total masked tokens: 8577958; finite losses: True
- checkpoint files: {'chck_20M': True, 'chck_40M': True}
- last log: {'step': 1012, 'loss': 3.0799901485443115, 'lr': 0.0007099445507801323, 'batch_words': 22068, 'cumulative_word_exposure': 40000000, 'seq_len': 256, 'masked_tokens': 4835, 'effective_mask_rate': 0.1522, 'mask_mode': 'wwm', 'mask_prob_nominal': 0.15, 'elapsed_sec': 1519.9}

## Ordered minus scrambled
- paired steps: 1012; step count equal: True; batch words equal: True
- total masked-token delta: -514
- per-batch masked-token delta stats: {'n': 1012, 'mean': -0.5079051383399209, 'median': 0.0, 'p05': 0.0, 'p25': 0.0, 'p75': 0.0, 'p95': 0.0, 'min': -147.0, 'max': 110.0}
- per-region deltas: {'changed_full': {'steps': 44, 'delta_masked_tokens': -445, 'delta_loss_mean': -0.5849895910783247}, 'changed_mixed': {'steps': 4, 'delta_masked_tokens': -52, 'delta_loss_mean': -0.40836864709854126}, 'unchanged': {'steps': 964, 'delta_masked_tokens': -17, 'delta_loss_mean': -0.019018927797736965}}

## lead route assessment after source use probe changed-block tokenization/CPU-proxy context
- Δ active tokens 0.0; Δ candidate groups 8.0; Δ view-active tokens 165.0; CPU-proxy Δ masked tokens 557.0; CPU-proxy Δ masked view/source 472.0/183.0.

JSON: `experiments/archive/frontier_consolidation/data/compact_order_training_log_mask_audit/training_log_mask_audit.json`
