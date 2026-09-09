# shuffle control audit conclusion dual-view training audit — aligned_vs_shuffled_completed

## aligned
Run: `experiments/archive/frontier_consolidation/training/runs/dualview_aligned_20M_seed43022`
metrics exists: `True`; log lines: `481`; chck_20M: `False`; final: `True`
charged/main/aux words: `19988304` / `19021584` / `966720`; updates `481`; final loss `4.379467487335205`

## shuffled
Run: `experiments/archive/frontier_consolidation/training/runs/dualview_shuffled_20M_seed43022`
metrics exists: `True`; log lines: `481`; chck_20M: `False`; final: `True`
charged/main/aux words: `19988304` / `19021584` / `966720`; updates `481`; final loss `4.439597129821777`

## aligned_vs_shuffled
```json
{
  "metric_equality": {
    "updates": {
      "a": 481,
      "b": 481,
      "equal": true
    },
    "total_main_word_exposure": {
      "a": 19021584,
      "b": 19021584,
      "equal": true
    },
    "total_aux_word_exposure": {
      "a": 966720,
      "b": 966720,
      "equal": true
    },
    "total_charged_words": {
      "a": 19988304,
      "b": 19988304,
      "equal": true
    },
    "schedule_total": {
      "a": 2529,
      "b": 2529,
      "equal": true
    },
    "total_params": {
      "a": 35463008,
      "b": 35463008,
      "equal": true
    },
    "adapter_params": {
      "a": 995584,
      "b": 995584,
      "equal": true
    },
    "aux_micro_batch_size": {
      "a": 8,
      "b": 8,
      "equal": true
    },
    "aux_loss_batches": {
      "a": 481,
      "b": 481,
      "equal": true
    }
  },
  "config_equality": {
    "seed": {
      "a": 43,
      "b": 43,
      "equal": true
    },
    "extra_init_seed": {
      "a": 43022,
      "b": 43022,
      "equal": true
    },
    "train_rng_seed": {
      "a": 43023,
      "b": 43023,
      "equal": true
    },
    "batch_size": {
      "a": 256,
      "b": 256,
      "equal": true
    },
    "seq_length": {
      "a": 256,
      "b": 256,
      "equal": true
    },
    "aux_max_length": {
      "a": 464,
      "b": 464,
      "equal": true
    },
    "learning_rate": {
      "a": 0.001,
      "b": 0.001,
      "equal": true
    },
    "warmup_fraction": {
      "a": 0.06,
      "b": 0.06,
      "equal": true
    },
    "weight_decay": {
      "a": 0.01,
      "b": 0.01,
      "equal": true
    },
    "mask_prob": {
      "a": 0.15,
      "b": 0.15,
      "equal": true
    },
    "lr_total_steps": {
      "a": 2529,
      "b": 2529,
      "equal": true
    },
    "adapter_bottleneck": {
      "a": 128,
      "b": 128,
      "equal": true
    },
    "adapter_scale": {
      "a": 1.0,
      "b": 1.0,
      "equal": true
    },
    "aux_lambda": {
      "a": 1.0,
      "b": 1.0,
      "equal": true
    },
    "aux_pair_shuffle_seed": {
      "a": 43022,
      "b": 43022,
      "equal": true
    }
  },
  "log_equality": {
    "n_a": 481,
    "n_b": 481,
    "n_common": 481,
    "length_equal": true,
    "diff_counts_by_key": {
      "batch_words": 0,
      "aux_words": 0,
      "cumulative_main_words": 0,
      "cumulative_aux_words": 0,
      "cumulative_charged_words": 0,
      "masked_tokens": 0,
      "aux_targets": 0,
      "aux_units": 0,
      "aux_conditioned_views": 0,
      "aux_free_views": 0,
      "lr": 0
    },
    "first_diff_examples": {},
    "all_accounting_equal": true
  }
}
```

JSON: `experiments/archive/frontier_consolidation/data/dualview_training_audit/aligned_vs_shuffled_completed.json`
