# compact order result and roberta transfer decision RoBERTa full-100M compact-vs-repeat transfer scaffold

Status: OK

No training or selected evaluation was launched by this scaffold.

## Matched data pair
- compact100: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl` rows 647400 words 100000000 sha `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- repeat100: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl` rows 647400 words 100000000 sha `91e8817e25f234b87481fa6ca96d968d6c460b84317273629eda21ea3be24eaa`
- row id mismatches: 0
- word-count mismatches: 0
- text-different rows: 30050 (10M diff rows 3005 x10)
- text-equal rows: 617350

## Scientific reason
The ordered-vs-scrambled 40M result removes coherent compact order as a downstream-positive cause, but it does not test the natural compact-vs-repeat data marginal. Because the original compact benefit emerged late, this transfer test must train to 100M and evaluate the late trajectory.

## Future training commands (not launched)
```
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer --tokenizer_label compliant16k_reinvest10M --model_family roberta --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 --batch_size 256 --seq_length 256 --max_seq_length 256 --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 --mask_prob 0.15 --max_word_exposure 100000000 --checkpoint_words 10000000 --lr_total_steps 2529 --log_every 50 --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_100M.jsonl --example_jsonl_label compact_reinvest100M_roberta_transfer --output_dir experiments/archive/frontier_consolidation/training/runs/roberta_compact_reinvest_100M_seed43022
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer --tokenizer_label compliant16k_reinvest10M --model_family roberta --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 --batch_size 256 --seq_length 256 --max_seq_length 256 --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 --mask_prob 0.15 --max_word_exposure 100000000 --checkpoint_words 10000000 --lr_total_steps 2529 --log_every 50 --example_jsonl experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_repeat_compact_reinvest_100M.jsonl --example_jsonl_label repeat_compact_reinvest100M_roberta_transfer --output_dir experiments/archive/frontier_consolidation/training/runs/roberta_repeat_compact_reinvest_100M_seed43022
```

Primary late readout: compact-minus-repeat over chck_60M/chck_70M/chck_80M/chck_90M/chck_100M on cheap6_no_GlobalPIQA, cheap5_no_GlobalPIQA_Reading, EWoK+Entity, Supplement, Entity, and COMPS.

## Tokenization note
Estimated compact-minus-repeat over 100M: active tokens 240000, candidate tokens 240000, word groups -1960, truncated rows 320. This is part of the natural treatment.
