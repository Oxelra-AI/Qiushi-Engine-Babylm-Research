# roberta transfer pair scaffold ready RoBERTa transfer pair scaffold

Status: OK

No training or selected evaluation was launched.

## Data identities
- compact40: `experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_40M.jsonl` sha `48313173a4cd93e768f490c8a5eaaa850fc828fabc9cfb21ebad73651bef85dc` rows 258960 words 40000000
- repeat40: `experiments/archive/frontier_consolidation/data/roberta_transfer_pair_scaffold/repeat_compact_reinvest_40M.jsonl` sha `ed0eb4ca34e7057f0e7a401008eb4163112a557973516ac86930142b41a7dca5` rows 258960 words 40000000
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer` sha `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`

## Pair invariants
- row id mismatches: 0
- word-count mismatches: 0
- changed rows different: 12020 / 12020
- changed rows equal topup: 4 / 4
- filler text mismatches: 0

## Tokenizer active-token/group differences (one 10M pass; compact minus repeat)
- total active tokens: 24000
- total candidate tokens: 24000
- total word groups: -196
- changed-block active tokens: 24000
- changed-block candidate tokens: 24000
- changed-block word groups: -196
- changed-block active tokens/word: 0.056668

## Future commands (not launched)
```
CUDA_VISIBLE_DEVICES=0 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer --tokenizer_label compliant16k_reinvest10M --model_family roberta --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 --batch_size 256 --seq_length 256 --max_seq_length 256 --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 --mask_prob 0.15 --max_word_exposure 40000000 --checkpoint_words 20000000 --lr_total_steps 2529 --log_every 50 --example_jsonl experiments/archive/frontier_consolidation/data/compact_order_factorial_pool_scaffold/compact_ordered_40M.jsonl --example_jsonl_label compact_reinvest_ordered40M_roberta_transfer --output_dir experiments/archive/frontier_consolidation/training/runs/roberta_compact_reinvest_40M_seed43022
CUDA_VISIBLE_DEVICES=1 python3 -B experiments/archive/frontier_consolidation/scripts/roberta_mlm_transfer_trainer.py --tokenizer_path experiments/archive/frontier_consolidation/data/compliant_tokenizer --tokenizer_label compliant16k_reinvest10M --model_family roberta --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 --batch_size 256 --seq_length 256 --max_seq_length 256 --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 --mask_prob 0.15 --max_word_exposure 40000000 --checkpoint_words 20000000 --lr_total_steps 2529 --log_every 50 --example_jsonl experiments/archive/frontier_consolidation/data/roberta_transfer_pair_scaffold/repeat_compact_reinvest_40M.jsonl --example_jsonl_label repeat_compact_reinvest40M_roberta_transfer --output_dir experiments/archive/frontier_consolidation/training/runs/roberta_repeat_compact_reinvest_40M_seed43022
```

Future launch remains conditional on the pending compact-order evidence.

Warnings are contextual rather than failures when legacy 100M row order differs or pass-prefixed source labels differ while text/id/word invariants hold.
