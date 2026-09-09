# scale1p75 pre80m state scale1.75 embedded-prefix reproduction

Run20: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M20M_seed43022`
Run50: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M50M_seed43022`
Checkpoint compared: `hf_model/chck_20M`

## Conclusion
- Prefix reproduces exactly (training log except elapsed + tensors): **True**
- Model safetensors hash equal: True
- Training-log prefix equal ignoring elapsed seconds: True
- State-dict tensors exactly equal: True

## Training-log prefix
- records compared through standalone chck_20M: 506 vs 506 (50M total train rows: 1265)
- first mismatches after dropping elapsed_sec: 0
- last 20M-run prefix record: `{"batch_words": 39823, "cumulative_word_exposure": 20008711, "effective_mask_rate": 0.1507, "loss": 3.765514850616455, "lr": 0.0009460119426666896, "mask_mode": "wwm", "mask_prob_nominal": 0.15, "masked_tokens": 8546, "seq_len": 256, "step": 506}`
- last 50M-run prefix record: `{"batch_words": 39823, "cumulative_word_exposure": 20008711, "effective_mask_rate": 0.1507, "loss": 3.765514850616455, "lr": 0.0009460119426666896, "mask_mode": "wwm", "mask_prob_nominal": 0.15, "masked_tokens": 8546, "seq_len": 256, "step": 506}`

## Tensor comparison
- keys: 218 vs 218, common 218
- exact tensor equal: True
- global L2 diff: 0.0
- max abs diff: 0.0
- max per-tensor rel L2: 0.0

## Selected hash equality
| relative file | equal |
|---|---:|
| `example_order_manifest.json` | False |
| `hf_model/chck_20M/config.json` | True |
| `hf_model/chck_20M/model.safetensors` | True |
| `hf_model/chck_20M/special_tokens_map.json` | True |
| `hf_model/chck_20M/tokenizer.json` | True |
| `hf_model/chck_20M/tokenizer_config.json` | True |
| `hf_model/config.json` | True |
| `hf_model/special_tokens_map.json` | True |
| `hf_model/tokenizer.json` | True |
| `hf_model/tokenizer_config.json` | True |
| `scientific_metrics.json` | False |
