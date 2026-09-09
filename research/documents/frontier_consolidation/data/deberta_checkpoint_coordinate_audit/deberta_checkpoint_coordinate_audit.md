# deberta grid tooling and topology packing confound DeBERTa checkpoint coordinate audit

Overall OK: **True**

| label | ok | ckpt dirs | missing common endpoints | adapter scale | params | tokenizer label | 100M exposure | source words total |
|---|---:|---:|---|---:|---:|---|---:|---:|
| scale1p75_seed43022_reference | True | 100 |  | 1.75 | 35463008 | compliant16k_reinvest10M | 100000000 | 100000000 |
| scale1p25_seed43022_dense | True | 50 |  | 1.25 | 35463008 | compliant16k_reinvest10M | 100000000 | 100000000 |
| scale1p75_seed43122_dense | True | 50 |  | 1.75 | 35463008 | compliant16k_reinvest10M | 100000000 | 100000000 |

## Shared file hashes at chck_82M

| file | shared | n unique | sha256 |
|---|---:|---:|---|
| tokenizer.json | True | 1 | a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a |
| tokenizer_config.json | True | 1 | 488d9617b98746cfdb862c62e02c2935bb086a2d0620307e75b9bac967d26d01 |
| special_tokens_map.json | True | 1 | 2be97b602cd0e4a6c2874cbf03c0d1025b3af5474666da931bc04f6c23a9a39d |
| adapter_scaled_modeling.py | True | 1 | 9fd4ef104f5f8e7d458c7ed10b996d238a6af53a866e8dea907a3cc0e4e156d2 |

## Issues

### scale1p75_seed43022_reference
- none
### scale1p25_seed43022_dense
- none
### scale1p75_seed43122_dense
- none

The config hash may differ across scale settings, but tokenizer and custom adapter modeling code should be shared. Trusted-code loading remains required for these checkpoints.

JSON: `experiments/archive/frontier_consolidation/data/deberta_checkpoint_coordinate_audit/deberta_checkpoint_coordinate_audit.json`
