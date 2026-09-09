# earlier analysis static-prior masking validation

CPU validation that generated static-prior trainer preserves WWM mask budget and reallocates mask pressure on real legal-pool examples before any GPU training is considered.

This is a CPU-only validation of the generated trainer's actual masking function; it is not a model-training result.

## Samples and mode comparisons

### front_changed_plus_early_filler
- rows: 1024; words: 143191; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 1024}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1501 | 0.1510 | 1.0129 | 0.9994 | 0.9862 | 0.9889 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1501 | 0.1510 | 1.0129 | 0.9994 | 0.9862 | 0.9889 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1501 | 0.1510 | 1.0205 | 0.9999 | 0.9862 | 0.9851 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1501 | 0.1510 | 1.0205 | 0.9999 | 0.9862 | 0.9851 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1499 | 0.1470 | 1.0586 | 1.0444 | 1.5884 | 1.7586 | -0.0003 | -0.0040 |
| wwm_static_prior:relation_only_v1 | changed | 0.1499 | 0.1470 | 1.0586 | 1.0444 | 1.5884 | 1.7586 | -0.0003 | -0.0040 |
| wwm_static_prior:relation_only_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_info_v1 | all | 0.1497 | 0.1494 | 1.0439 | 1.0229 | 1.3826 | 1.5104 | -0.0005 | -0.0015 |
| wwm_static_prior:relation_info_v1 | changed | 0.1497 | 0.1494 | 1.0439 | 1.0229 | 1.3826 | 1.5104 | -0.0005 | -0.0015 |
| wwm_static_prior:relation_info_v1 | other | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### stride_full_pool
- rows: 512; words: 79179; source_counts: {'cleanqwen_fineweb_compact_view_reinvest': 24, 'qwen_pair_packed': 93, 'gutenberg': 101, 'childes': 114, 'simple_wiki': 64, 'open_subtitles': 84, 'bnc_spoken': 31, 'switchboard': 1}

| mode | split | group mask | token mask | selected mean prior | prior lift | rel lex lift | high1.50 lift | Δ group vs fixed | Δ token vs fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| wwm_fixed:relation_only_v1 | all | 0.1504 | 0.1507 | 1.0005 | 0.9994 | 0.9991 | 0.9788 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | changed | 0.1480 | 0.1475 | 1.0134 | 0.9966 | 0.9912 | 0.9473 | 0.0000 | 0.0000 |
| wwm_fixed:relation_only_v1 | other | 0.1505 | 0.1508 | 1.0000 | 0.9995 | 1.0002 | 0.9813 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | all | 0.1504 | 0.1507 | 1.0002 | 0.9994 | 0.9991 | 0.9774 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | changed | 0.1480 | 0.1475 | 1.0192 | 0.9973 | 0.9912 | 0.9226 | 0.0000 | 0.0000 |
| wwm_fixed:relation_info_v1 | other | 0.1505 | 0.1508 | 0.9994 | 0.9995 | 1.0002 | 0.9813 | 0.0000 | 0.0000 |
| wwm_static_prior:relation_only_v1 | all | 0.1503 | 0.1486 | 1.0391 | 1.0379 | 1.6021 | 1.7715 | -0.0001 | -0.0021 |
| wwm_static_prior:relation_only_v1 | changed | 0.1486 | 0.1455 | 1.0660 | 1.0484 | 1.6440 | 1.7888 | 0.0007 | -0.0020 |
| wwm_static_prior:relation_only_v1 | other | 0.1504 | 0.1487 | 1.0379 | 1.0374 | 1.6003 | 1.7711 | -0.0002 | -0.0021 |
| wwm_static_prior:relation_info_v1 | all | 0.1501 | 0.1503 | 1.0206 | 1.0198 | 1.3981 | 1.5008 | -0.0003 | -0.0004 |
| wwm_static_prior:relation_info_v1 | changed | 0.1480 | 0.1473 | 1.0447 | 1.0223 | 1.3665 | 1.4341 | 0.0001 | -0.0002 |
| wwm_static_prior:relation_info_v1 | other | 0.1502 | 0.1504 | 1.0196 | 1.0197 | 1.4001 | 1.5050 | -0.0004 | -0.0004 |

## Input files

- trainer: `experiments/archive/frontier_consolidation/scripts/masking_curriculum_trainer_static_prior.py`
- pool_10m: `experiments/archive/frontier_consolidation/data/density_cleanqwen_overlay_medium_riskhard/cleanqwen_fineweb_compact_view_reinvest_10M.jsonl`
- tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer`
- prior_json: `experiments/archive/frontier_consolidation/data/static_token_mask_prior/static_token_mask_prior.json`
- seq_length: `256`
- batch_size: `256`
- num_seeds: `3`
- seed0: `43023`
- rows_front: `1024`
- rows_stride: `512`

Full JSON: `experiments/archive/frontier_consolidation/data/static_prior_masking_validation/static_prior_masking_validation.json`
