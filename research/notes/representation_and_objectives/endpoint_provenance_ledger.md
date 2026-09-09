# changed block overlap ancestry current official — Endpoint provenance ledger for compact_view_reinvest seed43022

Machine-readable ledger: `experiments/archive/representation_and_objectives/data/endpoint_provenance_ledger/compact_reinvest_seed43022_endpoint_provenance_ledger.json`

## Endpoint
- Official-coordinate Overall: **42.0331347900748**
- Margin over visible 41.8 leader: **0.23313479007479998**
- Collated JSON sha256: `a85aae3b29b3cecceb67e21792f75048af478cc9191011c9872a69a639fec192`
- Columns: `{'BLiMP': 66.87232315173485, 'Supplement': 63.27576417952158, 'EWoK': 53.536575594886855, 'Entity': 27.745741097952372, 'COMPS': 51.968828052457084, 'SuperGLUE': 71.03604952825312, 'GlobalPIQA': 35.62135922330097, 'Reading': 8.241572282566393, 'AoA': 0.0}`

## Training/data
- 10M pool: 64740 rows / 10000000 words, sha `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- 100M stream: 647400 rows / 100000000 words, sha `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`
- Source words in 10M pool: `{'cleanqwen_fineweb_compact_view_reinvest': 423511, 'neutral_cleanqwen_topup_compact_reinvest::open_subtitles': 9, 'qwen_pair_packed': 1656800, 'childes': 2574880, 'gutenberg': 1859360, 'open_subtitles': 1829440, 'simple_wiki': 1059040, 'bnc_spoken': 575840, 'switchboard': 21120}`
- Compact pairs selected: 12155 pairs, 423511 pair words, mean content recall 0.6631

## Model/tokenizer
- Architecture: DeBERTa-v2 MLM 8x480, 34467424 parameters, vocab 16384
- Tokenizer hash: `9cc4f9073675da3f817a6020e7b0833cf3234c0131515d68f4065b00f14933ac`
- chck_100M model hash: `6aa03d469281558d5a6d2fd490698a33584e8d3a157dc5040bdfaffe3a450a44`
- Required AoA checkpoint ladder missing: `[]`

## Evaluation coordinate
- Pristine strict root: `experiments/archive/representation_and_objectives/data/pristine_official_coordinate/babylm-eval/strict`
- GlobalPIQA lineage checks: `{'official_dl_exists': True, 'official_dl_sha256': 'be1a0bb84fc4efee4278d140279523cd1da0962c9598298cc3cef98f1f24da8c', 'collator_sha256': 'bbb36ebc1090873bbd526c385bb39af6b9a6289f55a5f5fbce5fe8e7f74a84ff', 'initial_model_studies_dl_identical_to_pristine_dl': True, 'official_dl_returncode': 0, 'all_generated_files_exist': True, 'all_generated_counts_match_collator_expectation': True, 'all_generated_match_inherited_bytes': True, 'generated_full_equals_fast': True, 'inherited_full_equals_fast': True}`
- Collated shape has null keys: `[]`; EWoK total 7618; AoA row counts [8005]

## Remaining
- Exact compact-rewrite generator identity/log and source license lineage still need a focused pass.
- seed43122 full official-coordinate result is still pending.
