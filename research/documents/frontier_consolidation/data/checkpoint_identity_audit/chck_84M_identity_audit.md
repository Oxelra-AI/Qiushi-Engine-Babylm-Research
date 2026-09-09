# partial deberta grid and endpoint branch checkpoint identity audit: chck_84M

Run: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder`

Endpoint actual words: `84028405`; epochs against 10M pool: `8.4028405`.

Model SHA: `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`; size `141878192` bytes.

Static files same as chck_82M except model: `True`.

CPU trusted-code loadability: `True`; class `AdapterDebertaV2ForMaskedLM`; params `35463008`.

JSON: `experiments/archive/frontier_consolidation/data/checkpoint_identity_audit/chck_84M_identity_audit.json`
