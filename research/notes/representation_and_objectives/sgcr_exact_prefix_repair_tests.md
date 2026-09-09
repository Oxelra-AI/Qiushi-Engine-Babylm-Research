# depth vector decision SGCR exact-prefix repair tests

Status: `SGCR_EXACT_PREFIX_REPAIR_TESTS_PASSED`.

- Exact decomposition histogram: `{'1': 16384, '2': 19333, '3': 3629, '4': 571, '5': 63, '6': 16, '7': 4}`; map SHA `b450cc45f7d66564a894fb8cc12e0b0ab8e3ba7db0339cad5eec6f30c6edfd8b`.
- Counts: 13942644 legal40k pool tokens across 39320 used types; 14669276 legal16k component-token counts. Special ids 0-4 legal40k counts `{'0': 0, '1': 0, '2': 0, '3': 0, '4': 0}`.
- Decomposition buffer shape is `[40000, 7]`, not the old 40k x 256 padded map.
- K=50 treatment mass-weighted residual is 0.0642800704; uniform control residual matches at 0.0642800927.
- Low-support used legal40k types with all exact components >=50: 23089/24854 = 0.928985.
- Cold exact table/logit diff 0.0; projection gradient live at first backward (0.0401983); component gradient live after one update (0.000648881).
- Baked checkpoint loads as standard HF model with no `_sgcr` keys and sidecar contains `base_word_embeddings`.
- JSON: `experiments/archive/representation_and_objectives/data/sgcr_exact_prefix_repair_tests/sgcr_exact_prefix_repair_tests.json`
