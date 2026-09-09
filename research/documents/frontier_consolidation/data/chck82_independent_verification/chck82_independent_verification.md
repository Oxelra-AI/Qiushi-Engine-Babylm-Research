# chck82 frozen private tail design independent verification — scale1.75 chck_82M
Status: **PASS**

## Score arithmetic
- Recomputed Overall: `41.942481167385985`; hardened summary Overall: `41.942481167385985`.
- Recomputed cheap7: `43.95944987645173`; hardened summary cheap7: `43.95944987645173`.
- Margin vs displayed 41.80: `0.14248116738598782`.
- Margin vs recomputed public leader columns: `0.14470338960820328`.
- Measurement repeatability: Overall delta `0.00022695351662349594`, max column delta `0.005893446487505116`.

## Legal exposure and provenance
- Legal pool words `10000000`, SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`.
- 100M stream words `100000000`, SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`.
- Selected checkpoint actual words `82012495` = `8.2012495` epochs of the 10M pool; target `82000000`.
- Tokenizer training SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`; vocab-map identity `True`.

## Artifact identity and loadability
- Source model SHA `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`; snapshot SHA `93ceb76adf5a33d349f1de33e988e6ed0c2b2a547dbd92cf83cc952f8e2591b3`.
- CPU HF loadability OK: `True`; class `AdapterDebertaV2ForMaskedLM`; params `35463008`; logits shape `[1, 10, 16384]`.

## Core checks
- `all_nine_scores_present`: `True`
- `cheap7_matches_summary`: `True`
- `overall_matches_summary`: `True`
- `overall_above_reported_41p8`: `True`
- `overall_above_recomputed_public_leader`: `True`
- `repeatability_pass`: `True`
- `preflight_pass`: `True`
- `snapshot_manifest_status`: `True`
- `source_model_sha_matches_expected`: `True`
- `snapshot_model_sha_matches_source`: `True`
- `pool_hash_matches_expected`: `True`
- `stream_hash_matches_expected`: `True`
- `tokenizer_training_hash_expected`: `True`
- `tokenizer_vocab_maps_identical`: `True`
- `pool_words_eq_10M`: `True`
- `stream_words_eq_100M`: `True`
- `selected_checkpoint_target_82M`: `True`
- `selected_checkpoint_actual_within_10_epochs`: `True`
- `selected_checkpoint_before_100M_endpoint`: `True`
- `loadability_cpu_forward_ok`: `True`
- `all_core_checks_pass`: `True`

## Scientific reading
The chck_82M endpoint is a legal-exposure, loadable, repeated-score above-frontier checkpoint candidate if the fixed from-corpus reproduction also matches or is re-evaluated. It establishes a practical score-bearing endpoint strategy and proves late 82M-to-100M competence loss in the scale1.75 trajectory, but it does not by itself settle the general source-free correspondence-learning mechanism.

JSON: `experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json`
