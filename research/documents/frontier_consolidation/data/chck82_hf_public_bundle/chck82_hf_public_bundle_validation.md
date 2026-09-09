# earlier analysis protected chck_82M HF public bundle validation

Status: `PUBLIC_BUNDLE_OR_SUBMIT_PATH_HAS_ISSUES`

## Public revision
- Repo: `leslie721007/babylm-strict-small-scale1p75-chck82`
- Revision/commit: `f49775dc5eafbf3a14d6f2107ec5368188d2b1dd`
- Private: `False`

## Function preservation
- Local trusted load class: `AdapterDebertaV2ForMaskedLM`; params `35463008`; max logit diff vs protected `0.0`.
- Public trusted load class: `AdapterDebertaV2ForMaskedLM`; params `35463008`; max logit diff vs protected `0.0`.
- Public native non-trust fallback: `DebertaV2ForMaskedLM` with params `34467424` and unexpected adapter tensors `48`; this is the wrong function.

## Prediction carrier and submit-path validation
- Prediction SHA matches expected: `True`.
- Current Space `is_valid_predictions(..., strict-small)`: `True` / `Upload successful.`.
- Public model-card helper: `None`; hub config/tokenizer trust_false helper: `None`.
- Local monkey-patched live submit dry-run ok: `False`; counted-message present: `None`; dummy uploads `None`.

## Scientific interpretation
- The uploaded public revision preserves the 35.46M-parameter adapter function under trusted-code loading; it does not silently replace the score-bearing function by the 34.47M native DeBERTa fallback.
- The native non-trust AutoModel fallback still loads a different function and is recorded as a hazard, not as a valid model. Any official or scientific scoring must use `trust_remote_code=True` or the submitted prediction file.
- The current live Space submit code validates the strict-small prediction file and metadata locally without executing the model; with uploads monkey-patched, the path accepted the package as a challenge-counted submission. This is evidence for submission-path compatibility, not an actual leaderboard submission.

JSON: `experiments/archive/frontier_consolidation/data/chck82_hf_public_bundle/chck82_hf_public_bundle_validation.json`
