# earlier analysis chck_84M HF bundle validation

Status: `CHCK84_HF_MODEL_READY`

## Endpoint score arithmetic
- cheap7: `44.12357142857143`.
- SuperGLUE: `69.30521676796288`.
- Projected Overall with AoA=0: `42.0189129742181`.
- Delta vs submitted chck82: `0.0764318068321117`.
- Delta vs coherent86 alpha0.75 projected: `-0.10211173574850108`.

## Function identity
- Source checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M`.
- Model SHA256: `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`.
- Local trusted params: `35463008`; max logit diff vs source `0.0`.
- Local native fallback: `DebertaV2ForMaskedLM` params `34467424`, wrong function `True`.

## Public revision
- Repo: `leslie721007/babylm-strict-small-scale1p75-chck84`.
- Revision: `040284de9ac49eee6dc1b30cf65aea97ec17d86e`.
- Public status: `PUBLIC_CHCK84_REVISION_VALIDATED`.
- Public trusted params: `35463008`; max logit diff vs source `0.0`.

## Scientific boundary
This is an endpoint-carrier artifact for a legal same-trajectory 84M checkpoint. It is not a leaderboard submission package and does not settle cross-trajectory robustness.

JSON: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/chck84_hf_public_bundle_validation.json`
