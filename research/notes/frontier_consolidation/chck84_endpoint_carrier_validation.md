# chck84 endpoint carrier validation synthesis: `chck_84M` endpoint carrier and tokenizer-hash repair

## What changed

The `chck_84M` endpoint branch is now carried by a public Hugging Face model repository with exact trusted-code validation:

- Repository: `leslie721007/babylm-strict-small-scale1p75-chck84`
- Final validated revision: `040284de9ac49eee6dc1b30cf65aea97ec17d86e` (supersedes the first chck84 endpoint carrier validation upload `b3eabdf8e324790ba1b0ac5469275c6af7b28f5c` after a README wording repair)
- Source checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_84M`
- Model SHA256: `2217917c687faf4de26ef6f381be3048d0bd66b2025c06382683c24d78e8d8c9`
- Checkpoint-carried tokenizer SHA256: `a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a`
- Trusted-code class: `AdapterDebertaV2ForMaskedLM`
- Trusted parameter count: `35,463,008`
- Public/source logits on the validation prompts: max abs diff `0.0`, mean abs diff `0.0`, exact tensor equal `true`
- Native `trust_remote_code=False` fallback is the wrong function: native `DebertaV2ForMaskedLM`, `34,467,424` parameters, 48 adapter tensors unexpected/dropped.

Validation records:

- Main JSON: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/chck84_hf_public_bundle_validation.json`
- Main Markdown: `research/documents/frontier_consolidation/data/chck84_hf_public_bundle/chck84_hf_public_bundle_validation.md`
- Local validation JSON: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/local_bundle_validation.json`
- Public validation JSON: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/public_upload_validation.json`
- Bundle directory: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/hf_repo_bundle`
- Redownloaded public revision: `experiments/archive/frontier_consolidation/data/chck84_hf_public_bundle/downloaded_public_revision`

No leaderboard submission was performed. The uploaded repo is a model artifact only, not a BabyLM submission package and not a complete fast/AoA carrier.

## Endpoint score arithmetic carried by the repo

The endpoint arithmetic remains the earlier analysis/partial deberta grid and endpoint branch result:

- cheap7: `44.12357142857143`
- SuperGLUE: `69.30521676796288`
- AoA used for the local projection: `0.0`
- projected Overall(AoA0): `42.0189129742181`
- delta vs protected/submitted `chck_82M`: `+0.0764318068321117`
- delta vs coherent86 alpha0.75 projected endpoint: `-0.10211173574850108`

This means `chck_84M` is a legal same-trajectory >42.0 endpoint branch and a cleaner ordinary-training endpoint than coherent86 alpha0.75, but it is still a single seed/mask trajectory and does not by itself establish a general learning principle.

## Tokenizer-hash repair

The first local no-upload validation failed before bundle construction because the script expected the raw spatial repair route status tokenizer-training artifact hash (`91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`) as the checkpoint tokenizer hash. That was a carrier-script error, not a model or tokenizer-compliance failure.

Evidence:

- spatial repair route status tokenizer metadata records the raw legal-tokenizer training artifact at `experiments/archive/frontier_consolidation/data/compliant_tokenizer/tokenizer.json` with SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`.
- deberta grid tooling and topology packing confound coordinate audit and partial deberta grid and endpoint branch identity audit show the actual scored checkpoints carry `tokenizer.json` SHA `a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a`; this SHA is shared by `chck_82M`, `chck_84M`, the reference grid, scale1.25, and seed43122 trajectories.
- `experiments/archive/frontier_consolidation/data/tokenizer_serialization_check/tokenizer_serialization_check.json` compared a small set of encodings between the raw spatial repair route status tokenizer directory and the checkpoint tokenizer directory; all tested encodings matched and vocab sizes were both 16,384. The hash difference is best treated as saved-tokenizer serialization/provenance distinction unless a later exhaustive comparison is needed.

The bundle script now validates the checkpoint-carried tokenizer SHA `a9cbb830...` while preserving the raw spatial repair route status tokenizer SHA as provenance.

## Scientific boundary

This artifact strengthens the endpoint-delivery branch but should not steer the next scientific work away from the robustness question. `chck_84M` is inside one scale1.75 seed43022 trajectory; chck84 item movement synthesis showed it is a real narrow late competence-allocation peak with broad-but-shallow cheap-task improvement and high item churn. The pending seed43122 common-grid scoring and directional-fork evidence remain the decisive inputs for whether the late phase is robust and what mechanism to pursue next. If the seed43122 phase does not recur, the next research should pivot toward stabilizing broad competence across stochastic trajectories rather than tuning the seed43022 peak.
