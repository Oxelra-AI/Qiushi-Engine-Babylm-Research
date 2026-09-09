# coherent86 endpoint robustness synthesis reviewer status — coherent86 endpoint and mechanism

## Endpoint-carrier status

The coherent86 endpoint is now a faithful public HF carrier candidate, even though the first public-bundle script summary says `COHERENT86_BUNDLE_HAS_ISSUES`. The issue is isolated to the in-process live Space helper binding Hugging Face caches to a read-only literal path, not to model identity or prediction validity.

Evidence:

- Truthful local carrier: `data/truthful_coherent86_carrier/all_full_preds_truthful_coherent86_mlm.json`, SHA `4a0278a689ea48bdb88215090e34a94caa3fc8ba10df78d19a9195d3698b533e`.
- Coherent86 model checkpoint: `training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final`, SHA `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8`; frozen anchor coherent replay item reading exact replay produced the same SHA.
- Local bundle validation: `data/coherent86_hf_public_bundle/local_bundle_validation.json` reports `LOCAL_BUNDLE_VALIDATED`; both carrier filenames pass the current Strict-Small prediction validator; trusted-code logits match the source checkpoint exactly (max/mean diff 0.0); native non-trusted fallback loads native `DebertaV2ForMaskedLM` with 34,467,424 parameters and 96 unexpected adapter/private-adapter tensors, so it is correctly identified as the wrong scientific function.
- Public revision: `leslie721007/babylm-strict-small-coherent86`, revision `ad128352c154702704e8b29c24cf94d986fe5c7f`.
- Public upload validation: `data/coherent86_hf_public_bundle/public_upload_validation.json` shows upload OK, snapshot download OK, every downloaded file SHA matches the local bundle, public trusted-code logits match source checkpoint exactly, public prediction alias passes current Strict-Small validator, and native fallback is the wrong function. The only failing field is `live_space_public_helper_checks`, with an OSError from a read-only cache path.
- Standalone helper validation: `data/coherent86_hf_public_bundle/public_helper_validation_standalone.json` runs the current Space helper in a fresh process with writable caches and returns `PASS`: model card OK, Hub config/tokenizer OK with `trust_remote_code=False`, and Hub config/tokenizer OK with `trust_remote_code=True`.

Thus the public HF carrier is ready for live submission if the independent repeat SuperGLUE result keeps the projected Overall margin materially positive.

## Score status before repeat SuperGLUE

Current candidate-native arithmetic from the first coherent SuperGLUE run:

- BLiMP 68.52
- Supplement 63.65
- EWoK 49.91
- Entity 28.44
- COMPS 51.99
- GlobalPIQA 38.065
- Reading 8.17
- SuperGLUE 69.77796826428681
- AoA represented as scalar 0.0
- Overall(AoA0) 42.058107584920755, +0.1156264175347701 vs submitted `chck_82M`.

Repeat SuperGLUE evaluation remains the decisive endpoint-robustness check. Fine-tuning variance was already implicated by the shuffled86 repeat; the new result tests whether the coherent86 endpoint margin persists. It does not validate the mechanism.

## Mechanism status

The coherent86 fast path is not a settled general slow-fast learning advance. Item and overlap evidence show competence redistribution:

- `data/fastpath_item_family_analysis/fastpath_item_family_analysis.md`: coherent vs protected `chck_82M` gains 3,116 and loses 3,231 discrete items over 170,722 common rows, net -115; Supplement and GlobalPIQA improve, but EWoK and COMPS decline. Coherent vs shuffled86 is net -1,113 items, with different column tradeoffs.
- `data/fastpath_flip_overlap/fastpath_flip_overlap.md`: coherent and shuffled86 have low shared gain/loss overlap relative to the anchor (all-discrete gain Jaccard 0.0875, loss Jaccard 0.0884; EWoK-fragile gain Jaccard 0.0658), so they are not two confirmations of the same added-decision set.

The endpoint can be submitted if robust, but the research cannot close around it. Subsequent science must show seed-stable added decisions in the enabled function without renewed erosion of anchor-correct behavior, or move to another mechanism that preserves the verified 82M competence vector while adding new capability.
