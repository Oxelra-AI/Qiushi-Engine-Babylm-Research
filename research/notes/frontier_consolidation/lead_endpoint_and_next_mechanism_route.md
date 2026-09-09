# context dependence ig probe result Lead synthesis: endpoint hardening and next mechanism route

## What changed

### Current public surface

I refreshed the live BabyLM 2026 leaderboard using only `/refresh_leaderboards`; no upload or submission call was made.

Artifact: `data/live_leaderboard_snapshot/leaderboard_snapshot.{json,md}`.

Fresh Strict-Small surface:

- top displayed Overall: `41.94`
- `Qiushi-BabyLM-35M-Strict-Small-v3` and `leslie721007/babylm-strict-small-scale1p75-chck82` have identical public component vectors and displayed Overall `41.94`
- our `chck82` row remains present, displayed rank `2` among tied `41.94` rows
- `coherent86` public repo is not on the live leaderboard

The displayed public frontier has not moved above `41.94`. The local alpha0.75 endpoint remains numerically well above the displayed surface; this comparison does not establish a new leaderboard submission.

### Alpha0.75 artifact identity

I reconciled the alpha0.75 private-scale endpoint using file/CPU work only.

Artifact: `data/alpha075_artifact_reconciliation/alpha075_artifact_reconciliation.{json,md}`.

Key measurements:

- model SHA256: `e14d757ae51b41e33bf0813f841248fecd1eefeb9e040f520c4c6203343b15c8` (full 64 hex)
- config SHA256: `ca6792fe1842f0e89e7080e7709e0619fdcaf5ce48a65f69bf7137ac26ce9bb7`
- carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`
- recomputed model/config/carrier digests match the alpha0.75 truthful carrier manifest
- alpha0.75 weights are bit-identical to coherent86 alpha1, coherent replay, and alpha0.5; only `private_adapter_scale` differs in config (`0.75` vs `1.0` for alpha1)
- trusted-code CPU load gives `36,458,592` parameters
- alpha0.75 logits differ from alpha1/alpha0.5 as expected because the config scale changes the private residual amplitude
- using the model method `set_private_enabled(False)` exactly recovers the protected chck82 logits on the CPU probe (max/mean diff `0.0`)

Scientific reading: alpha0.75 is a well-defined config-scaled inference function over the single learned coherent86 private residual. It is not a distinct trained-weight endpoint. That is not by itself a disqualifier, because the model function is fully specified by weights plus config and reloads through `trust_remote_code=True`; it must be described truthfully as a no-training amplitude-scaled function, not as an independently trained continuation.

### Mechanism status after edit state probe chck82 synthesis

edit state probe chck82 synthesis closed the edit-state correspondence family on the score-bearing chck82 checkpoint. The source-absent true-vs-decoy signal remains decodable by a detached private readout, but it is generic changed-token distributional information, dominated by content words, and does not survive source removal in the mature frozen-tail tests. Combined with the broad dual-view and frozen-tail failures, this route should not receive another trainer.

The private-scale family remains a practical endpoint family only. Its score movement is amplitude-controlled redistribution over existing official-item margins: GlobalPIQA movement comes from a few examples, EWoK/Entity movement is small and mixed, and anchor-margin magnitude did not separate useful residual changes from damages.

The coherence-margin/block-shuffle route is closed by the destructive 4M pilot (`cheap7 36.6921`, down `7.2673` from chck82) and the absence of the intended coherent-over-disrupted NLL separation.

## Endpoint and science must now be separated

### Practical endpoint branch

The strongest local endpoint is alpha0.75:

- cheap7: `44.18142857142857`
- SuperGLUE: `69.81922238969935`
- Overall(AoA0): `42.1210247099666`
- delta vs protected chck82 local Overall: `+0.17854354258061278`
- truthful local carrier: `data/truthful_private_scale_carriers/coherent86_alpha0p75/all_full_preds_truthful_coherent86_alpha0p75_mlm.json`
- carrier SHA256: `40181994810e21bc823474a3e4ac84c8eb42213e03904d36a60a4698477d1994`

Next endpoint work should harden this object without submitting it:

1. Build a public HF bundle for alpha0.75 by adapting the coherent86 endpoint robustness synthesis coherent86 bundler, using the same `model.safetensors` SHA and the alpha0.75 `config.json`.
2. Validate local and public `trust_remote_code=True` loads, parameter count, config scale, and representative logits.
3. Validate that `trust_remote_code=False` is a wrong native fallback, as for chck82 and alpha1.
4. Include the truthful scalar-AoA/no-fast-history carrier and exact score arithmetic in the bundle materials.
This branch improves submission readiness, not the scientific mechanism.

### Mechanism branch

A new full-retraining route is not yet justified. The evidence does not support restarting source-present edit learning, exact-swap innovation masking, retention-KL, anchor-confidence rules, alpha tuning, or block-shuffle variants. A schedule that simply trains to `82M` is also not a new learning hypothesis: the endpoint was found through official-score laddering, and no legal internal landing signal has yet been shown to select it.

The next scientific object should be upstream of frozen-endpoint amplitude: determine whether the fixed 15% MLM target budget is spent on tokens whose prediction is locally solvable while under-supervising tokens whose identity depends on wider context.

## Leading next mechanism: context-dependence information-gain target allocation

Mechanism idea: for ordinary legal-corpus text, score candidate target positions by the model's own context-dependence:

`IG(i) = NLL_local_window(y_i) - NLL_full_context(y_i)`

where both contexts are legal corpus contexts and the target token/piece is removed from the predictor input. High positive `IG` means the token needs wider syntactic, semantic, discourse, or event context; low `IG` means local co-occurrence nearly solves it. A future trainer would keep the same legal corpus and word exposure but allocate more of the fixed mask/loss budget to high-IG targets, with frequency controls and a matched ordinary-MLM arm.

This is different from closed routes:

- not source/rewrite correspondence: no true source text, no decoy source, no source-present inference
- not innovation masking: no special rewrite-innovation label pool and no source-absent edit targets
- not coherence-margin: no corrupted passage negative and no block-shuffle likelihood task
- not private-scale/anchor-confidence: it changes training supervision upstream, not inference amplitude
- not benchmark-shaped: the score is computed on legal training/held-out corpus text without official labels

## Lowest-cost discriminator before training

Before any training, run a forward-only probe on saved checkpoints and legal-corpus text.

Initial small probe:

- checkpoints: `chck_82M` and `chck_100M` from the scale1.75 ladder; optionally spatial repair route status legal baseline after the smoke passes
- sample: a deterministic held-out slice of the legal 10M pool excluding generated pair rows if desired, e.g. 512 rows / a few thousand target pieces
- contexts: full row context versus local windows of radius 4, 8, and 16 tokens, with the target piece masked or removed consistently
- outputs: per-target full NLL, local NLL, `IG`, unigram/token frequency, target kind, source row/source family, and checkpoint identity
- analysis: test whether high-IG tokens still have large residual error at chck82, whether they degrade or remain underfit at 100M, and whether the relation survives controls for token frequency and piece rarity

What the small probe decides:

- If `IG` is mostly frequency/rarity or high-IG targets are already well predicted, stop this route.
- If high-IG targets are a meaningful underlearned tail at chck82 and remain sensitive between 82M and 100M, expand the probe across the 77M-83M ladder and a spatial repair route status/legal baseline.
- If the expanded probe shows a stable under-supervised high-IG tail, then design a small matched-exposure training screen (4M-8M first, not 100M) comparing dynamic IG target allocation against ordinary WWM with equal word exposure/update count.

H100 use for the probe is acceptable only because it is forward-only and much cheaper than training; it decides whether any new trainer should exist. A single-GPU run over a small deterministic sample is enough for the first decision.

## Secondary mechanism candidates to preserve but not launch first

independent_review suggested several alternatives. They should not be launched before the context-dependence reader because they either require heavier engineering or have closer contact with closed routes:

1. dependency-span target ordering, using a parse-free span proxy; likely related to the context-dependence measure and can be added after the first reader
2. curvature-weighted consolidation of early-formed structure; only worth pursuing if a Fisher-like importance separates late helpful and harmful movement better than the already-null squared-gradient relation
3. span-level joint-consistency MLM; promising if independent-token predictions are often jointly incoherent in high-IG contexts
4. corpus-internal minimal-pair contrasts; must be induced without official item structure and kept separate from leaderboard labels

## Recommended next work

1. The proposed forward-only context-dependence probe on `chck_82M` and `chck_100M` requires exact corpus samples and hashes, context-construction assertions, and frequency controls; it does not require training.
2. Public HF validation of the alpha0.75 endpoint bundle remains separate from submission.
3. Exact-swap, edit-correspondence, coherence-margin, retention-KL, and alpha-tuning are closed routes. Further full retraining is not justified before the context-dependence result is available; a complementary no-training measurement remains a possible control.
