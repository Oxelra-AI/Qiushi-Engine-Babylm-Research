# Inverse-priority full evaluation and proposed endpoint controls

## Authoritative new result

Full-evaluation measurements are available for the two seed43022 inverse-priority endpoints:

| endpoint | submit-facing status | Overall | SuperGLUE | AoA raw | AoA leaderboard | no-AoA equal7 |
|---|---|---:|---:|---:|---:|---:|
| inverse_priority chck_95M | endpoint-frozen scientific measurement, not submit-ready | 39.961218980736284 | 69.78503419725679 | -0.1719406337063024 | -17.19406337063024 | 43.86571428571428 |
| inverse_priority chck_100M | true standard endpoint, submit-ready locally | 39.714154598526164 | 69.83717271817582 | -0.18344781331440352 | -18.344781331440352 | 43.705 |

This closes inverse-priority masking as an immediate SOTA route. Its no-AoA gains were real but did not survive official scoring because AoA became strongly negative. The true `chck_100M` endpoint would have Overall about 41.752 if AoA were zero, still just below the visible 41.8 row; the endpoint-frozen 95M would exceed 41.8 under AoA=0 but is not submit-facing and also has negative AoA.

## Existing correction still matters

all mask endpoint interpretation recovered the true exposure-matched no-AoA contrast before the full-eval result arrived:

- uniform_control `chck_100M` equal7 43.2114;
- inverse_priority `chck_100M` equal7 43.7050;
- inverse-minus-uniform at 100M equal7 +0.4936 and equal6 without GlobalPIQA +0.4875.

So the mask-target/no-AoA signal is not fake and not mainly a GlobalPIQA artifact at the true endpoint. It is simply not AoA-safe in the current form.

Scientific evidence files:

- `data/endpoint_matched_mask_contrast/endpoint_matched_mask_contrast.json`
- `notes/endpoint_matched_mask_contrast.md`
- `data/inverse_priority_full_eval_interpretation/inverse_priority_full_eval_interpretation.json`
- `notes/inverse_priority_full_eval_interpretation.md`

## Evidence-visible follow-up measurements pending in this record

The comparison requires full evaluation of the existing evidence-visible endpoints:

- `chck_90M` (best no-AoA evidence-visible endpoint; endpoint-frozen scientific measurement);
- `chck_100M` (true standard endpoint).

Scientific purpose: determine whether the AoA collapse is inverse-specific or common to tail/mask continuation. Evidence-visible has a different mask-target distribution and near-threshold no-AoA columns, but its endpoint-matched gain is GlobalPIQA-driven; it is not expected to be final unless SuperGLUE/AoA surprises positively.

## Proposed conditional controls

- Uniform full-evaluation control: proposed only if evidence-visible also shows strongly negative AoA and a comparison is needed to determine whether common restart/rephasing alone causes AoA damage. No completed uniform measurement is established here.
- All-family interpreter: `scripts/interpret_all_mask_endpoint_full_eval.py`. It summarizes available mask endpoint full-eval payloads and AoA-zero counterfactuals from aggregate scores; its interpretation requires the completed evidence-visible measurements.
- Decoupled masking controls: `scripts/decoupled_mask_control_trainer.py` and `scripts/summarize_decoupled_mask_controls_noaoa.py`. These remain proposed controls, conditional on a full-score-surviving or replication-worthy masking variant; otherwise they are mechanism assets, not a path to immediate SOTA.

## Current decision posture

The completed inverse full-eval result alone does not justify the seed43122 inverse-vs-uniform replication. The full-score result is far below clean-Qwen and the visible leader. The evidence-visible comparison is needed to determine whether it has non-negative AoA or whether tail/mask continuations generally damage AoA. If evidence-visible also fails, the proposed alternatives are uniform restart AoA measurement or a continuation design that preserves clean-Qwen's AoA=0 while retaining the no-AoA gains.
