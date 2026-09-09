# chck84 endpoint carrier validation stabilization scaffold: same-trajectory late-weight averaging

## Why this exists

The earlier correction is that if the seed43122 common-grid does not reproduce the seed43022 late 82-84M competence-allocation phase, stop tuning the original seed43022 peak and pursue a mechanism that stabilizes broad competence across stochastic trajectories. earlier analysis history review established that the prior earlier analysis-92 recombination work was prediction-level selection/voting, not parameter averaging or training-time EMA/SWA/LAWA. existing ladder checkpoint route state did contain a simple two-checkpoint arithmetic parameter average, but in the older noncompliant-tokenizer coordinate; it gained only +0.0243 equal7 over its old 100M fast reference and hurt GlobalPIQA, so it warns against naive averaging but does not close the current adapter-coordinate stabilization question.

chck84 endpoint carrier validation therefore prepared a same-trajectory late-iterate averaging scaffold as a possible low-cost next experiment after the pending seed43122 evidence is read. It is not cross-seed weight interpolation and not a leaderboard/submission path.

## Literature grounding

Three retrieved/read sources motivate but do not prove the idea:

- LAWA (`\cite{kaddour2022stop}`): late weight averaging can be read as a low-pass filter over optimizer iterates; the averaging window and late-training region matter.
- Switch EMA (`\cite{li2024switch}`): training-time switching between online and EMA weights is a slow/fast stability mechanism related to flatter solutions.
- SWA for PLM fine-tuning (`\cite{lu2022improving}`): stochastic weight averaging can improve generalization in compact pretrained LM fine-tuning settings.

The BabyLM result must come from official-compatible selected scoring; these sources only justify a mechanistic candidate.

## Built artifacts

Script:

- `experiments/archive/frontier_consolidation/scripts/late_weight_average_builder.py`

Dry-run manifest:

- `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/late_weight_average_scaffold_manifest.json`
- `research/documents/frontier_consolidation/data/late_weight_average_scaffold/late_weight_average_scaffold_manifest.md`

The dry run enumerates 15 buildable same-trajectory candidates: 5 templates for each of the reference scale1.75 seed43022, scale1.25 seed43022, and scale1.75 seed43122 trajectories. All candidates preserve the trusted-code adapter coordinate and shared checkpoint tokenizer hash `a9cbb830495cb92bbb2996adc256207746282ee67f4f8d40a4f646a634ec139a`.

Built mechanical smoke candidate:

- Candidate: `reference_scale1p75_seed43022__center_80_82_84_uniform`
- Candidate directory: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform`
- Averaged endpoints: `chck_80M`, `chck_82M`, `chck_84M`
- Weights: uniform `[1/3, 1/3, 1/3]`
- Averaged model SHA256: `d47c15f96e3424fc0946cfa4e747f9a6374006d9a3cf68d58d5031509585ac50`
- Trusted-code load: `AdapterDebertaV2ForMaskedLM`, `35,463,008` params, adapter scale `1.75`, vocab `16,384`, finite logits.
- Native fallback still wrong: `DebertaV2ForMaskedLM`, `34,467,424` params, 48 adapter tensors unexpected/dropped.
- Manifest: `experiments/archive/frontier_consolidation/data/late_weight_average_scaffold/candidates/reference_scale1p75_seed43022__center_80_82_84_uniform/averaging_manifest.json`

No official-compatible evaluation was run for this average, and no upload/submission was performed for it. It is a mechanical candidate only.

## How to use it later

Only evaluate same-trajectory averages if the delivered seed43122 grid suggests that peak instability, not data/objective failure, is the main obstruction. The first useful screen would be one or two selected cheap-task evaluations, not a full endpoint package. The natural first candidate is the already built reference `center_80_82_84_uniform` average because it asks whether low-pass smoothing of the rising late phase preserves the broad cheap6/cheap5 gains while reducing the immediate 86M churn.

Do not average across different random initializations with this script. Independent seed trajectories are not weight-aligned and require a separate functional-alignment argument before parameter-space averaging could be meaningful.
