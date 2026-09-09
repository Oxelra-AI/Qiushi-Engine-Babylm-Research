# masking schedule preflight and resource decision masking-schedule preflight and resource decision

## Why this work was worth doing now

The pending cached FineWeb source-breadth trajectory is still the central data evidence path, and no further FineWeb packet or filter construction should happen before that trajectory arrives. The orthogonal unresolved leader factor is the late switch from whole-word masking to token-level masking. A fixed-data WWM→token model was already trained on the same clean-Qwen data and protected DeBERTa-v2 8×480 / baseline16k / AdamW recipe, so converting it to downstream evidence can change whether this masking schedule is later combined with a surviving data mechanism.

## CPU result

`experiments/archive/representation_and_objectives/scripts/masking_schedule_training_measurement.py` produced:

- JSON: `experiments/archive/representation_and_objectives/data/wwm_to_token_training_measurement/wwm_to_token_training_measurement.json`
- Note: `research/notes/representation_and_objectives/wwm_to_token_training_measurement.md`

The training logs are identical to the COMPACT_EXPERIENCE fixed-WWM clean-Qwen reference through earlier analysis, at cumulative exposure 70,023,296 words. The first scientific difference is earlier analysis, cumulative exposure 70,062,953 words, where the target run switches from `wwm` to `token`. The target run has 1761 WWM steps and 754 token-mask steps; the fixed-WWM reference has 2515 WWM steps. This supports the smallest downstream conversion: evaluate only `chck_80M`, `chck_90M`, and `chck_100M`, with `chck_70M` unnecessary unless a later reader wants an identity anchor.

The post-switch objective loss is lower, but this is not BabyLM competence evidence because the prediction target changes. Downstream zero-shot plus Reading results are still needed before this factor should be carried into any combined run.

## GPU decision

I checked current H100 use because the strategist allowed this conversion only if a GPU could be isolated without delaying the FineWeb task or frontier_consolidation. GPU0 is clearly occupied by high-memory work. GPU1 was not stably isolated: a direct dry run saw GPU1 at 78,583 MiB free but 49% utilization, and a short protected sampler exited without evaluation because GPU1 was not stable enough. A 30-second sample also showed GPU1 alternating between idle-looking and active-looking states. Therefore I did not launch the downstream evaluation.

## Prepared launcher, not used for actual evaluation

I wrote two scripts for later immediate use when GPU1 is truly stable:

- `experiments/archive/representation_and_objectives/training/scripts/eval_wwm_to_token_postswitch_minimal.sh`
- `experiments/archive/representation_and_objectives/training/scripts/guarded_eval_wwm_to_token_postswitch_gpu1.sh`

The first script verifies the fixed-data run and evaluates only the three post-switch checkpoints. The second samples GPU1 and exits without evaluation unless it is stably idle. Both passed shell syntax checks; the protected sampler was run once and correctly refused to evaluate under unstable GPU1 utilization.

## Scientific Consequence

Do not extend FineWeb packet/filter machinery. If the repaired FineWeb result is delivered, inspect it immediately. If it is still pending and GPU1 becomes stably isolated, run the masking schedule preflight and resource decision protected launcher to convert the WWM→token factor into downstream evidence. If GPU1 remains unstable, keep concentration on the delivered managed results rather than starting another route.
