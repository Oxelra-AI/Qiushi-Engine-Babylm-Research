# register rate and spread interpretation register-removal rate and reference-rate spread interpretation

This note records a CPU-only pre-score measurement made before using any new childspeech register scores. It responds to the removal-side form of the active-error/rate account.

## Why this was necessary

The MAX-register pair admits the identical FineWeb compact-view block and differs only in which clean rows are sacrificed:

- `childspeech_removed`: removes 6,992 developmental/speech rows = 1,118,720 words (`childes` 578,080; `open_subtitles` 411,040; `bnc_spoken` 124,160; `switchboard` 5,440).
- `adultprose_removed`: removes 6,992 adult-prose rows = 1,118,720 words (`gutenberg` 710,560; `simple_wiki` 408,160).

The score contrast is `childspeech_removed - adultprose_removed`. Positive means the arm that removed child/speech rows scores higher, so removing adult prose was costlier.

The distribution proximity prediction static-profile model already predicts this contrast should be positive: about +0.748 exEntity5 and +0.817 cheap6, with strict task-text extraction +0.770 exEntity5/+0.844 cheap6. Word-unigram JS predicts near zero/slightly negative.

But a removal-side rate account can also predict the same sign if adult-prose rows are still an active error source late in the clean training trajectory while child/speech rows have flattened. Therefore register scores alone cannot be interpreted cleanly unless the removed-block rate is frozen before the childspeech results are read.

## New register rate and spread interpretation measurement

Script: `experiments/archive/frontier_consolidation/scripts/register_removal_rate_and_reference_spread.py`

Output directory: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread`

The script loads the clean DeBERTa checkpoint ladder and computes forward-only MLM loss on sampled rows from the two removed blocks under 3 independent row/mask replicates. It also recomputes repeat/breadth/view reference changed-block rates under the same three-replicate design.

### Register removed-block rate

Clean 60M→100M loss reduction:

- child/speech removed block: mean 0.150145, sd 0.003279, values [0.149885, 0.153546, 0.147004]
- adult-prose removed block: mean 0.212764, sd 0.008398, values [0.215882, 0.219157, 0.203253]
- adult-minus-child rate: mean 0.062619, sd 0.005520, values [0.065997, 0.065611, 0.056249]

This freezes a positive rate-based sign prediction for `childspeech_removed - adultprose_removed`: adult-prose rows were still being learned more late by the clean model, so removing them should be costlier.

## Consequence for interpretation

The distribution proximity prediction static-profile account and the register rate and spread interpretation removal-side rate account now agree on the sign of the register contrast. Therefore:

- If the final official register contrast is positive, it supports that adult-prose removal is more damaging, but it does **not** by itself distinguish target-profile proximity from removal-side late-error opportunity cost.
- If the contrast is near zero or negative, it challenges both the static-profile prediction and the removal-side rate prediction, with word-JS remaining only a weak negative control.
- If the magnitude is much smaller than the distribution proximity prediction +0.75 exEntity5 profile prediction but has the correct sign, this points toward a mixed or weak opportunity-cost effect rather than a clean surface-profile law.
- Entity must remain separated into zero/nonzero-operation rows before being used in interpretation, because previous Entity effects were operation-mixture allocation, not balanced tracking.

This makes the in-corpus adult-prose cell the stronger remaining discriminator among static target proximity, out-of-corpus novelty, and active-error persistence. The in-corpus admission-side late rate should be interpreted using the reference spread below, not a sharp hardcoded midpoint.

## Reference rate spread for repeat/breadth/view

Historical single-sample 60M→100M changed-block rates from earlier analysis were:

- repeat 0.148226
- breadth 0.222660
- view 0.295710

register rate and spread interpretation three-replicate resampling gives:

- repeat: mean 0.154030, sd 0.007174, range [0.147678, 0.161811]
- breadth: mean 0.229361, sd 0.010410, range [0.220871, 0.240976]
- view: mean 0.303934, sd 0.019549, range [0.292553, 0.326507]

Midpoint bands:

- repeat/breadth midpoint: mean 0.191695, sd 0.006574, range [0.184275, 0.196788]
- repeat/view midpoint: mean 0.228982, sd 0.013206, range [0.220116, 0.244159]

The reference ordering is stable under resampling, but the bands are narrow. Future in-corpus rate placement should treat ~0.18–0.20 as the repeat/breadth transition region and ~0.22–0.24 as the repeat/view transition region, rather than treating the original 0.185443 and 0.221968 midpoints as sharp constants.

## Files

- Summary markdown: `research/documents/frontier_consolidation/data/register_removal_rate_and_reference_spread/register_removal_rate_and_reference_spread.md`
- Full JSON: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/register_removal_rate_and_reference_spread.json`
- Compact commitment: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/prediction_commitment.json`
- Measurements CSV: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/loss_measurements.csv`
- Rate rows CSV: `experiments/archive/frontier_consolidation/data/register_removal_rate_and_reference_spread/rate_rows.csv`
