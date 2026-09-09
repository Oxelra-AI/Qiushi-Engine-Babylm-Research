# earlier analysis combined decisive readout

This file-only readout uses only the corrected earlier analysis split scorer roots and excludes failed partial scorer outputs.

## Roots

- experiments/archive/frontier_consolidation/data/register_decisive_gpu_eval
- experiments/archive/frontier_consolidation/data/incorpus_decisive_gpu_eval

## Arms seen

{
  "regmax_adultprose": [
    "chck_100M",
    "chck_80M"
  ]
}

## Contrast table

| contrast | checkpoint | BLiMP | Supplement | EWoK | COMPS | Entity | exEntity4 | cheap5 | complete families |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| regmax_adultprose_minus_clean | chck_80M | +0.6600 | +1.1800 | +0.3700 | +0.8600 | +2.0600 | +0.7675 | +1.0260 | 5 |
| regmax_adultprose_minus_clean | chck_100M | +0.5200 | +1.9000 | -0.2400 | +0.6000 | +2.1400 | +0.6950 | +0.9840 | 5 |

## Frozen prediction anchors

- Register childspeech-minus-adultprose profile exEntity5 prediction: 0.7482735781151597; word-control exEntity5: -0.06922283357166191; strict task-text profile exEntity5: 0.7697313461466507.
- Register removal-rate commitment: positive_register_contrast_predicted_by_rate: adult-prose block has larger clean late loss reduction, so removing adult prose should be costlier and childspeech_removed - adultprose_removed should be positive
- Register removed-block late rates: child/speech {'n': 3, 'mean': 0.150145, 'sd': 0.003279, 'min': 0.147004, 'max': 0.153546, 'values': [0.149885, 0.153546, 0.147004]}; adult prose {'n': 3, 'mean': 0.212764, 'sd': 0.008398, 'min': 0.203253, 'max': 0.219157, 'values': [0.215882, 0.219157, 0.203253]}; adult-minus-child {'n': 3, 'mean': 0.062619, 'sd': 0.00552, 'min': 0.056249, 'max': 0.065997, 'values': [0.065997, 0.065611, 0.056249]}
- Register interpretation consequence: If this rate sign and the static profile sign coincide, a positive register score cannot by itself distinguish target-profile proximity from removal-side active-error opportunity cost. If they diverge, the register score adjudicates between them. In either case, Entity must remain separated from broad ex-Entity interpretation.
- In-corpus rate commitment: conditional_rate_commitment: official in-corpus broad late gain should be small/repeat-like if the same-script in-corpus admitted-block loss reduction from 60M to 100M is at or below the repeat-distinct midpoint; it should approach the view/breadth late-retention regime only if that 60M->100M reduction remains distinct-like. The current clean-prior level is recorded now and must not be used alone as the mechanism.
- In-corpus clean-prior position: {'incorpus_clean_prior_100M': 3.49535, 'reference_clean_prior_100M': {'view': 3.26586, 'repeat': 2.949448, 'breadth': 5.266488}, 'nearest_reference_by_level': 'view', 'distance_to_nearest': 0.22949, 'distance_to_breadth_prior': 1.771138}
- In-corpus static profile commitment: {'incorpus_minus_clean_exEntity4_noReading': 0.27543065398934624, 'incorpus_minus_clean_exEntity5_withReading': 0.28927327530058433, 'incorpus_minus_clean_cheap5_noReading': 0.3115962258155697, 'incorpus_minus_full1x_exEntity4_noReading': 0.020123307236453336, 'incorpus_minus_full1x_cheap5_noReading': 0.042285993744071604}
- In-corpus word-JS control commitment: {'incorpus_minus_clean_exEntity4_noReading': -0.024001987762539246, 'incorpus_minus_full1x_exEntity4_noReading': -0.01742863003330492}
- In-corpus measured-band interpretation: status=False, classification=pending_future_incorpus_60M_100M_rate, rate_60_to_100=None.
- In-corpus measured-band note: The post-training in-corpus rate ladder remains pending; repeat the analysis when it is available.

## Files

- summary_json: `experiments/archive/frontier_consolidation/data/combined_decisive_readout/combined_decisive_readout.json`
- summary_md: `research/documents/frontier_consolidation/data/combined_decisive_readout/combined_decisive_readout.md`
- arm_csv: `experiments/archive/frontier_consolidation/data/combined_decisive_readout/arm_scores_vs_clean.csv`
- contrast_csv: `experiments/archive/frontier_consolidation/data/combined_decisive_readout/decisive_contrasts.csv`
