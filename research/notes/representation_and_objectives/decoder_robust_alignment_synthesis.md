# decoder alignment route boundary decoder-robust alignment synthesis

No model weights were updated. Each target fitted layerwise statistic/ridge maps on legal-corpus masked states only; the adjacent-depth/equal-blend readout was selected on the non-official earlier analysis naturalistic bridge, then applied unchanged to EWoK and GlobalPIQA.

Selected bridge-fixed rule: decoder_mode=raw band=L1-L2 selector_score=1.0 persistent actual-swap=1 band actual-swap=0.

## Final-head and alignment checks

- `matched_base_80M`: final-head max |logit diff|=0.0, target logprob diff=0.0; adapter placement=None.
- `scale1p75_live_80M`: final-head max |logit diff|=0.0, target logprob diff=0.0; adapter placement={'source': 'experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M80M_seed43022/hf_model/chck_80M/adapter_scaled_modeling.py', 'post_layer_adapter_addition_line_present': True}.
- `scale1p75_disabled_80M`: final-head max |logit diff|=0.0, target logprob diff=0.0; adapter placement=None.

## Fixed official readout

- `matched_base_80M` EWoK: both=0.08324324324324324, persistent both=0.017297297297297298, swapped null=0.01945945945945946, stable=0.3005405405405405, final-not-both persistent recovered=16.
  GlobalPIQA hard52: true acc=0.11538461538461539, persistent true=0.07692307692307693, rotated null=0.3076923076923077, final-wrong persistent recovered=3.
- `scale1p75_live_80M` EWoK: both=0.08216216216216216, persistent both=0.024864864864864864, swapped null=0.011891891891891892, stable=0.30486486486486486, final-not-both persistent recovered=18.
  GlobalPIQA hard52: true acc=0.15384615384615385, persistent true=0.07692307692307693, rotated null=0.2692307692307692, final-wrong persistent recovered=3.
- `scale1p75_disabled_80M` EWoK: both=0.08864864864864865, persistent both=0.02054054054054054, swapped null=0.005405405405405406, stable=0.3081081081081081, final-not-both persistent recovered=13.
  GlobalPIQA hard52: true acc=0.15384615384615385, persistent true=0.07692307692307693, rotated null=0.2692307692307692, final-wrong persistent recovered=3.

## Paired deltas

- `scale1p75_live_80M_minus_matched_base` EWoK net persistent both=7, net stable=4; GP hard52 net persistent true=0, delta top-minus mean=0.06707400529217687.
- `scale1p75_disabled_80M_minus_matched_base` EWoK net persistent both=3, net stable=7; GP hard52 net persistent true=0, delta top-minus mean=0.07120322378203896.
- `scale1p75_live_minus_disabled` EWoK net persistent both=4, net stable=-3; GP hard52 net persistent true=0, delta top-minus mean=-0.004129218489862081.

Interpretation should be based on whether the bridge-fixed aligned band shows persistent official-row recovery beyond target-swap/rotated-label nulls and whether scale1.75 specifically loses such rows late relative to the matched base.
