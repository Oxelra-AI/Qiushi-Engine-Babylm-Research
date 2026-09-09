# exported base threeway official interpretation — WESS transfer route update after three-way official scoring and distillation

## Evidence files

- Three-way exported-base official-compatible table: `data/noaddress_available_ewok_scores.json`
- Plain/WESS auxiliary exported-base screen: `data/wess_aux_base_transfer_screen.json`
- No-address export control: `data/noaddress_export_control.json`
- WESS-to-base distillation screen: `data/wess_to_base_distill_screen.json`
- Detailed interpretation before distillation: `notes/exported_base_threeway_official_interpretation.md`

## What the official-compatible exported-base comparison showed

The three exported plain HF backbones were evaluated with WESS disabled:

| column | plain mixed | WESS aux exported base | no-address exported base |
|---|---:|---:|---:|
| BLiMP | 53.31 | 52.15 | 53.56 |
| Supplement | 48.95 | 48.83 | 48.73 |
| Entity | 16.63 | 16.48 | 16.75 |
| COMPS | 49.95 | 49.91 | 49.84 |
| EWoK | 50.40 | 50.17 | 49.79 |
| GlobalPIQA mean | 31.81 | 33.265 | 33.295 |
| Reading mean | 5.66 | 6.30 | 5.775 |
| Available-7 mean | 36.673 | 36.729 | 36.820 |

The GlobalPIQA movement is not specific to persistent entity addressing: no-address matches or slightly exceeds WESS aux on GlobalPIQA mean. WESS aux has a larger Reading gain, but it loses BLiMP and does not improve Entity/EWoK. The target cluster for the BabyLM gap therefore does not move in the desired direction.

## What the synthetic transfer measurements showed

Annotated WESS remains powerful when span/routing metadata is supplied:

- WESS-on: pair accuracy 1.000, log-odds +9.04, slot swap +16.41, write-removal +10.63.

But the exported base with WESS disabled has no controlled binding:

- WESS aux base-only: pair accuracy 0.000, log-odds +0.0006.
- plain mixed base-only: pair accuracy 0.160, log-odds -0.0627.
- no-address base-only: pair accuracy 0.140, log-odds -0.0492.

The nonzero top-1 hits in plain/no-address are not robust paired binding because correct-vs-counterfactual log-odds are negative.

## Distillation attempt

To test whether the WESS-correct signal can be pushed into the official-compatible backbone, exported base threeway official interpretation trained `wess_aux_distill` with:

- WESS-on MLM loss;
- base-only CE on the synthetic binding targets;
- KL from detached WESS-on logits to base-only logits;
- the same 384×4 shape and 50%→25% schedule.

Result:

| measurement | value |
|---|---:|
| WESS-on pair accuracy | 1.000 |
| WESS-on log-odds | +6.431 |
| WESS-on slot swap | +11.850 |
| WESS-on write-removal | +7.666 |
| base-only pair accuracy | 0.140 |
| base-only log-odds | -0.069 |
| official-text MLM loss base-only | 3.836 |

The distillation attempt did not transfer controlled binding into the ordinary backbone. It also worsened ordinary-text MLM loss relative to plain (3.602), WESS aux (3.480), and no-address (3.583). The likely reason is that the base-only branch can fit local target frequencies or template bias without acquiring entity-indexed persistent state; the WESS module still solves the task, so the auxiliary path absorbs the structured credit assignment rather than forcing it into the encoder.

## Scientific conclusion

The route “train annotated WESS as an auxiliary module, export the plain backbone, and expect official Entity/EWoK improvement” is not supported by the current evidence. The address-specific mechanism exists and is stable when spans/routes are supplied, but its benefit does not naturally become a standard DeBERTa capability, and a simple CE/KL transfer loss did not fix this.

The next meaningful problem is therefore **unlabeled addressing**, not larger-scale annotated auxiliary training.

## Next executable route

Build a learned-address WESS bridge:

1. Train a text-only span/router head on the synthetic episodes, using gold spans/routes during training only.
2. At held-out evaluation, run WESS with predicted entity/state positions and predicted entity-to-slot routes, not gold metadata.
3. Compare gold-route WESS, predicted-route WESS, no-address memory, and random route on independent templates.
4. Measure pair accuracy, log-odds, slot swap, write-removal, and routing/span accuracy separately.
5. Only if predicted-route WESS approaches the gold-route result should it be integrated into official-compatible BabyLM training/evaluation.

A second, lower-priority route is to redesign transfer loss so base-only predictions must solve counterfactual pairs jointly rather than single examples, for example pairwise swapped-state margin with matched same-bag examples. But exported base threeway official interpretation suggests simple WESS-logit distillation is not enough.

The BabyLM SOTA goal remains open. No WESS model has improved the official Entity/EWoK target cluster or produced a complete 9-column score.
