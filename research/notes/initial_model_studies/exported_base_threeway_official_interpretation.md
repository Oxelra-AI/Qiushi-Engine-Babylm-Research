# exported base threeway official interpretation — Three-way exported-base official-compatible interpretation

## Evidence

- earlier analysis WESS auxiliary screen: `data/wess_aux_base_transfer_screen.json`
- Exported available-column scores: `data/exported_available_scores.json`
- EWoK scores for plain vs WESS-exported base: `data/exported_ewok_scores.json`
- No-address synthetic/export control: `data/noaddress_export_control.json`
- No-address available+EWoK official-compatible scores: `data/noaddress_available_ewok_scores.json`
- Exported HF checkpoints:
  - plain: `training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal/plain_mixed/hf_model`
  - WESS aux exported base: `training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal/wess_aux/hf_model`
  - no-address exported base: `training/runs/wess_aux_base_transfer_screen/smoke_384x4_anneal/no_address/hf_model`

## Why this comparison matters

Annotated WESS success cannot be assumed to be a usable BabyLM architecture because official training/evaluation inputs do not contain gold entity/state spans or routing labels. Therefore earlier analysis-194 separated:

1. **WESS-on annotated synthetic inference** — whether the slot mechanism works when spans/routes are known;
2. **exported base-only transfer** — whether the ordinary HF DeBERTa backbone improves when WESS is disabled at inference;
3. **no-address exported control** — whether any official-facing gains are specific to persistent entity addressing or just generic auxiliary/synthetic/fusion training.

## Synthetic mechanism and base-transfer results

On the held-out independent-template synthetic binding suite:

| model / mode | example acc | pair acc | log-odds | intervention |
|---|---:|---:|---:|---|
| plain_mixed base-only | 0.4925 | 0.160 | -0.0627 | none |
| WESS aux base-only | 0.0000 | 0.000 | +0.0006 | none |
| **WESS aux WESS-on** | **1.0000** | **1.000** | **+9.0395** | swap +16.41, ablation +10.63 |
| no-address base-only | 0.5050 | 0.140 | -0.0492 | none |
| no-address-on | 0.4875 | 0.135 | -0.0515 | swap 0.0, ablation +0.19 |

Interpretation: annotated WESS remains a powerful causal mechanism. However, the WESS-trained exported base does **not** retain synthetic binding when WESS is disabled. Plain and no-address get some top-1 hits but negative log-odds, so this is not robust paired binding.

## Official-compatible exported-base scores (384×4 smoke)

| column | plain_mixed | WESS-exported base | no-address exported base |
|---|---:|---:|---:|
| BLiMP | 53.31 | 52.15 | 53.56 |
| Supplement | 48.95 | 48.83 | 48.73 |
| Entity | 16.63 | 16.48 | 16.75 |
| COMPS | 49.95 | 49.91 | 49.84 |
| EWoK | 50.40 | 50.17 | 49.79 |
| GlobalPIQA mean | 31.81 | 33.265 | 33.295 |
| Reading mean | 5.66 | 6.30 | 5.775 |
| Available-7 mean | 36.673 | 36.729 | 36.820 |

Deltas versus plain:

| column | WESS-exported - plain | no-address - plain |
|---|---:|---:|
| BLiMP | -1.16 | +0.25 |
| Supplement | -0.12 | -0.22 |
| Entity | -0.15 | +0.12 |
| COMPS | -0.04 | -0.11 |
| EWoK | -0.23 | -0.61 |
| GlobalPIQA mean | +1.455 | +1.485 |
| Reading mean | +0.640 | +0.115 |
| Available-7 mean | +0.056 | +0.147 |

WESS-exported minus no-address:

| column | delta |
|---|---:|
| BLiMP | -1.41 |
| Supplement | +0.10 |
| Entity | -0.27 |
| COMPS | +0.07 |
| EWoK | +0.38 |
| GlobalPIQA mean | -0.03 |
| Reading mean | +0.525 |

## Scientific conclusion

The current exported-base official-facing signal is **not sufficient evidence** that persistent entity addressing transfers into standard encoder weights.

Key reasons:

1. **Base-only synthetic binding fails for WESS aux.** When WESS is disabled, the exported base has pair accuracy 0 and log-odds ≈0 on the controlled binding suite. Thus the entity-state binding mechanism does not reside in the ordinary backbone in the current auxiliary setup.
2. **Official target-cluster columns do not improve.** Entity and EWoK are slightly lower for WESS-exported than plain, while no-address is actually slightly higher on Entity. This is the opposite of what would be expected if entity-addressed binding had transferred into official-relevant behavior.
3. **GlobalPIQA gain is not address-specific.** WESS and no-address have essentially identical GlobalPIQA mean gains (+1.455 vs +1.485), so this movement is likely generic auxiliary/synthetic training, data mixture, schedule, or regularization rather than persistent entity addressing.
4. **Reading gain may be WESS-specific but is small and isolated.** WESS improves Reading more than no-address (+0.64 vs +0.115), but this does not solve the Entity/EWoK gap and may not survive scale/seeds.
5. **Available-7 mean favors no-address over WESS in this smoke.** WESS has +0.056 over plain; no-address has +0.147 over plain. Neither is a BabyLM candidate.

Therefore the WESS route has now split into two distinct unresolved problems:

- **Inference-time WESS route:** make WESS operate on unlabeled official inputs by learning entity/state span extraction and routing, then evaluate WESS-on official inputs without gold annotations.
- **Auxiliary-transfer route:** redesign the auxiliary objective so the binding knowledge transfers into the standard DeBERTa backbone, not only into the WESS slot module.

The current annotated WESS auxiliary recipe should **not** be scaled directly to S1/100M as a SOTA candidate. It would likely spend budget on a powerful annotation-dependent module whose ability disappears when official inputs are unlabeled.

## Best next research action

The immediate next experiment should be a **learned-address WESS bridge** rather than more exported-base scaling:

1. Train a lightweight span/address router from the synthetic episodes using only text tokens, not gold routing at inference.
2. Evaluate WESS-on held-out independent templates with predicted spans/routes.
3. Include gold-route WESS as an upper bound and random/no-address as controls.
4. Test whether the predicted-router WESS keeps large slot-swap/write-ablation effects.
5. Only if learned routing approaches the gold-route mechanism should it be integrated into an official-compatible trainer.

A parallel auxiliary-transfer experiment can also be useful: add a distillation/consistency loss from WESS-on logits to base-only logits on binding-critical masks, so the base encoder is explicitly trained to imitate the slot-correct prediction. That would directly attack the failure observed here: WESS-on is perfect, but base-only is zero.

The highest BabyLM SOTA goal remains unsolved. The protected internal best remains the DeBERTa 8×480 WWM 100M coordinate (Overall ~40.53), below the public leader (~41.80). No WESS model has improved the official Entity/EWoK target cluster or produced a 9/9 coordinate.
