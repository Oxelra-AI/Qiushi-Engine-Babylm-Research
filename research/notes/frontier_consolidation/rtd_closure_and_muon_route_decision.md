# rtd closure and muon route decision — RTD state probe and next whole-system route

## Why this note exists

The prior plan left open a delayed mature-tail MLM+RTD-GDES experiment from the spatial repair route status 80M checkpoint. The limitation is that delayed consolidation was not established merely because the positive RTD mechanism probe used an 80M encoder while the score-bearing run used RTD from initialization. Before any second RTD training task, I ran the same low-cost RTD/MLM interaction probe on existing checkpoints.

The active goal remains unchanged: a fully legal BabyLM 2026 Strict-Small Overall SOTA from real training and official-compatible evaluation. The best fully legal complete endpoint is still spatial repair route status compact-view reinvest, Overall 41.257770896404615, below the 41.8 target. The protected substrate remains compact semantic second views plus reinvested source diversity.

## rtd closure and muon route decision state-dependent RTD/MLM probe

Artifacts:

- Script: `scripts/state_dependent_rtd_probe.py`
- Result: `data/state_dependent_rtd_probe/state_dependent_rtd_probe.{json,md}`

The probe reused the same 5,184 examples / 799,882 words / 21 batches as the earlier analysis balanced RTD probe. For each checkpoint it fitted a fresh class-balanced frozen-encoder RTD head for 80 steps, measured held-out hard-vs-random corruption discrimination, and measured trained-head MLM-vs-RTD gradient geometry over six batches.

| checkpoint | hard bal acc | hard AUROC | random-hard AUROC gap | replacement rate | trunk cosine | trunk RTD/MLM | rel cos | lambda1 rotation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| spatial repair route status MLM 20M | 0.6477 | 0.7159 | 0.2136 | 0.6968 | +0.0218 | 0.3186 | -0.0591 | 17.56° |
| spatial repair route status MLM 80M | 0.6871 | 0.7622 | 0.1884 | 0.5224 | +0.0338 | 0.2506 | -0.2359 | 13.95° |
| mlm rtd gdes 20m screen plan RTD 20M | 0.7023 | 0.7913 | 0.1276 | 0.6925 | +0.0266 | 0.2258 | +0.1901 | 12.65° |

Direct comparisons:

- 80M MLM vs 20M MLM: hard AUROC +0.0463, random-hard AUROC gap -0.0252, trunk cosine +0.0121, trunk RTD/MLM ratio -0.0680, lambda1 rotation -3.61°.
- RTD20 vs matched MLM20: hard AUROC +0.0754, random-hard AUROC gap -0.0860, trunk cosine +0.0048, trunk RTD/MLM ratio -0.0928, lambda1 rotation -4.91°.

## RTD route judgment

The delayed-tail RTD explanation is not supported strongly enough for another expensive training run.

The decisive comparison is RTD20 vs MLM20. The replacement rates are essentially matched, and the RTD-trained encoder has exactly the kind of improved RTD readout one might have wanted: hard AUROC rises from 0.7159 to 0.7913 and the random-hard shortcut gap shrinks from 0.2136 to 0.1276. Yet the corresponding mlm rtd gdes 20m screen plan official-compatible behavior was only cheap7 +0.1107, with EWoK -0.66, GlobalPIQA -0.97, and Reading -0.235. This disconnect shows that improving the RTD readout is not predictive of the broad BabyLM likelihood competence required by the goal.

The 80M checkpoint does not show a qualitative transition from orthogonal side signal to cooperative consolidation signal. Its trunk cosine remains very small (+0.0338), its RTD/MLM norm ratio remains material (0.2506), and its relative-position embedding gradient is more negative (-0.2359) than at 20M. Maturity makes hard corruptions somewhat more contextual, but it does not turn RTD into an MLM-aligned update. A delayed tail would need a qualitative reversal of the observed EWoK/GlobalPIQA/Reading tradeoff, and rtd closure and muon route decision gives no positive mechanism for that reversal.

Decision: do not launch a mature-tail RTD continuation now. The from-scratch lambda=1 route is closed by mlm rtd gdes 20m screen plan, and the delayed-tail variant is set aside unless a future non-training measurement directly shows preserved MLM semantic/polarity/event margins under a mature RTD perturbation. Further H100 training on RTD is not justified by this evidence.

## independent_review read after rtd closure and muon route decision

independent_review files:

- Generator: 
- Verifier: 

The verifier independently agreed that the same-family RTD route should be set aside: RTD improved its own hard-discrimination probe without broad BabyLM movement; rtd closure and muon route decision showed no state-dependent shift strong enough to justify delayed-tail training.

The generator proposed the strongest distinct next route: preserve the compact-view reinvest substrate and official MLM interface, but change hidden-weight update geometry using Muon/AdaMuon-style orthogonalized updates on the core attention/FFN matrices while leaving embeddings, MLM head, norms, and DeBERTa relative-position channel on AdamW. This is distinct from the closed LAMB, RTD, data, tokenizer, and masking routes. It is supported by external BabyLM 2026 evidence that BabySteps and RecGPT both used Muon-family optimizers at this data scale, but that external evidence is not itself enough because those are causal/GPT-style stacks rather than our DeBERTa-v2 MLM stack.

## rtd closure and muon route decision Muon spectral update probe

Artifacts:

- Script: `scripts/muon_spectral_update_probe.py`
- Result: `data/muon_spectral_update_probe/muon_spectral_update_probe.{json,md}`

This zero-training measurement replayed 12 legal compact-view batches at spatial repair route status 20M and 80M. Because the original trainer did not save optimizer buffers, the measurement uses local replayed gradient/momentum estimates, not historical optimizer state. It asks whether AdamW hidden-matrix updates are concentrated into a few singular directions.

Core hidden matrices here mean attention query/key/value/output and FFN input/output weights, excluding token/position/relative embeddings, the MLM head, and DeBERTa position projection weights.

| checkpoint | momentum stable-rank mean | AdamW-step stable-rank mean | momentum top-8 Frobenius share | cos(momentum, polar) | cos(AdamW-step, polar) |
|---|---:|---:|---:|---:|---:|
| spatial repair route status MLM 20M | 2.66 | 3.55 | 0.738 | 0.365 | 0.378 |
| spatial repair route status MLM 80M | 5.06 | 7.54 | 0.482 | 0.545 | 0.568 |

Family details:

- At 20M: attention_qkv stable rank 2.92, attention_output 2.30, ffn_in 3.08, ffn_out 1.85; DeBERTa position projections 1.90 and relative-position embedding 1.31.
- At 80M: attention_qkv stable rank 4.21, attention_output 5.53, ffn_in 6.53, ffn_out 5.69; DeBERTa position projections 2.64 and relative-position embedding 1.51.

The top eight singular directions carry 73.8% of core update energy at 20M and 48.2% at 80M. These ranks are tiny compared with the 480-dimensional hidden-matrix rank capacity. This supports real headroom for a Muon-style spectral update intervention on the core hidden matrices.

A small post-probe scale calculation estimated the Muon matrix learning rate that would match the spatial repair route status AdamW relative step size on these replayed updates:

- spatial repair route status 20M core mean 0.0127, median 0.0106; attention/FFN-in groups around 0.010-0.011; FFN-out around 0.024.
- spatial repair route status 80M core mean 0.00845, median 0.00718; attention/FFN-in groups around 0.00698-0.00728; FFN-out around 0.0148.

Thus an imported BabySteps matrix LR 0.02 would not be the clean first test for this DeBERTa-v2 system. A calibrated first screen should use roughly 0.008-0.012 on core matrices, with the AdamW auxiliary group initially kept at the spatial repair route status LR 0.001 to isolate hidden update geometry.

## Next route: calibrated Muon-hidden / AdamW-interface screen

Scientific hypothesis:

Under the legal 10M-word / 10-epoch budget, the compact-view substrate may be failing to accumulate further because AdamW installs MLM evidence through low-rank hidden-matrix update directions. Muon-style momentum orthogonalization on attention and FFN matrices could spread finite evidence across more hidden dimensions, making the compact semantic second-view signal compound rather than rotate between BLiMP, Supplement, EWoK, GlobalPIQA, and Reading.

Why this route is worth one bounded screen:

1. It leaves the legal data, tokenizer, architecture interface, MLM objective, and official evaluation path unchanged.
2. It is distinct from the closed LAMB route: LAMB changes per-parameter/trust-ratio scaling, not the singular-direction geometry of matrix updates.
3. It is supported by rtd closure and muon route decision local spectral evidence in our exact system, not just by external Muon examples.
4. It attacks a whole-learning-system layer after data, tokenizer, masking, checkpointing, and RTD routes have been narrowed.

First implementation should use:

- Model/corpus/tokenizer/WWM/seeds/schedule exactly matching spatial repair route status legal compact-view reinvest.
- Optimizer parameter groups:
  - Muon: only `attention.self.{query,key,value}_proj.weight`, `attention.output.dense.weight`, `intermediate.dense.weight`, `output.dense.weight` for all eight DeBERTa layers.
  - AdamW: token embeddings, absolute position embeddings, `encoder.rel_embeddings`, DeBERTa `pos_key_proj` and `pos_query_proj`, MLM head, LayerNorms, all biases, and all scalar/vector parameters.
- Global grad clipping before optimizer step, as in spatial repair route status.
- Same cosine schedule horizon and warmup fraction as spatial repair route status, applied to both groups.
- Primary matrix LR bracket: calibrated `0.008` and `0.012` if two H100s are available; if only one arm is allowed, use `0.010` as the median early-scale compromise. AdamW auxiliary LR initially `0.001`.
- Checkpoints at 5M/10M/20M for the screen; no 100M endpoint without score evidence.

Expensive-work admission for the next training:

- What it decides: whether hidden-matrix update orthogonalization produces broad official-compatible improvement on the protected compact-view substrate, unlike RTD's local-discrimination improvement and unlike closed data/tokenizer/masking routes.
- Lowest reliable cost: a 20M screen with calibrated matrix LR bracket and existing spatial repair route status 20M baseline, after CPU/GPU smoke validates exact first-batch loss, data order, WWM surface, save/load, and parameter grouping. A 5M-only result is not reliable for BabyLM behavior; a 100M run is not justified before the 20M screen.
- Continue only if the best Muon arm is broad: cheap7 at 20M improves meaningfully over spatial repair route status 20M and the movement is not a BLiMP-only or GlobalPIQA-only redistribution. Special attention: EWoK, GlobalPIQA, and Reading must not repeat the RTD damage pattern. A strong signal would be cheap7 >= +0.35 over spatial repair route status 20M with at least five of seven cheap columns nonnegative; a smaller but broad, non-damaging signal may justify extension to 80M only if fine-grained anatomy supports compounding rather than tradeoff. If movement is < +0.15 cheap7 or redistributive, close Muon for this stack and return to the identity-initialized gating/inductive-bias direction.

## Scientific Implications

Do not launch more RTD. The next constructive step should build and smoke-test the calibrated Muon-hidden / AdamW-interface trainer by adapting the existing spatial repair route status trainer, not by changing data or objective. If the trainer validates cleanly and GPUs are free, the first expensive action should be the bounded 20M screen above, not a full endpoint.
