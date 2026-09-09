# r1 learning response result — R1 ordered-dynamic vs static-indirect learning-response result

## Purpose

A matched learning-response experiment tests the prerequisites for any 10M/full official R1 run:

- do not infer that ordinary WWM will learn composition merely because the generated data requires composition;
- do not use a tiny ~0.5M random-init screen;
- do not train a new readout layer;
- compare ordered-dependent experience against a control that preserves indirect-reference style and answer statistics but removes dynamic cross-step dependence.

This step completed that experiment.

## Training runs

Both arms used protected DeBERTa-v2 8×480 WWM geometry, baseline16k tokenizer from the protected run, seed 42 / init seed 456 / train RNG 789, batch 64, micro-batch 32, AdamW, 5M word exposure, and checkpoints at 1M intervals.

Evidence:

- Ordered dynamic run: `training/runs/r1_ordered_dynamic_5M/`
- Static indirect control: `training/runs/r1_static_indirect_control_5M/`
- Matched corpora: `data/r1_static_control/`
- Frozen MLM-head curve: `data/r1_mlm_counterfactual_learning_curve.json`
- Train-vocab / heldout-vocab split: `data/r1_vocab_split_learning_curve.json`

Training comparability:

| arm | exposure | optimizer steps | masked tokens | final loss | generated source words |
|---|---:|---:|---:|---:|---:|
| ordered dynamic | 5,000,000 | 890 | 947,273 | 6.2132 | 1,000,550 |
| static indirect control | 5,000,000 | 918 | 951,259 | 5.9381 | 1,001,003 |

The control has lower loss, consistent with the intended easier no-dynamic-dependence distribution. Both runs saved `chck_1M` through `chck_5M`.

## Frozen MLM-head readout

Readout: for each held-out same-word-bag counterfactual pair, mask the answer phrase and compare the model's own MLM-head log-likelihood for the true answer versus the paired counterfactual answer. No trained probe/readout is used. Candidate answer pairs are filtered to equal tokenizer length to avoid the length confound found in earlier analysis.

Protected reference on held-out vocabulary: paired case accuracy 0.5025, mean margin +0.0015, positive fraction 0.5025. The protected complete 100M model is essentially chance on this probe.

### Held-out vocabulary curve

Ordered-dynamic minus static-indirect control:

| checkpoint | pair-summed margin delta | both-contexts-correct delta | positive-pair-sum fraction delta |
|---|---:|---:|---:|
| chck_1M | -2.19e-7 | 0.00 | -0.055 |
| chck_2M | -1.93e-7 | 0.00 | -0.065 |
| chck_3M | -7.15e-8 | 0.00 | +0.085 |
| chck_4M | -1.00e-7 | 0.00 | +0.065 |
| chck_5M | -2.38e-8 | 0.00 | -0.005 |

### Training vocabulary curve

Ordered-dynamic minus static-indirect control:

| checkpoint | pair-summed margin delta | both-contexts-correct delta | positive-pair-sum fraction delta |
|---|---:|---:|---:|
| chck_1M | 0.00 | 0.00 | +0.060 |
| chck_2M | -9.54e-9 | 0.00 | +0.070 |
| chck_3M | 0.00 | 0.00 | +0.010 |
| chck_4M | -4.29e-8 | 0.00 | -0.125 |
| chck_5M | +2.62e-8 | 0.00 | +0.020 |

The strict pair-level success metric — both counterfactual contexts correct — is 0 for both arms at every checkpoint in both train-vocab and heldout-vocab settings.

## Interpretation

The learning-response criterion is **not passed**.

The ordered-dynamic arm does not produce a stable or meaningful advantage over the static-indirect control on the frozen MLM-head counterfactual readout. This holds even on the training vocabulary, so the result is not merely a failure of lexical transfer to held-out items.

The sample rows show a characteristic failure: trained checkpoints assign answer likelihood mostly by answer-token prior/preferences (e.g. preferring `vault` over `pantry`, or `shed` over `loft`) rather than by the passage order. Because each counterfactual pair is evaluated in both directions, these priors cancel to case accuracy around 0.5 and pair-summed margins around zero. This is exactly the kind of shortcut/credit-assignment failure these experiments repeatedly show: WWM can fit the distribution without using the dynamic state variable.

R1 data-side dependency is real (earlier analysis heuristics and counterfactuals), but under ordinary WWM and this mixture/scale it is not enough to make the MLM head prefer the correct order-dependent answer. The problem has moved from data construction to learning signal / architecture / credit assignment.

## Consequence

Do not proceed to a 10M official-column R1 screen in the current form. The next route should rebuild either:

1. the learning signal, so the model is directly penalized for confusing same-word-bag counterfactual states without relying on a new trained readout; or
2. the architecture/representation, so state variables become accessible to the ordinary MLM head; or
3. the experience design, so ordinary WWM masks the state-dependent answer much more often and cannot win by answer priors.

Any repair must keep the static-indirect control and frozen MLM-head counterfactual readout as a first response test before official-scale compute.
