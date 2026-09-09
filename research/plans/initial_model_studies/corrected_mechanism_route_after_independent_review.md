# Corrected mechanism route after the repaired A/B gradient probe

## Why the route changed again

The intermediate-layer order probe did not justify its route conclusion. It measured an important representation pattern, but its gradient part was incomplete: it used only context A, its span scoring was not position-aligned, and it did not measure A/B joint gradient alignment. Therefore intermediate layer order probe result does not justify opening a full RecGPT mainline.

corrected order gradient probe result repaired the probe:

- position-aligned span score: `sum_i log p(answer_i at mask_position_i)`;
- joint paired objective: `I=(s_A(a)-s_A(b))+(s_B(b)-s_B(a))`;
- gradients through both context A and context B;
- separation-gradient alignment with `delta=h_A-h_B`;
- direct intermediate-head scores as a readout test, not as a trained readout.

Evidence files:

- Corrected probe script: `scripts/corrected_order_gradient_probe.py`
- Corrected probe result: `data/corrected_order_gradient_probe.json`
- Corrected interpretation: `notes/corrected_order_gradient_probe_result.md`

## Corrected evidence

The repaired result supports only the following narrow facts for the R1 `chck_5M` checkpoint and this pair family.

1. The masked-query A/B hidden difference decays sharply with depth:
   `0.1898 -> 0.0619 -> 0.0251 -> 0.00518 -> 0.00084 -> 0.000098 -> 0.000009 -> 0.000001`.

2. Final exact paired scores remain mirrored:
   - `m_A_mean=-0.00345`
   - `m_B_mean=+0.00345`
   - `I_pair_sum_mean=-4.77e-08`
   - `both_correct=0.0`

3. The joint paired objective has little local hidden-state leverage at shallow layers where the small A/B difference exists:
   - layer 0 separation-gradient norm `1.29e-09`;
   - layer 1 `6.27e-09`;
   - layer 2 `2.82e-08`;
   - layer 3 `1.99e-07`.

4. The alignment between hidden A/B difference and paired separation gradient is near zero or slightly negative at all layers.

5. Applying the existing final MLM head directly to intermediate states does not reveal a ready answer-discriminating direction: direct-head paired `I` is near zero and `both_correct=0` at every layer.

## What this does and does not settle

This does settle:

- Final-answer MLM ranking through the current readout has weak local leverage in this checkpoint.
- Running more final-answer ranking variants is unlikely to be a high-value path.
- The earlier analysis data-ratio route under ordinary WWM should not continue.

This does **not** settle:

- whether bidirectional WWM can ever learn order under other supervision locations or checkpoints;
- whether a trained layer-mixed readout can use shallow/context token information;
- whether explicit intermediate state/order supervision can force the relevant information to remain available;
- whether causal scoring or recursion is the decisive part of RecGPT's phenotype.

The right next step is therefore not full RecGPT training. It is a small shared-task learning-response comparison that separates readout/supervision/interface effects.

## Next experiment: common R1 learning-response comparison

Use the same R1 ordered-dynamic generator and held-out separation discipline from related experiments. Keep train/dev/test separated by entity words, location words, templates, and operation combinations as much as current generator allows.

All arms should use the same generated train stream, same official-text interleaving if used, same exposure/update budget, and same evaluation pairs.

### Arm 1 — bidirectional final-readout control

Current DeBERTa-style MLM with the final paired objective or ordinary WWM control. Purpose: reference failure mode.

### Arm 2 — bidirectional trained layer-mixing readout

Add a small trainable scalar layer mixer over query hidden states, e.g.

`z = sum_l softmax(alpha)_l * LN_l(h_query^l)`

then use a lightweight answer scorer or the existing MLM transform on `z` for the paired answer task. Train only the mixer/readout first, then optionally allow the encoder to update.

Purpose: test whether shallow/mid representations contain enough usable order information if the readout is trained to use them.

### Arm 3 — bidirectional layer-mixing + intermediate order/state supervision

Same as Arm 2, but add explicit supervision before the final query:

- prefix/state query after each operation;
- operation-local entity/location update target;
- noncausal swap invariance where possible;
- held-out entity/location/template combinations.

Purpose: test whether the bottleneck is not readout but the need to actively train the representation to preserve order/state.

### Arm 4 — dense causal control

A small causal model with similar parameter scale or a carefully chosen smaller pilot. It sees only prefixes and predicts next tokens or masked-next targets using causal attention.

Purpose: test whether prefix-conditioned scoring makes the R1 order task easier without recursion.

### Arm 5 — recursive causal screen

A lightweight RecGPT-style shared-block causal model, not the full package initially. Compare against Arm 4 to isolate recursion/weight-sharing.

Purpose: test whether recursion adds compositional/sample-efficiency benefit beyond causality.

## Evidence to collect

For every arm:

- R1 held-out `both_correct`, `m_A`, `m_B`, paired `I`, and same-candidate A/B score difference;
- train vs held-out gap, so template or answer memorization is visible;
- per-layer or per-depth A/B hidden differences before and after training;
- gradient/alignment summaries if feasible;
- small official-compatible column probes: BLiMP, Supplement, Entity, COMPS, GlobalPIQA, Reading, and EWoK if runtime permits.

## Route decision after the comparison

- If Arm 2 succeeds on held-out R1 with minimal encoder updating, the problem is mainly readout/layer mixing; bidirectional repair becomes live.
- If Arm 3 succeeds but Arm 2 fails, the problem is representation shaping; bidirectional intermediate supervision remains live but must prove official transfer before scale-up.
- If Arms 4/5 succeed where Arms 2/3 fail, the causal interface becomes the strongest route.
- If Arm 5 improves over Arm 4, recursion/weight sharing is likely load-bearing and a RecGPT-style screen is justified.
- If R1 success does not move small official probes, R1 is not a good route proxy and the main SOTA path must be chosen from direct official-column evidence.

## Practical priority

The smallest high-value first construction is Arms 1–4 on a compact R1 dataset, because it directly resolves whether bidirectional layer-mixing/intermediate supervision is viable before implementing the full recursive stack. Arm 5 should follow once dense causal baseline behavior is known, unless the original RecGPT code supports a faithful parallel small screen.

Do not launch a full 100M run, and do not start another ordinary corpus-ratio variant, before this mechanism comparison changes the evidence.
