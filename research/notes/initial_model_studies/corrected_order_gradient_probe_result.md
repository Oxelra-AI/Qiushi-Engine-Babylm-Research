# Corrected order-gradient probe

## Why intermediate layer order probe result needed repair

The intermediate layer order probe result probe established a useful representation fact: the masked-query A/B hidden difference decayed strongly across layers in the R1 chck_5M checkpoint. But its route conclusion was too strong because the gradient part was incomplete:

- it backpropagated only through context A;
- its score computation used cross-mask-position matrix indexing in the diagnostic code;
- it did not compute a joint A/B paired objective;
- it did not measure answer-direction alignment between hidden A/B differences and paired gradients.

Therefore intermediate layer order probe result should not be used to claim that WWM cannot train order information, and it should not by itself justify a RecGPT mainline.

## Corrected probe

Script: `scripts/corrected_order_gradient_probe.py`

Result: `data/corrected_order_gradient_probe.json`

The corrected probe uses:

- position-aligned span scoring: `sum_i log p(answer_token_i at mask_position_i)`;
- joint paired objective: `I = (s_A(a)-s_A(b)) + (s_B(b)-s_B(a))`;
- gradients through both context A and context B;
- hidden A/B representation differences at every layer;
- direct intermediate-head scores as a diagnostic, not a trained readout;
- alignment between `delta = h_A - h_B` and `sep_grad = grad_A - grad_B`.

## Main numbers

For 60 equal-token R1 counterfactual pairs, final exact paired scores remain mirrored:

- `m_A_mean = -0.00345`
- `m_B_mean = +0.00345`
- `I_pair_sum_mean = -4.77e-08`
- `both_correct_mean = 0.0`

Layer-wise representation and corrected gradient summaries:

| layer | rep L2 | rep cosine | direct-head I | both-correct | sep-grad norm | delta/sep-grad cosine |
|---|---:|---:|---:|---:|---:|---:|
| embedding | 0.000000 | 0.999999994 | 0.000000 | 0.000 | 6.21e-10 | 0.000 |
| layer_0 | 0.189796 | 0.999850815 | -0.000621 | 0.000 | 1.29e-09 | -0.003 |
| layer_1 | 0.061883 | 0.999985732 | -0.000349 | 0.000 | 6.27e-09 | -0.040 |
| layer_2 | 0.025139 | 0.999996853 | -0.000165 | 0.000 | 2.82e-08 | -0.036 |
| layer_3 | 0.005184 | 0.999999862 | -0.000018 | 0.000 | 1.99e-07 | -0.029 |
| layer_4 | 0.000840 | 0.999999994 | +0.000001 | 0.000 | 1.69e-06 | -0.018 |
| layer_5 | 0.000098 | 1.000000017 | ~0 | 0.000 | 1.66e-05 | -0.018 |
| layer_6 | 0.000009 | 0.999999999 | ~0 | 0.000 | 2.36e-04 | +0.001 |
| layer_7 | 0.000001 | 0.999999993 | ~0 | 0.000 | 3.97e-02 | -0.014 |

## Correct interpretation

The corrected evidence supports three narrower statements.

### 1. The masked-query A/B hidden difference decays strongly with depth

Layer 0 contains a small contextual difference between operation orders (`L2=0.1898`), but deeper layers compress it to numerical zero (`L2=1.4e-6` at layer 7). This is a real representation fact for this checkpoint and task family.

### 2. The existing final-answer paired objective has weak local leverage on the early order difference

The joint paired objective's separation gradient at early layers is extremely small (`~1e-9` to `~1e-7` through layer 3). The layers with any nontrivial A/B representation difference are not the layers receiving usable gradient from the final paired score. The layers receiving large gradient have essentially no A/B hidden difference.

The delta/sep-gradient cosine is near zero or slightly negative at every layer, so the local gradient does not align with amplifying the existing A/B representation difference.

### 3. Directly applying the existing MLM head at intermediate layers does not reveal a usable hidden answer-direction signal

The direct intermediate-head paired score `I` is near zero at every layer, and both-contexts-correct remains 0. This does not prove a trained intermediate readout could not work, but it means there is no obvious unexploited order-sensitive answer direction in the current hidden states.

## What the corrected evidence does NOT prove

- It does not prove that DeBERTa or all bidirectional encoders structurally ignore order.
- It does not prove that WWM can never learn order under other data, checkpoint states, or objectives.
- It does not close all bidirectional repairs: a deliberately trained intermediate auxiliary, layer mixing, or explicit trajectory supervision could still be tested.
- It does not by itself justify full RecGPT reproduction.

## Route implication

The corrected evidence rules out a weak route: simply reshaping final-answer MLM ranking losses is unlikely to solve the state/order problem in this checkpoint, because the local gradient does not reach or align with the early order difference.

It leaves two serious mechanism routes:

1. **Causal/recursive interface screen.** A causal scorer makes order part of the prefix-conditioned state and aligns with the RecGPT phenotype. But RecGPT itself is weak on Entity, so this should begin as a controlled 10M mechanism screen, not full SOTA training.

2. **Bidirectional intermediate-supervision repair.** If we still want a bidirectional route, it must explicitly train an intermediate representation or layer-mixed readout to preserve order. It cannot rely on the existing final MLM head. This should be a small R1 learning-response test before any official-scale work.

Given the leaderboard evidence that RecGPT reaches near-leader Overall through BLiMP/COMPS/GlobalPIQA/Reading, and given repeated failures of bidirectional final-answer routes, the next high-value work is a **small controlled causal/recursive mechanism screen** while preserving the possibility of a short intermediate-supervision bidirectional test if its information value justifies the comparison.

## Supersession note

This note supersedes the over-strong conclusions in `notes/intermediate_layer_order_probe_result.md`. The representation-decay numbers from intermediate layer order probe result remain useful; the route conclusion from intermediate layer order probe result should be replaced by the corrected, narrower conclusion here.
