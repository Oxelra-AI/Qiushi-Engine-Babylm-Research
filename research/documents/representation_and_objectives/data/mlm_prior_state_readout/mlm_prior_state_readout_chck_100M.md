# corrected orientation route and next experiments MLM prior state readout (chck_100M)

Inference-only cloze probe: premise + `After the event, [MASK] had the object.` Candidate scores are MLM logits for the two names. Positive margin means the MLM prefers the row label's owner.

- scored candidate groups: 896
- checkpoint: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_100M`

## By source
| source | n | accuracy | margin mean | margin std |
|---|---:|---:|---:|---:|
| eval/cross_template_state_readout | 128 | 0.477 | 0.006 | 2.657 |
| eval/name_permutation_counterfactual | 128 | 0.555 | 0.033 | 2.166 |
| eval/paired_state_conservation | 256 | 0.492 | 0.007 | 2.538 |
| train/aligned_state_bridge | 64 | 0.422 | 0.018 | 1.729 |
| train/inverted_state_bridge | 64 | 0.625 | 0.060 | 1.668 |
| train/neutral_decoupled | 256 | 0.508 | -0.307 | 2.821 |

## By relation
| relation | n | accuracy | margin mean |
|---|---:|---:|---:|
| h0_dax | 192 | 0.521 | 0.004 |
| h1_mep | 128 | 0.477 | -0.063 |
| h2_norp | 192 | 0.526 | 0.112 |
| h3_ziv | 128 | 0.492 | -0.020 |
| s_give | 128 | 0.516 | -0.312 |
| s_receive | 128 | 0.500 | -0.302 |

## By inverted_bridge_label
| inverted_bridge_label | n | accuracy | margin mean |
|---|---:|---:|---:|
| False | 832 | 0.499 | -0.085 |
| True | 64 | 0.625 | 0.060 |

## Reading
Chance-level held-eval accuracy would mean the complete orientation probe results state gain is not directly visible in the MLM before supervised state-format fine-tuning. Above-chance true-direction held-eval accuracy would support a directly accessible pretrained/event-surface prior. Poor accuracy on inverted training labels would be expected if the base model favors the original true direction rather than the intentionally inverted labels.

- full JSON: `experiments/archive/representation_and_objectives/data/mlm_prior_state_readout/mlm_prior_state_readout_chck_100M.json`
