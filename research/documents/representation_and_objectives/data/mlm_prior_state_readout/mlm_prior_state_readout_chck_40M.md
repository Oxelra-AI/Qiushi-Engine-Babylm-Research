# corrected orientation route and next experiments MLM prior state readout (chck_40M)

Inference-only cloze probe: premise + `After the event, [MASK] had the object.` Candidate scores are MLM logits for the two names. Positive margin means the MLM prefers the row label's owner.

- scored candidate groups: 896
- checkpoint: `experiments/archive/representation_and_objectives/training/runs/qwen_8x480_16k_wwm_to_token_100M_seed43022/hf_model/chck_40M`

## By source
| source | n | accuracy | margin mean | margin std |
|---|---:|---:|---:|---:|
| eval/cross_template_state_readout | 128 | 0.484 | -0.240 | 2.829 |
| eval/name_permutation_counterfactual | 128 | 0.469 | -0.023 | 2.491 |
| eval/paired_state_conservation | 256 | 0.469 | -0.215 | 2.790 |
| train/aligned_state_bridge | 64 | 0.438 | 0.159 | 2.952 |
| train/inverted_state_bridge | 64 | 0.484 | -0.267 | 3.260 |
| train/neutral_decoupled | 256 | 0.441 | -0.806 | 4.180 |

## By relation
| relation | n | accuracy | margin mean |
|---|---:|---:|---:|
| h0_dax | 192 | 0.453 | -0.180 |
| h1_mep | 128 | 0.438 | -0.218 |
| h2_norp | 192 | 0.500 | -0.075 |
| h3_ziv | 128 | 0.484 | -0.146 |
| s_give | 128 | 0.500 | -0.798 |
| s_receive | 128 | 0.383 | -0.814 |

## By inverted_bridge_label
| inverted_bridge_label | n | accuracy | margin mean |
|---|---:|---:|---:|
| False | 832 | 0.460 | -0.342 |
| True | 64 | 0.484 | -0.267 |

## Reading
Chance-level held-eval accuracy would mean the complete orientation probe results state gain is not directly visible in the MLM before supervised state-format fine-tuning. Above-chance true-direction held-eval accuracy would support a directly accessible pretrained/event-surface prior. Poor accuracy on inverted training labels would be expected if the base model favors the original true direction rather than the intentionally inverted labels.

- full JSON: `experiments/archive/representation_and_objectives/data/mlm_prior_state_readout/mlm_prior_state_readout_chck_40M.json`
