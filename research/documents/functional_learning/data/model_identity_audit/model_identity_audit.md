# repaired relation first acquisition synthesis coherent86 model identity audit

Model endpoint: `models/frontier`

Config architectures: `['FrozenSlowPrivateDebertaV2ForMaskedLM']`; auto_map: `{'AutoModelForMaskedLM': 'frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM'}`

State dict private-adapter tensors: 48

## generic_auto_no_trust

- class: `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2ForMaskedLM`
- total params: 34467424; private params: 0 in 0 tensors
- default trainable params: 34467424 in 170 tensors
- executed private scales: `[]`
- loading missing/unexpected/private-unexpected: 0/96/48

## generic_auto_trust_remote_code

ERROR: `OSError(30, 'Read-only file system')`

## trusted_loader

- class: `frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM`
- total params: 36458592; private params: 995584 in 48 tensors
- default trainable params: 36458592 in 266 tensors
- executed private scales: `[0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]`
- after private freeze trainable params: 995584 ; non-private trainable tensors: 0

## Consequence for multiseed and relation first

Step039b/Step039d used generic AutoModelForMaskedLM and Step039d optimized model.parameters(). Any completed or partial Step039d training would not be interpretable as private-adapter coherent86 acquisition unless the loaded class and trainable set are repaired.

Script scan:

- `experiments/archive/functional_learning/scripts/revision_039b_robust_scorer.py`: {'uses_generic_auto_model': True, 'uses_trust_remote_code': False, 'uses_model_parameters_optimizer': False, 'has_token_search_fallback_drops_first_token': True, 'uses_individual_position_scoring_loop': True, 'constructs_distinct_new_values': False}
- `experiments/archive/functional_learning/scripts/revision_039d_answer_only_training.py`: {'uses_generic_auto_model': True, 'uses_trust_remote_code': False, 'uses_model_parameters_optimizer': True, 'has_token_search_fallback_drops_first_token': True, 'uses_individual_position_scoring_loop': True, 'constructs_distinct_new_values': False}
- `experiments/archive/functional_learning/scripts/relation_first_constructor.py`: {'uses_generic_auto_model': True, 'uses_trust_remote_code': False, 'uses_model_parameters_optimizer': False, 'has_token_search_fallback_drops_first_token': False, 'uses_individual_position_scoring_loop': False, 'constructs_distinct_new_values': True}
