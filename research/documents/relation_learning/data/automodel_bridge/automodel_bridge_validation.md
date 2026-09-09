# stage3 execution synthesis AutoModel bridge validation

## chck82_scale1p75
- repaired: `experiments/archive/relation_learning/data/automodel_bridge/repaired_chck82_scale1p75`
- trusted MLM: `transformers_modules.repaired_chck82_scale1p75.adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM` params {'total': 35463008, 'adapter': 995584, 'private_adapter': 0, 'trainable': 35463008}
- repaired AutoModel: `transformers_modules.repaired_chck82_scale1p75.adapter_scaled_modeling.AdapterDebertaV2Model` params {'total': 35214784, 'adapter': 995584, 'private_adapter': 0, 'trainable': 35214784}
- AutoModel vs MLM encoder max diff `0.000000e+00`, mean diff `0.000000e+00`, identical `True`
- source AutoModel without bridge: `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model` params {'total': 34219200, 'adapter': 0, 'private_adapter': 0, 'trainable': 34219200}, max diff vs trusted `1.512444e+00`
- executed scales: `{'adapter': [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75], 'private_adapter': []}`

## coherent86_alpha075
- repaired: `experiments/archive/relation_learning/data/automodel_bridge/repaired_coherent86_alpha075`
- trusted MLM: `transformers_modules.repaired_coherent86_alpha075.frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM` params {'total': 36458592, 'adapter': 995584, 'private_adapter': 995584, 'trainable': 36458592}
- repaired AutoModel: `transformers_modules.repaired_coherent86_alpha075.frozen82_private_modeling.FrozenSlowPrivateDebertaV2Model` params {'total': 36210368, 'adapter': 995584, 'private_adapter': 995584, 'trainable': 36210368}
- AutoModel vs MLM encoder max diff `0.000000e+00`, mean diff `0.000000e+00`, identical `True`
- source AutoModel without bridge: `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model` params {'total': 34219200, 'adapter': 0, 'private_adapter': 0, 'trainable': 34219200}, max diff vs trusted `1.484657e+00`
- executed scales: `{'adapter': [1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75], 'private_adapter': [0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]}`

All validated: `True`
