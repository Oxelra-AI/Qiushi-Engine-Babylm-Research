# SuperGLUE loader identity audit

Does AutoModel-based SuperGLUE finetuning load the custom private-adapter checkpoints, or does it fall back to the frozen DeBERTa encoder and ignore private tensors?

## chck82
- path: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
- auto_map: `{'AutoModelForMaskedLM': 'adapter_scaled_modeling.AdapterDebertaV2ForMaskedLM'}`; architectures: `['AdapterDebertaV2ForMaskedLM']`; private state tensors/params: `0` / `0`
- AutoModel: ok `True`, class `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model`, private named params `0`, unexpected keys `53`, examples `['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'encoder.layer.0.adapter.down.bias', 'encoder.layer.0.adapter.down.weight', 'encoder.layer.0.adapter.layer_norm.bias', 'encoder.layer.0.adapter.layer_norm.weight', 'encoder.layer.0.adapter.up.bias', 'encoder.layer.0.adapter.up.weight', 'encoder.layer.1.adapter.down.bias']`
- AutoModelForMaskedLM: ok `False`, class `None.None`, private named params `None`, private on/off probe maxdiff `None`

## coherent86_train_scale1p0
- path: `experiments/archive/frontier_consolidation/training/runs/frozen82_fastpath4M_coherent_seed43022/hf_model/final`
- auto_map: `{'AutoModelForMaskedLM': 'frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM'}`; architectures: `['FrozenSlowPrivateDebertaV2ForMaskedLM']`; private state tensors/params: `48` / `995584`
- AutoModel: ok `True`, class `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model`, private named params `0`, unexpected keys `101`, examples `['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'encoder.layer.0.adapter.down.bias', 'encoder.layer.0.adapter.down.weight', 'encoder.layer.0.adapter.layer_norm.bias', 'encoder.layer.0.adapter.layer_norm.weight', 'encoder.layer.0.adapter.up.bias', 'encoder.layer.0.adapter.up.weight', 'encoder.layer.0.private_adapter.down.bias']`
- AutoModelForMaskedLM: ok `False`, class `None.None`, private named params `None`, private on/off probe maxdiff `None`

## dense62064_u0080
- path: `experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080`
- auto_map: `{'AutoModelForMaskedLM': 'frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM'}`; architectures: `['FrozenSlowPrivateDebertaV2ForMaskedLM']`; private state tensors/params: `48` / `995584`
- AutoModel: ok `True`, class `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model`, private named params `0`, unexpected keys `101`, examples `['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'encoder.layer.0.adapter.down.bias', 'encoder.layer.0.adapter.down.weight', 'encoder.layer.0.adapter.layer_norm.bias', 'encoder.layer.0.adapter.layer_norm.weight', 'encoder.layer.0.adapter.up.bias', 'encoder.layer.0.adapter.up.weight', 'encoder.layer.0.private_adapter.down.bias']`
- AutoModelForMaskedLM: ok `False`, class `None.None`, private named params `None`, private on/off probe maxdiff `None`

## dense62065_u0080
- path: `experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080`
- auto_map: `{'AutoModelForMaskedLM': 'frozen82_private_modeling.FrozenSlowPrivateDebertaV2ForMaskedLM'}`; architectures: `['FrozenSlowPrivateDebertaV2ForMaskedLM']`; private state tensors/params: `48` / `995584`
- AutoModel: ok `True`, class `transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Model`, private named params `0`, unexpected keys `101`, examples `['cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.predictions.transform.dense.weight', 'encoder.layer.0.adapter.down.bias', 'encoder.layer.0.adapter.down.weight', 'encoder.layer.0.adapter.layer_norm.bias', 'encoder.layer.0.adapter.layer_norm.weight', 'encoder.layer.0.adapter.up.bias', 'encoder.layer.0.adapter.up.weight', 'encoder.layer.0.private_adapter.down.bias']`
- AutoModelForMaskedLM: ok `False`, class `None.None`, private named params `None`, private on/off probe maxdiff `None`

## Pairwise probe differences
- AutoModel_hidden_coherent86_train_scale1p0_minus_chck82: `{'max_abs': 0.0, 'mean_abs': 0.0}`
- AutoModel_hidden_dense62064_u0080_minus_chck82: `{'max_abs': 0.0, 'mean_abs': 0.0}`
- AutoModel_hidden_dense62065_u0080_minus_chck82: `{'max_abs': 0.0, 'mean_abs': 0.0}`
- AutoModel_hidden_dense62065_minus_dense62064: `{'max_abs': 0.0, 'mean_abs': 0.0}`
- MLM_diff_error: `OSError(30, 'Read-only file system')`

Interpretation: if AutoModel has zero private named parameters and its hidden states are identical to chck82 while AutoModelForMaskedLM sees nonzero private-on/off logits, then SuperGLUE finetuning based on AutoModel cannot be used as evidence about dense/private continuation. Zero-shot and Reading remain separate because they call AutoModelForMaskedLM with trust_remote_code.

JSON: `experiments/archive/relation_learning/data/superglue_loader_identity_audit/loader_identity_audit.json`
