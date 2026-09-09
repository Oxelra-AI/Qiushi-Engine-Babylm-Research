# superglue loader repair SuperGLUE loader diagnostic

## dense_seed62064_u0080
- Checkpoint: `experiments/archive/functional_learning/data/unchanged_dense_focus_train/correspondence_focus_weighted/checkpoints/update_0080`
- AutoModel → `DebertaV2Model`, 34219200 params, 0 adapter, 0 private_adapter

## dense_seed62065_u0080
- Checkpoint: `experiments/archive/functional_learning/data/dense_focus_rep_seed62065_train/correspondence_focus_weighted/checkpoints/update_0080`
- AutoModel → `DebertaV2Model`, 34219200 params, 0 adapter, 0 private_adapter

## coherent86_alpha075
- Checkpoint: `models/frontier`
- AutoModel → `DebertaV2Model`, 34219200 params, 0 adapter, 0 private_adapter

## Verdict
- dense_seed62064_u0080: AutoModel drops ALL adapters (? adapter + ? private)
- dense_seed62065_u0080: AutoModel drops ALL adapters (? adapter + ? private)
- coherent86_alpha075: AutoModel drops ALL adapters (? adapter + ? private)

**Repair needed:** `True`
