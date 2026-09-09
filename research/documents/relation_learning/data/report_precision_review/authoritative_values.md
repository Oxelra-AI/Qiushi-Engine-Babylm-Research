# earlier analysis authoritative values for report precision

## (M,S) construction counts
- ms64_train: total_focus_targets=28590, total_ordinary_targets=592858, focus_candidate_groups_total=132283, focus_selected_groups_total=21479; final update focus=298, ordinary=6290, candidate_groups=1416, selected_groups=235, qwen_rows=41, ordinary_wwm_rows=176.
- ms65_train: total_focus_targets=28476, total_ordinary_targets=592836, focus_candidate_groups_total=None, focus_selected_groups_total=None; final update focus=289, ordinary=6271, candidate_groups=1416, selected_groups=224, qwen_rows=41, ordinary_wwm_rows=176.

## Official exported-directory entry scores
- v4: all_valid=True, primary_score=61.53846153846154, metrics={'accuracy': 61.53846153846154, 'f1': 0.0, 'mcc': 0.0}, AutoModel_params=36210368, private_params=995584, slow_params_probe=0, adapter_scale_values_probe=[], private_scale_values_probe=[].
- v5: all_valid=True, primary_score=63.46153846153846, metrics={'accuracy': 63.46153846153846, 'f1': 38.70967741935484, 'mcc': 17.124171528991518}, AutoModel_params=36210368, private_params=995584, slow_params_probe=0, adapter_scale_values_probe=[], private_scale_values_probe=[].

## Seed64 ladder

## Split admission
- o62065: complete=False, Overall=None, scores={'BLiMP': None, 'Supplement': 63.62, 'EWoK': 49.81, 'Entity': 28.09, 'COMPS': None, 'SuperGLUE': None, 'GlobalPIQA': 39.535, 'Reading': 8.195, 'AoA': 0.0}, SuperGLUE tasks={'boolq': 67.4006116207951, 'multirc': None, 'rte': 64.02877697841727, 'wsc': 65.38461538461539, 'mrpc': 87.58620689655172, 'qqp': 71.6198210773428, 'mnli': 60.45232273838631}, missing=['BLiMP', 'COMPS', 'SuperGLUE'].
- ms62065: complete=True, Overall=42.17887384024569, scores={'BLiMP': 68.09, 'Supplement': 63.09, 'EWoK': 49.82, 'Entity': 29.29, 'COMPS': 52.14, 'SuperGLUE': 68.93486456221126, 'GlobalPIQA': 40.05, 'Reading': 8.195, 'AoA': 0.0}, SuperGLUE tasks={'boolq': 67.95107033639144, 'multirc': 68.23432343234323, 'rte': 63.30935251798561, 'wsc': 63.46153846153846, 'mrpc': 87.88927335640139, 'qqp': 71.40917027743646, 'mnli': 60.28932355338223}, missing=[].

Detailed JSON: `experiments/archive/relation_learning/data/report_precision_review/authoritative_values.json`
