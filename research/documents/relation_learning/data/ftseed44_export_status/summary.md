# earlier analysis ftseed44 and export status

This note consolidates a completed downstream SuperGLUE repeat and the export-package location.

## Clean-preservation seed62064 downstream seed44

- SuperGLUE seed44: `69.62659512687073`; seed42 canonical: `69.047710989885`; delta `0.5788841369857209`.
- Overall with only SuperGLUE swapped to seed44 and all other canonical clean64 components fixed: `42.31073279187453`.
- Delta vs faithful coherent86 seed42 coordinate: `0.28676480055942477`; delta vs clean64 seed42 coordinate: `0.06432045966508326`.
- Per-task primary deltas seed44 minus clean64 seed42: `{"boolq": -0.24464831804282028, "mnli": 0.40749796251018466, "mrpc": -0.38658871256413363, "multirc": -0.2475247524752433, "qqp": 0.6911340190956423, "rte": 5.755395683453237, "wsc": -1.9230769230769198}`.

## Export package

- Export package: `experiments/archive/functional_learning/data/v5_candidate_export`; exists `True`; listed entries `17`.
- Source-level verifier: `experiments/archive/functional_learning/data/v5_export_evidence_verifier/v5_export_evidence_verifier.json`.

Full JSON: `experiments/archive/relation_learning/data/ftseed44_export_status/summary.json`
