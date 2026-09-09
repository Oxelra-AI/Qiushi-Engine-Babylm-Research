# eval path hardening and health — scale1.75 full-eval path hardening and health audit

This is a file-only audit of the active scale1.75 full-evaluation output tree. It did not run evaluation, or invoke CUDA.

Base output: `experiments/archive/frontier_consolidation/data/scale1p75_100M_full_eval_hardened`
Endpoint-ready artifact exists: **True**
Final full-eval summary exists: **False**
Completed split parts: `['BLiMP', 'Supplement', 'EWoK', 'Entity', 'COMPS', 'GP_parallel', 'GP_nonparallel', 'Reading']`
Missing/incomplete split parts: `['SuperGLUE', 'AoA']`
Read-only/HF-cache issue detected in visible logs: **True**

Interpretation: Some split parts complete and can be reused by the hardened evaluator when force is not set; only incomplete parts should run in a later resume. A cache/path issue appears in logs; patched eval path hardening and health evaluators now attach writable HF_HOME/HF_MODULES_CACHE at the outer subprocess boundary for future reruns.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_eval_health_audit/scale1p75_eval_health_audit.json`
