# endpoint consumption and u256 tail composition — scale1.75 100M HF load smoke

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_100M`

Loaded through `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` on CPU: **True**.

Total params: 35,463,008; adapter-named params: 995,584.

Tiny deterministic masked-LM probe loss: `4.396473407745361`; finite logits: `True`.

This is not a BabyLM score. It verifies official-compatible package loadability and a finite forward pass while the managed full evaluator computes the nine-column surface.

JSON: `experiments/archive/frontier_consolidation/data/scale1p75_hf_load_smoke/scale1p75_hf_load_smoke.json`
