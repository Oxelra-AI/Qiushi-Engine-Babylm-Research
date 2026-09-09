# earlier analysis candidate sensitivity and method gap

Source comparison: `experiments/archive/functional_learning/data/same_coordinate_comparison_after_clean_superglue/same_coordinate_comparison.json`

## GlobalPIQA-zero sensitivity

- `dense_seed62064` complete `True` official delta `0.1251231280522731`; delta with GlobalPIQA contribution set to zero `-0.03987687194772723`; remaining-component mean `-0.04486148094119313`.
- `dense_seed62065` complete `True` official delta `0.14445471098841267`; delta with GlobalPIQA contribution set to zero `-0.020545289011588846`; remaining-component mean `-0.02311345013803745`.
- `clean_pres_lambda1_eval_seed62064` complete `True` official delta `0.2224443408943415`; delta with GlobalPIQA contribution set to zero `0.05744434089434039`; remaining-component mean `0.06462488350613294`.
- `clean_pres_lambda1_eval_seed62065` complete `False` official delta `None`; delta with GlobalPIQA contribution set to zero `None`; remaining-component mean `None`.

## Acquisition-to-preservation method gap

The fully evaluated dense controls are (M,M), while clean preservation is built on (M,S) acquisition plus deterministic parent anchoring. The exact acquisition-only (M,S) endpoint must receive compatible official zero/Reading, repaired SuperGLUE, and measured AoA before attributing the SuperGLUE/Overall increment specifically to preservation.
