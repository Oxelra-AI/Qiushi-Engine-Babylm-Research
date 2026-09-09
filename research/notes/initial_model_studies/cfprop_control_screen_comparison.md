# cfprop control screen comparison — counterfactual propagation control screen comparison

Evidence JSON: `experiments/archive/initial_model_studies/data/cfprop_control_screen_comparison.json`

| quantity | linked v3 | unlinked v2 | linked-unlinked |
|---|---:|---:|---:|
| WWM words | 200000 | 200000 | 0 |
| aux pair words | 131448 | 199960 | -68512 |
| counted words pair-rule | 331448 | 399960 | -68512 |
| final MLM loss | 7.194316864013672 | 7.189154148101807 | 0.005162715911865234 |
| final CF loss | 1.0593039989471436 | 1.0508233308792114 | 0.008480668067932129 |
| train pair acc | 0.4944044764903333 | 0.4760206102663404 | 0.018383866223992906 |
| tail mean pair acc | 0.5391447365283966 | 0.48125 | 0.05789473652839655 |
| tail mean random acc | 0.5004934221506119 | 0.46875 | 0.03174342215061188 |
| heldout pair acc | 0.578125 | 0.5357142885526022 | 0.04241071144739783 |
| heldout random acc | 0.625 | 0.5119047647430783 | 0.11309523525692167 |

## Interpretation
- Linked-v3 and unlinked-v2 heldout pair accuracies are similar; the screen does not separate true linked dependency from leakage/control artifacts.

HF cleanliness: linked hf_pollution=False, unlinked hf_pollution=False.
