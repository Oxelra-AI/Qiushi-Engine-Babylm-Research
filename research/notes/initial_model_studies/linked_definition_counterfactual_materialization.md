# downstream counterfactual materialization v2 — linked definition counterfactual v3 materialization

JSONL: `experiments/archive/initial_model_studies/data/counterfactual_revision_109/linked_definition_counterfactual_v3_seed1093_target100000_actual.jsonl`
Summary: `experiments/archive/initial_model_studies/data/counterfactual_revision_109/linked_definition_counterfactual_v3_seed1093_target100000_actual.summary.json`

Rows: **1267**
Counted aux words: **99993**
Dependent starts: {'it': 971, 'they': 144, 'these': 27, 'this': 112, 'that': 11, 'those': 2}
Determiners: {'the': 1174, 'an': 11, 'a': 82}

High-precision subset: sentence 1 begins with determiner-headed subject before a short anchor verb; sentence 2 begins with pronoun/deictic dependent. Edit is the subject head, so sentence 2 explicitly uses the edited variable.

This v3 subset supersedes arbitrary v2 swaps for the intended state-propagation trainer; v2 should remain a matched leakage/unlinked control pool.
