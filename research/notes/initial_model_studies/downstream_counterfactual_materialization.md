# downstream counterfactual materialization — downstream counterfactual state-propagation materialization

JSONL: `experiments/archive/initial_model_studies/data/counterfactual_revision_108/downstream_counterfactual_seed108_target200000_actual.jsonl`
Summary: `experiments/archive/initial_model_studies/data/counterfactual_revision_108/downstream_counterfactual_seed108_target200000_actual.summary.json`

Rows: **1855**
Counted aux words if original+perturbed+random streams are processed: **199994**
Sources: ['gutenberg.train.txt', 'simple_wiki.train.txt']
By source: {'gutenberg.train.txt': 1006, 'simple_wiki.train.txt': 849}

This materializes true within-line adjacent sentence pairs from official raw data. Each row contains original `[s1;s2]`, perturbed `[s1';s2]`, and a random-nonadjacent downstream control `[s1';s2_random]` with source/line IDs and matched rough replacement metadata. It is a data object for the next custom trainer, not a final result.
