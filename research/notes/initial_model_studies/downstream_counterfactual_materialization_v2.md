# downstream counterfactual materialization v2 — downstream counterfactual state-propagation v2 materialization

JSONL: `experiments/archive/initial_model_studies/data/counterfactual_revision_109/downstream_counterfactual_v2_seed109_target200000_actual.jsonl`
Summary: `experiments/archive/initial_model_studies/data/counterfactual_revision_109/downstream_counterfactual_v2_seed109_target200000_actual.summary.json`

Rows: **1679**
Counted aux words (original+perturbed+random): **199970**
By source: {'gutenberg.train.txt': 987, 'simple_wiki.train.txt': 692}
By edit kind: {'common_after_determiner': 621, 'common_after_prep': 275, 'proper_mid': 584, 'common_after_article': 199}

v2 restricts edits to proper-noun or noun-like determiner/article/preposition contexts, preserves article-vowel/plural/capitalization coarse buckets, requires downstream pronoun/deictic marker, and source+length-matches random downstream controls.

This replaces the downstream counterfactual materialization v1 materialization for training, because v1 had obvious syntactic-artifact edits.
