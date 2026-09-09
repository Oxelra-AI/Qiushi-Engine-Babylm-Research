# earlier analysis causal-GPT transfer contrast redesign
This is a CPU-only redesign of the companion analysis earlier analysis transfer contrast after the a02 causal gpt transfer audit audit and earlier analysis strategist note. It preserves the clean aligned pair/filler/tokenizer assets but does not launch training.
## Arms
- `own_compact`: source paired with its own compact rewrite; semantic compression + correspondence, low literal copy.
- `adjbreak_compact`: source paired with a deranged compact rewrite; same rewrite multiset/token mass/boundary novelty, broken correspondence.
- `semantic_extractive`: source paired with source-token extract selected using the compact rewrite; copyable + semantically selected.
- `random_extractive`: source paired with a same-source random extract matched in word count and near-matched in BPE tokens; copyable + weak semantic selection.
## Matched-factor checks
- own vs adjbreak side-token delta: `{'a': 'own_compact', 'b': 'adjbreak_compact', 'a_side_tokens': 254671, 'b_side_tokens': 254671, 'diff_a_minus_b': 0, 'pct_of_b': 0.0}`
- semantic vs random extractive side-token delta: `{'a': 'semantic_extractive', 'b': 'random_extractive', 'a_side_tokens': 251564, 'b_side_tokens': 250957, 'diff_a_minus_b': 607, 'pct_of_b': 0.24187410592252856}`
- extractive per-pair abs token diff: `{'n': 12155, 'mean': 0.052570958453311396, 'median': 0.0, 'p10': 0.0, 'p90': 0.0, 'min': 0.0, 'max': 5.0}`; <=1 token pairs `12052/12155`.
- adjbreak local same rewrite-word length: `12151/12155`; local mismatches are rare long-tail cases while global rewrite multiset is exact.
## Reading the result if trained later
Only train after the causal evaluator is repaired to the official `evaluation_data` roots and fail-closed all-column scoring. A small pilot should precede 100M. The two contrasts should be read separately: own-vs-adjbreak for local correspondence under matched compact token mass and boundary novelty; semantic-vs-random extractive for information-dense selection under matched copy opportunity.
