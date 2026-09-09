# corpus directional pair census — Corpus directional pair census

## Purpose
Before any mechanism training, quantify the density of usable paired directional
evidence in the 10M compact-view corpus. This determines whether a four-cell
interaction loss can find training signal, or whether we must rely on weaker mechanisms.

## Key results
- Total sentences (≥5 words): 690,663
- With any relational marker: 342,159 (49.5%)
- With directional pivot token: 437,599 (63.4%)
- Unique directional pairs found: 150
- Pair types: 36
- Top pair types: {"isn't/is": 22, "is/isn't": 14, 'up/down': 14, 'down/up': 11, 'open/close': 10, 'yes/no': 8, 'positive/negative': 6, 'no/yes': 6, 'out/in': 5, 'first/last': 5}

## Pivot-visible dependent masking
- Eligible sentences: 341,950 (49.5%)
- Maskable dependents: 2,587,733
- Mean per eligible sentence: 7.57

## Files
- Census JSON: `experiments/archive/representation_and_objectives/data/corpus_directional_census/corpus_directional_pair_census.json`
- Directional pairs JSONL: `experiments/archive/representation_and_objectives/data/corpus_directional_census/directional_pairs.jsonl`
