# span matcher review and next span discovery SpanMatcher initialization and matching audit

## Condition-specific initialization

- learned shared_trunk hash same when untied is also in the run: True
- oracle shared_trunk hash same when untied is also in the run: True
- aggregate hash differs when condition list differs: True

Interpretation: span matcher construction and design's displayed aggregate init hashes should not be used as a sign-pair equality check, but the reconstructed shared_trunk initial parameters are identical across the relevant condition-list variants.

## Untrained matcher identity behavior

| dataset | mean acc across seeds | min acc | min true prob over seeds |
|---|---:|---:|---:|
| train_all_pairs | 0.945 | 0.912 | 0.198 |
| eval_all_pairs | 0.950 | 0.896 | 0.159 |
| fresh_alpha_all_pairs | 0.908 | 0.825 | 0.102 |
| near_names_all_pairs | 0.828 | 0.729 | 0.155 |
| colliding_digit_adjacent | 0.500 | 0.500 | 0.500 |

Interpretation: if train/eval/fresh-alpha matching is already perfect before relational training, span matcher construction and design should be described as using a context-invariant equality-style matcher that supplies stable candidate coordinates from character strings. The relational experiment verifies that this interface feeds gauge transport, not that sparse state/comparison labels taught a broad name-matching algorithm.

## Character collision note

- train: 16/16 unique char sequences; collisions=[]
- eval: 16/16 unique char sequences; collisions=[]
- fresh_alpha: 16/16 unique char sequences; collisions=[]
- near_names: 16/16 unique char sequences; collisions=[]
- colliding_digit: 1/4 unique char sequences; collisions=[['Xname0', 'Xname1', 'Xname2', 'Xname3']]

fresh_rename_state_queries uses Xname0... but name_to_char_ids ignores digits, collapsing all to the same 'xname' character sequence if --fresh-rename is used.
