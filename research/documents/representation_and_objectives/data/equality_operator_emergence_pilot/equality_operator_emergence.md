# posalign repair and equality emergence equality-operator emergence pilot

This CPU pilot tests the identity route upstream of relational gauge transport.

## Character coverage

Train-name letters: `aefhijklmnoprstvx`

Official held-name letters: `abceghijlmnorstuvyz`

Held letters absent from train names: `bcguyz`

## Assignment accuracy

Hard assignment is correct only when the candidate query selects the true candidate token among all raw tokens, not just over the other name.

| model | train events | official held | fresh known chars | fresh unseen chars | near known names | held rel-name | held true-maxnonname |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared_pos_initial | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.9107 |
| dual_pos_initial | 0.485 | 0.341 | 0.312 | 0.312 | 0.188 | 0.590 | -0.0171 |
| dual_pos_name_level_train | 1.000 | 0.818 | 1.000 | 0.562 | 1.000 | 0.872 | 0.4963 |
| dual_pos_train_alphabet_char_pairs | 1.000 | 0.903 | 1.000 | 0.625 | 1.000 | 1.000 | 0.5656 |
| dual_pos_full_alphabet_char_pairs | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.7101 |

## Interpretation hooks

- `shared_pos_initial` measures the built-in same-symbol prior supplied by shared character embeddings plus position-wise product.
- `dual_pos_initial` removes that prior while preserving position-sensitive structure.
- `dual_pos_name_level_train` asks whether finite name-level equality evidence from train events can align token/query character coordinates.
- `dual_pos_train_alphabet_char_pairs` asks whether lower-level equality anchors for only observed train letters are enough.
- `dual_pos_full_alphabet_char_pairs` asks whether complete alphabet-level finite equality evidence restores broad held-name assignment.

