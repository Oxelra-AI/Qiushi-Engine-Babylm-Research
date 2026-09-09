# compliant tokenizer retrain status — tokenizer shift analysis

Compares old 100M-trained baseline16k tokenizer to new compliant16k trained only on the 10M compact_view_reinvest pool. This is a CPU segmentation diagnostic, not model evidence.

## Vocabulary overlap

- old vocab size: 16384
- new vocab size: 16384
- shared token strings: 13809 (0.843 of new)
- special IDs old: `{'<pad>': 3, '<s>': 1, '</s>': 2, '<unk>': 0, '<mask>': 4}`
- special IDs new: `{'<pad>': 3, '<s>': 1, '</s>': 2, '<unk>': 0, '<mask>': 4}`

## Pool-level segmentation

| pool | rows | words | old tok/word | new tok/word | new/old token ratio | old trunc@256 | new trunc@256 | new-only trunc | visible group Δ mean | total token Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reinvest_10M | 64740 | 10000000 | 1.4756 | 1.4627 | 0.9915 | 15883 | 15117 | 232 | 0.2306 | -124979 |
| clean_qwen_10M | 64381 | 10000000 | 1.4771 | 1.4652 | 0.9922 | 16630 | 15872 | 251 | 0.2354 | -116161 |

## Relational marker segmentation changes

Changed relational forms: 12 / 204.

| form | old tokens | new tokens | Δlen |
|---|---|---|---:|
| `Inside` | `['ĠInside']` | `['ĠIn', 'side']` | 1 |
| `␠Inside` | `['Ġ', 'ĠInside']` | `['Ġ', 'ĠIn', 'side']` | 1 |
| `Behind` | `['ĠBehind']` | `['ĠBe', 'hind']` | 1 |
| `␠Behind` | `['Ġ', 'ĠBehind']` | `['Ġ', 'ĠBe', 'hind']` | 1 |
| `Into` | `['ĠInto']` | `['ĠInt', 'o']` | 1 |
| `␠Into` | `['Ġ', 'ĠInto']` | `['Ġ', 'ĠInt', 'o']` | 1 |
| `Leads` | `['ĠLe', 'ads']` | `['ĠLead', 's']` | 0 |
| `␠Leads` | `['Ġ', 'ĠLe', 'ads']` | `['Ġ', 'ĠLead', 's']` | 0 |
| `heavier` | `['Ġheavier']` | `['Ġheav', 'ier']` | 1 |
| `␠heavier` | `['Ġ', 'Ġheavier']` | `['Ġ', 'Ġheav', 'ier']` | 1 |
| `Different` | `['ĠDifferent']` | `['ĠD', 'ifferent']` | 1 |
| `␠Different` | `['Ġ', 'ĠDifferent']` | `['Ġ', 'ĠD', 'ifferent']` | 1 |

Machine-readable JSON contains top token-delta example rows for each pool.
