# dual compliant tokenizer endpoint policy byte-alphabet repair-token training frequency

This analysis asks whether byte-alphabet tokens that replace spatial repair route status `<unk>` spans in official inputs were actually seen in the legal 10M pretraining pool. It is interpretation-only and does not alter the frozen tokenizer design.

Allowed pool tokens under byte-alphabet tokenizer: 14798756 over 64740 rows; unique tokens seen 16268.

## zero_shot

Official repair-token occurrences: 1892; unique repair tokens: 37; occurrences with training frequency 0: 946; <10: 946; <100: 962.

Top repair tokens:

| token | official occurrences | train frequency |
|---|---:|---:|
| `Ċ` | 946 | 0 |
| `ĠSarah` | 327 | 2030 |
| `ĠB` | 290 | 8508 |
| `ĠNo` | 90 | 8981 |
| `ĠYes` | 86 | 4402 |
| `ĠA` | 36 | 12884 |
| `ĠI` | 23 | 185796 |
| `ĠScience` | 7 | 205 |
| `ĠIn` | 7 | 11213 |
| `ĠAn` | 7 | 1657 |
| `ĠDin` | 6 | 73 |
| `ĠHistory` | 6 | 487 |
| `ĠL` | 6 | 6184 |
| `ĠPast` | 5 | 72 |
| `ĠSeveral` | 5 | 116 |
| `ĠDavid` | 4 | 1321 |
| `ĠM` | 4 | 8008 |
| `ĠThe` | 3 | 36754 |
| `ĠThis` | 3 | 7936 |
| `ĠEvery` | 3 | 498 |

## superglue

Official repair-token occurrences: 1148; unique repair tokens: 81; occurrences with training frequency 0: 523; <10: 595; <100: 1069.

Top repair tokens:

| token | official occurrences | train frequency |
|---|---:|---:|
| `Ð` | 122 | 0 |
| `Ø` | 120 | 0 |
| `Ì` | 55 | 0 |
| `Ö` | 51 | 0 |
| `Ñ` | 44 | 0 |
| `ç` | 41 | 0 |
| `è` | 37 | 0 |
| `±` | 29 | 39 |
| `§` | 27 | 21 |
| `¨` | 26 | 9 |
| `ģ` | 26 | 84 |
| `µ` | 24 | 14 |
| `´` | 22 | 3 |
| `¾` | 21 | 81 |
| `ª` | 20 | 17 |
| `©` | 19 | 120 |
| `°` | 19 | 131 |
| `Ģ` | 19 | 68 |
| `Į` | 16 | 47 |
| `¯` | 16 | 6 |

## Zero-shot by family

| family | spatial repair route status unk texts | unk spans | newline spans | repair-token occ | freq0 frac | <10 frac | <100 frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| Supplement | 946 | 946 | 946 | 1892 | 0.500 | 0.500 | 0.508 |

## Interpretation

If the byte-alphabet endpoint differs from the spatial repair route status endpoint, the score movement should be read as a real interaction between legal tokenizer construction and learning. Removing `<unk>` does not guarantee improvement when the replacing byte tokens are rare or absent as positive MLM targets in the 10M pool; conversely, spatial repair route status's `<unk>` token is also unseen as a pretraining target. The official endpoint results remain decisive.

Full JSON: `experiments/archive/frontier_consolidation/data/bytealpha_repair_token_frequency/bytealpha_repair_token_frequency.json`
