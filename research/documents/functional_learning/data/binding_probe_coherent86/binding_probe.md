# credit allocation binding pilot entity-binding cloze probe

Model: `models/frontier` (private_adapter, scale=0.75)

## Aggregate

- Pairs: 8
- UPDATE correct: 7/8 (88%)
- RETAIN correct: 8/8 (100%)
- Both correct: 7/8 (88%)
- Mean UPDATE margin: +11.826
- Mean RETAIN margin: +12.238

## Per-pair detail

| pair_id | UPDATE margin | RETAIN margin | Both correct |
|---|---:|---:|:---:|
| tp_001 | +2.793 | +4.391 | ✓ |
| tp_002 | +16.443 | +16.866 | ✓ |
| tp_003 | +16.620 | +3.392 | ✓ |
| tp_004 | -4.728 | +28.978 | ✗ |
| tp_005 | +17.266 | +13.045 | ✓ |
| tp_006 | +6.042 | +10.099 | ✓ |
| tp_007 | +18.958 | +6.472 | ✓ |
| tp_008 | +21.212 | +14.665 | ✓ |

## Per-packet detail

| pair_id | type | answer | foil | margin | top1 | correct |
|---|---|---|---|---:|---|:---:|
| tp_001 | UPDATE | green | red | +2.793 | Ġgreen | ✓ |
| tp_001 | RETAIN | red | green | +4.391 | Ġred | ✓ |
| tp_002 | UPDATE | digital tablets | leather-bound journals | +16.443 | Ġa | ✓ |
| tp_002 | RETAIN | leather-bound journals | digital tablets | +16.866 | Ġthe | ✓ |
| tp_003 | UPDATE | white truck | silver sedan | +16.620 | Ġcar | ✓ |
| tp_003 | RETAIN | silver sedan | white truck | +3.392 | Ġsilver | ✓ |
| tp_004 | UPDATE | converted warehouse | small apartment | -4.728 | Ġsmall | ✗ |
| tp_004 | RETAIN | small apartment | converted warehouse | +28.978 | Ġgarden | ✓ |
| tp_005 | UPDATE | spotted rabbit | tabby cat | +17.266 | Ġlittle | ✓ |
| tp_005 | RETAIN | tabby cat | spotted rabbit | +13.045 | Ġt | ✓ |
| tp_006 | UPDATE | mushroom risotto | spicy tofu | +6.042 | Ġday | ✓ |
| tp_006 | RETAIN | spicy tofu | mushroom risotto | +10.099 | Ġdish | ✓ |
| tp_007 | UPDATE | electric violin | acoustic guitar | +18.958 | Ġjazz | ✓ |
| tp_007 | RETAIN | acoustic guitar | electric violin | +6.472 | Ġac | ✓ |
| tp_008 | UPDATE | denim vest | wool coat | +21.212 | Ġden | ✓ |
| tp_008 | RETAIN | wool coat | denim vest | +14.665 | Ġway | ✓ |
