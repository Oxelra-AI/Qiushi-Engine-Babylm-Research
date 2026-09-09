# bridge atlas geometry result bridge geometry comparison

Total bridge candidates: 103

## Atlas benchmarks

- **compact**: gap1=0.7775, skip=0.2225, span=0.8072, absent=0.1729
- **extractive_balanced**: gap1=0.5278, skip=0.4722, span=0.9715, absent=0.0000
- **extractive_wide**: gap1=0.6465, skip=0.3535, span=0.9281, absent=0.0000

## Bridge group geometry

| group | n | gap1 | skip | source_span | absent_frac | content_density | func_frac | compression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_bridge | 103 | 0.4040 | 0.5960 | 0.8963 | 0.0215 | 0.6441 | 0.3559 | 0.7482 |
| extraction_like | 48 | 0.4411 | 0.5589 | 0.8948 | 0.0028 | 0.6237 | 0.3763 | 0.8270 |
| transformation_like | 55 | 0.3715 | 0.6285 | 0.8976 | 0.0378 | 0.6619 | 0.3381 | 0.6794 |
| substantive_only | 8 | 0.3432 | 0.6568 | 0.8502 | 0.1075 | 0.6875 | 0.3125 | 0.6063 |

## Matched natural compact geometry (same pairs)

| group | n | gap1 | skip | source_span | absent_frac | content_density | func_frac | compression |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all_bridge | 103 | 0.3498 | 0.6502 | 0.8728 | 0.1517 | 0.6817 | 0.3183 | 0.6567 |
| extraction_like | 48 | 0.3889 | 0.6111 | 0.8568 | 0.1302 | 0.6723 | 0.3277 | 0.7060 |
| transformation_like | 55 | 0.3156 | 0.6844 | 0.8867 | 0.1704 | 0.6900 | 0.3100 | 0.6136 |
| substantive_only | 8 | 0.3688 | 0.6312 | 0.8644 | 0.1088 | 0.6869 | 0.3131 | 0.6077 |

## Proximity diagnostic (transformation-like)

- **gap1**: bridge_transform=0.3715, compact=0.7775, ext_bal=0.5278, ext_wide=0.6465 → closest to **ext_balanced**
- **skip**: bridge_transform=0.6285, compact=0.2225, ext_bal=0.4722, ext_wide=0.3535 → closest to **ext_balanced**
- **source_span**: bridge_transform=0.8976, compact=0.8072, ext_bal=0.9715, ext_wide=0.9281 → closest to **ext_wide**

## Interpretation

If transformation-like bridge is closest to extractive on gap1/skip/span, then structural operations under the source-attested constraint produce extraction-like geometry despite surface restructuring — the route adds no geometric dimension beyond extractive, which already failed.

If transformation-like bridge has gap1/skip closer to compact than to extractive, fluent restructuring genuinely changes the data geometry and a matched training comparison is justified.

