# cross data binding comparison result: Cross-data entity-event-state binding comparison

## 80M exposure comparison

| Model | Treatment | Affected | Unaffected | EEBF | Multi-event | Aff margin |
|-------|-----------|----------|------------|------|-------------|------------|
| clean_qwen_stock_80M | clean_qwen | 0.2656 | 0.7500 | 0.5078 | 0.5 | -1.2657 |
| compact_reinvest_stock_80M | compact_reinvest | 0.3958 | 0.6250 | 0.5104 | 0.5 | -0.3459 |
| extractive_balanced_stock_80M | extractive_balanced | 0.4062 | 0.6250 | 0.5156 | 0.5052 | -0.6506 |
| extractive_wide_stock_80M | extractive_wide | 0.3385 | 0.6927 | 0.5156 | 0.5 | -0.8182 |

## 100M exposure comparison

| Model | Treatment | Affected | Unaffected | EEBF | Multi-event | Aff margin |
|-------|-----------|----------|------------|------|-------------|------------|
| compact_reinvest_stock_100M | compact_reinvest | 0.3958 | 0.6510 | 0.5234 | 0.5104 | -0.3994 |
| extractive_balanced_stock_100M | extractive_balanced | 0.4010 | 0.6615 | 0.5312 | 0.5052 | -0.6289 |
| extractive_wide_stock_100M | extractive_wide | 0.3333 | 0.6875 | 0.5104 | 0.4948 | -0.7914 |

## Compact-reinvest minus others at 80M

- **compact_minus_clean_qwen**: affected Δ=+0.1302, unaffected Δ=-0.1250, EEBF Δ=+0.0026
- **compact_minus_extractive_balanced**: affected Δ=-0.0104, unaffected Δ=+0.0000, EEBF Δ=-0.0052
- **compact_minus_extractive_wide**: affected Δ=+0.0573, unaffected Δ=-0.0677, EEBF Δ=-0.0052

## Scale1.75 adapter results (from earlier analysis, for reference)

| Checkpoint | Affected | Unaffected | EEBF | Multi-event |
|------------|----------|------------|------|-------------|
| scale1p75_adapter_82M | 0.3177 | 0.7552 | 0.5365 | 0.5000 |
| scale1p75_adapter_84M | 0.2865 | 0.7708 | 0.5286 | 0.4896 |
| scale1p75_adapter_100M | 0.2969 | 0.7813 | 0.5391 | 0.4896 |
