# subdose ladder state: Sub-1x dose ladder in MAX geometry

All arms share the MAX rowholdout row-word-count sequence (65,313 rows, 10M words).
The rho=0 reference is the already-trained MAX-geometry DeBERTa clean control.

## Row classification

- Pure 1x rows: 3004 (423405 words)
- Mixed rows: 1 (154 words, 1x portion: 106)
- Pure increment rows: 4918

## Dose ladder

| Dose | Rows | 1x pairs | Inc pairs | Total PW | 1x PW | ρ | Docs | Types |
|------|------|----------|-----------|----------|-------|---|------|-------|
| clean | 0 | 0 | 0 | 0 | 0 | 0.0 | 0 | 0 |
| quarter_1x | 758 | 2861 | 0 | 105962 | 105962 | 0.010596 | 2861 | 11202 |
| half_1x | 1513 | 5856 | 0 | 211853 | 211853 | 0.021185 | 4247 | 16264 |
| full_1x | 3005 | 12155 | 1 | 423559 | 423511 | 0.042356 | 4529 | 22734 |
| dose1p82 | (existing) | | | 771199 | | 0.077120 | 5089 | 36016 |
| MAX | (existing) | | | 1118587 | | 0.111872 | 5261 | 43540 |

## Files

### quarter_1x
- 10M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_quarter_1x_view_10M.jsonl`
- 100M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_quarter_1x_view_100M.jsonl`
- SHA 10M: `c89659cc0a4a6612c293bcaa83606a0921a527d743a0795432780c9b01da9e61`
- SHA 100M: `991a80e57553b6c73b7341d587691ea420e87898ba7daa4dc7fbc905fc411ca5`

### half_1x
- 10M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_half_1x_view_10M.jsonl`
- 100M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_half_1x_view_100M.jsonl`
- SHA 10M: `6ebf93b308054c2c5b6f5bc278e672437fe94874dfa529c18b34f4871e63fe2b`
- SHA 100M: `77c7b57c58df50200c5a3d880bd4152bfe6ba42a51aacfc07693f781727679ff`

### full_1x
- 10M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_full_1x_view_10M.jsonl`
- 100M: `experiments/archive/frontier_consolidation/data/subdose_ladder_maxgeom_pools/subdose_full_1x_view_100M.jsonl`
- SHA 10M: `c7ac0864176e13c784c519718e7327590408ee6aee495b97df922f45877cfb81`
- SHA 100M: `4b6f008a86eb23230b8f3a2f68705a2a1be20ef4147ea491fcb70c0d5861f36c`

## Coverage saturation

See `subdose_cumulative_coverage.csv` for the per-row cumulative coverage curve.
The key question: does the score knee coincide with the coverage knee?
