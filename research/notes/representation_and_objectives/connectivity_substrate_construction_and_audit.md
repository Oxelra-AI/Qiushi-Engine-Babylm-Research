# connectivity substrate construction and audit — Coordinate-connectivity substrate construction and audit

## What was done

1. **Ran coordinate connected experience route audit** (`data/fit_and_signed_margin_audit/`):
   - Confirmed k16 same-initial bridge rows were fit locally (1.000 exact choice in training)
   - Confirmed eval shows anti-copy pattern: same-initial changed 0.009–0.010, opposite 0.990–0.995
   - Confirmed signed mixed orientation absent: aligned−inverted gap = 0.058±0.051 (noise)
   - This validates the premise for the connectivity experiment

2. **Built coordinate-connectivity substrate** (`data/coordinate_connectivity_substrate/`):
   - **Connected**: bridge state rows share name pairs with held-held comparison rows (4 CONN pairs appear in seen-coordinate, held-held, AND bridge)
   - **Disconnected**: bridge state rows use disjoint name pairs from held-held (4 DISC pairs appear in seen-coordinate AND bridge, but NOT held-held)
   - Both conditions have identical formal information (same satisfying assignments)
   - Global checks verified: formal_equiv=True for all arms, connected has 4 full chains, disconnected has 0
   - Initial-pattern balance: every (ip × rel × ss × voice) cell has exactly 4 rows — no diagonal confound
   - Eval includes BOTH initial patterns (opposite + same) for paired_state_conservation (1024 rows) and cross_template_state_readout (512 rows)

3. **Built and smoke-tested learned probe** (`training/scripts/connectivity_probe.py`):
   - Adapted earlier analysis's DeBERTa encoder + linear head runner for connectivity substrate construction and audit's shared eval directory
   - Seeds: 28600, 28601, 28602 (fresh, no overlap with earlier analysis)
   - Smoke test (1 epoch) completed successfully, confirming path logic

## Row counts (matching static slot confounded factorial and balanced budget preparation k16 structure)

| File | Count |
|---|---:|
| common_seen_train (both) | 1024 |
| aligned supervised | 320 |
| inverted supervised | 320 |
| heldheld_only supervised | 192 |
| eval paired_state_conservation | 1024 |
| eval mixed_held_seen_orientation | 512 |
| eval cross_template_state_readout | 512 |
| eval heldheld_unseen_edge_closure | 128 |

## Name exposure (noted imbalance)

| Condition | CONN names | DISC names |
|---|---:|---:|
| Connected aligned | 256 | 128 |
| Disconnected aligned | 224 | 160 |

This is an inherent feature: bridge pairs have more supervised exposure. The manipulation is connectivity, not exposure dose. If the result is positive, a follow-up control can add matched unsupervised exposure.

## Predictions

| Outcome | Connected | Disconnected | Interpretation |
|---|---|---|---|
| same-initial changed exact | HIGH → event-role | LOW → anti-copy | Connectivity enables reusable coordinate |
| opposite-initial changed | HIGH | HIGH | Both fit standard patterns |
| pair-both same-initial | HIGH | LOW | Connected builds joint conservation |
| mixed aligned−inverted | POSITIVE gap | NO gap | Connected propagates signed coordinate |
| Both fail | - | - | Architecture/interface factorization needed |
| Both succeed | - | - | Connectivity not the causal variable |

## Next step

Launch the full learned probe split across GPUs:
- GPU0: `--conditions connected --out connectivity_probe_connected`
- GPU1: `--conditions disconnected --out connectivity_probe_disconnected`
- 3 arms × 3 seeds each = 9 runs per GPU, ~30 min total

## Files
- Audit: `data/fit_and_signed_margin_audit/`
- Substrate: `data/coordinate_connectivity_substrate/`
- Construction script: `scripts/coordinate_connectivity_substrate.py`
- Probe runner: `training/scripts/connectivity_probe.py`
- Smoke test: `data/connectivity_probe_smoke/`
- Route note: `notes/coordinate_connected_experience_route.md`
