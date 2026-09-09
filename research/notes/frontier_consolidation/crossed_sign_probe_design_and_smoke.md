# crossed sign probe design and smoke: Crossed-sign interaction probe — design and smoke findings

## Design

2×2 factorial: Two contexts (C1, C2) × Two alternatives (alt_A, alt_B).
C1 makes alt_A correct; C2 makes alt_B correct.
Interaction Δ = [PLL(A|C1) - PLL(B|C1)] - [PLL(A|C2) - PLL(B|C2)]
Expected Δ > 0. Crossed-sign = margin(C1) > 0 AND margin(C2) < 0.

Probe: 250 items, 5 families, content SHA `376e0e1f11c5f71fc3ac819d1b83510c755bb205e3acbf45a2ca0f0f1e584762`
- entity_state: 70 items (35 d2, 25 d3, 10 d4) — multi-step location tracking
- spatial_put: 45 items — simple placement binding
- property_bind: 45 items — "the adj one is the entity"
- agent_action: 45 items — "the role is name"
- temporal_order: 45 items — "the first to verb was name"

Files:
- Probe builder: `scripts/build_crossed_sign_probe.py`
- Probe JSON: `data/crossed_sign_probe/crossed_sign_probe.json`
- Scorer: `scripts/crossed_sign_scorer.py`
- Smoke results: `data/crossed_sign_scores/smoke_82M_100M.{json,md}`
- Smoke detail: `data/crossed_sign_scores/smoke_82M_100M_detail.json`

## Smoke results (82M vs 100M)

| Family | 82M Δ | 82M cs | 100M Δ | 100M cs | 82M better? |
|--------|-------|--------|--------|---------|-------------|
| spatial_put | +16.09 | 1.000 | +16.24 | 1.000 | No (ceiling) |
| agent_action | +1.30 | 0.311 | +0.61 | 0.267 | **YES** |
| property_bind | +0.77 | 0.200 | +0.85 | 0.156 | Mixed |
| entity_state | -10.82 | 0.000 | -11.06 | 0.000 | Slightly (floor) |
| temporal_order | -9.10 | 0.044 | -9.25 | 0.044 | Slightly (floor) |
| **Composite** | **-0.350** | **0.280** | **-0.520** | **0.264** | **YES** |

## Key findings from smoke

1. **82M beats 100M on composite and crossed-sign accuracy** — correct direction.
2. **spatial_put** at ceiling (cs=1.0, 45/45 correct) — not discriminating.
3. **entity_state** shows 100% ANTI-crossed-sign (70/70 items have m1<0, m2>0):
   the model systematically prefers the INITIAL location over the TRACKED final location.
   This is a genuine state-tracking failure, not noise.
4. **temporal_order** similarly anti-crossed (m1<0 on 38/45, m2>0 on 36/45):
   the model cannot infer "X was first" from "X verb before Y."
5. **agent_action** is the most discriminating family: 21/45 Δ>0 at 82M vs presumably fewer at 100M;
   26/45 get m_C1 right, 26/45 get m_C2 right.
6. The composite picks 82M as best checkpoint (correct).
7. Pearson requires 3+ points — the full 8-checkpoint panel is needed.

## Next: full 8-checkpoint panel

Run the scorer with --checkpoints chck_77M..83M chck_100M on GPU0.
Expected ~2.5 min total. Then compute per-family Pearson/Spearman with cheap7.

If the composite or agent_action Pearson is positive and meaningful (>0.5),
this is the first successful legal corpus-derived selector identifying the 82M peak.
If it fails, selector construction should stop and return to mechanism intervention.
