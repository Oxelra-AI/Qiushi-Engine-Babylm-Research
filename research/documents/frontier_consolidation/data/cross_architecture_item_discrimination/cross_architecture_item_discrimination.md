# cross architecture item discrimination synthesis: Cross-Architecture Item-Level Discrimination Analysis
Created: 2026-09-03T20:53:49Z
Common 100M items: 170722

## Analysis 1: Cross-architecture delta correlations at 100M
| Pair | r | CI 95% | Agreement | Mean Δ₁ | Mean Δ₂ | n |
|------|---|--------|-----------|---------|---------|---|
| full_deberta_vs_cc_nodis | 0.0217 | [0.0175, 0.0272] | 0.534 | -0.0019 | 0.0037 | 170722 |
| full_deberta_vs_plain_nodis | 0.0120 | [0.0064, 0.0169] | 0.519 | -0.0019 | 0.0030 | 170722 |
| full_deberta_vs_roberta | -0.0092 | [-0.0148, -0.0038] | 0.485 | -0.0019 | 0.0017 | 170722 |
| cc_nodis_vs_plain_nodis | 0.0638 | [0.0578, 0.0693] | 0.589 | 0.0037 | 0.0030 | 170722 |
| cc_nodis_vs_roberta | 0.0282 | [0.0226, 0.0338] | 0.540 | 0.0037 | 0.0017 | 170722 |
| plain_nodis_vs_roberta | 0.0079 | [0.0020, 0.0138] | 0.512 | 0.0030 | 0.0017 | 170722 |

## Analysis 2: Key per-column correlations (full_deberta vs plain_nodis)
- BLiMP: r=0.0086 (n=59875)
- COMPS: r=0.0163 (n=91028)
- EWoK: r=-0.0123 (n=7618)
- Entity: r=0.0113 (n=6780)
- Supplement: r=-0.0129 (n=5218)

## Analysis 4: DeBERTa 80M correlations
- full_deberta_vs_cc_nodis: r=0.0303 (n=170722)
- full_deberta_vs_plain_nodis: r=0.0081 (n=170722)
- cc_nodis_vs_plain_nodis: r=0.0516 (n=170722)

## Analysis 5: Stable-column accuracy profiles
- full_deberta: compact=0.5684, repeat=0.5702, delta=-0.0019
- cc_nodis: compact=0.5280, repeat=0.5243, delta=0.0036
- plain_nodis: compact=0.5323, repeat=0.5293, delta=0.0030
- roberta: compact=0.5256, repeat=0.5239, delta=0.0017

## Interpretation
See candidate_explanations in JSON for hypothesis comparison.

## Boundary
CPU-only analysis on saved evaluation artifacts; no model loading, training, evaluation, SuperGLUE, AoA, upload, or leaderboard submission.
