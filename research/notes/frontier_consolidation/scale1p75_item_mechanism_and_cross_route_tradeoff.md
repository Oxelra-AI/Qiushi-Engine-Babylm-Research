# scale1p75 item mechanism and cross route tradeoff — scale1.75 residual-adapter 50M item-mechanism and cross-route tradeoff

## Findings (CPU-only, no GPU, no new training)

Two independent CPU analyses over existing official-compatible saved outputs, both payload-validated.

### 1. Cross-route score-vector geometry (`score_vector_tradeoff`)

14 legal-coordinate intervention records (RTD, Muon 20M/80M, Muon-switch, LAMB, word-mean,
minfreq50, strict-innovation, FineWeb compact/breadth, adapter scale1.75/2.0 at 20M/50M),
each delta vs matched spatial repair route status legal reference at the same exposure.

- Across the 8 mature (70/80/100M) records: **mean cheap7 delta -0.320; none has all seven columns positive.**
- The best mature cheap7 movement is Expanded-FineWeb compact-view 100M at only +0.176, far below the
  ~+0.697 cheap7 equivalent needed for Overall 41.8 if SuperGLUE/AoA stay flat.
- PCA of the delta vectors: **the mature failures are not one scalar axis.** PC1 (43% var) is a broad
  Supplement/BLiMP/GlobalPIQA magnitude axis; PC2 (28%) is GlobalPIQA-up / Entity-down; PC3 (17%) is Entity-up.
  So "competence redistribution" is real but multi-directional: broad underlearning (LAMB), GlobalPIQA-for-Entity
  (Muon, word-mean), and Supplement/EWoK damage (strict-innovation, minfreq50) are separable failure modes.

### 2. scale1.75 vs spatial repair route status 50M item-flip mechanism (`scale1p75_50m_item_flips`)

Reconstructed column deltas match the payload exactly (cheap7 +0.4114; BLiMP +1.835, Entity +2.790,
COMPS +0.728, Supplement -0.040, EWoK -0.733, GlobalPIQA -1.471). Item-level gain/loss flips over 170,722 rows:

- The 50M gain is **not broad independent complementarity**; it is a structured decision-boundary shift.
- BLiMP: net +1132 items, concentrated in `superlative_quantifiers_1` (+236/979), `superlative_quantifiers_2`
  (+195/986), `wh_questions_object_gap` (+181/859), `regular_plural_subject_verb_agreement` (+118/890);
  offset by NPI (`npi_present_2` -125), islands (`left_branch_island_echo_question` -98, `adjunct_island` -92),
  `wh_vs_that_with_gap` -76. The adapter systematically reorders quantifier/wh material — the example flips show
  it prefers the gapped/filler-final ordering, which helps object-gap and superlatives but hurts island constraints.
- Entity: net +167, in high-operation `regular_0/2/4_ops`, `move_contents_0/5_ops`; small negatives in low-op
  `regular_1_ops`, `ambiref_0/1_ops`. Consistent with the adapter strengthening multi-operation state chains.
- EWoK: net -57, gains in `physical-dynamics` (+19) but losses in `material-properties` (-16),
  `quantitative-properties` (-19), `social-interactions` (-16), `material-dynamics` (-31).
- COMPS: near-symmetric churn (17566 gains, 16691 losses, net +875), essentially high-variance not structural.
- GlobalPIQA: tiny sample (203 items), net -3, entirely within-noise item count, but scored -1.47 due to the
  small-n mean form; this is a fragile column at this exposure.

## Scientific reading

The scale1.75 residual side-path does not add broad new competence at 50M; it **rotates the model's
preference toward a particular syntactic/state-tracking strategy**. It wins where that strategy is
correct (superlative/wh-gap ordering, multi-op entity chains) and loses where it conflicts (NPI/island
locality, EWoK material/quantitative world facts). This is the same multi-directional redistribution the
cross-route PCA shows, now localized to specific benchmark families rather than being a generic "capacity" gain.

Implication for the pending 70/80M decision: a mature *repair* would require the EWoK material/quantitative
and island/NPI losses to recover while the superlative/wh/high-op-entity gains persist. If instead the same
families stay traded at 70/80M, fixed scale1.75 is another localized redistribution and should be closed as a
SOTA endpoint, with the residual-capacity idea handed to a construction that constrains *which* families the
side path is allowed to move (norm-budgeted per-family or depth-local amplitude), not a single global scalar.

## Artifacts
- `data/score_vector_tradeoff/score_vector_tradeoff.{json,md}`
- `data/scale1p75_50m_item_flips/scale1p75_vs_step35_50M.{json,md}`
- `scripts/score_vector_tradeoff_analysis.py`
- `scripts/pairwise_item_flip_analysis.py` (generic; rerun with --base/--candidate on 70/80M when available)

## Still pending (not decided here)
- scale1.75 80M training and 70/80M cheap eval remain running.
- Best complete legal endpoint remains spatial repair route status Overall 41.257770896404615; no improved SOTA claim established.
