# dual view corrected design — corrected dual-view (aligned vs shuffled) private-pathway trainer

## Why the dual view trainer design trainer is discarded
The dual view trainer design auxiliary loss saw only a **rewrite-only** view, so `dual_view_true` vs `mlm_only`
mixed two things: (a) the intended source→rewrite alignment signal and (b) extra
rewrite-target exposure that `mlm_only` never received. It also (c) let aux gradients touch
all stock weights, (d) did not charge auxiliary words toward the 100M budget, and (e) used
`total_steps = len(loader)` for the LR cosine schedule instead of the fixed 100M horizon
(2529 steps), so a 20M prefix would decay LR to zero at 20M instead of following the mature
horizon. All five are corrected here.

## Reproduced object = the source free transfer synthesis probe object, made train-time
source free transfer synthesis established transfer using items with identical `masked_rew`, differing only by the
conditioning source: `ids_true = [CLS] source_true [SEP] rewrite_masked [SEP]` vs
`ids_shuffled = [CLS] source_shuffled [SEP] rewrite_masked [SEP]`. The shared-readout
condition (train jointly on source-conditioned and source-free hidden states) was the only
one where true alignment beat shuffled (all-edit −1.007 NLL; source-absent −0.323 NLL).

The train-time mechanism therefore is, for each pair row:
- **Main view (charged, official-shaped):** standard WWM on the full packed row through
  stock backbone + adapter → MLM loss. Updates stock + adapter (exactly spatial repair route status recipe).
- **Auxiliary source-conditioned view (private):** build
  `[CLS] SOURCE [SEP] rewrite_masked [SEP]` where the rewrite word groups masked in the
  main view are masked identically here, targets identical. Forward through the SAME
  backbone + adapter. Aux CE on the same rewrite targets. Gradient restricted to the
  **adapter (private) parameters only** by temporarily freezing stock `requires_grad`.
- **Aligned vs shuffled arms differ only in which SOURCE string is used** (true source of
  the pair vs a length-matched different-doc source). Identical rewrite text, identical
  masks, identical targets, identical initial tensors, identical schedule, identical total
  charged exposure. This is the clean correspondence contrast.

## Correspondence-only shuffle
Pair source_words distribution has 39 unique values, all with ≥2 members (no singletons),
so a within-length-group derangement (each pair mapped to a different pair's source of the
**same source_words count**, deterministic, no self-map) is always feasible. This guarantees
the shuffled arm changes only source correspondence, not source length or token budget.

## Charging auxiliary exposure
Auxiliary source-conditioned words are real additional token exposure and must be counted.
Each step's charged words = main-view words + auxiliary source words used that step. The
100M cap is enforced on this combined count so aligned and shuffled arms consume identical
total legal exposure and neither exceeds 10 epochs of the 10M pool. Because pair rows are
4.6% of data and aux uses only the source portion (~87 words/pair-row-source), the aux
charge is small but real and identical across arms.

## Zero-output equivalence + private gradient
Adapter `W_up`/bias zero-init ⇒ augmented model == stock spatial repair route status-arch at init (verified in
adapter 20M closure). Main MLM updates stock+adapter. Auxiliary loss backprops with stock params
`requires_grad=False` so only `.adapter.*` receives the source-conditioned correspondence
gradient — the private pathway. This keeps the official-view function driven by MLM while
the private pathway learns the transferable source→rewrite structure.

## LR horizon
`lr_total_steps` fixed to the mature 100M horizon (2529). A 20M prefix uses the same cosine
schedule truncated at the 20M step count, matching how scale1.75/U256 20M prefixes were
exact prefixes of the 100M runs.

## Arms for the 20M gate panel
- `aligned`  : aux uses true source (treatment)
- `shuffled` : aux uses length-matched different-doc source (correspondence control)
- `mlm_only` : no aux forward, adapter present zero-init (matched-capacity MLM baseline)

Gate (before any 100M): `aligned` must beat `shuffled` on cheap7 while preserving sentinels
(Supplement subject-aux inversion, Reading, EWoK material/spatial/quantitative). `aligned`
vs `mlm_only` is informative but the decisive contrast is aligned−shuffled because those two
share identical aux text/target exposure and differ only in correspondence.

## Mechanical checks before committing GPUs
1. At earlier analysis the augmented model logits == stock (adapter zero) — reuse adapter 20M closure result.
2. On a forced pair-heavy micro-batch: aux_loss > 0, and after aux backward only `.adapter.*`
   grads are nonzero (stock grads all None/zero from the aux pass).
3. aligned and shuffled micro-runs produce identical main-view stream/masks/targets and
   identical charged-word totals; only aux source ids differ.
