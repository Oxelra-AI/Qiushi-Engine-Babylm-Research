# token value private readout synthesis — token-value and private-readout discriminators for the next route

## Question
After scale1.75 (Overall 41.5707) and U256 (41.3292) closed below 41.8, the proposed route preserves function space with complementary acquisition through a private pathway. Before any
training, this analysis measured, on the frozen legal spatial repair route status checkpoint, whether directed edit-state or
intra-row discourse structure supplies prediction-connected information that a detached,
capacity-matched private residual readout can use to explain spatial repair route status residual masked-token errors,
without changing reference logits.

Refinement: raw frozen-NLL gains only locate structure spatial repair route status already uses when handed
it; the decisive cheap signal is the detached private readout true-vs-false residual advantage.

All runs used the legal pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
and tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, frozen spatial repair route status
`chck_100M`, no LM parameter updates, no official evaluation. One-mask items are built at the token-id
level so every context has identical length and exactly one target piece.

## Results

### A. Directed edit-state (source → compact rewrite), 4,096 items
- Raw frozen NLL: true source lowers changed-token NLL vs matched decoy source by **+2.742** (boot CI
  2.63–2.86); true context top1 0.411 vs base 0.062. spatial repair route status strongly uses aligned source when present.
- Detached private residual readout (equal 64-dim bottleneck, base logits as unchanged offset):
  true-vs-decoy held-out advantage **+2.831** (CI 2.56–3.09), high-baseline-error advantage **+3.515**.
- Data: `data/edit_state_token_value/`.

### A′. Source-token-ABSENT edit-state, 3,072 items (target piece not present in source ids)
- Raw frozen NLL: true source is now **worse** than decoy (−0.312), i.e. the large all-edit raw gain
  was substantially exact-copy of source tokens.
- But the detached private readout still extracts a real true-vs-decoy residual advantage
  **+0.836** held-out (CI 0.65–1.01) and **+1.240** on high-error items (CI 0.97–1.51).
- Meaning: beyond copying, the true-source hidden state carries transformation/edit-state information
  that a private channel can read to reduce residual token errors while leaving reference logits fixed.
- Data: `data/edit_state_token_value_source_absent/`.

### B. Intra-row discourse (natural rows only), 4,096 items, source-balanced
- Raw frozen NLL: true-vs-shuffled **+0.493** (mostly topic/register), true-vs-reversed only **+0.018**.
- Detached private readout: true-vs-reversed **+0.123** (CI 0.03–0.20, weakly positive),
  true-vs-shuffled **+0.015** (CI −0.11–0.12, not significant); high-error not significant.
- Meaning: within-row order carries little order-specific residual signal beyond topic matching; a
  discourse-order private pathway is a weak successor on this legal corpus.
- Data: `data/discourse_token_value/`.

### C. Squared-gradient importance vs endpoint displacement (bounded, 769 legal rows)
- Element-level log grad² vs log delta² pearson is near zero for scale1.75 (−0.016), U256 (−0.007),
  and normal spatial repair route status 80M→100M movement (−0.025); within-tensor-shuffle nulls also near zero.
- Group level: embeddings carry the largest displacement fraction in all three, with delta/importance
  ratios (scale1.75 3.11, U256 4.12, normal late 4.67) — the closed routes are **not** more enriched
  in high-MLM-importance directions than normal late movement.
- Interpretation limit: this is only evidence about MLM-sensitive displacement. It does
  not identify representations to protect for broad competence, and it does **not** motivate a simple
  global importance-damping mechanism as the fix.
- Data: `data/importance_displacement_relation/`.

## Route implication
- The strongest genuinely unsaturated, prediction-connected legal signal is **directed edit-state**:
  true source→compact transformation structure is readable by a detached private channel even when it
  is not copyable, and it concentrates on exactly the changed spans that define the compact-view data
  mechanism (the protected substrate that already gave +1.35 mean7 at 80M).
- Discourse-order and simple importance-damping are weak; do not build the next route around them.
- This does not yet authorize a long run. A short matched continuation is warranted only if a
  train-time private edit-conditioned pathway, coupled to the frozen/slow spatial repair route status function, uniquely
  improves a held-out changed-span token target vs matched false-structure controls AND preserves the
  score sentinels (Supplement subject-aux inversion, Reading, small SuperGLUE, EWoK
  material/spatial/quantitative) — because scale1.75/U256 showed early gains that did not survive.

## Scripts
- `scripts/token_value_private_readouts.py` (modes: edit, discourse; id-level masking;
  `--edit-require-target-id-absent-source` isolates non-copy structure).
- `scripts/importance_displacement_relation.py`.
