# evidence revision source trigger test matched source-trigger experiment plan

## Purpose

The next controlled synthetic experiment should find the condition that distinguishes two facts:

- The earlier matched-support experiment already had same-window identity practice but did not produce a BabyLM-like true-source-specific recurrence term.
- The BabyLM relation-learning integration and split controls show that BabyLM original REPEAT has a true-source-triggered identity competitor and that split repetition preserves content exposure while removing the local shortcut.

The experiment should test when identity practice improves reusable rewrite identification and when it creates a source-triggered competing prediction.

## Training corpus

Use a small causal-LM sequence task derived from Step10b, but with tighter matching.

### Common rows

Every arm receives the same correspondence backbone:

```text
BOS [context events: ENT(e_i) HAS SRC(a_i)] SEP ENT(e_q) REWRITE_CUE IS RWT(a_q) PAD
```

Backbone size: start with 100 rows/epoch as in Step10b. Use held-out query entities for probes; do not train the held-out entities as query targets.

### Main extra-row arms

Use the same total rows and the same target-family rates. The central comparator is `UNPAIRED_SRC`, not neutral noise.

1. `IDENT_FULL`
   - extra rows train local identity `SRC(a) -> SRC(a)`.
   - target/query segment can attend to the source event during training.

2. `IDENT_BLOCKED`
   - identical token sequences to `IDENT_FULL`.
   - training attention mask blocks the query/target segment from reading the matched source event keys. The block should apply in every transformer layer via the attention mask.

3. `IDENT_SPLIT`
   - same source and identity target material, but source event and copy target are not in the same sequence/window.
   - This is the synthetic analogue of the BabyLM REPEAT_SPLIT arm and is important for separating local shortcut from exposure/residual work.

4. `UNPAIRED_SRC`
   - same number of SRC-family target rows as identity arms.
   - source token is `SRC(a)` but target token is a balanced derangement `SRC(b)`, `b != a`.
   - Balance all ordered pairs as evenly as possible across attributes.
   - Same cue rates, positions, entity rates, source tokens, target family, and budget as identity.

Optional reference only:

5. `NEUTRAL`
   - Step10b-style random RWT or separate neutral cue rows. Do not use this as the primary identity comparator.

### Task specification regimes

Run at least two versions:

1. `TYPED`
   - COPY_CUE for copy/SRC-target rows.
   - REWRITE_CUE for backbone rewrite rows.
   - neutral cue if neutral reference is included.

2. `UNINFORMATIVE`
   - cue token present but not predictive of requested relation. Keep cue frequencies matched across relation families.

For typed models, also evaluate the same saved model under inference-time cue swaps: `REWRITE_CUE`, `COPY_CUE`, and an uninformative cue if available.

## Corrected T/U probes

For each held-out query entity `h` and attribute `a`, construct paired probes for target `RWT(a)`:

- `T`: window contains `ENT(h) HAS SRC(a)` in a designated source slot.
- `U`: same length, same slot layout, and same source-token multiset, but the query entity no longer carries `SRC(a)`. Put `SRC(a)` under another entity and give `ENT(h)` another source token through a balanced derangement. Target remains `RWT(a)`.

Also include copy probes with target `SRC(a)` using the same T/U construction.

Keep the old no-source probe only as an auxiliary reference; do not use it for the source-specific term.

## Measurements

Save raw logits for all probes. From logits compute:

- total NLL and probability for `RWT(a)`;
- RWT-family mass and family NLL;
- within-RWT NLL from a direct log-softmax over only the RWT tokens, plus top-1 and MRR;
- exact source-token probability `p(SRC(a))`;
- SRC-family mass;
- within-SRC NLL for `SRC(a)`;
- copy-probe total NLL for `SRC(a)`;
- old no-source metrics as reference;
- ordinary no-support/control NLL to capture residual prediction work.

Verify numerically that

```text
total_RWT_NLL = RWT_family_NLL + within_RWT_NLL
```

for every aggregate and row-level metric within small floating error.

## Interaction quantities

For a metric `m`, define source-specific identity pairing effect:

```text
D_m(L, Q) = (m_IDENT,T - m_UNPAIRED,T) - (m_IDENT,U - m_UNPAIRED,U)
```

where `L` is `FULL`, `BLOCKED`, or `SPLIT`, and `Q` is the cue regime. For NLL, positive `D` means extra true-source damage. For probabilities, read the sign directly.

Local access contribution:

```text
D_m(IDENT_FULL, Q) - D_m(IDENT_BLOCKED, Q)
D_m(IDENT_FULL, Q) - D_m(IDENT_SPLIT, Q)
```

Task-specification contribution:

```text
D_m(L, TYPED with rewrite cue) - D_m(L, UNINFORMATIVE)
```

Within one typed model, cue swaps provide the stronger test of whether the cue redirects the learned prediction on the same weights.

## Immediate follow-up if source-specific harm appears

If `IDENT_FULL` produces T-specific elevation of `p(SRC(a))` or T-specific damage to `RWT(a)`:

1. Re-score with an evaluation mask blocking the query/target positions from the true source event. Track whether harm and within-RWT ranking disappear together or separately.
2. Recompute logits after setting only the exact copied-source logit `SRC(a)` to `-inf` and renormalizing. This estimates how much damage is carried by the direct source competitor.
3. Recompute within-RWT-only probabilities. This isolates ranking among correct-family targets.
4. In the same typed model, swap to the rewrite cue and test whether family/SRC competition is reduced while within-RWT ranking stays improved.
5. Continue a harmful identity model with small additions of matched correspondence rows (25, 50, 100) and measure whether source-specific harm decays at a different rate from within-RWT ranking.
6. If the decisive result is cheap enough, rerun the smallest cells with untied input/output embeddings to test whether Step10b's within-family benefit depends on embedding/output-head tying.

## Expected scientific readings

- Earlier matched-support-like outcome: T and U move together. Local identity is still insufficient in this synthetic setup.
- BabyLM-like outcome: only `IDENT_FULL` raises exact source-token mass and damages `RWT(a)` under T; blocked/split/unpaired remove it.
- Coexistence of benefit and harm: within-RWT ranking improves while family/SRC competition worsens. Net NLL follows the sum.
- Shared-path outcome: blocking source access removes both within-RWT improvement and source competition, arguing against a clean internal separation.
- Residual-work outcome: split or blocked exposure improves ordinary no-support/control NLL compared with full local copies, mirroring the budget-matched relation-learning comparison.

## Suggested first run

Use a fast smoke run first:

- seeds: `[42]`
- epochs: `20`
- arms: `IDENT_FULL`, `IDENT_BLOCKED`, `UNPAIRED_SRC`
- cue regimes: typed only

Then run the first decisive batch:

- seeds: `[42, 43, 100]`
- epochs: `300`
- arms: `IDENT_FULL`, `IDENT_BLOCKED`, `IDENT_SPLIT`, `UNPAIRED_SRC`
- cue regimes: typed and uninformative
- save models at final epoch for path-removal/logit-removal rescue analyses.

The proposed bounded comparison prioritized `IDENT_FULL`, `IDENT_BLOCKED` and `UNPAIRED_SRC` under typed cue with corrected T/U probes; `IDENT_SPLIT` remained the additional exposure-versus-locality control motivated by the budget-matched relation-learning result.
