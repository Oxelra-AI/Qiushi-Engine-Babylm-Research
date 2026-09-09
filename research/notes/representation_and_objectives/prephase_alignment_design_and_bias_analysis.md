# prephase alignment design and bias analysis — Prephase alignment probe: design, bias threat, and interpretation

## Scientific purpose

address bootstrap and role query established that tag-only prephase training rescues inline-role direct acquisition
for changed focal records that scratch training can fail on. This warm-start result could be one of three things:

1. **Genuine aligned coordinate reuse**: the model transfers specific tag→record associations
   from the prephase so that the same tag tokens in the inline continuation already have
   the right state-record binding. This would mean coordinate bootstrapping is the real
   mechanism, not generic warm-up.

2. **Tag-token familiarity without semantic alignment**: the model has seen the continuation's
   tag tokens in the prephase (same namespace), so they are not out-of-vocabulary OOV-like
   tokens during continuation. Seeing them before learning any associations creates a lighter
   optimization landscape. Permuted prephase (same tokens, wrong associations) would behave
   similarly.

3. **Generic warm-up / format learning**: any prephase that puts the model in a favorable
   optimization basin for the temporal-change entailment task rescues continuation. Disjoint
   tags (different tokens but same task) would help equally.

## Three-arm design

All three conditions do IDENTICAL inline-role direct continuation (same correct aligned
tag→record mapping, same worlds, same labels).

- **Aligned arm**: prephase uses correct tag→record mapping with the continuation's namespace.
  The model learns TAG_X → before-state, TAG_Y → after-state exactly as required.
  Prediction: rescues changed focal continuation fit.

- **Permuted arm**: prephase uses SWAPPED tag→record mapping with the SAME namespace.
  TAG_X → after-state (before_state in continuation), TAG_Y → before-state.
  This creates maximally conflicting associations for the continuation.
  Prediction under coordinate reuse hypothesis: HURTS continuation (must unlearn).
  Prediction under generic warm-up hypothesis: still rescues (basin is similar).

- **Disjoint arm**: prephase uses correct mapping but DIFFERENT tag tokens.
  No specific tag association transfers; only task and format familiarity carries over.
  Prediction under token familiarity hypothesis: DOES NOT rescue.
  Prediction under generic warm-up hypothesis: still rescues.

- **Scratch arm**: no prephase at all. This is the existing address bootstrap and role query baseline.

## Decision table

| Result | Interpretation |
|--------|---------------|
| Only aligned rescues | Genuine coordinate reuse: specific tag→record bindings transfer |
| Aligned + permuted both rescue, disjoint fails | Token familiarity: seen tokens help regardless of association direction |
| Aligned + permuted + disjoint all rescue | Generic warm-up: task familiarity is sufficient |
| Aligned rescues, permuted hurts, disjoint intermediate | Pure coordinate reuse: conflicting associations actively interfere |
| Nothing rescues consistently | The address bootstrap and role query rescue was seed/initialization-specific, not reliable |

## operation-bias threat and our construction

companion analysis showed that first-basin Entity slope at MAX is dominated by nonzero-op items:
late 80/90/100M mean +4.08 pp, zero-op -9.71 pp, nonzero-op +6.84 pp. This suggests
the model may be learning an operation-prior (update has occurred) rather than genuine
correspondence between source and rewrite views.

In our bridge construction, the analogous risk is: a model that learns to always predict
"the after-state" for changed-focal queries would score well on focal_after rows but
fail on focal_before rows. This is a temporal-position bias.

### Why our balanced design provides partial protection

Our contrastive evaluation pairs AB vs BA hypotheses within each query family. A simple
directional prior (always predict player A is higher) would fail on BA pairs. This is the
contrastive_acc column.

The remaining bias risk is a "temporal/update position prior": always predicting the
after-state is different from the before-state. This could succeed for changed focal worlds
(where before≠after) while failing for stable focal worlds (where before=after).

Our training set contains **16 sparse changed focal** AND **16 sparse stable focal** worlds.
Stable focal queries have the SAME correct answer for before and after queries.
A temporal-position-bias model would predict different answers for before and after
even when they should be the same, thereby failing on sparse_stable_focal rows.

### The micro-EEBF check

Define micro-EEBF = focal_after_acc - focal_before_acc on held changed/stable worlds.

- Near zero: the model genuinely discriminates before vs. after states
- Large positive: the model is biased toward predicting the after-state as correct
- Near zero even when con_acc = 1.0: symmetric genuine discrimination

This check complements the binary contrastive accuracy. A model with micro-EEBF ≈ 0
and high con_acc is genuinely accessing both states. A model with micro-EEBF >> 0
and moderate con_acc may be exploiting update bias.

### What the permuted arm's changed-focal behavior specifically tells us

The permuted arm trains the model to associate:
  TAG_X → after-state (continuation will use TAG_X for before-state)
  TAG_Y → before-state (continuation will use TAG_Y for after-state)

For CHANGED focal worlds, these associations directly contradict the continuation labels.
For STABLE focal worlds, before = after, so the contradiction is less severe.

If permuted rescues changed-focal continuation despite this contradiction, the mechanism
is NOT specific tag→record association transfer. The model is learning something else
(task format, basin structure, generic temporal reasoning) that happens to generalize.

If permuted fails specifically on changed-focal rows while succeeding on base/stable rows,
the mechanism IS specific association: the model's prephase-learned associations for those
exact tags are actively interfering with contradiction.

## Connection to permuted-companion

permuted companion control operates at the same logical level but at BabyLM scale:
- Aligned view: source S paired with compact rewrite of S (source-conditioned correspondence)
- Permuted companion: same compact rewrites, but S1's rewrite is paired with S2's source

If the DeBERTa gain comes from the correspondence structure (S paired with its own
semantically faithful compact form), permuted companion would NOT produce the same benefit.
If the gain comes from the compact text distribution itself (independent of pairing), or
from a generic regularization/diversity effect, permuted companion would produce similar gain.

This is exactly the macro-scale analog of our aligned/permuted tag prephase:
- Our tags are arbitrary address tokens that only carry meaning through pairing
- DeBERTa's compact rewrites carry semantic content through their pairing with sources

The shared question: does source-conditioned correspondence (knowing which rewrite goes
with which source) contribute the scientific value, or is the pairing incidental?

## What prephase alignment design and bias analysis will establish or fail to establish

**Establishes if aligned uniquely rescues**: coordinate bootstrapping under sparse interference
is a real phenomenon, not a generic warm-up artifact. The specific associations formed
during prephase learning provide a non-trivial head start for accessing contradictory
changed records.

**Establishes if permuted also rescues**: the effect is not specific to aligned associations
but is compatible with the token-familiarity or generic warm-up accounts. Coordinate
reuse cannot be claimed from warm-start effects alone.

**Does not establish regardless of result**: architecture-general transfer to larger models,
temporal-semantic grounding, natural language role-to-address composition, or a BabyLM-scale
version of the same principle. Those require separate experiments at different scales.

## Next steps after prephase alignment design and bias analysis

If aligned uniquely rescues:
  → Build the M(role→address) selector as a separate module that outputs a tag selection
    rather than directly outputting state labels. Keep R(tag→state) fixed or lightly updated.
    Test whether M can be learned from sparse role-language without disturbing R.

If permuted also rescues (generic warm-up):
  → The bootstrapping framing is weaker. Need to test whether permuted-companion
    also matches aligned view on Entity rows. If so, the DeBERTa benefit may also be
    generic, which would challenge the correspondence mechanism across both studies.
  → Alternative route: make R(tag→state) architecturally separable (frozen encoder + fresh
    selection head), so prephase and continuation use provably different parameter subsets.

If nothing rescues consistently:
  → The address bootstrap and role query warm-start results are seed-sensitive. The changed-focal bottleneck is
    more fundamental than optimization-path dependence. Route toward architectural solutions:
    separate selector and reader parameters, or explicit multi-head decomposition.

## Files

- Script: training/scripts/prephase_alignment_probe.py
- Analysis: scripts/analyze_prephase_alignment.py
- Results (pending): data/step271_{aligned,permuted,disjoint,scratch}_seed27100/
- This note: notes/prephase_alignment_design_and_bias_analysis.md
