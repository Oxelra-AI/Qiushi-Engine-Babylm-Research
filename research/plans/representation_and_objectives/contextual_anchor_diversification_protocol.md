# earlier analysis — Contextual diversification around supervised shared anchors

This note revises the cross-realization interpretation and specifies an ordinary-WWM comparison before any new H100 training.

## Why the route must be reframed

cross realization probe result did **not** prove that compact input distribution alone is the dominant cause. Its absolute genuine-rewrite-over-counterfactual advantage is large in every model, including `compact_repeat`, so only the **between-training increments** carry mechanism information:

- `full − compact_repeat` on rewrite advantage: about +0.131 nats.
- `adjbreak − compact_repeat`: about +0.230 nats.
- `full − drop_abs`: about +0.061 nats.
- `drop_abs − drop_copied_word`: about +0.0075, interval crossing zero.

All compact-input 100M arms (`full`, `drop_abs`, `drop_copied_word`) still train with dense copied/source-shared target gradients on the same compact contexts. Thus `drop_abs ≈ drop_copied_word` means source-absent labels are unnecessary for that cross-realization skill; it does **not** separate varied compact context exposure from repeatedly predicting shared anchors in those compact contexts.

stock RoBERTa result keeps this route scientifically demanding: natural compact-vs-repeat in a different masked-LM coordinate is essentially flat on cheap7 (+0.042) and negative on stable selected families excluding GlobalPIQA/Reading (`cheap6_no_GlobalPIQA` -0.115, `cheap5_no_GlobalPIQA_Reading` -0.194, Supplement -2.21), despite positive local compact-side pseudolikelihood. The next experiment must therefore use **downstream transfer in an independent MLM coordinate** as the consequential readout, not another local NLL probe.

## Current scientific object

The strongest hypothesis consistent with matched triangle and route judgment--238 is:

> Faithful compact rewrites may help sample-efficient masked learning by placing semantically stable source-shared anchors into diverse, compressed contexts, so ordinary WWM repeatedly asks the model to reconstruct or use the same anchors across different contextual bases.

This object has separable factors:

- `D`: diversity/compression of the second context around source content.
- `R`: whether the second context shares semantic anchors with the paired source.
- `T`: how often those anchors themselves receive MLM loss.
- `C`: how often visible anchors condition prediction of surrounding material.

Ordinary WWM entangles `T` and `C`; deliberately zeroing content labels or forcing anchor masks changes the learning problem too much to be the first separation. A target-location double-mask experiment would estimate a narrow credit-assignment effect, not the unresolved exposure-versus-shared-supervision object.

## Primary separation: ordinary-WWM factorial

Construct four arms over the same 12,155 compact-pair objects and the same filler/reinvestment budget, then train in an independent RoBERTa/BERT-style MLM coordinate with ordinary WWM:

| arm | second-context type | source-own anchor recurrence | row form |
|---|---|---|---|
| HS | compact faithful rewrite | same anchors | `source_i + compact_i` |
| LS | literal source repeat | same anchors | `source_i + repeat_i` |
| HD | compact faithful rewrite | different anchors | `source_i + compact_deranged(i)` |
| LD | literal source repeat | different anchors | `source_i + repeat_deranged(i)` |

The deranged arms preserve second-context marginals while breaking source-own recurrence. Because they introduce source/second-context mismatch, only the paired factorial contrasts are interpretable; LD is needed to subtract the generic mismatch effect from HD.

Downstream estimands on preregistered RoBERTa selected columns:

- Same-anchor context-diversity effect: `HS − LS`.
- Different-anchor compact-marginal effect: `HD − LD`.
- Anchor-conditioned diversification interaction: `(HS − LS) − (HD − LD)`.

Scientific readings:

- `HS − LS > 0` and `HD − LD ≈ 0`: compact benefit requires repeatedly supervised source-shared anchors in varied contexts.
- `HS − LS ≈ HD − LD > 0`: compact context marginal helps even without source-own anchor recurrence.
- Positive main effect plus positive interaction: both generic compact-context exposure and shared-anchor recurrence matter.
- Local NLL/representation movement without positive selected-family transfer: another local--global split, not the sought transferable principle.

## CPU feasibility evidence produced in earlier analysis

### 1. Context/anchor derangement audit

Script: `experiments/archive/representation_and_objectives/scripts/context_anchor_factorial_preflight.py`.
Result: `experiments/archive/representation_and_objectives/data/context_anchor_factorial_preflight`.

- 12,155 pairs; zero self-pairs.
- Exact view-word length matches: 12,152/12,155 = 0.999753.
- Total absolute view-word delta: 4; net delta 0.
- Receiver-source content overlap is destroyed as intended in deranged arms:
  - own compact 0.6583 vs wrong compact 0.0031.
  - own repeat 0.6342 vs wrong repeat 0.0029.
- Legal16k BPE differences are symmetric by construction:
  - `HS − LS` mean +2.007 BPE pieces per pair.
  - `HD − LD` mean +2.007 BPE pieces per pair.
  - `HS − HD` and `LS − LD` mean 0.

This shows the four-arm construction is mechanically feasible and the factorial interaction will not be biased by different compact-vs-repeat BPE load in the same versus deranged branches. It does **not** yet solve the compact-vs-repeat BPE load itself.

The earlier wrong-partner mapping offers a slightly less exact but domain-matched derangement: exact rewrite length 12,151/12,155, total absolute delta 6, same primary domain 99.819%, zero self-pairs. The final mapping should use a domain-first, length/BPE-second objective, informed by both earlier audits.

### 2. Low-diversity repeat BPE matching audit

Script: `experiments/archive/representation_and_objectives/scripts/repeat_bpe_match_audit.py`.
Result: `experiments/archive/representation_and_objectives/data/repeat_bpe_match_audit`.

The historical hash-rotated repeat view is BPE-lighter than compact:

- old repeat minus compact view: mean -2.007 BPE pieces per pair;
- total -24,398 BPE pieces over the pair objects;
- exact BPE matches only 15.82% of pairs.

A BPE-matched cyclic repeat selection can reduce the mismatch while preserving whitespace length and literal-source repetition:

- best cyclic repeat minus compact view: mean -0.590 BPE pieces per pair;
- total -7,175 BPE pieces;
- exact BPE matches 62.60%; mean absolute delta 0.658 pieces.

This is a useful construction improvement, but it changes the repeated source segment start for 94.54% of pairs. Before replacing the historical repeat view, audit source-position coverage/content coverage so the repeat arm remains a low-diversity literal-source control rather than an accidental tail-coverage repair.

## Construction requirements before H100 training

The proposed first stage is CPU/file construction only:

1. Build the final derangement with zero self-pairs, near-exact view length, domain balance, compact/repeat BPE symmetry, and recorded content-overlap collapse.
2. Build candidate 10M pools and 100M streams for HS/LS/HD/LD with identical filler rows, row counts, word counts, example ordering geometry, tokenizer, and source budget.
3. Decide the repeat view rule. Preferred if source-position audit remains acceptable: BPE-matched cyclic repeat, because it better preserves supervised mass under ordinary WWM. If it alters source-position coverage too much, keep the historical repeat and explicitly carry the +2 BPE/pair compact load as a natural-treatment component that cancels in the factorial interaction.
4. Precompute a shared ordinary-WWM mask manifest for all four arms in the RoBERTa trainer coordinate, or prove tensor equality where possible. Record per-step and cumulative corrupted groups, supervised pieces, replacement classes, active tokens, source/second-view target mass, anchor mass, BPE length bins, and target-position distributions.
5. Tensor smoke on several mixed batches: same batch word counts, same attention geometry, same optimizer/LR/init, finite gradients, reproducible first update, and labels differing only because the four text arms differ.

No new H100 run should start from this note alone. The next spend decision depends on whether the constructed streams preserve ordinary WWM and supervised mass closely enough that downstream differences are readable.

## Minimum downstream training program if construction passes

Train the four RoBERTa/BERT-style MLM arms from identical initialization and data-order seeds, using the existing RoBERTa coordinate as much as possible: legal16k tokenizer, 8x480 model, AdamW/ordinary WWM, same batch/sequence/LR horizon and checkpoints. Use two H100s in parallel.

Lowest reliable staged path:

1. Run HS/LS/HD/LD to a mid-late checkpoint with saved optimizer/RNG state and selected official-compatible readout at prespecified checkpoints (for example 20M/40M/60M). The readout is cheap5 without GlobalPIQA/Reading, cheap6 without GlobalPIQA, cheap7, and individual Supplement/EWoK/Entity/COMPS/GlobalPIQA/Reading columns.
2. Continue the same runs toward 100M only if the downstream factorial effect is directionally coherent on stable selected families, or if CPU construction is exceptionally clean and available H100 time makes the full readout cheaper than another proxy. Do not use local NLL alone to continue.
3. If a positive interaction or main effect appears at 100M, repeat at least the load-bearing contrast with an independent seed before treating it as a transferable learning principle. The expected effect is small: natural RoBERTa compact-vs-repeat result is flat/negative, and the cross realization probe result between-arm increments are much smaller than the absolute rewrite-context advantage.

## Work not chosen as the next primary route

- A zero-content-target arm is too extreme as the immediate next experiment: it would remove much of the content loss and force reallocation decisions that change the objective away from ordinary WWM.
- A paired double-mask source-vs-rewrite target-location swap is a valid narrow credit-assignment screen, but it still leaves compact context and identical anchor prediction in both arms. It does not answer the present varied-context-versus-shared-anchor question.
- Source-absent target up-weighting remains closed as a SOTA route: it produced a real local channel but did not mediate the 100M downstream compact-view advantage.

## Relation to the Other Model Coordinate

The stock RoBERTa compact-vs-repeat replication was independently closed as a stable transfer route, motivating natural compact-data factor decomposition. The anchor-recurrence/context-diversity ordinary-WWM factorial is complementary to RoBERTa tests of compression density, source-position spread, and coherence; these constructions should vary distinct factors.

## Addendum after corrected repeat source-position measurement

After writing the protocol I inspected the historical corpus builder. The actual semantic full eval route update repeat construction is **not** a prefix-first-N repeat: it uses a hash-rotated cyclic source segment with the same word count as the compact rewrite. I therefore added a corrected CPU measurement:

- script `experiments/archive/representation_and_objectives/scripts/repeat_position_coverage_audit.py`
- result `experiments/archive/representation_and_objectives/data/repeat_position_coverage_audit`

Main numbers:

| view | source-content coverage | tail-content coverage | content fraction | compact-minus-view BPE |
|---|---:|---:|---:|---:|
| compact | 0.6633 | 0.6992 | 0.6928 | — |
| prefix repeat | 0.6002 | 0.0000 | 0.5329 | +2.574 |
| historical hash repeat | 0.6244 | 0.6201 | 0.5537 | +2.007 |
| BPE-matched cyclic repeat | 0.6516 | 0.8326 | 0.5786 | +0.590 |

The earlier source-position story inherited from earlier analysis atlas was therefore too prefix-like for the actual training stream. The real historical repeat already spreads across source positions; compact differs more by fluent compression, higher content fraction, and source-shared anchors in a distinct sentence context than by simple late-source recovery alone.

Implication for the next construction: the primary HS/LS/HD/LD factorial should stay close to the historical hash-rotated repeat unless measurements show that the BPE-matched cyclic repeat improves supervised-mass matching without turning the low-diversity control into a tail-recovery control. The four-arm interaction cancels the compact-vs-repeat BPE load symmetrically (`HS−LS` and `HD−LD` have the same +2.007 BPE/pair under the current preflight), so exact BPE equality is less important than preserving the historical contrast and reading downstream interaction rather than local loss. If a BPE-matched repeat is used, it should be a secondary construction with explicit source-position/content measurements.
