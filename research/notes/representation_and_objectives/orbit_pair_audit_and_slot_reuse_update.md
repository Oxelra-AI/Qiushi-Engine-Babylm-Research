# orbit pair audit and slot reuse update update: stable-arm audit, global-compatible rebuild, and token familiarity vs role-slot reuse

## Why earlier analysis training was stopped

The strategist note identified a real structural risk in the earlier analysis language-scale pair: `stable` used a stable candidate order but did not force a single corpus-global key -> alias map after row-level exclusions and whole-row tokenizer-geometry checks. I cancelled the running background task `s259_t18_tool1` before the two H100 arms could become misleading evidence. The task had already built the 10M/100M corpora and launched training, but no score or model movement from that run should be interpreted as the decisive language-scale comparison.

## Independent audit of the earlier analysis corpora

Audit script: `experiments/archive/representation_and_objectives/scripts/audit_orbit_pair.py`  
Output: `experiments/archive/representation_and_objectives/data/orbit_pair_audit/orbit_pair_audit.json` and `.md`.

The audit replayed the earlier analysis constructor against the produced 10M corpora.

- Rows/words audited: 64,740 / 10,000,000.
- Replay matched the actual outputs exactly: `per_row_output_mismatch=0`, `stable_output_mismatch=0`.
- The per_row and stable arms transformed the same row/key support: `support_diff_rows=0`.
- However, the stable arm was not truly global: 148 of 9,681 transformed lexical keys received more than one alias.
- 2,497 / 23,980 assigned key-events lay on multi-alias keys (10.41%), though only 236 / 23,980 key-events used a non-modal alias (0.984%).
- Mean entropy per changed key was 0.0107 bits. Top examples include `edna` -> mostly Ange but sometimes Nepal, `bessie` -> mostly Dest but sometimes Murray, `dorothy` -> mostly Glasgow but sometimes Romans.

Interpretation: earlier analysis corpora were more paired than initially feared at the support level, but the stable arm fails the exact global-identity contract. Therefore per_row-minus-stable training from earlier analysis cannot be treated as a clean identity-persistence contrast.

## Corrected global-compatible corpus builder

Builder script: `experiments/archive/representation_and_objectives/scripts/global_compatible_orbit_pair.py`.

The corrected builder first chooses a corpus-global alias for each lexical key and the row/key support on which that alias is compatible. If a single alias cannot preserve token/WWM geometry in every candidate row, unsupported row/key occurrences are dropped from both arms. It then writes:

- `det_orbit_stable_global_*`: one alias per lexical key across the entire corpus;
- `det_orbit_per_row_global_support_*`: row-specific aliases on exactly the same transformed row/key support.

A 3,000-row pilot passed the intended contract:

- Candidate rows/key-events/keys: 1,372 / 1,993 / 1,503.
- Global aliases selected: 1,501 keys; coverage classes full=1,498, partial=3, zero=2.
- Final changed rows/key-events/spans: 1,369 / 1,986 / 4,090.
- Rows dropped after paired row-level geometry/injectivity checks: 1.
- Stable keys with multiple aliases: 0.
- Per-row keys with multiple aliases: 285 / 1,499.
- Whitespace word count and full legal16k WWM word-start geometry were preserved.

The full CPU corpus build was submitted as background task `s260_t11_tool1`. It should be interpreted only as a corpus-construction result. Training should not start until its manifest is read and verifies exact 10M/100M counts, zero stable multi-alias keys, identical support hash, correct reference row order, and token/WWM geometry.

## Controlled raw-text separation: token familiarity vs relation-level role-slot reuse

Script: `experiments/archive/representation_and_objectives/scripts/token_familiarity_vs_role_reuse.py`  
Output: `experiments/archive/representation_and_objectives/data/token_familiarity_vs_role_reuse/token_familiarity_vs_role_reuse_summary.json` and `.md`.

Purpose: earlier analysis showed transfer appears when argument fillers are drawn from a reusable trained type system. This could mean merely ordinary token familiarity, or it could mean that fillers must be trained as reusable argument slots in oriented relation contexts. The orbit pair audit and slot reuse update controlled diagnostic used disjoint train and held alias pools. Base anchor and sparse probe-anchor rows used the train pool; held-family evaluation used the held pool. Held-pool tokens were either absent, neutrally exposed in non-relational/coreference rows, exposed in true anchor-relation contexts, or exposed in flipped anchor-relation contexts.

Training all cells fit: 4/4 seeds in every mode/arm.

Held-pool token exposure:

| mode | held types covered | held-pool occurrences in training |
|---|---:|---:|
| absent | 0/32 | 0 |
| neutral | 32/32 | 19,200 |
| anchor_true | 32/32 | 19,200 |
| anchor_flip | 32/32 | 19,200 |

Held-family probe accuracy with held-pool fillers:

| mode | zero | true sparse probe anchors | shuffled/anti sparse probe anchors | true - shuffled | true - zero |
|---|---:|---:|---:|---:|---:|
| absent | 0.625 | 0.745 | 0.368 | +0.377 | +0.119 |
| neutral | 0.716 | 0.812 | 0.342 | +0.471 | +0.096 |
| anchor_true | 0.754 | 0.860 | 0.320 | +0.540 | +0.106 |
| anchor_flip | 0.260 | 0.141 | 0.573 | -0.432 | -0.119 |

Interpretation:

1. Ordinary token familiarity helps: neutral held-pool exposure raised the zero held-probe baseline from 0.625 to 0.716.
2. Token familiarity alone does not explain coordinate orientation: true sparse probe anchors still beat anti-aligned anchors strongly after neutral exposure (+0.471), and this separation is larger under true relation-level held-pool exposure (+0.540).
3. Relation-level slot orientation can override token familiarity: flipped anchor exposure for the held pool drove anchor-held accuracy to ~0.10 and inverted the sparse-probe relation (`true - shuffled = -0.432`).
4. The stronger current principle is therefore not simply "randomize identities" and not merely "make tokens familiar." Limited-data relation learning becomes reusable when argument fillers participate in a shared, trained slot type system whose relation coordinate is oriented by sparse reliable evidence. Identity randomization, family-stable aliases, and shared pools are different interfaces that create this condition; fixed/disjoint natural names suppress transfer because the held argument slots are poorly connected to that type system.

## What still is not established

- The raw-text evidence remains a controlled BiGRU sports-derived diagnostic. It does not yet show that BabyLM-scale MLM endpoints improve on relational EWoK/Entity.
- The deterministic language-scale detector covers only repeated capitalized lexical atoms: earlier analysis coverage was 16,683 changed rows and 54,783 replaced spans, about 0.55% of whitespace words. The global-compatible rebuild will have slightly smaller support. A null result may therefore reflect weak coverage rather than failure of the slot-type principle.
- The language-scale per_row-vs-stable contrast separates cross-document alias persistence under a fixed detector, but by itself it does not separate ordinary token familiarity from relation-level reuse. The interpretation must include token exposure summaries, broad unaffected tasks, Entity/world-knowledge movement, and item-level relational EWoK. If possible, a future language-scale control should add a same-support token-familiarity arm or mine relation-heavy subsets before committing another full run.

## Immediate continuation

1. Wait for automatic delivery of `s260_t11_tool1` only when the next decision depends on it; do not poll.
2. On delivery, inspect `global_orbit_pair_manifest.json` before training. Required checks: 10M/100M counts, row order, support identity, stable multi-alias keys = 0, word/token/WWM geometry, changed rows/key-events/spans, output hashes.
3. If valid and coverage remains scientifically meaningful, decide whether to train the corrected pair or first build an even cheaper readout/control that estimates relation-rich coverage of the transformed tokens against the 1,756 relational EWoK panel and Entity items.
4. Preserve the refined mechanism in all interpretation: reusable argument-slot type sharing plus sparse absolute-coordinate orientation, with ordinary token familiarity as a helpful but insufficient component.

## independent_review verifier correction and refined control

independent_review integration (`data/external/independent_review01_verifier1_integration.md`) agreed that the earlier analysis stable-arm defect invalidates an exact zero-entropy identity-persistence interpretation, while noting the contamination was numerically small rather than making the corpus meaningless. It also sharpened the raw-text interpretation: the previous held-token experiment proves filler-pool polarity control, but `absent` transfer above chance, vocabulary construction including eval tokens, ordered-neutral exposure, and possible pool-switch/mention-order strategies mean it should not be overstated as necessary relation-slot training or abstract predicate transfer.

I therefore ran a matched-update, order-balanced refinement.

Script: `experiments/archive/representation_and_objectives/scripts/slot_reuse_control_refinement.py`  
Outputs: `experiments/archive/representation_and_objectives/data/slot_reuse_control_refinement/slot_reuse_control_refinement_summary.json` and `.md`.

Design changes:

- all modes receive the same number of extra training rows;
- anchor/probe/held template sets are balanced for winner-first and loser-first surfaces;
- `absent_control` uses extra train-pool identity rows so update count matches;
- `pure_match` gives held-pool tokens same/different identity exposure without relation predicates;
- `ordered_neutral` gives held-pool ordered non-sports pair exposure;
- `anchor_random` gives held-pool relation-anchor text with balanced uninformative labels;
- `anchor_true` and `anchor_flip` give matched true/complemented relation-anchor exposure.

All interpretable non-random cells fit training; `anchor_random` intentionally did not fit well because labels are unlearnable (`train≈0.82–0.85`), and its held-probe accuracies stayed near chance.

Balanced held-pool probe results:

| mode | zero | true sparse | anti sparse | true - anti | true - zero | notes |
|---|---:|---:|---:|---:|---:|---|
| absent_control | 0.583 | 0.747 | 0.416 | +0.331 | +0.164 | eval tokens in vocab but unseen; relation/template structure still transfers partly |
| pure_match | 0.671 | 0.777 | 0.419 | +0.358 | +0.107 | pure token identity matching helps zero baseline |
| ordered_neutral | 0.685 | 0.763 | 0.359 | +0.404 | +0.078 | ordered/coreference exposure helps but does not remove coordinate effect |
| anchor_random | 0.512 | 0.557 | 0.472 | +0.085 | +0.045 | random relation cooccurrence is not enough and prevents train fit |
| anchor_true | 0.689 | 0.815 | 0.310 | +0.505 | +0.127 | true relation-slot exposure gives strongest positive orientation |
| anchor_flip | 0.401 | 0.270 | 0.560 | -0.290 | -0.131 | complemented relation-slot exposure reverses held-pool polarity |

This substantially strengthens the interpretation: ordinary token familiarity and identity matching help, but random relation-token cooccurrence does not reproduce the effect; true and complemented relation-level slot exposure drive opposite held-pool orientations under matched row count and balanced mention-order templates. The evidence still remains within controlled synthetic alias pools and trained probe templates; held-template transfer is weaker, so the full principle should emphasize conditions under which reusable slot interfaces plus reliable orientation evidence support transfer, not claim universal abstract relation semantics.

## Postbuild audit of corrected global-compatible corpus

The completed orbit-pair manifest was independently reread and audited.

Postbuild audit: `experiments/archive/representation_and_objectives/scripts/global_orbit_postbuild_audit.py`  
Outputs: `experiments/archive/representation_and_objectives/data/global_compatible_orbit_pair/global_orbit_pair_postbuild_audit.json` and `.md`.

The global-compatible build itself is valid on the forward contract:

- 64,740 rows / 10,000,000 words and 647,400 rows / 100,000,000 words;
- changed rows/key-events/spans: 16,909 / 24,211 / 56,112;
- stable `H(alias|key)=0`: stable multi-alias keys = 0;
- per-row differs strongly: 3,444 keys with multiple aliases, `P_same(alias|key pair)=0.1199`, same-as-stable events 254 / 25,839 = 0.0098;
- repeated-key dose: 18,638 / 25,839 events on repeated keys = 0.7213;
- row order and word counts match the reference.

But it exposed a new confound: the stable arm maps 10,649 transformed keys onto only 3,173 aliases; 1,979 aliases are shared by multiple source keys, and 21,481 events lie on multi-key aliases, with mean `H(key|alias)=1.1311`. This means the global-compatible corpus isolates zero `H(alias|key)` but mixes persistence with alias fusion and stable alias-frequency concentration. It should not be the preferred expensive training substrate if a globally injective mapping is feasible.

A feasibility check (`experiments/archive/representation_and_objectives/scripts/global_injective_feasibility.py`) showed profile deficits only for long rare token profiles and a greedy unique-alias trial assigned 9,568 / 9,723 keys. A 3,000-row globally injective pilot then passed:

- candidate rows/key-events/keys: 1,372 / 1,993 / 1,503;
- stable unique aliases selected: 1,503 keys;
- final changed rows/key-events/spans: 1,371 / 1,990 / 4,098;
- stable multi-alias keys = 0 and stable aliases shared by multiple source keys = 0;
- per-row multi-alias keys = 285 / 1,503;
- word/token/WWM geometry preserved.

The full globally injective corpus build was launched on CPU, without training. If the resulting support is adequate, this one-to-one pair is proposed to replace the global-compatible but many-to-one pair in a later training comparison.

## Current scientific state after orbit pair audit and slot reuse update

The current mechanism is now sharper and more conditional:

- Persistent identity is not intrinsically bad: shared-pool and family-stable identities can transfer when fillers inhabit a reusable trained argument-slot type system.
- Token familiarity helps but is not sufficient: pure identity matching and ordered neutral exposure raise baselines, but random relation cooccurrence does not fit or orient; reliable relation-labeled exposure and complemented exposure steer held-pool predictions in opposite directions.
- Sparse probe anchors orient trained probe surfaces strongly, but held-template transfer remains weaker; the current raw-text evidence is best read as conditional reuse over trained relation surface families, not yet full abstract predicate generalization.
- Language-scale intervention coverage is modest and semantically mixed. Even a valid pair will test a natural-corpus identity/alias-persistence manipulation, not the full synthetic slot-orientation principle unless interpreted with token-marginal, inverse-collision, repeated-key, relation-rich exposure, EWoK/Entity item movement, and broad unaffected-task analyses.

The completed globally injective manifest requires an independent postbuild audit analogous to `global_orbit_postbuild_audit.py` before any H100 training is scientifically justified.
