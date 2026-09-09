# evidence revision source trigger test evidence revision and next source-trigger test

## Scientific position after reviewing task ambiguity decomposition result

The strongest current result is not a finished two-channel theory. It is a set of measurable effects that must be separated by matched interventions:

1. In the Step10b synthetic task, identity substitution improves correct rewrite-token ranking inside the RWT output set relative to the present neutral control. The exact decomposition is

\[
-\log p(r_a)=-\log p(\mathrm{RWT})-\log p(r_a\mid \mathrm{RWT}).
\]

At epoch 300 across seeds 42/43/100:

| cue condition | identity minus neutral, total NLL | identity minus neutral, within-RWT | identity minus neutral, RWT-family |
|---|---:|---:|---:|
| typed cue | -3.281 ± 1.496 | -3.384 ± 1.592 | +0.104 ± 0.114 |
| constant cue | +3.402 ± 0.286 | -3.002 ± 0.909 | +6.404 ± 0.871 |

The equality `hc_s = within_s + family_s` holds to about 1e-5 in the saved Step10b rows. Thus task specification changes the RWT-family mass term in this synthetic mixed-output setting, while the within-RWT ranking improvement survives both cue regimes.

2. The Step10b result does not establish an internal split between a non-attention weight channel and an attention routing channel. All learned behavior is stored in parameters, and Step10b contains no masked-identity arm. The older corr acquisition result `ident_full`/`ident_masked` contrast used a different measurement and cannot by itself show that the newly measured within-RWT improvement survives removal of local identity access. The safer statement is operational: identity practice changed later conditional prediction; the current data separate family assignment and within-family ranking, not internal mediation.

3. The word `always` should be removed from the facilitation statement. Identity improves within-RWT ranking in the Step10b setup relative to its neutral control, but the net target NLL is worse under the constant cue because RWT-family probability collapses. Wrong correspondence in corr acquisition result also shows that relational content can damage later use. Experience value is conditional on relation content, task specification, output geometry, and what the examples replace.

4. Step8b is a crucial limiting case. It did train in-window, unblocked local identity: `repeat_full` trained `s_a -> s_a` in the same sequence, and `repeat_masked` used the same token data with target-to-source access blocked. Yet the nonoverlap rewrite probe found almost no true-source-specific recurrence term:

| contrast | T delta | U delta | T-specific excess |
|---|---:|---:|---:|
| `R_full - support` | +1.827 | +1.881 | -0.054 ± 0.045 |
| `R_full - R_masked` | +1.350 | +1.376 | -0.026 ± 0.034 |

So local identity practice can be necessary in BabyLM and still not sufficient in this synthetic substrate. The existing broad rewrite harm in Step8b is a target-space or final-slot shift shared by T and U, not a recognized-source-triggered competitor. Future synthesis must explain why BabyLM DeBERTa shows a source-triggered effect while Step8b does not.

5. The earlier relation-learning integration supplies evidence for a BabyLM source-triggered effect, but the effect had not yet been independently reproduced in the synthetic instrument. In DeBERTa, REPEAT_SPLIT nearly removes the nonoverlap recurrence term (`-0.031` versus original magnitudes `0.75-1.04`) and removes the held-out copy benefit. The source-mass readout shows R-C elevation under true source T (`+0.055` to `+0.097`) but nearly none under unrelated source U (`+0.005` to `+0.008`), with target probability depressed mainly under T. This is hard to reduce to opportunity cost or frequency hedging. It motivates a synthetic test that makes the source-triggered term appear or disappear by design.

## Scientific question for the next synthetic construction

When does identity practice improve reusable source-to-rewrite use, and when does it install a source-triggered competing prediction that remains after task specification? The experiment should not assume two internal channels. It should measure three observable effects under matched marginals:

- within-RWT ranking of the correct rewrite token;
- RWT-family versus SRC-family mass;
- true-source-specific interaction relative to an unrelated-source control with the same window shape and token multiset.

## Matched experiment design

### Training factors

Use the Step10b backbone idea but repair the missing factors.

**Common backbone for every arm**

- A fixed number of correspondence rows per epoch: context contains `ENT(e) HAS SRC(a)`, query asks for `RWT(a)`.
- Held-out query entities remain absent from training query targets.
- Same model initialization, same seed list, same budget, paired contexts where possible.

**Identity pairing and local access**

- `IDENT_FULL`: extra rows train `s_a -> s_a` in-window, with the target position able to access the source event.
- `IDENT_BLOCKED`: identical tokens and pairings, but from the query segment onward the target cannot attend to the source-event keys during training. The block should cover all layers through the attention mask, not only the immediate target token.
- `IDENT_SPLIT` if cheap: same source and copy target material placed in separate sequences, synthetic analogue of the BabyLM REPEAT_SPLIT arm.
- `UNPAIRED_SRC`: same SRC-family target count, same source tokens, same entities, same slot/cue rates, but target token is a balanced derangement `s_b` with `b != a`. Over attributes, every ordered pair should appear evenly. This is the central paired baseline: identity differs by joint source-target pairing, not by SRC exposure or family rate.

Keep `NEUTRAL` only as a reference, not as the main comparator.

**Task specification**

- `TYPED`: COPY, REWRITE, and NEUTRAL cues distinguish requested relation.
- `UNINFORMATIVE`: cue token is present but carries no relation information; copy and rewrite examples have matched cue rates.
- In typed models, run inference-time cue swaps on the same probe: rewrite cue, copy cue, and uninformative cue. This tests whether a cue can redirect a learned behavior inside one trained model rather than comparing different models.

### Corrected T/U probes

For each held-out query item and attribute `a`, build paired probes with identical target token `r_a`:

- T: the window contains the true source event `ENT(h) HAS SRC(a)` at a specified slot.
- U: the window keeps the same length, positions, cue, and source-token multiset, but destroys the relation between query entity and `SRC(a)`. A balanced derangement can put `SRC(a)` under another entity and give `ENT(h)` a different source token. The target remains `RWT(a)`, making U a no-support or wrong-support reference rather than a different task.

The old Step10b `ns` probe, which deletes the held entity from the context, can stay as an extra reference but should not be used as the source-specific term.

### Metrics

For every arm, access condition, cue condition, inference cue, seed, and T/U probe, save logits and report:

- total NLL for `r_a` and probability `p(r_a)`;
- RWT-family mass and family NLL;
- within-RWT NLL computed by direct log-softmax over the RWT tokens, plus top-1 and MRR;
- probability of the exact matched source token `p(s_a)`;
- SRC-family mass and within-SRC NLL for `s_a`;
- copy-probe NLL for `s_a` under matched T/U probes.

Define the source-specific effect for metric `m` as

\[
D_m(L,Q)=\big(m_{I,T}-m_{UP,T}\big)-\big(m_{I,U}-m_{UP,U}\big),
\]

where `I` is an identity arm, `UP` is `UNPAIRED_SRC`, `L` is local access, and `Q` is cue regime. For NLL, positive `D` means identity pairing creates extra true-source damage; for probabilities, the sign should be interpreted directly. The total-NLL source-specific effect should equal the sum of the family and within-RWT source-specific effects up to numerical tolerance.

Local-access dependence is measured by `D_m(IDENT_FULL) - D_m(IDENT_BLOCKED)` and, if run, `D_m(IDENT_FULL) - D_m(IDENT_SPLIT)`. Task specification is measured by within-model cue swaps and by typed versus uninformative training.

### Removal and rescue if a selective source term appears

If `IDENT_FULL` produces a positive T-specific NLL term or a T-specific increase in `p(s_a)`, immediately run selective tests on the saved model:

1. **Path removal at evaluation**: mask target/query positions from the true source event only. If T-specific harm disappears while within-RWT ranking remains improved, the two observable effects use different current paths. If both disappear, the same contextual retrieval path supports both.
2. **Exact competitor removal in logits**: set only the copied-source logit `z(s_a)` to `-inf`, renormalize, and recompute `p(r_a)` and total NLL. Then separately renormalize within RWT only. This quantifies how much of the damage is carried by the direct source-token competitor versus broader family mass.
3. **Cue rescue**: in the same typed model and same T/U probes, swap to rewrite cue. A true rescue would reduce the T-specific family/SRC competitor while preserving within-RWT ranking.
4. **Small correspondence addition**: continue from the harmful identity model with 25/50/100 extra matched correspondence rows and track whether source harm decays faster than within-RWT ranking changes.
5. **Untied output head control**: rerun the smallest decisive condition without tied input/output embeddings. If the within-RWT effect vanishes, Step10b's improvement may be an embedding/output-head geometry effect rather than a general representation result.

## Outcomes and interpretation

- If the experiment repeats the earlier matched-support comparison after tighter matching, with T and U moving together, then in-window identity remains insufficient in this synthetic causal-LM. The missing condition may lie in natural text, token overlap/frequency structure, MLM objective geometry, scale, or training phase.
- If `IDENT_FULL` alone increases T-specific `p(s_a)` and worsens total `r_a` NLL while `IDENT_BLOCKED` or `IDENT_SPLIT` removes that term, the synthetic instrument will have reproduced the BabyLM source-triggered competing prediction under controlled pairing.
- If identity improves within-RWT ranking in both full and blocked access while only full access creates T-specific source-token competition, the result supports behavioral separability without naming fixed internal channels.
- If blocking access removes both within-RWT improvement and source competition, the correct account is a shared contextual retrieval computation, not a weights-versus-attention split.
- If typed cue removes family/SRC competition inside the same model while preserving within-RWT ranking, task specification can rescue the synthetic competitor. If not, the BabyLM-like token-level competitor survives relation cues and is closer to the BabyLM effect.

## Consequence for route choice

A depth-crossover experiment or a formal additive model should wait until this matched source-trigger test is run. The hardest current question is not whether a crossover can be made, but what additional condition turns local identity practice from broad target-prior damage into true-source-triggered competition while allowing reusable correspondence to improve. This is the next test most likely to convert the accumulated synthetic and BabyLM evidence into a predictive data-efficient learning principle.


## Addendum after relation_learning budget matched design note

The budget-matched relation-learning analysis adds a second mechanism that should be included in the next synthetic design: local exact recurrence also changes how much residual prediction work the repeated material contributes. In seed43022, comparing REPEAT_SPLIT directly with original REPEAT over the same selected tokens and budget, split repetition is much better on compact-rewrite nonoverlap targets: RS−R true-source NLL delta `-0.995` and unrelated-source NLL delta `-0.274` over 80M/90M/100M. Relative to CLEAN, RS has no original-R sign reversal (`T=-0.547`, `U=-0.578`). Pair-level and strict-word filters keep RS−C near zero (`pair gain -0.047`, strict-word gain `-0.096`) while original R−C strict-word excess remains large (`+0.588`).

The corrected copy reading also matters: raw RS−C copy gain is `-0.200`, but RS has lower unrepeated-control NLL, so after gain/control normalization RS−C is only about `-0.031`. The safe statement is that original REPEAT's large copy advantage is local; split repetition should not be described as intrinsically destroying copy ability.

This strengthens the next-test requirements. A synthetic experiment should separate at least four effects:

1. **source-triggered identity competitor**: T-specific elevation of exact source-token probability and suppression of `r_a`;
2. **reusable correspondence identification**: improved within-RWT ranking/probability of `r_a`;
3. **broad target/family prior shift**: T and U moving together, as in Step8b;
4. **residual-prediction-work effect**: exact local copies make some masked/predicted targets easy through local lookup, whereas spaced or blocked exposure preserves ordinary prediction gradients.

Therefore `IDENT_SPLIT` is not optional if compute permits. It is the synthetic analogue of the BabyLM RS condition and can show whether preserving token exposure while removing local co-occurrence improves ordinary rewrite/correspondence fit without source-specific misfire. The design should compare `IDENT_FULL`, `IDENT_BLOCKED`, `IDENT_SPLIT`, and `UNPAIRED_SRC` under identical token/family/slot budgets, not only identity versus neutral noise. The residual-work term should be measured by ordinary no-support/control NLL and by learning curves, not inferred from the T-specific interaction alone.

A compact interpretation consistent with both studies is now:

> A training window is an efficient unit for relation learning only when the lowest-loss local relation is the relation later needed. Exact local duplication can be efficient for direct copying but inefficient for transformed/nonidentical use because it both practices a source-triggered identity readout and removes residual work from the repeated row. Spaced repetition can keep exposure while avoiding the local shortcut. Partial-overlap or varied restatement may be valuable when it preserves source recognizability while making the cheapest local relation a content correspondence rather than identity.

This is still a hypothesis requiring the matched synthetic source-trigger experiment. It should not be promoted to a general law until the synthetic conditions that distinguish the earlier matched-support comparison's broad harm from the BabyLM T-specific misfire are identified and the BabyLM VIEW_SPLIT/natural-variation results are incorporated.


## Literature positioning after focused retrieval

Focused retrieval added source paths and citation keys that should constrain future wording:

- Olsson et al. define induction heads through prefix matching and copying on repeated random-token sequences, with strong causal evidence in small attention-only Transformers and weaker correlational evidence in larger models with MLPs (`\cite{olsson2022context}`). Source-triggered copying is therefore not, by itself, a new kind of circuit. The possible advance is to connect relation content and data placement under fixed budget to when such contextual copying helps or hurts nonidentical targets.

- Hernandez et al. show that repeated data can create a double-descent-like performance trough and disproportionately degrade copying and prefix-matching/induction-head-associated behavior (`\cite{hernandez2022scaling}`). This is adjacent but not identical to the present BabyLM signal: the relation-learning study separates exact same-window recurrence from spaced repetition on held-out source-conditioned content, and measures a T/U sign reversal. The new contribution should be framed as a local relation-practice and residual-work decomposition inside a fixed corpus substitution, not as another repetition-saturation result.

- Lee et al. identify exact and near duplicates in C4/RealNews/Wiki/LM1B and show deduplication improves memorization, leakage, and efficiency, but their deduplication interventions do not manipulate duplicate spacing, window co-occurrence, or semantic restatement relation (`Knowledge/objects/papers/Deduplicating-Training-Data-Makes-Language-Models-Better--d1cf3b5347d7--dcfa46d7aced/object.md`, `\cite{lee2021deduplicating}`). SemDeDup extends duplicate removal to semantic similarity and redundancy (`Knowledge/objects/papers/SemDeDup-Data-efficient-learning-at-web-scale-through-semantic-deduplica--b9dae4cf39a3--2bb0ab3f969f/object.md`, `\cite{abbassemdedup}`), but likewise does not separate identity, partial-overlap correspondence, and residual prediction work inside local windows.

- Chan et al. show that burstiness, rarity, dynamic meanings, and Zipfian skew can induce in-context learning in Transformers while trading off with in-weights learning (`\cite{chan2022data}`). This supports the broad importance of distributional structure and local recurrence, but it does not specify the relation-content sign: identity copy versus content correspondence. The combined synthetic and BabyLM route can add value by measuring how the relation practiced inside the burst controls later source-conditioned readout.

The strongest current research direction is therefore not a generic data-diversity rule and not an induction-head rediscovery. It is a predictive account of **relation practice under fixed budgets**: local co-occurrence makes a contextual source relation learnable; exact identity, partial-overlap restatement, wrong correspondence, and unpaired exposure install different output behavior and leave different amounts of residual prediction work. The next synthetic experiment should identify the extra condition that makes exact identity produce source-triggered harm in BabyLM but not in the earlier matched-support comparison.
