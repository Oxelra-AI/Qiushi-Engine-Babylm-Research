# The Relation-Practice Principle: empirical statement, mechanism, and open branches

## 1. The measured phenomenon

Under a fixed 10M-word/100M-exposure budget, replacing a subset of training examples with source–companion pairs installs cross-span competences that depend on the *relation* between source and companion, not merely on the added content.

Three measured quantities carry the principle across three DeBERTa seeds (43022, 43122, 43222) and, with weaker resolution, one RoBERTa seed (43022):

### 1a. Active recurrence cost (primary, cross-architecture)

On held-out compact-rewrite token-nonoverlap targets, where the masked target token does not appear in the source token set, the source-conditioned gain  
$$G = U_{\mathrm{NLL}} - T_{\mathrm{NLL}}$$  
(unrelated-source NLL minus true-source NLL at the masked target) is lower for REPEAT than for CLEAN:

| seed | arch | R−C gain Δ | T NLL Δ | U NLL Δ |
|---:|---|---:|---:|---:|
| 43022 | DeBERTa | −0.752 | +0.448 | −0.304 |
| 43122 | DeBERTa | −1.043 | +0.693 | −0.350 |
| 43222 | DeBERTa | −0.850 | +0.416 | −0.435 |
| 43022 | RoBERTa | −0.398 | +0.492 | +0.094 |

The DeBERTa sign reversal is isolating: REPEAT is *better* than CLEAN with an unrelated source (lower $U$) yet *worse* with the true source (higher $T$). This cannot be explained by overall model quality loss; it indicates an installed computation that interferes specifically when a related nonidentical source is present.

Pair-level and strict word-level controls preserve the cost with attenuated magnitude: 66–70% of held-out pairs show positive excess true-source cost; strict word filtering gives R−C costs of +0.588, +0.830, +0.251 for the three DeBERTa seeds and RoBERTa.

### 1b. Content-conditioning benefit (three-seed DeBERTa, weak in RoBERTa)

On the same targets, VIEW−CLEAN gain Δ is positive:

| seed | V−C gain Δ | T NLL Δ | U NLL Δ |
|---:|---:|---:|---:|
| 43022 | +0.684 | −1.179 | −0.495 |
| 43122 | +0.894 | −1.354 | −0.461 |
| 43222 | +0.839 | −1.368 | −0.529 |

True-source improvement exceeds unrelated-source improvement in every seed, so VIEW's benefit survives register-fit decomposition. At seed 43222, V−R gain Δ ≈ +1.69 is almost entirely true-source-specific (T delta −1.784, U delta −0.094).

RoBERTa's V−C residual is only +0.067 on tokenizer nonoverlap and +0.044 under strict word filtering — much of its raw improvement is broad register fit. The positive side is therefore three-seed DeBERTa evidence, not a universal architecture law.

### 1c. Natural-copy benefit (three-seed DeBERTa, unresolved in RoBERTa)

On held-out natural source-repeat packets from never-trained rows, copy gain is defined as NLL without the repeated source minus NLL with the source repeated. In raw gain, the order is REPEAT > VIEW > CLEAN:

| seed | R−C copy Δ | V−C copy Δ | R−V copy Δ |
|---:|---:|---:|---:|
| 43022 | +0.500 | +0.327 | +0.173 |
| 43122 | +0.668 | +0.358 | +0.310 |
| 43222 | +0.559 | +0.268 | +0.291 |

The advantage is content- and structure-dependent: it appears on natural source-repeat packets but not random token spans. The raw gain suggests VIEW retains some source-present benefit, but compact mixture model and predictions/independent_review showed this statement needs component decomposition: at seed43022 VIEW's repeated-source NLL is nearly identical to CLEAN and its raw gain advantage largely comes from worse no-source fit. Therefore VIEW's copy competence should be treated as an open component-level question until decomposed across seeds.

RoBERTa's copy contrast is near zero (R−C ≈ −0.049) with checkpoint sign changes and should be treated as unresolved at its learner phase and resolution.

## 2. Behavioral face: Entity retrieval vs. update crossover

On the BabyLM Entity Tracking task, where success requires maintaining the state of a queried entity across operations, three DeBERTa seeds show:

- **R−C at zero relevant updates**: +8.78, +9.09, +3.11 (all positive)
- **R−C at ≥3 relevant updates**: −3.61, −1.30, −4.50 (all negative)

This crossover is the behavioral expression of the cost center: practiced identity retrieval helps when the queried state has not changed, but can hurt discrimination once the state requires updating beyond the initial mention. Entity is directionally robust but magnitude-variable across seeds; the held-out copy/rewrite probes are the primary installed-competence measurements.

## 3. Source-token mass and the mechanism question

At the same held-out nonoverlap targets, the initial all-token readout showed REPEAT placing more probability mass on source-span token types than CLEAN: +0.103, +0.142, +0.093 across DeBERTa seeds. Because CLEAN already assigns substantial mass to source-token sets dominated by common words, relation practice principle ran the decisive content-token source-specificity control.

The content-token 2×2 result confirms that the mass shift is triggered by the recognized true source, not frequency hedging. R−C true-source content-mass deltas are +0.0727, +0.0972, and +0.0550 under the true-source window condition, but only +0.0054, +0.0083, and +0.0050 when an unrelated source occupies the window. R−C unrelated-window content mass is near zero, and target probability is suppressed only under the true-source condition. Thus the active cost is computationally source-specific: exact recurrence installs a source-recognition/copy readout that fires when related content is in the window and competes with nonidentical target prediction.

VIEW has a different readout: it also raises true-source content mass under the true-source condition, but it raises rather than suppresses the correct nonoverlap target probability. The dissociation is therefore identity readout versus content readout, not merely source attention versus no source attention.

## 4. The explanatory hypothesis: shared routing capacity

The measurements are consistent with the hypothesis that a learner with shared routing capacity installs whichever cross-span relation minimizes loss on the pairs it practices, then applies that relation when contextual similarity triggers it at evaluation:

- Exact recurrence makes **identity** the practiced relation. The model learns to route attention/output mass toward tokens matching the source span. This helps exact retrieval (copy benefit, Entity zero-update) but misfires when the answer requires a *nonidentical* transformation of the source content, pulling probability toward source tokens instead of the correct nonidentical target.
- Varied restatement retains high surface overlap (~82% of source tokens preserved in VIEW rewrites) while changing enough content to make identity alone insufficient. The model must learn content-conditioned correspondence — using source content to predict modified content. This simultaneously practices both identity (for overlap tokens) and content transformation (for nonoverlap tokens), explaining VIEW's intermediate copy competence combined with strong content-conditioning.
- CLEAN has no companion in the same window, so neither identity nor content-correspondence relation is locally practiced.

**This is a hypothesis, not an established mechanism.** The measurements establish the relation-specific competence pattern; the routing explanation accounts for why exact recurrence actively degrades nonidentical use (identity routing fires and competes) rather than merely failing to teach it. The hypothesis becomes testable through the split controls and the source-specificity measurement.

## 5. The split control: Branch A confirmed for the cost side

The split arms (REPEAT_SPLIT, VIEW_SPLIT) preserve the same source/companion text, total token budget, and suffix/filler material, but place source and companion in separate training rows so they never co-occur in one training window.

**REPEAT_SPLIT result (relation practice principle)**: Both the copy benefit and the recurrence cost are eliminated.

| quantity | original R−C range (3 seeds) | REPEAT_SPLIT−CLEAN |
|---|---:|---:|
| token-nonoverlap rewrite gain | [−1.043, −0.752] | **−0.031** |
| held-out natural copy gain | [+0.500, +0.668] | **−0.201 raw; −0.031 normalized vs CLEAN** |

The T/U decomposition is decisive: the original sign reversal (T worse +0.45, U better −0.30) is completely eliminated in the split arm (T better −0.55, U better −0.58). Cross-row exposure produces a generic LM improvement; the source-specific copy misfire requires same-window co-occurrence.

A second finding appears when REPEAT_SPLIT is compared directly with original REPEAT rather than CLEAN. They have the same selected source/repeat tokens and the same budget, but REPEAT_SPLIT is much better on token-nonoverlap compact rewrites: true-source NLL is lower by about 0.995 nats and unrelated-source NLL by about 0.274 nats over 80M/90M/100M. In-window exact recurrence therefore both installs the misfire and reduces residual prediction work from the repeated content itself: when an exact copy sits in the same MLM window, many masks can be solved by copying, whereas spaced cross-row repetition preserves a harder ordinary prediction signal.

The raw RS−C copy gain is negative because RS has a lower unrepeated-control NLL than CLEAN; after normalizing gain by each arm's own control NLL, RS−C is only −0.0307, while original R−C is +0.0497 and RS−R is −0.0804. Therefore the defensible copy statement is that the large original REPEAT copy advantage requires in-window co-occurrence, not that split repetition has intrinsically worse copy ability than CLEAN.

**Branch A is confirmed for the recurrence-cost side and for the original copy advantage**: same-window relation practice is the causal mechanism for source-specific identity misfire; spaced repetition of identical text at this dose can improve ordinary content fit without producing the local cost. The principle for the recurrence cost is:

> *Under a fixed experience budget, the relation between spans inside a training window determines which cross-span computation is installed. Exact identity between co-occurring spans installs a copy computation that competes with content-conditioned use. Partial overlap (restatement) practices both relations, balancing copy and correspondence.*

The training implication is that **the window is the unit of relation practice**: what matters for data efficiency is not token exposure count but how related spans are structured *within* each context window. Variation sets (consecutive partially overlapping utterances in child-directed speech) would be the natural instance of this principle.

### Branch B: Split result inside original range

If duplicated-token budget without in-window co-occurrence reproduces the original effects, then **cross-row repeated exposure suffices**. The principle must be restated:

> *Under a fixed experience budget, repeated versus varied token exposure adjusts the model's default output distribution toward identity versus content-conditioned use. The mechanism operates through weight-level learning, not attention-mediated in-context relation practice.*

The window structure would then be irrelevant; what matters is token-type frequency and context diversity at the epoch level. This would align more closely with Hernandez et al.'s repeated-data findings and Allen-Zhu & Li's augmentation work, narrowing the novelty to the active-cost demonstration.

### Branch C: Intermediate result

If split effects are attenuated but not eliminated, then **both exposure and local relation practice contribute**. This is the most likely outcome for at least one side. The principle should state partial window-dependence:

> *Under a fixed experience budget, duplicated or varied token exposure shifts the weight-level output distribution, while in-window co-occurrence amplifies the effect by locally practicing the relation between spans. The recurrence cost is partially driven by in-window identity practice and partially by broader repeated-token exposure.*

The degree of attenuation determines the relative contribution of each mechanism and whether the practical training advice emphasizes window-level relation construction or corpus-level deduplication.

## 6. The training principle (contingent on split outcome)

Assuming Branch A (or the A-leaning portion of Branch C):

**Data-efficient learning under finite budgets should structure experience so that related spans co-occur in training windows with partial rather than exact overlap.** Specifically:

1. **Avoid exact in-window recurrence** of content spans. Each window's repeated exposure to identical text trains identity routing that competes with content-conditioned use.
2. **Varied restatement** — re-expressing the same content in partially overlapping form within the same window — is preferable because it practices both copy (on overlapping tokens) and content correspondence (on non-overlapping tokens).
3. **Surface overlap** between related spans is the graded variable. At 100% overlap (identity), only copy is practiced. At partial overlap (~50–80%), both relations are practiced. At near-zero overlap, neither relation connects the two spans.
4. **Variation sets** in child-directed speech exemplify this structure naturally: consecutive caregiver utterances share communicative intent and many words while varying form and content. The prediction is that variation-set frequency improves not because it increases diversity *per se*, but because it practices content-conditioned relations at partial overlap within the infant's processing window.

This connects the BabyLM finding to a general principle about data-efficient learning: *the efficiency of a fixed experience budget depends not only on what content is presented but on how related content is structured within the learner's attention window.*

## 7. What remains unresolved

1. **VIEW_SPLIT** (running at this stage): Determines whether VIEW's positive content-conditioning benefit also requires in-window source/rewrite co-occurrence or can arise from cross-row paraphrase exposure.
2. **Second split seeds**: REPEAT_SPLIT seed43122 was training; VIEW_SPLIT seed43122 was planned. These make locality a seed-stable spine rather than a one-seed causal result.
3. **Natural variation-set probe** (in progress): Tests whether the competence pattern transfers from synthetic compact rewrites to natural child-directed speech and whether surface-overlap bins produce graded identity/content readout weights.
4. **Architecture generality**: RoBERTa shows a weaker but aligned cost; further architectures and objectives (causal LM) are useful as boundaries once the split results are integrated.
5. **Lexical controls**: Strict word filtering preserves the cost but with attenuation; target-absent unrelated sources, shuffled sources, and lexical decoys would strengthen the claim.
6. **Mechanism below the output readout**: Source-specificity establishes recognized-source output competition, but the internal pathway—attention routing, representation, or readout-layer coupling—still requires activation or intervention work.

## Files

- seed43222 entity clean integration misfire data: `data/source_token_misfire_mass/`
- mechanism floor and behavioral refinement synthesis: `notes/mechanism_floor_and_behavioral_refinement.md`
- seed43222 entity clean integration split prestatement: `notes/split_control_numeric_prestatement.md`
- pair level relation robustness decomposition: `notes/principle_center_after_decomposition.md`
- seed43222 threearm probe decomposition three-arm probe: `notes/seed43222_threearm_probe_decomposition.md`
- Entity refinement: `data/mechanism_floor_integration/entity_r_minus_c_by_relevant_updates.csv`
