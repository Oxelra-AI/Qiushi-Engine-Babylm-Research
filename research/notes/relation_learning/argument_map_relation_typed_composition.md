# Argument Map: Relation-Typed Composition Under Fixed Budget

This argument map reframes the
stable DeBERTa relation-locality evidence as **relation-typed window
composition**, positioning it against the direct literature neighbor (In-Context
Pretraining, Shi et al. 2023) and the deduplication literature (Lee et al. 2022).

## 1. The Positioning Triangle

Three existing lines of work manipulate the relationship between pretraining
text and context windows, but none measures the relation-type dependence:

**In-Context Pretraining (ICLM)** \cite{shi2023context}: Groups semantically
related documents in the same context window while keeping corpus content and
training budget fixed. Reports +8% ICL, +15% reading comprehension, +16%
context faithfulness. Crucially, ICLM **explicitly filters near-duplicates**
using retrieval scores before composing windows (§2.1): "we found that the
pretraining corpus contains many near duplicate documents... we further leverage
the retrieval scores to eliminate near duplicate documents from the pretraining
corpus." Their ablation (Figure 5) shows removing deduplication degrades PPL
from 7.3 to 8.3, and they explain: "When near duplicate documents are present
in the same context, language models might merely copy from the prior document,
leading to training instability."

**Corpus-level deduplication** \cite{lee2022deduplicatinga}: Removes repeated
substrings (suffix array, ≥50 BPE tokens) and near-duplicate documents (MinHash,
edit similarity >0.8) from the training corpus before any window arrangement.
Leaves perplexity unchanged or improved, reduces verbatim generation from 1% to
0.1% of tokens. Operates at corpus/document level, does not study window
composition or ordering.

**Rephrased/augmented pretraining** (WRAP, Allen-Zhu & Li): Adds paraphrased or
synthetic restatements to the training data, generally reporting downstream
improvements. These change the corpus content or size and are not fixed-budget
composition manipulations.

**What none of these can see:** Whether the sign and kind of installed
cross-span computation depends on the *relation type* practiced between
co-occurring spans. ICLM removes near-duplicates while composing related documents, but its ablation
confounds corpus-level duplicate removal, in-window adjacency, and training
stability. Our work supplies the matched-content instrument and T/U/N readout
that ICLM lacked: it isolates a window-internal component and shows that the
practiced relation type changes the installed source-conditioned computation.

## 2. Central Result: Relation-Typed Composition with Asymmetric Reach

Under a fixed experience budget, the relation practiced between spans sharing a context window determines which cross-span computation is installed, and different installed routines travel different distances. In the designed compact intervention family, C/R/V/RS/VS share the inherited COMPACT_EXPERIENCE/REPRESENTATION_FRONTIER_STUDIES qwen-clean substrate; CLEAN is the reference for added compact pairs, not a relation-free corpus. Use the neutral-anchor polarity `A_T=N−T` and `A_U=N−U`, where positive `A_T` means better true-source use relative to a neutral ordinary source.

| Relation practiced | Compact nonoverlap ΔA_T=N−T | Compact nonoverlap ΔA_U=N−U | Effect on source use |
|---|---:|---:|---|
| Exact recurrence (REPEAT) | **−0.8138 ± 0.1729** | +0.0679 ± 0.0340 | Weakens changed-form true-source support while helping exact/unchanged retrieval |
| Compact restatement (VIEW) | **+0.7970 ± 0.0758** | −0.0085 ± 0.0536 | Strengthens true-source use in the matching compact register |
| HALF_VIEW, seed43022 | +0.0177 | −0.1505 | The 16.7K compact-restatement subset did not install the full compact source-use routine |
| No added compact companion above qwen-clean substrate (CLEAN) | 0 (designed-family reference) | 0 (reference) | Baseline context use plus inherited local original--rewrite practice |
| Split same content across rows (REPEAT_SPLIT/VIEW_SPLIT) | → near CLEAN | → near CLEAN | Source-specific residual strongly attenuated |

T = NLL with true related source in context; U = NLL with unrelated compact source; N = NLL with length-matched ordinary held-out text. `A_T` isolates true-source-specific use from general compact-context fit.

The designed-family sign reversal is the core local result. Same added compact dose and budget, same window position: exact recurrence lowers `A_T` on changed-form compact targets, whereas compact restatement raises it. The unrelated-source terms are near zero compared with true-source terms, so the effect is specifically about the related source rather than any compact neighbor. Because all designed-family arms inherit `qwen_pair_packed`, this is a marginal relation-composition effect above an already relation-practicing substrate rather than a comparison against a corpus with no aligned restatement practice.

## 3. The Causal Manipulation: Locality

Row-split controls preserve all selected source/companion text, suffix/filler
content, row-length sequence, and 100M-word budget, but move source and
companion to separate rows. This removes within-window co-occurrence while
keeping content exposure identical.

**Two-seed DeBERTa compact nonoverlap rewrite gain:**

| Seed | R−C | RS−C | V−C | VS−C |
|---|---|---|---|---|
| 43022 | −0.75 | −0.03 | +0.68 | +0.03 |
| 43122 | −1.04 | −0.19 | +0.89 | −0.04 |

At seed43122, T/U decomposition confirms the split arms improve both T and U
vs CLEAN, eliminating the original arms' source-specific residuals rather than
merely shrinking a gain statistic. The large source-specific effects require
within-window relation practice.

## 4. The Measured Recurrence Cost — The In-Window Component ICLM Did Not Isolate

ICLM's observation that "near duplicate documents in the same context" cause
"training instability" and hurt perplexity is a phenomenological report under
a corpus-level duplicate-removal and composition pipeline. Our T/U/N
decomposition does not explain that broad PPL loss by itself; it isolates a
targeted in-window component at BabyLM dose:

- REPEAT installs a **recognized-source content pull**: source-content mass
  increases 11–14× more under the true related source than under an unrelated
  source (relation practice principle source-specificity 2×2). Target probability is suppressed
  only in the true-source condition for nonoverlap rewrite targets.
- This source-recognition routine **weakens use of the same content for
  nonidentical prediction**: Δ(T−N) = +0.81 ± 0.17 nats means the true source
  helps less than in CLEAN, not that it hurts absolutely, but the relative loss
  is real and source-specific. Direct target-probability summaries show the
  precise answer-copy readout is format-bound: in nonidentical rewrite contexts
  R−C lowers source-conditioned gain even on overlap targets where the answer
  token appears in the source, and lowers true-source target probability on
  nonoverlap targets.
- The unrelated-source term is neutral (Δ(U−N) ≈ −0.07), so the cost is not
  generic context sensitivity but specifically a mismatch between the practiced
  identity relation and the nonidentical evaluation targets.

This is not yet the measured cause of ICLM's broad PPL degradation. It is the
component that their design cannot separate: when related spans are exact
near-recurrences within a window, the model practices a source-recognition
routine that can compete with related nonidentical prediction. Whether ICLM's
broad harm comes from corpus duplication, the same component at a much higher
adjacency fraction, or instability remains a dose/generalization question.

## 5. Behavioral Face: Entity Tracking (REPEAT Side)

The compact mechanism has a behavioral downstream correlate in BabyLM Entity
tracking, but only for the REPEAT side:

- Across 3 original DeBERTa seeds, REPEAT beats CLEAN at zero relevant
  queried-entity updates (+3.1 to +8.8 points) but falls below at deep
  updates (rel≥3: −1.3 to −4.5).
- Splitting removes this in both evaluated split seeds: seed43022 RS−C at zero
  updates is −0.54 (vs R−C +8.78), and seed43122 RS−C is −2.57 (vs R−C
  +9.09). Deeper-update REPEAT costs are also reduced or reversed by splitting.
- VIEW_SPLIT is not a symmetric behavioral face. Seed43022 retained much of
  VIEW's positive-update Entity behavior, but seed43122 did not (rel_ge3 VS−C
  −3.25 while V−C +5.56). The VIEW Entity behavior should not be used as proof
  of the compact restatement residual.

The Entity crossover is a grounded behavioral instance of the compact mechanism:
exact recurrence helps when the queried state is unchanged (identity retrieval)
but hurts discrimination when it has changed (nonidentical use).

## 6. The Fixed-Budget Price is Narrow

The price of within-window pairing is redistributed within the compact companion
family, not broad language degradation:

| Measurement | R local−split | V local−split |
|---|---|---|
| Compact N anchor (seed43022 local−split) | +0.30 nats | +0.27 nats |
| Ordinary held-out MLM (6,992 rows, seed43022 local−split) | +0.021 nats | +0.020 nats |
| Ordinary held-out total vs CLEAN | R−C +0.107 = RS−C +0.086 + R−RS +0.021 | V−C +0.099 = VS−C +0.079 + V−VS +0.020 |

Same-window pairing reallocates prediction work within the companion/rewrite
subdistribution while purchasing a relation-specific computation. It does not
measurably degrade ordinary held-out language modeling at this scale.

## 7. Asymmetric Evidence Maturity

The argument is deliberately asymmetric:

**REPEAT side (strongest):**
- Compact T/U/N: 3 seeds, opposite sign from baseline, source-specific
- Source-output mass: 3 seeds, 11-14× true-source specificity
- Natural copy: 3 seeds, REPEAT > VIEW > CLEAN
- Entity behavioral face: 3-seed crossover, zero-update advantage causally localized by splitting in two seeds
- Two-seed split locality replication

**VIEW side (compact-strong and naturally form-robust for source-recurring restatement tokens):**
- Compact T/U/N: 3 seeds, opposite sign from REPEAT
- Two-seed split locality replication
- Wikipedia/Simple-English overlap targets show natural transfer of VIEW source use: V−C gain_T_vs_N is +0.238 ± 0.071 and V−R is +0.129 ± 0.053, with V−C gain_U_vs_N near zero (+0.007 ± 0.016)
- Wikipedia nonoverlap V−C gain_T_vs_N is small (+0.0316 ± 0.0398), while V−R remains positive (+0.249 ± 0.065) because REPEAT is actively below CLEAN on substituted/source-absent target tokens
- Entity face is not symmetric across split seeds

The reach pattern is now sharper than a two-relation contrast. Exact recurrence has a broad positive side for exact/unchanged targets and a broad liability for changed-form/source-absent targets: designed DeBERTa R−C compact nonoverlap `ΔA_T=-0.8138±0.1729`, COMPACT_EXPERIENCE duplicate-vs-official compact nonoverlap `ΔA_T=-1.9263`, COMPACT_EXPERIENCE duplicate-vs-official Wikipedia nonoverlap `Δ(N−T)=-1.1840±0.0659`, and causal-GPT R−C compact nonoverlap `ΔA_T=-1.1532` with R−RS `ΔA_T=-1.2599`. Restatement competence travels less freely: designed VIEW is large on the matching compact readout but modest on Wikipedia overlap, while COMPACT_EXPERIENCE aligned Qwen is huge on near-register Wikipedia source-recurring targets in two seeds and near-zero or negative on compact nonoverlap. The asymmetry is therefore not “VIEW fails to transfer”; it is a reach rule: exact recurrence liability generalizes farther than the positive competence installed by a corresponding restatement relation, whose use remains tied to practiced register, format, and target form.

## 8. Relation to the Goal

The research goal asks for a "data-efficient learning principle" that explains
how "limited data and limited training budget form reusable, composable,
generalizable, and transferable knowledge."

The relation-typed composition principle gives a specific answer for one axis:

**Under a fixed experience budget, the kind of cross-span source-use routine a model
learns depends not only on what selected source family is available but on what
relation between co-occurring spans the model practices.** R versus RS and V
versus VS preserve the relevant source/companion content and budget while changing
window adjacency, and the large source-specific residuals are strongly attenuated
under splitting. R versus V are not same-content arms because their companion
surfaces instantiate different relations.

This has actionable implications:
- **For sequence composition:** compose related nonidentical text when the desired competence is form-robust source use, and avoid exact near-duplicate adjacency when later use requires changed or substituted wording; keep separate the unmeasured causes of corpus-level duplicate harm in ICLM
- **For data selection:** the value of an experience pair depends on whether it
  trains identical-surface retrieval, changed-form restatement use, or a mixed
  local diet whose readouts may reflect both relation dose and relation competition
- **For curriculum design:** the same content can serve opposite training
  purposes depending on how it is arranged and which target forms receive
  repeated reinforcement

## 8A. SOTA Ingredient as a Relation-Typed Design

The inherited COMPACT_EXPERIENCE clean-Qwen ingredient is itself a relation-composition object.
It contributes 37,594 filtered original--Qwen-rewrite pairs, 1,656,800 words per
10M pool, mean content overlap about 0.63, and preserved pair boundaries. The
clean natural nearduplicate adjacency refined count found about 22,117 high-overlap local pairs in
`qwen_pair_packed` per 10M, so qwen-clean is already a high-dose local
restatement substrate.

COMPACT_EXPERIENCE also trained five zero-new-training arms that instantiate relation type on
this SOTA ingredient at seed43022:

| Arm | Relation practiced inside or outside the window | Use in current mechanism readouts |
|---|---|---|
| `selected_original_dup_all` | selected originals duplicated locally; no Qwen words | exact-recurrence analogue after verifying in-window duplication |
| `qwen_clean_aligned` | each selected original paired with its own Qwen rewrite locally | own restatement at the SOTA ingredient dose |
| `qwen_shuffled_control` | selected original plus wrong Qwen rewrite locally, with original/rewrite multisets preserved | separates correspondence from mere rewrite-register adjacency |
| `qwen_separated_pair` | selected originals and their Qwen rewrites both present but placed in different rows/windows | lower-locality coexistence point; row-count mismatch requires N and ordinary-heldout checks |
| `official_lengthmatched` | qwen block replaced by official filler under matched recipe | no selected Qwen-pair practice at this dose |

This turns the old qwen-clean SOTA result into a direct test of the same principle rather than an analogy, but the COMPACT_EXPERIENCE ingredient and the designed compact intervention are different registers. The compact FineWeb-register T/U/N probe asks whether the qwen restatement routine transfers across register into the compact probe family. The Wikipedia/Simple-English probe is closer to the qwen substrate because 10,822 selected qwen pairs are from `simple_wiki`; the observed large Wikipedia readout with a small compact readout means the installed restatement use is register-shaped, not absent. Official-style aggregate scores from COMPACT_EXPERIENCE should not be used for this mechanism comparison because AoA accounting differs across these arms; the current readouts are compact T/U/N, Wikipedia target classes, natural copy, Entity relevant-update structure, N anchors, and ordinary held-out loss.

causal gpt relation boundary/032 results now make the COMPACT_EXPERIENCE crossing durable for ALN--OFF: aligned Qwen versus official-only gives Wikipedia overlap `Δ(N−T)=+3.5975±0.1236` at seed43022 and `+4.1998±0.1379` at seed43122, while compact nonoverlap `ΔA_T` is only `+0.0592` and `−0.1931`; ordinary held-out loss is slightly better for ALN in both seeds (`−0.0125` and `−0.0128`). SHUF and DUP are seed43022-only but supply the missing relation contrasts there: SHUF keeps local rewrite adjacency while breaking correspondence and is below OFF on Wikipedia overlap (`−0.9942±0.0682`) with small broad-fit movement, while DUP is exact-recurrence-like, helping copy/overlap targets and hurting compact and Wikipedia source-absent targets.

The inherited Qwen block is not a point on the designed-register compact dose curve. For compact nonoverlap, all designed C/HV/V/HM arms share the inherited qwen-clean substrate; the added compact-restatement curve is C (0 added compact pairs) → HALF_VIEW (16,731 added compact pairs) → VIEW (33,291 added compact pairs), with hash-mix off-axis because it combines the HALF_VIEW rewrite subset with 16,560 exact-recurrence companions. frequency concentration probe places HALF_VIEW near C on compact nonoverlap `A_T` (`HV−C=+0.0177`) and full VIEW far above C (`V−C=+0.7104`). Hash-mix sits between them (`HM−C=+0.3456`) while carrying exact-copy behavior and a Wikipedia source-absent cost; it is a mixed-relation composition point, not a clean restatement-dose point.

## 9. Load-Bearing Figure Plan

**Figure 1 (Central):** Three-seed true-source advantage `A_T=N−T` and unrelated-source advantage `A_U=N−U` for R/V/C, with two-seed split RS/VS overlaid. Use the numerical repair and halfview prestate polarity where positive means better true-source use. This single figure shows:
- Relation-type sign dependence (R lower than C, V higher than C on `A_T`)
- Near-zero `A_U` contrasts compared with the true-source terms
- Split arms strongly attenuated toward zero (causal locality)
This is the load-bearing figure because it encapsulates the measured principle without relying on sign-flipped Δ notation.

**Figure 2 (Seed-level locality):** Two-seed compact nonoverlap gain plus T/U
components. Shows the causal manipulation at individual-seed level. Side panels
for T and U components make the source-specificity visible.

**Figure 3 (Entity behavioral face):** Entity accuracy by relevant-update
group for R/C/RS at seeds 43022 and 43122. Shows unchanged-state benefit for
local REPEAT and its removal under splitting; VIEW/VS appears only as a caution
that the positive-update face is not symmetric across seeds.

**Figure 4 (Source-output mechanism):** Target probability and source-content
mass in true-source vs unrelated-source contexts for R−C and V−C, paired with
the overlap/nonoverlap target-class readout. Shows the output-level
phenomenology: a transferring source-recognition trigger with non-transferring
precise readout for REPEAT, versus content-conditioned support for VIEW.

**Figure 5 (Natural-domain form robustness):** Wikipedia/Simple-English arm ×
target-class panel: source-recurring targets, source-absent targets, class-balanced mean, and token-weighted mean for R−C, V−C,
and V−R. This is the natural-domain figure because it shows the relation-arm
ordering depends on target surface relation: exact repeats favor REPEAT in the separate natural-copy probe; source-recurring restatement tokens favor VIEW in Wikipedia; substituted/source-absent tokens penalize REPEAT while VIEW stays near CLEAN.

**Figure 6 (Price narrowness):** Compact N vs ordinary held-out local-vs-split
gaps. Shows the price is compact-family redistribution, not broad degradation.

**Figure 7 (Positioning):** Schematic showing ICLM → our work → actionable
principle. ICLM manipulates composition, we add relation-type measurement,
together this gives a grounded sequence-composition rule.

## 10. Integrated Boundaries and Open Work

| Result | Current reading |
|---|---|
| COMPACT_EXPERIENCE aligned Qwen vs official-only | Two-seed near-register source-recurring gain with compact nonoverlap near-null: ALN--OFF Wikipedia overlap `+3.60/+4.20`, compact nonoverlap `+0.059/−0.193`, ordinary held-out `−0.0125/−0.0128`. This is the main reach crossing. |
| COMPACT_EXPERIENCE SHUF and DUP | Seed43022-only relation controls. SHUF supports practiced non-correspondence in the near-register source-recurring readout but not compact nonoverlap; DUP reproduces exact/unchanged benefit plus changed-form/source-absent cost. |
| COMPACT_EXPERIENCE SEP | Useful lower-locality coexistence point but broadly degraded (`SEP−OFF` ordinary held-out `+0.1384`), so it should not carry the causal comparison. No immediate Qwen split training is needed. |
| HALF_VIEW and hash-mix | HALF_VIEW is near C on compact nonoverlap `A_T` (`+0.0177`) while full VIEW is large (`+0.7104`). Hash-mix is off-axis (`+0.3456`) and combines copy behavior with incomplete compact restatement competence plus Wikipedia source-absent cost. |
| RoBERTa REPEAT_SPLIT | Compact recurrence true-source deficit collapses strongly under splitting in one RoBERTa seed, supporting recurrence-cost locality beyond DeBERTa; exact-copy behavior remains learner/phase dependent. |
| Causal GPT C/R/RS | Exact-recurrence changed-form cost and locality survive next-token pretraining on the compact readout (`R−C ΔA_T=-1.1532`, `R−RS=-1.2599`), while natural-domain effects still depend on target form and left-context deployment. |
| Token-frequency concentration | DeBERTa/RoBERTa/GPT coordinates tie input/output rows, but COMPACT_EXPERIENCE ALN--OFF Wikipedia effects do not correlate with simple target-token frequency or marginal ALN--OFF frequency change. |

The integrated results strengthen rather than overturn the DeBERTa relation-locality spine, while revising the transfer statement into a reach-asymmetry principle. RoBERTa N-anchor scoring and deeper contextual-source-recognition probes remain useful, but the central argument no longer depends on a new Qwen split or on treating inherited Qwen as a compact-dose point.

## 11. Independent Contributions vs. Prior Work

What this study has that ICLM, deduplication, and rephrased pretraining lack:

1. **Matched-content causal controls**: Same selected content, same budget, same
   model, different only in within-window arrangement. ICLM changes document
   ordering globally and evaluates on downstream tasks; we isolate the window-
   internal relation.

2. **The T/U/N decomposition**: Separates true-source use, unrelated-source
   sensitivity, and neutral ordinary-text fit. ICLM reports perplexity and
   downstream accuracy; we measure the installed source-conditioned computation.

3. **Relation-type sign reversal**: Exact recurrence and nonidentical
   restatement produce opposite-sign compact effects on Δ(T−N) within the same
   experimental framework. ICLM treats near-duplicate filtering as a design
   requirement but does not isolate the window-internal relation component;
   corpus deduplication treats repeated content outside the composition question.

4. **Source-specific output mass**: The mechanism goes beyond aggregate scores
   to show how the model's output distribution changes with the relation type.

5. **The active recurrence cost**: Not just "duplicates don't help" but
   "practicing exact recurrence within a window installs a source-recognition
   routine that can actively degrade use of the same related content for
   nonidentical targets below the no-companion baseline."

6. **Fixed-budget price anatomy**: The cost is concentrated in the companion
   subdistribution, not broad language modeling, providing a specific account of
   where finite prediction work is reallocated.

## Evidence carrier files

See `notes/relation_practice_synthesis_spine.md` for the complete
file inventory. The key additions for the ICLM positioning:
- `Knowledge/objects/papers/In-context-Pretraining-Language-Modeling-Beyond-Document-Boundaries--2eba432a0d4c--f03ba11a8c64/object.md` — \cite{shi2023context}
- `Knowledge/objects/papers/Deduplicating-Training-Data-Makes-Language-Models-Better--0c8024957129--d6d539a355b3/object.md` — \cite{lee2022deduplicatinga}
