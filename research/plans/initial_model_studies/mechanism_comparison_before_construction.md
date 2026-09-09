# earlier analysis — Mechanism comparison before construction

## Experimental constraints

The route reset away from BSM must not be shrunk into ordinary "high-quality text row selection." Before any training investment, the next route must identify:

1. A transferable learning mechanism hypothesis.
2. Why that mechanism would produce compositional generalization and multi-column synergy.
3. How to distinguish "text was easier to fit" from "learning produced more reusable/composable representations."

If the mechanism is only "cleaner text → lower loss → hope for better scores," it does not meet the standard for the new scientific mainline.

## Three genuinely different experience-organization mechanisms

### Mechanism A: Syntactic-compositional completeness pressure

**Hypothesis:** Under WWM, the model's prediction problem is fundamentally different on compositionally complete text vs fragments. On a complete sentence like "The tall boy who found the red ball threw it to his sister," the masked prediction of "threw" requires integrating a relative clause, a subject, and an object structure. On a fragment like "yeah okay so uh," the prediction requires only local co-occurrence. By concentrating the 10M-word budget on rows with higher compositional structure density, we create more prediction problems that require syntactic composition to solve, directly training the composition machinery that BLiMP, Supplement, COMPS, EWoK, and GlobalPIQA test.

**Why compositional generalization:** The core columns are themselves tests of compositional ability — subject-verb agreement across intervening clauses (BLiMP), semantic acceptability of composed meanings (Supplement/COMPS), inference from composed world-knowledge statements (EWoK), and physical reasoning from composed descriptions (GlobalPIQA). Training on text that exercises composition directly trains the skill the evaluation tests.

**Why multi-column:** All NLP evaluation columns require composing meaning from parts. A model that better represents syntactic composition should improve broadly, not only on one column.

**Control against "easier fitting":** Compositionally complete text is NOT easier to fit than fragments — it is harder because the prediction depends on long-range structure. A fragment-heavy arm should have LOWER loss but WORSE BLiMP/Supplement/COMPS. If we observe that the compositionally complete arm has higher loss but better multi-column scores, we have evidence for the mechanism (composition pressure) rather than the confound (ease of fitting).

**Observable prediction:** clarity_selected arm has higher final loss than fragment-heavy arm but better BLiMP+Supplement+COMPS+EWoK scores. If both loss and scores are better, we cannot distinguish mechanism from confound without further controls.

### Mechanism B: Predictive diversity pressure (information-theoretic)

**Hypothesis:** Text with higher lexical/semantic diversity per word-budget creates more diverse prediction targets, preventing the model from relying on local bigram/trigram statistics and forcing it to use compositional structure and world knowledge to succeed at WWM. Under this mechanism, selecting text that maximizes type-token diversity and conceptual coverage per word budget produces broader generalization.

**Why compositional generalization:** When local statistics are unreliable (because the vocabulary is diverse and repetition is low), the model must rely on structural regularities (grammar rules, semantic composition) rather than memorized co-occurrences. This is why diverse language exposure is believed to help human language acquisition.

**Why multi-column:** Diverse exposure means the model encounters more distinct facts (EWoK, GlobalPIQA), more diverse grammatical constructions (BLiMP, Supplement), and more conceptual distinctions (COMPS).

**Control against "easier fitting":** High-diversity text is harder (higher entropy targets). If the diverse arm has higher loss but better multi-column scores, we have mechanism evidence.

**Observable prediction:** Same as Mechanism A — high-diversity arm should have higher loss and better multi-column scores. Importantly, this should be distinguishable from Mechanism A by using text that is diverse but fragmentary (e.g., diverse vocabulary dialogue fragments vs. complete but repetitive narrative).

### Mechanism C: Progressive exposure ordering (true curriculum)

**Hypothesis:** The protected DeBERTa-v2 builds representations bottom-up: early training forms token/bigram associations, then syntactic patterns, then semantic/world-knowledge patterns. If the training order is aligned with this developmental trajectory — shorter/simpler/more predictable text first, longer/more complex/less predictable text later — the model builds stronger lower-level representations before attempting higher-level composition. Misaligned order forces simultaneous learning of all levels, leading to less stable representations.

**Why compositional generalization:** Compositional representations require stable building blocks. A curriculum ensures each compositional level is learned before it is required as a component of the next level.

**Why multi-column:** Early syntax consolidation helps BLiMP/Supplement; later semantic complexity exposure helps EWoK/COMPS/GlobalPIQA.

**Control against "easier fitting":** A curriculum arm has the SAME total text content as a shuffled arm (same rows, just reordered). If curriculum improves scores over shuffled with identical content, the effect is pure ordering, not content selection.

**Observable prediction:** curriculum_ordered minus content_matched_shuffled > 0 on multi-column proxy. Loss should be lower early and higher late; endpoint loss similar; scores different.

**Prior evidence:** The earlier experiments tested a short-to-long length schedule early (wwm cross seed stability interpretation-46) and it was harmful (Supplement -3.00, COMPS -0.61). BUT that was length-only ordering on BERT, not complexity/predictability ordering on the protected DeBERTa. The leader's name includes "curriculum" — this route has not been tested on DeBERTa under the protected recipe with a better ordering criterion.

## Comparison and selection

| Property | Mechanism A | Mechanism B | Mechanism C |
|---|---|---|---|
| Changes content? | Yes (selects rows) | Yes (selects rows) | No (reorders same rows) |
| Changes training dynamics? | Indirectly | Indirectly | Directly |
| Prior experimental evidence against? | No (not tested) | No (not tested) | Partial (length-only on BERT was negative) |
| Confound with "easier fitting"? | Controlled: complete sentences are harder | Controlled: diverse text is harder | None: same content |
| Leaderboard support? | Implicit (leader uses "simplification" which often means clearer/complete) | Indirect | Direct (leader name includes "curriculum") |
| Smallest experiment? | 3-arm 4M screen | 3-arm 4M screen | 2-arm 4M screen (same content) |

## Which mechanism should be the scientific mainline?

**Mechanism C (curriculum ordering) has the cleanest experimental design** because it uses the same content and separates ordering from everything else. It also has the most direct leaderboard motivation. It failed once on BERT with length-only ordering, but that was a primitive criterion on a weaker backbone.

**Mechanism A (compositional completeness) has the strongest mechanistic argument** for WHY multi-column improvement would occur, but it changes content, which makes attribution harder.

**A 2×2 design combining A and C is most informative:**

| | Shuffled | Curriculum-ordered |
|---|---|---|
| **Official flat (all rows)** | Arm 1: official_flat_shuffled (existing reference) | Arm 2: official_flat_curriculum |
| **Compositionally selected** | Arm 3: composition_selected_shuffled | Arm 4: composition_selected_curriculum |

This separates:
- Arm 3 minus Arm 1 = pure composition-selection effect (content change, no order change)
- Arm 2 minus Arm 1 = pure curriculum-ordering effect (order change, no content change)
- Arm 4 minus Arm 1 = combined effect
- Arm 4 minus Arm 3 = curriculum effect on top of selected content
- Arm 4 minus Arm 2 = selection effect on top of curriculum order

**This is the right design.** Four arms, same DeBERTa-v2 8×480 WWM recipe, same seed, same word budget, at 4M first. If any arm improves weighted multi-column proxy, scale to full budget.

## Mechanism A operationalization: compositional completeness score

NOT "clarity" or "readability." The score must measure compositional structure density:

- Presence of finite verb (sentence-level predication)
- At least one dependent clause (subordination, relative clause, complement clause)
- Multiple arguments (subject + object)
- Modifier depth (adjective phrases, prepositional phrases)
- Absence of: dialogue tags, incomplete fragments, bare lists, single-word utterances
- Source: parse-free heuristics from POS-tag patterns and punctuation

This is computable from official text without any external model or data.

## Mechanism C operationalization: curriculum ordering

NOT length-only. The ordering criterion should be:

- **First 25% (0–2.5M words):** Shorter rows with simple syntax, high word frequency, few unknown tokens — builds basic token associations and common syntactic patterns.
- **Next 25% (2.5M–5M words):** Medium-length rows with moderate complexity, introduces less common vocabulary and more diverse constructions.
- **Next 25% (5M–7.5M words):** Longer rows with embedded clauses, lower-frequency words, more conceptual content.
- **Final 25% (7.5M–10M words):** Most complex, longest, most diverse rows.

Ordering feature: a composite of word count, mean word rank (from corpus frequency), type-token ratio, and syntactic-complexity proxy.

## What this experiment will resolve

1. If curriculum order helps on DeBERTa: "the leader's curriculum advantage is real and reproducible."
2. If compositional selection helps: "concentrating on structurally rich text improves compositional generalization."
3. If the combination is best: both mechanisms contribute independently.
4. If neither helps: the leader's advantage lies elsewhere (tokenizer interaction, specific text transformations, or model shape).

In case (4), the next step would be a tokenizer-curriculum factorial, which is more expensive but addresses the remaining confounded variable.

## Immediate construction task

Proposed construction:
1. A compositional-completeness scorer for official text rows.
2. A curriculum-ordering scorer.
3. A materializer that produces the four JSONL arms from the official corpus.
4. Smoke-verify that all four arms have exactly matched word budgets and the same source-file coverage.

Do NOT build a "clarity" scorer. Build a compositional-structure scorer.
