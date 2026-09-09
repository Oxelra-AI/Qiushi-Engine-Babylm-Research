# earlier analysis — Mechanism chain: causal interface, order-blindness, and intermediate-layer evidence

## Scientific question

The proposed comparison examines changes to the learning interface or experience structure: close the mechanism chain between the RecGPT-style causal route and the counterfactual order diagnosis order-blindness finding, using intermediate-layer and paired-gradient evidence, before choosing a main training route.

## What the evidence actually says (synthesis)

### counterfactual order diagnosis finding (checkpoint-specific, not over-generalized)

On the R1 ordered-dynamic chck_5M DeBERTa checkpoint, for same-word-bag counterfactual pairs differing by mean 16.5 tokens:
- The final-layer query-position hidden state has cosine 0.99999999 between the two operation orders.
- The same candidate answer scores identically in both contexts (|diff| ~2e-7 log-prob).

This proves the **final query position** is operation-order-invariant in this checkpoint. It does NOT prove DeBERTa structurally cannot encode order; it does not prove intermediate layers are order-blind; it does not prove the model never learned order information (it may have been learned then removed).

### RecGPT phenotype (from the model card and experimental notes)

RecGPT-10M: 34.17M params, shared Transformer block recursively for 16 iterations, hidden 768, embedding 192, FFN 12,288, 32,768-token BPE, Muon optimizer for recursive block + AdamW for embeddings, token batch 32,768, seq 256, custom 10M corpus.

Scores: BLiMP 73.11, Supplement 61.73, EWoK 52.62, Entity 16.59, COMPS 55.43, GlobalPIQA 40.68, SuperGLUE 66.64, Reading 6.92, NLP Average 52.40, Overall 41.53.

**Critical fact:** RecGPT's Entity (16.59) is WORSE than our protected model's Entity (22.62). RecGPT's EWoK (52.62) is comparable to ours (52.19). RecGPT's main advantages are BLiMP (+6.35), COMPS (+3.24), GlobalPIQA (+5.045), SuperGLUE (−1.38). RecGPT is weaker on Supplement (−3.85 vs our 59.88) and Reading (−0.70 vs our 7.62).

### The mechanism chain

1. **Causal interface → order-sensitive prefix representation.** A causal model where position k sees only the prefix must encode order because different operation orders produce different prefixes. This is structural, not checkpoint-specific. The counterfactual order diagnosis finding (final-layer order-blindness in bidirectional DeBERTa) is the *complement*: a bidirectional encoder can become order-blind at the query position because it sees both directions.

2. **Order-sensitive prefix → better BLiMP/COMPS/GlobalPIQA.** RecGPT confirms this: strong BLiMP, COMPS, GlobalPIQA. The causal interface forces the model to maintain sequential structure, which helps grammar, composition, and commonsense reasoning that depend on argument/event order.

3. **Order sensitivity alone ≠ Entity tracking.** RecGPT's Entity is weak (16.59). Entity tracking requires entity-identity persistence across operations — knowing that "the broom" is the same entity after being moved from loft to vault to pantry. A causal prefix can encode the sequence of operations, but entity identity must be bound to a persistent representation that survives re-mention. RecGPT's recursive shared-block architecture may not provide this.

4. **The combined phenotype we want:** protected Entity (22.62) + protected Supplement (59.88) + protected Reading (7.62) + RecGPT-like BLiMP/COMPS/GlobalPIQA. This would be Overall well above 41.80.

### The missing evidence: intermediate-layer order information

The scope limitation is that counterfactual order diagnosis only measured the final layer. Before using counterfactual order diagnosis to justify architecture choices, we need to know:

- **Do intermediate layers encode operation order?** If layer 4 or 6 has order-separable hidden states but layer 8 (final) collapses them, the order information was learned then removed — a representational bottleneck, not a learning failure. This would suggest a different fix (e.g., residual connection from an intermediate layer to the MLM head, or a shallower model).
- **Do paired gradients for order-relevant tokens flow differently?** If the gradient for tokens that differ between operation orders is near-zero or anti-correlated, the model is not being trained to use order even when it could.

## Proposed diagnostic: intermediate-layer order probe

### Design

Take the same 60 held-out counterfactual pairs from counterfactual order diagnosis (or regenerate with same seed). For each pair (A, B) with answers a, b:

1. For each layer l ∈ {0, 1, ..., 7} (embedding through final hidden):
   - Extract the hidden state at the masked query position for context A: h_A^l
   - Extract the hidden state at the masked query position for context B: h_B^l
   - Compute cosine similarity cos(h_A^l, h_B^l)
   - Compute the difference in same-candidate score if we replace the final-layer hidden with h_A^l vs h_B^l (requires a linear probe or using the LM head directly)

2. For each layer l, also compute:
   - The L2 norm of the hidden state difference ||h_A^l - h_B^l||
   - Whether the difference is structured (aligned with answer direction) or random

3. Paired gradient analysis:
   - For a small batch of counterfactual pairs, compute the gradient of the MLM loss w.r.t. the hidden states at each layer
   - Check whether the gradient for context A and context B are anti-correlated (which would explain the mirror) or independent

### What different outcomes would mean

- **If intermediate layers also have cosine ~1.0:** order information was never learned. The DeBERTa encoder genuinely does not distinguish operation order at any depth for this task. This would strengthen the case for a causal interface (Route C) because bidirectional encoding may be structurally order-insensitive for this kind of task at this scale.
- **If intermediate layers have cosine << 1.0 but final layer collapses:** order information exists but is removed. The fix could be a skip connection from an intermediate layer to the MLM head, or training with a shallower model, or an auxiliary loss at intermediate layers.
- **If paired gradients are anti-correlated:** the mirror (m_A ≈ -m_B) is enforced by optimization, not just representation. The loss landscape itself pushes the model to be order-invariant.

## Route decision framework

After the intermediate-layer diagnostic:

| Intermediate-layer finding | Paired-gradient finding | Recommended route |
|---|---|---|
| All layers order-blind | Anti-correlated gradients | Route C (causal interface) — bidirectional encoder cannot carry this signal |
| All layers order-blind | Independent gradients | Route C — representation, not optimization, is the bottleneck |
| Mid layers order-sensitive, final collapses | Anti-correlated | Hybrid: causal prefix + intermediate skip to MLM head |
| Mid layers order-sensitive, final collapses | Independent | Skip connection or shallower model under WWM |

## What NOT to do

- Do not launch a full RecGPT reproduction as the main route. RecGPT's Entity is weaker than ours, and the leader's Entity advantage (28.45) is one of the largest gaps.
- Do not treat counterfactual order diagnosis as a structural theorem about DeBERTa. The intermediate-layer diagnostic must come first.
- Do not design another custom-corpus variant. The earlier analysis result confirmed that data-side changes under plain WWM do not move the target cluster.
- Do not reopen closed routes (WESS, R1, SCMLM-R, surface adapter, MNTP, structure-density).

## Immediate next action

The proposed diagnostic is the intermediate-layer order probe and paired-gradient diagnostic on the R1 ordered-dynamic chck_5M checkpoint. This is a small, cheap experiment (no training needed) that directly answers the scientific question about whether order information was never learned or was accidentally removed. The result determines whether Route C (causal interface) is the right next main route or whether a hybrid/skip-connection repair of the bidirectional encoder is viable.
