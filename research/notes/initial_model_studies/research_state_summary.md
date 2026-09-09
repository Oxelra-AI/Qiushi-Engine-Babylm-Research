# Synthesis of baseline, data and binding-mechanism evidence

## Confirmed Facts (Established by multi-seed or controlled experiments)

1. **Protected backbone:** DeBERTa-v2 8×480, baseline16k, WWM p=0.15, batch 256, official corpus. Outperforms BERT by BLiMP +12.6, Supplement +9.0, Entity +6.0, COMPS +2.1, GlobalPIQA +3.9. [34m training validation and deberta repair]

2. **WWM > token masking:** Cross-seed positive in BLiMP, EWoK, Entity. Mean BLiMP +2.15, EWoK +2.50, Entity +0.93. [related experiments]

3. **Current best internal coordinate:** wwm_seed43 chck_80M, Overall 40.703 (validated — existing runs exposure trajectory evaluator uses direct checkpoint paths). Gap to public leader: 1.098. [new best verification and assessment]

4. **Public leader phenotype is real:** Locally reproduced — BLiMP 67.20, Entity 28.45, EWoK 56.07. [related experiments]

5. **Leader advantage is data-dependent, not architecture-only:** RecGPT architecture on official corpus → Overall 34.27, far below both our model and the leader. [earlier analysis]

6. **Cross-sentence history is recoverable but not used for prediction:** Probes show full-history > local-only at all checkpoints (AMLM +11.58 at 100M). Corrected counterfactual shows entity history does NOT causally predict later content. [official remention probe]

7. **Evaluation artifact:** HuggingFace ignores `revision` for local directories. fineweb random quality 3m profile trajectory invalid for Entity/Reading; existing runs exposure trajectory trajectory is valid (uses path joining). [related experiments]

## Closed Routes (Do NOT reopen without genuinely new evidence)

| Route | Result | Evidence |
|---|---|---|
| Paired-restatement | true_pair worse than orig/shuffled on Entity/EWoK | pair discriminator tokenaware 1m profile |
| Relation-explicit scaling | EWoK non-persistent (+3.63→−2.09→+0.54), Entity flat | fineweb relation vs random 3m direct checkpoint trajectory |
| WESS | Predicted routing collapsed, no transfer | wess route closed |
| R1/SCMLM-R | Order-invariant final representations, both-correct=0 | related experiments |
| MCEC | History signal not converted to content prediction | corrected history content probe |
| AMLM standalone | 10M peak decays by 100M, seven-column mean −0.09 | related experiments |
| Route B (morphology adapter) | No Entity/EWoK signal beyond generic capacity | related experiments |
| Custom corpus under plain WWM | EWoK +1.15, Entity −0.06, not scaled | earlier analysis |
| RecGPT on official corpus | Overall 34.27 | earlier analysis |
| 40k tokenizer | Entity 22.16 < 16k Entity 22.62 | new best verification and assessment |
| State-counterfactual mining | Too sparse/contaminated for training | related experiments |
| Relation-word priority masking (RMEC) | Rejected: relabels prior negatives | official remention probe |
| Delayed-state/binding mask | Rejected: relabels RMEC+MCEC | fineweb random quality 3m profile |

## Deficits to Public Leader (from new best verification and assessment coordinate)

| Column | Our Best | Leader | Deficit |
|---|---:|---:|---:|
| Entity | 22.20 | 28.45 | −6.25 |
| EWoK | 50.44 | 56.07 | −5.63 |
| GlobalPIQA | 37.59 | 39.67 | −2.07 |
| SuperGLUE | 68.26 | 69.79 | −1.53 |
| BLiMP | 66.54 | 67.20 | −0.66 |
| Supplement | 61.00 | 56.04 | +4.96 |
| COMPS | 53.00 | 53.57 | −0.57 |
| Reading | 7.30 | 54.25 | large gap (different evaluator?) |

## Remaining Explanations for the Gap

After 300 steps of evidence, only these explanations survive:

1. **Training signal efficiency:** Standard MLM provides gradient to only 15% of tokens per exposure. The model under-uses 85% of seen positions. More efficient objectives (e.g., replaced token detection) could extract more learning per word.

2. **Leader's custom data recipe:** The leader uses FineWeb + something that our deterministic reproduction failed to replicate. Possibly LLM-generated rewrites rather than spaCy simplifications, or a different training schedule/interface on its custom data.

3. **Cross-sentence credit assignment:** Models encode cross-sentence state but the standard MLM gradient doesn't reward using it (local context suffices for 15% masked prediction). A mechanism that forces cross-sentence context to influence predictions at ALL positions would change credit assignment fundamentally.

## Next Experiment: Hybrid MLM + Replaced Token Detection (RTD)

**Why this is genuinely different from all closed routes:**
- Not a data filter or composition change
- Not a masking schedule change (RMEC)
- Not an auxiliary loss on unmasked downstream content (MCEC)
- Genuinely multiplies effective training signal from 15% to 100% of tokens
- Changes credit assignment: model must detect contextual anomalies at every position
- Literature support: ELECTRA is proven 4-10× more sample-efficient than MLM

**Why it might fix Entity/EWoK specifically:**
- Entity: model must detect whether entity-state tokens are plausible given earlier operations
- EWoK: model must detect whether world-knowledge tokens fit the context
- GlobalPIQA: model must distinguish plausible from implausible completions

**Experiment specification:**
- Scientific question: Does adding RTD to WWM improve Overall, especially Entity/EWoK?
- Hypothesis: RTD at all positions increases per-word learning signal, improving tasks requiring contextual sensitivity
- Single changed factor: Training objective (WWM → WWM + RTD hybrid)
- Matched baseline: Pure WWM, same architecture/data/tokenizer/batch/steps/seed
- Generator: 2-layer DeBERTa-v2 hidden 128 (minimal compute overhead)
- RTD weight: 50.0 (ELECTRA default, to be validated)
- Seeds: 42, 43
- Endpoint: 100M exposure, evaluate at 20M/40M/60M/80M/100M
- Success: Overall ≥ +0.5 over matched WWM at best endpoint, sustained at 80-100M, positive Entity or EWoK
- Stop: If 20M shows identical or worse Overall AND identical MLM loss profile
