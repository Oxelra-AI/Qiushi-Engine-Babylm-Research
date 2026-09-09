# earlier analysis — Route after R1 WWM failure: state-counterfactual MLM scoring objective

## Starting evidence

R1 v3 succeeded at the data side: generated passages can contain same-word-bag, non-commuting state dependencies that defeat text-visible heuristics. But the matched 5M learning-response experiment failed the predefined learning-response criterion:

- ordered-dynamic WWM did not beat static-indirect control on frozen MLM-head counterfactual margins;
- pair-level success was zero even on training vocabulary;
- checkpoints mainly expressed candidate answer priors rather than context-order-sensitive state composition.

Evidence: `notes/r1_learning_response_result.md`, `data/r1_vocab_split_learning_curve.json`.

Therefore the problem is no longer data construction alone. The next route must change credit assignment or representation so the ordinary exported scoring head is directly trained to prefer the correct state-dependent answer over the same-word-bag counterfactual answer.

## Literature and rule anchors

- BabyLM rules permit any training objective/regime if data restrictions are followed and the submitted model can score word sequences without extra fine-tuning. See `BabyLM-Turns-3` FAQ lines 236–238.
- ELECTRA shows that discriminative replaced-token training is more sample-efficient than MLM because supervision is defined over many/all input tokens rather than 15% masked tokens, especially for small models \cite{clark2020electra}.
- DeBERTaV3 shows RTD-style objectives can be attached to DeBERTa and outperform MLM DeBERTa; gradient-disentangled embedding sharing avoids generator/discriminator embedding conflict \cite{he2021debertava}.
- BabyLM 2025 findings report that contrastive synthetic data with paired Good/Bad completions improves reasoning and robustness more reliably than vanilla synthetic sampling when mixed with natural data, but diversity and natural-data balance matter.

## Primary route: State-Counterfactual MLM Ranking (SCMLM-R)

### Mechanism hypothesis

R1 WWM failed because the answer span was seldom a decisive supervised event and the loss did not compare the correct state answer against the paired wrong state answer. A model can minimize ordinary WWM while retaining answer-token priors.

SCMLM-R changes the objective on generated state pairs so that the *same ordinary MLM head* is directly penalized when it scores the paired counterfactual answer above the true state answer under the masked query context.

For each counterfactual pair `(A, B)` with same/similar word bag and answers `a` and `b`:

- mask the answer span in passage A and compute `s_A(a)` and `s_A(b)` with the MLM head;
- mask the answer span in passage B and compute `s_B(b)` and `s_B(a)`;
- optimize a symmetric logistic ranking loss:

`L_state = - log sigmoid((s_A(a) - s_A(b))/tau) - log sigmoid((s_B(b) - s_B(a))/tau)`.

This is not a trained probe. It is exactly the frozen-head evaluation signal used in r1 learning response result, but used as training loss. The final exported model remains an ordinary masked LM scorer because the same MLM head is used.

Add ordinary WWM on natural BabyLM text and generated passages:

`L = L_WWM_official + λ_gen L_WWM_generated + λ_state L_state`.

A useful first setting is `λ_state` in `{0.1, 0.3, 1.0}` after loss-scale inspection; do not assume the largest weight is best.

### Why this differs from prior failed routes

- Unlike R1 WWM, it directly compares true vs counterfactual answer spans, so answer priors cannot satisfy the state-pair loss.
- Unlike naive contrastive binding from binding learning response results, the score is produced by the ordinary MLM head at the query answer span, not a separate classifier or a target position bug-prone contrast.
- Unlike WESS, there is no gold address module or inference-time custom mechanism. The exported checkpoint is ordinary DeBERTa MLM.
- Unlike ELECTRA/RTD as a generic objective, the negatives are not random generator samples; they are semantically plausible same-word-bag state counterfactual answers targeted exactly at the failed Entity/EWoK mechanism.

## Matched first experiment

Use the same R1 static-control discipline before any 10M official screen.

Model/scale:

- protected DeBERTa-v2 8×480 baseline16k geometry first, but a smaller smoke can be used only to check loss plumbing;
- 5M word exposure response curve with checkpoints `chck_1M`–`chck_5M`;
- same 400K official + ~100K generated unique pool repeated through 10 passes;
- same seeds and optimizer as earlier analysis unless loss-scale requires reducing LR.

Arms:

1. `WWM_static_control`: static-indirect WWM control from earlier analysis.
2. `WWM_ordered_dynamic`: ordered-dynamic WWM from earlier analysis (already run; can serve as reference).
3. `SCMLM_R_ordered_dynamic`: official WWM + ordered-dynamic generated WWM + state-counterfactual ranking on ordered dynamic pairs.
4. `SCMLM_R_static_pairs`: same objective form but on static-control pseudo-pairs whose answers do not require dynamic composition, if constructible. This tests whether the ranking loss merely trains answer/style preferences.
5. Optional: `SCMLM_R_label_shuffled`: ordered dynamic pairs but randomly swapped true/counterfactual labels. This should destroy the readout; useful as a shortcut/leakage check.

Readout:

- primary: frozen MLM-head counterfactual margins on held-out vocabulary and train vocabulary, using `eval_r1_vocab_split_curve.py` style pair-summed margins and both-contexts-correct fraction;
- secondary: same readout on a new surface template family not used in training;
- early official columns if primary passes: Entity Tracking, EWoK, BLiMP/Supplement/COMPS/Reading at 5M/10M.

Pass condition before 10M official screen:

- `SCMLM_R_ordered_dynamic` must beat both WWM ordered and static controls on pair-summed margin and both-contexts-correct fraction;
- the advantage must appear on held-out vocabulary, not only training vocabulary;
- label-shuffled or static-pair controls must not show the same improvement;
- BLiMP/Supplement/Reading smoke should not collapse.

## Main risks

1. **Synthetic-probe overfitting:** The model may learn the generated templates and answer vocabulary without transferring to official Entity/EWoK. Mitigation: held-out vocabulary, held-out templates, multiple domains, static and label-shuffled controls, early official Entity/EWoK checks.
2. **Loss/score mismatch:** Ranking span scores may distort MLM probabilities and hurt BabyLM zero-shot columns. Mitigation: small `λ_state` sweep and regular WWM retention on official text.
3. **Length/tokenization confound:** Candidate spans need equal tokenizer length or a careful length-normalized score. Start with equal-token pairs exactly as r1 learning response result did.
4. **No representation pathway:** Even direct ranking loss may be minimized by shallow pair/order features. The pair-summed both-contexts-correct metric and held-out template families are necessary to detect this.
5. **Objective implementation risk:** The symmetric loss must backpropagate through the same MLM logits and not accidentally train a side head.

## Alternative route: RTD/DeBERTaV3-style state replacements

A second route is to train a DeBERTa discriminator to detect state-inconsistent answer spans or corrupted operation tokens, inspired by ELECTRA/DeBERTaV3. It has dense supervision and strong sample-efficiency precedent, but BabyLM evaluation needs sequence scoring without extra fine-tuning. A pure RTD head may not provide the same pseudo-likelihood scoring used by official MLM evaluations unless combined with an MLM head or converted into a scoring function. Therefore RTD is promising as an auxiliary regularizer, not the first repair unless the scoring/export path is settled.

## Current recommendation

Build SCMLM-R first as the minimal repair of R1: it uses the proven R1 counterfactual data, the exact failed frozen MLM-head readout, and the ordinary MLM scoring head. If it cannot produce a held-out counterfactual-margin advantage over static controls, the route should move to architecture/slot/state representation rather than more data or generic objectives.
