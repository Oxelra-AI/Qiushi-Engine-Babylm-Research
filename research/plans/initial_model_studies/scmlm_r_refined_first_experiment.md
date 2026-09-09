# earlier analysis — Refined first experiment for state-counterfactual MLM ranking

## Why this route exists

R1 showed that legal generated passages can contain real same-word-bag non-commuting state dependence, but ordinary WWM did not train the MLM head to use that dependence. The next experiment should therefore change the loss attached to the ordinary MLM head, not add more of the same generated WWM exposure.

The useful scientific question is:

**Can a small amount of state-counterfactual training make the ordinary exported MLM head change answer preferences with the computed state, instead of preserving answer-token priors?**

## Core objective

For a paired state counterfactual `(A, B)` with answers `a` and `b`, compute MLM-head span scores in masked query contexts:

- `m_A = s_A(a) - s_A(b)`
- `m_B = s_B(b) - s_B(a)`

Train with a combination of:

1. **direction terms**: `softplus(-m_A/tau) + softplus(-m_B/tau)`
2. **interaction term**: `softplus((gamma - (m_A + m_B))/tau)`
3. **multi-candidate span softmax** over the paired counterfactual answer, other locations in the story, and balanced off-story locations
4. **prefix supervision** at several operation depths, so the state variable is trained along the chain rather than only at the final sentence
5. ordinary WWM on official BabyLM text and generated text to preserve language modeling calibration

The same vocabulary MLM head is used for both training and evaluation. No side readout is trained.

## Data needed before training

Use shortcut-hardened pairs rather than the initial adjacent-swap family alone:

- causal operation swaps that change the queried entity state;
- noncausal operation swaps that preserve the queried entity state;
- state-irrelevant swaps with the same surface exchange form but unchanged answer;
- query-entity permutation cases;
- variable edit position, variable distractor count, and non-adjacent operation swaps;
- held-out object/location words, held-out templates, held-out object-location combinations, and longer chains than seen in training;
- answer words balanced within batches so no location is usually true or usually false.

Candidate answers should have equal tokenizer length for the first implementation, using the r1 learning response result equal-length filtering protocol.

## Most decisive first experiment

Do not begin with another full 5M from random initialization. Start from the **same earlier analysis ordered-dynamic 5M checkpoint** and run short equal-budget continuations. This fixes initial language ability, answer priors, and early training history, isolating the loss effect.

Fork four continuations from:

`training/runs/r1_ordered_dynamic_5M/hf_model/chck_5M`

Arms:

1. `WWM_continuation`: same corpus, ordinary WWM continuation.
2. `Answer_MLM`: directly mask and predict true answer spans, but no counterfactual negative.
3. `Random_negative_ranking`: rank true answer above a frequency-balanced random location negative.
4. `SCMLM_R`: direction ranking + interaction + multi-candidate + prefix supervision on state-counterfactual pairs.

Match across arms:

- same word exposure / optimizer updates;
- same number of answer-span updates;
- same official/generated ratio;
- same seeds and checkpoint cadence;
- same ordinary MLM head and exported checkpoint format.

Readouts:

- both-contexts-correct fraction on train vocabulary, held-out vocabulary, held-out templates, held-out object-location combinations, and longer chains;
- mean pair-summed interaction margin `m_A + m_B`;
- sensitivity to causal swaps minus sensitivity to noncausal and state-irrelevant swaps;
- ordinary held-out WWM loss / pseudo-likelihood;
- early BLiMP, Supplement, Reading, Entity, and EWoK smoke only after synthetic response becomes state-sensitive.

Success pattern for route continuation:

- `SCMLM_R` beats all three comparison arms on both-contexts-correct fraction and pair-summed interaction margin;
- the effect appears on held-out vocabulary and held-out templates, not just matched training forms;
- causal swaps change the margin while noncausal and state-irrelevant swaps do not;
- natural WWM calibration and protected grammar/reading smoke do not collapse.

If this pattern does not appear, then the route should move to a state-directed RTD or exportable object-state representation rather than adding more R1 data.

## Why generic RTD is not first

ELECTRA and DeBERTaV3 strongly support discriminative pretraining for sample efficiency, and BabyLM rules permit non-MLM objectives if the model can score sequences. But generic RTD negatives can be solved through generator artifacts or local unnaturalness. The failure in r1 learning response result is specifically that the ordinary MLM head does not compare the correct state answer with a plausible same-word-bag counterfactual answer. SCMLM-R is the smallest direct repair of that failure. A state-directed RTD auxiliary becomes the next competitor if SCMLM-R trains only a template-specific answer interface.
