# credit allocation binding pilot credit-allocation mechanism for entity binding: pilot result

## Central finding

A paired pilot on coherent86 establishes that **credit allocation to the answer computation** is the decisive variable for entity-binding acquisition, not merely the presence of contrastive experience:

| condition | R−N gap (baseline) | R−N gap (epoch 200) | UPDATE margin | Both correct |
|---|---:|---:|---:|---:|
| Baseline (no training) | −4.91 | — | +4.33 | 5/8 |
| Arm A: Standard WWM (15%) | — | **−1.54** | **−1.06** | **3/8** |
| Arm B: Focused masking (always mask answer + 10% background) | — | **−0.04** | **+23.32** | **8/8** |

Standard WWM partially closes the R−N gap (−4.91 → −1.54) but **destroys** UPDATE detection (+4.33 → −1.06), dropping to 3/8 pairs correct. This is consistent with the synthetic full-objective result: generic prediction credit drowns binding-specific gradients.

Focused masking **closes the R−N gap to near zero** (−4.91 → −0.04) while **strengthening** UPDATE detection to +23.32, with 8/8 pairs correct. The model learns both:
- To follow entity-targeted updates (positive UPDATE margin)
- To ignore distractor updates (RETAIN ≈ NEUTRAL)

## Connection to the synthetic mechanism

This pilot is the natural-language analog of the earlier permutation-orbit findings:

| Synthetic setting | Natural language analog | Result |
|---|---|---|
| Query-first answer-only training → near-perfect binding | Focused masking (always mask answer) | R−N closes to −0.04, both correct 8/8 |
| Full-objective training → bag-level/seed-bimodal | Standard WWM (15% random) | UPDATE destroyed, 3/8 correct |
| Effective credit = 1/(3K+5) per answer position | P(answer masked) ≈ 15% per epoch under WWM | Sparse binding supervision ≈ 10% useful |

The common mechanism: **the fraction of training credit that reaches the answer position determines whether binding is acquired**. Under standard MLM, only ~15% of epochs provide answer-position supervision, and of those, many also mask critical evidence. Focused masking raises effective answer credit to ~100%, directly concentrating gradients on the computation that implements entity-conditioned selection.

## Connection to the relation-arm results

The earlier relation-arm analysis showed VIEW T-N vs CLEAN = +0.452 vs REPEAT T-N vs CLEAN = +0.161, confirming causal locality. The asymmetric binding result explains why: the model's existing detection route works for "which entity was mentioned in this update" but lacks negative selection for "this update is about a different entity." The cheapest-sufficient-relation principle operated through this architectural constraint — the learner improved what it could (detection) rather than installing what was missing (negative selection).

## Limitations and required next work

1. **Train-set overfitting**: 16 packets over 200 epochs. The model memorizes 8 entity-state pairs. Generalization to held-out entities and states is not tested.
2. **Focused masking requires answer annotations**: unlike standard WWM, focused masking needs explicit answer-position marking. This makes it a supervised intervention rather than a general MLM improvement.
3. **Downstream benchmark impact is unknown**: the pilot trains only private adapters; effect on BLiMP, Entity, EWoK, etc. is not measured.
4. **Scale of effect**: 16 packets cannot change benchmark-level behavior. A corpus-scale intervention needs 100-1000 paired packets embedded in the full training stream.
5. **The RETAIN result is on memorized pairs**: the model may learn "when Bob is mentioned in the update, Alice keeps her original state" rather than a general entity-conditioned filter.

## Next experiments (ordered by scientific value)

1. **Held-out binding generalization**: Generate 200 pairs, train on 160, test on 40. If focused-masking binding generalizes to unseen entity-state combinations, the mechanism is not just memorization.
2. **Mixed-objective pilot**: Train with focused masking on contrastive packets interleaved with standard WWM on ALN corpus. Measure whether binding transfers without destroying downstream competence.
3. **Credit titration**: Test intermediate masking probabilities (always mask answer with probability p ∈ {0.3, 0.5, 0.7, 1.0}) to find the minimum answer-credit needed.
4. **Benchmark-integrated training**: Embed ~500 contrastive pairs into the ALN 10M pool, train from coherent86 with parent anchoring, and evaluate on full official BabyLM.

## Files

- Training pilot: `data/binding_training_pilot/binding_training_pilot.json`
- Scripts: `scripts/binding_training_pilot.py`
- Counterfactual diagnostic: `data/counterfactual_binding/`
- MLM mask analysis: `data/paired_mlm_analysis/`
- Paired packets: `data/paired_packets/`
- Baseline probe: `data/binding_probe_coherent86/`
