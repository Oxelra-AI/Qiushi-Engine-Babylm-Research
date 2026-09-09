# broader binding probe conclusion — broader binding-substrate probe: conclusion

## Evidence

- Script: `scripts/probe_revision_167b_entity_swap_binding.py`
- Output: `data/broader_binding_probe_v2.json`

## Results summary

| metric | value |
|---|---:|
| Full-corpus binding windows (≥2 entities + predicates) | 12,939 |
| Bag-preserving entity-swap examples | 4,460 |
| Valid target-identical tokenization probes | 45/300 |
| Mean (correct - swapped) target LL | +0.032 |
| Median | 0.0 |
| Positive fraction | 0.467 (below chance) |
| p10, p90 | -0.225, +0.210 |

## Scientific conclusion

The entity-position binding probe is null:

1. **The protected model shows no binding-dependent state prediction.** When entity names are swapped throughout a window (preserving the word bag exactly), the target state/location tokens are equally likely under correct or wrong binding. The model predicts state words from local context and frequency, not from entity-state assignment.

2. **This is consistent with structure density conclusion and objective pivot:** plain WWM does not force the loss to depend on entity-state binding. The model learned grammar and local co-occurrence but not who-did-what tracking.

3. **Comparison with source swap antecedent probe copy probe:** source swap antecedent probe showed +2.44 correct-antecedent advantage for later entity-name targets. broader binding probe conclusion shows +0.03 for state/location targets. The entire source swap antecedent probe signal was surface copy.

4. **Quality of detected binding patterns is mixed.** Examples include stock tickers, modal verbs parsed as entities, traffic reports — not all are genuine entity-state relations. The 12,939 windows are an upper bound on usable binding material; the actual high-quality binding-dependent subset is substantially smaller.

## Route implications

### Official-corpus binding route: demoted, not closed

The official corpus has 12,939 windows with detectable multi-entity structure. Some are genuine entity-state patterns. But:
- The model doesn't use entity-state binding (null probe)
- Teaching binding from scratch in 10M with <5000 genuine binding examples repeated 2-4× requires the objective to carry the entire learning
- The detection heuristics are noisy; cleaner filtering would reduce the viable set further
- No evidence that even a perfect binding objective on this material would transfer to Entity Tracking evaluation

Status: not closed definitively, but LOW priority relative to the hybrid route.

### GPT-BERT/MNTP hybrid route: PROMOTED to main SOTA direction

Reasons:
1. Affects ALL tokens in ALL training examples, not just rare binding patterns
2. Strong BabyLM literature support: GPT-BERT reports +4.2% BLiMP, +2.0% EWoK with only 6.25% MNTP; AntLM reports +9.7 BLiMP, +14.1 EWoK on LTG-BERT 10M
3. Not tested in these experiments — a genuinely fresh direction
4. The mechanism (forcing left-to-right coherence while preserving bidirectional evaluation) attacks information-flow and dense supervision, not entity binding specifically — which may be the RIGHT level for the remaining Overall gap
5. Compatible with our established DeBERTa-v2 backbone, baseline16k tokenizer, and AdamW training infrastructure
6. No future-leak risk in standard DeBERTa architecture (causal attention mask suffices)
7. RecGPT shows causal modeling can help BLiMP/COMPS/GlobalPIQA; hybrid preserves our Supplement/Reading strengths

### What the hybrid route must NOT become

- Pure GlobalPIQA/COMPS chase while Entity/EWoK stagnate
- Causal-only training that loses Supplement/Reading advantages
- Complex multi-modification stack (no simultaneous attention gating + layer weighting + mask scheduling + batch scheduling)
- Start with the simplest form: low-ratio MNTP (6-12%) + standard WWM, DeBERTa-v2, official corpus

## Research implications

Build the GPT-BERT/MNTP hybrid trainer. The no-future-leak test is the central engineering requirement.
