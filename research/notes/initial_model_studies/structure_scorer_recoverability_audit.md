# structure density conclusion and objective pivot — structure-density scorer recoverability audit

Evidence JSON: `experiments/archive/initial_model_studies/data/structure_scorer_recoverability_audit.json`

This audits whether the high_entity_state scorer measured recoverable entity/state structure or mostly cue-token density. The proxies are heuristic, but they separate anchored cross-mention/state candidates from pronoun-rich or cue-rich text without an entity anchor.

| arm | unique windows | entity_state score mean | coref-candidate frac | entity+state recoverable frac | state-transition frac | pronoun-dialogue-like frac | cue-without-anchor frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| high_entity_state | 31325 | 0.266 | 98.4% | 69.0% | 45.1% | 27.8% | 1.6% |
| matched_low | 31325 | 0.112 | 86.5% | 42.2% | 19.4% | 14.9% | 11.3% |
| uniform | 31325 | 0.178 | 93.4% | 57.3% | 32.4% | 21.6% | 6.1% |

## Reading

High_entity_state raises the heuristic entity+state recoverable fraction from 42.2% (matched_low) and 57.3% (uniform) to 69.0%, but it also changes pronoun/cue distributions and still failed to improve Entity/EWoK. This suggests either the proxy captures only shallow recoverability, or ordinary WWM does not force use of the structure.

The next decision should be made from this audit plus the three-arm evaluation, not from score labels alone.
