# wess route closed — Decisive unlabeled-address WESS experiment: result and route closure

## Evidence

- Script: `scripts/unlabeled_address_decisive.py`
- Result JSON: `data/unlabeled_address_decisive.json`
- Prior route update: `notes/wess_transfer_route_update.md`

## Experiment design

The decisive experiment was to test whether WESS can operate on
unlabeled text using predicted entity/state positions and predicted entity-to-slot routing,
with no gold metadata at inference. The experiment must measure whether predicted-route
WESS retains causal slot-swap and write-removal effects, and must immediately connect to
official short-budget comparison if the mechanism survives.

The experiment used:
- 384×4 DeBERTa-v2 with WESS slot module and a token-level 4-class router (other/entity/state/query);
- Training: gold-route WESS for 60% of steps, then predicted-route for 40%, with router
  supervised by gold role labels;
- Held-out independent-template evaluation with predicted entity/state positions, predicted
  entity-to-slot addresses via token-identity matching, and no access to EpisodeFeat metadata;
- Four evaluation modes: gold (upper bound), predicted, no_address, random;
- Metrics: pair accuracy, log-odds, slot-swap delta, write-removal delta, router accuracy.

## Results

### Router performance

| metric | value |
|---|---:|
| token accuracy | 90.4% |
| entity recall | 100.0% |
| state recall | 100.0% |
| query recall | 0.0% |
| parse success rate | 0.0% |
| route exact rate | 0.0% |

The router cannot identify the query entity token in any held-out episode. Since the
query entity is needed to select which slot to read, the predicted-route pipeline fails
at the parse stage for every example.

### Binding and intervention results

| mode | pair acc | log-odds | swap delta | ablation delta |
|---|---:|---:|---:|---:|
| gold | 1.000 | +8.974 | -17.959 | -8.973 |
| predicted | 0.000 | -0.005 | 0.000 | 0.000 |
| no_address | 0.000 | -0.005 | 0.000 | 0.000 |
| random | 0.000 | -0.005 | 0.000 | 0.000 |

Gold-route WESS remains perfect with strong causal interventions. Predicted-route WESS
is indistinguishable from no-address and random routing: zero pair accuracy, zero log-odds,
zero intervention effects.

## Scientific conclusion

The unlabeled-address WESS route fails at the query-entity detection stage. The router
cannot distinguish the query entity token from other entity tokens in held-out templates,
and this single failure propagates to complete address collapse. The predicted-route
mechanism does not retain any causal slot effect.

The results do not justify further repairs to this WESS route. The mechanism does not
transfer to unlabeled evaluation inputs, and the annotated auxiliary route does not
transfer to official BabyLM columns (exported base threeway official interpretation). WESS is a real, stable mechanism when
entity/state spans and routing are known, but it is not a deployable BabyLM architecture
and cannot be assumed to close the official Entity/EWoK/GlobalPIQA gap.

## Next route: reopen parallel mechanisms, data, and representation

The 11.47 summed-point gap to the public leader (Overall 41.80 vs protected 40.53)
remains open. The WESS route has been rigorously tested and closed. The next research
should:

1. Re-examine the public leader's actual strengths (Entity 28.45, EWoK 56.07, GlobalPIQA
   39.665) and the protected model's deficits;
2. Search for mechanisms that do not require gold annotation at inference: data selection,
   representation learning, objective design, curriculum, or architecture that can improve
   entity tracking, world knowledge, and commonsense reasoning from the official corpus
   under the ≤10M word constraint;
3. Return to the literature and knowledge base for approaches that have demonstrated
   Entity/EWoK/GlobalPIQA movement in small-data regimes;
4. Run a new round of controlled short-budget screening on promising candidate mechanisms
   with official-compatible evaluation.

The BabyLM Strict-Small SOTA goal remains active. No WESS model has produced a complete
9-column score or improved the official target cluster. The protected internal best
remains DeBERTa-v2 8×480 WWM 100M (Overall ~40.53).
