# Integration of controlled binding and relation-typed composition

## Two complementary axes of the same principle

### Relation-learning evidence: What relation is practiced determines what computation is installed

Under fixed experience budget:
- Exact recurrence → source-recognition/content-pull (Δ(T−N) positive = weakened
  true-source use for nonidentical targets)
- Nonidentical restatement → content-conditioned support (Δ(T−N) negative = 
  strengthened true-source use)
- Both require within-window co-occurrence (split controls collapse effects)
- The same content, same budget, same model produces opposite computations
  depending on the practiced relation

Entity behavioral instance: REPEAT helps at zero relevant updates (identity
retrieval), hurts at deep updates (nonidentical discrimination). This mirrors
bag-level vs binding: bag-level knows what's present, binding knows what
belongs to whom.

### Controlled binding evidence: What causal pathway is available determines what level of computation forms

Same architecture, same entity-attribute content, same answer position:
- Original order (context before query): model converges to bag-level statistics
  regardless of objective allocation (orbit binding design)
- Query-first order (query before context): answer-only training reaches
  near-perfect counterfactual binding (query first binding compact summary)
- The difference is not capacity (153K-1.2M all fail in original order) or 
  information (same latent rows), but whether context states can condition on 
  the query during formation

### The shared principle (candidate)

**Data-efficient learning depends on two structural conditions:**
1. **Relational structure of co-occurring material** (relation-learning experiments): what computation
   the experience practices
2. **Causal/temporal pathway for credit assignment** (controlled binding experiments): whether the
   computation has a reachable acquisition route

Neither condition alone is sufficient:
- Right relation + wrong pathway = bag-level statistics (original-order orbit
  binding with entity-specific targets)
- Wrong relation + right pathway = bag-level statistics (query-first
  BAG_INDEP answer-only)
- Right relation + right pathway = reusable computation (query-first BOUND
  answer-only; the relation-learning VIEW arm under within-window co-occurrence)

This means a fixed experience budget's value depends jointly on the relational
structure practiced between co-occurring spans AND on whether the
causal/objective format routes sufficient credit to the target computation.
Changing either condition changes the installed computation while keeping
content, budget, and model fixed.

## What the integration candidate decides

The branching experiment is the hinge between a *format-specific acquisition
effect* and a *reusable computation principle*:

| Outcome | Implication |
|---|---|
| Binding persists under full objective | Selector is stable once formed; full training maintains but may not acquire |
| Binding requires ongoing answer pressure | The selector is reinforced, not structural; formation and maintenance are distinct |
| Blocking query-context attention destroys binding | Prospective marking: context states must form around the query |
| Bound init transfers to original order | **Key test.** The computation is reusable across formats; bootstrapping works |
| Bound init does NOT transfer | The computation is format-specific; query-first binding is a favorable-format artifact |

If transfer succeeds: the combined relation-structure and credit-pathway principle becomes actionable:
"Under limited data, compose related spans in formats that route credit to the
desired computation. If the direct format is difficult, first acquire the
computation in a favorable format; it can then make ordinary experience more
effective."

If transfer fails: the controlled binding experiments establish that causal order determines binding
acquisition but the acquired state is format-locked. The connection to relation-typed composition
narrows to shared phenomenology (bag-level vs relation-specific) rather than
a unified mechanism.

## What remains unresolved regardless of the integration candidate outcome

1. The orbit-binding task is synthetic. Bridging to natural language requires
   showing that natural text supplies analogous temporal/relational structure.

2. The relation-learning mechanism (source-recognition/content-pull vs content-conditioned
   support) operates at a different level from the controlled binding selector/bag distinction.
   They may be the same computation measured differently, or distinct effects
   of the same relational condition.

3. Neither study has yet shown that the identified conditions are necessary
   (only that they are sufficient in their respective substrates). A stronger
   claim requires showing that removing the favorable condition prevents the
   computation even when the content is present (the controlled binding study has this for causal order;
   the relation-learning study has this for within-window co-occurrence through split controls).

4. The "amount" of favorable relational structure needed under a fixed budget
   is unmeasured. The relation-learning effects are at 100% dose within the compact companion.
   The controlled binding is at 100% entity-specific targets. Real corpora have mixed
   doses.
