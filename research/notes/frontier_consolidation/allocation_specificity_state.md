# allocation specificity state allocation/specificity state

This note preserves the current scientific distinction before the long allocation specificity state jobs finish.

## What changed in allocation specificity state

The earlier analysis Entity balanced readout should not be read as simply retiring the compact-view mechanism.  It separates two different contrasts:

- `V-R`: compact source-conditioned view versus exact repeat.
- `V-B`: compact source-conditioned view versus same-population independent-sentence breadth.
- `B-R`: breadth versus exact repeat.

For the first-basin late Entity surface, the identity is exact at every stratum and checkpoint:

```text
V-R = (V-B) + (B-R)
```

Late 80/90/100M means from `data/entity_balanced_and_transfer_readout_full/`:

| Entity contrast | official all18 | zero-op | nonzero-op | neutral zero/nonzero |
|---|---:|---:|---:|---:|
| V-R | +4.0175 | -9.6609 | +6.7532 | -1.4539 |
| B-R | +1.0234 | -13.0372 | +3.8355 | -4.6008 |
| V-B | +2.9941 | +3.3763 | +2.9177 | +3.1470 |

Thus the offset-like changed-state allocation appears in `B-R` and therefore in `V-R`: zero-operation retention worsens while nonzero-operation/update rows improve under the official 5:1 nonzero-heavy mixture.  But `V-B` is same-sign positive in both zero-op and nonzero-op strata.  This is the current threshold-free proxy for a source-conditioned re-expression specificity effect, because the official prediction JSONs contain only selected strings, not option likelihoods.

The same data also show why Entity alone cannot be the general learning principle: DeBERTa `V-R` becomes negative under neutral zero/nonzero weighting in two basins, and RoBERTa MAX shows only a much smaller late Entity `V-R` (official +0.2587 pp; neutral +0.1367 pp).  Entity is an interpretable redistribution surface, not broad reusable record skill by itself.

## Active experimental split

allocation specificity state follows the corrected card allocation:

1. train the first-basin DeBERTa MAX-geometry clean control (`seed43022`).  Preflight verified the clean stream SHA `64e686d16e0d7d5e81acecc73494d8670a1d6f8ddaffb4ce517277c983a9bed8`, 653,130 rows, exact 100M words, 2,552 updates, tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`, and 34,467,424-parameter full DeBERTa.  This tests whether the surviving DeBERTa `V-C` signal is real under MAX row/update geometry rather than inherited from the older 1x-geometry clean arm.

2. score already-trained breadth and RoBERTa stable-family backlogs.  This is lower-cost than another training run and can decide whether `V-B` and `V-C` survive ex-Entity and outside DeBERTa.

3. After seed43022 clean training completes, score its stable families.

4. Second-basin DeBERTa MAX-geometry clean training (`seed43122`) remained planned, not completed.

The file-only merger `scripts/fixed_budget_allocation_readout.py` writes to `data/fixed_budget_allocation_readout/` and should be rerun after each relevant task delivery.  Its current baseline has 219 present score rows, 381 missing, 165 contrast rows, and only the three official Entity triangle rows complete; no broad ex-Entity `V-B/B-R` or RoBERTa geometry-clean `V-C` conclusion is available yet.

## Interpretation to preserve

A candidate principle is not "Entity score increases" and not "source-view addressability is established."  The live hypothesis is narrower and more generalizable if supported:

> Under a fixed finite word/update budget, exact duplicate recurrence can spend capacity on a brittle decision allocation; source-conditioned compact re-expression can convert part of that expenditure into more uniformly useful supervision than unrelated same-population breadth, while still carrying redistribution boundaries that must be measured by family and stratum.

The above remains a hypothesis.  It needs the broad ex-Entity `V-B/B-R` readout, RoBERTa geometry-matched `V-C`, and DeBERTa MAX-geometry clean controls.  The permuted companion correspondence control permuted-companion control remains scientifically high-quality but is not the next H100 priority unless a broad balanced aligned carrier appears that cannot be explained by breadth, row/update geometry, or operation allocation.

## research-loop side route

The completed research loop produced a mass-matched innovation-biased WWM prototype on the old 1x compact stream.  It is useful only as a possible later mechanism probe after the fixed-budget evidence clarifies whether source-conditioned re-expression has a broad signal.  It is not ready to divert H100s now: it targets source-absent rewrite innovations, requires trainer integration and CPU/tiny smokes, and the active uncertainty is currently MAX geometry/breadth/RoBERTa allocation.
