# pair route full result and closure — FineWeb paired-restatement route: full four-arm result and closure judgment

## Full 1M four-arm profile (pair discriminator 1m profile, corrected deranged data, GPU1, exact 1M exposure)

Absolute fast-profile scores (DeBERTa-v2 8×480, baseline16k, WWM p=0.15, b128, seed42, chck_1M):

| column | true_pair | hard_neg | orig_only | shuffled |
|---|---:|---:|---:|---:|
| blimp_fast | 53.17 | 52.25 | 53.87 | 53.22 |
| supplement_fast | 48.40 | 52.40 | 44.40 | 42.80 |
| ewok_fast | 49.18 | 48.36 | 49.18 | 49.09 |
| entity_tracking_fast | 17.88 | 16.60 | 17.73 | 17.60 |
| comps | 49.79 | 50.01 | 50.20 | 49.83 |
| Reading_mean | 3.545 | 3.71 | 6.295 | 4.06 |

Contrasts:

- true − hardneg: Entity +1.28, EWoK +0.82, BLiMP +0.92, Supplement −4.0, Reading −0.165
- **true − orig: Entity +0.15, EWoK 0.00, BLiMP −0.70, Supplement +4.0, Reading −2.75**
- **true − shuffled: Entity +0.28, EWoK +0.09, BLiMP −0.05, Supplement +5.6, Reading −0.515**

## Scientific interpretation

1. **The paired-restatement mechanism does NOT lift Entity or EWoK on a normal baseline.**
   Against `orig_only` (the FineWeb source/syntax baseline) and against `shuffled_pair_adjacent`,
   true_pair Entity is +0.15 / +0.28 and EWoK is 0.00 / +0.09 — indistinguishable from zero at 1M single-seed.

2. **The apparent +1.28 Entity / +0.82 EWoK "gain" over hard-negative is driven by the hard-negative
   arm being DAMAGED (Entity 16.60, lowest of all four), not by true_pair rising.** The hard arm's
   only real advantage is Supplement (52.40), which is likely noise given Supplement swings 42.8–52.4
   across arms with identical source material.

3. **Truncation rates are NOT matched across arms**, so the four models did not train on matched
   actual token content:
   - true_pair truncated_fraction 0.25696
   - hard_neg 0.23664
   - orig_only 0.21872
   - shuffled 0.13072
   The pair arms (orig+simplification concatenated) produce longer rows that truncate more often at
   seq_len 256. This means "adjacency correspondence" is confounded with "how much of each pair
   survived truncation." The primary contrast is not clean at the input level.

4. Training losses last: true 7.524, hard 7.590, orig 7.483, shuffled 7.612 — nearly identical,
   confirming the tiny Entity/EWoK differences are not optimization-driven.

## Decision

The evidence does not justify expanding the paired-restatement corpus on this evidence. The route as a
standalone lever does not jointly raise Entity and EWoK above a normal FineWeb baseline.

Two proposed alternatives:
- (A) If keeping the route: first run ONE decisive re-check with token-aware, no-truncation or
  equal-truncation packing, so every arm's actual model input is matched. Only then is the
  adjacency-correspondence contrast interpretable.
- (B) Otherwise: redirect compute to an experience mechanism that genuinely co-lifts Entity AND EWoK
  on the ordinary baseline.

## Follow-up decision

The full profile shows true≈orig≈shuffled on Entity/EWoK. Even a clean token-aware re-check would at
best recover a ~+0.15–0.28 Entity signal — far below the −6.25 Entity deficit to the leader. The route
is not a promising SOTA lever. The scientifically efficient move is (B): pivot compute to a mechanism
targeting joint Entity+EWoK lift on the normal baseline, while preserving this negative result as a
closed route.

However, the truncation confound also means the negative result itself is not fully clean. A single
cheap token-aware re-check (option A) would let us CLOSE the route with a clean control rather than an
ambiguous one, and costs only ~30 min of GPU. This is worth doing once before full pivot, so the
closure is decisive and not reopened later.

## Closed-route record

FineWeb paired-restatement decomposition (true/hard/orig/shuffled), 1M single-seed: true_pair does not
beat orig_only or shuffled on Entity (+0.15/+0.28) or EWoK (0.00/+0.09). Apparent hard-negative contrast
is hard-arm damage plus mismatched truncation. Not a standalone SOTA lever. Pending one token-aware
equal-packing re-check before final closure.
