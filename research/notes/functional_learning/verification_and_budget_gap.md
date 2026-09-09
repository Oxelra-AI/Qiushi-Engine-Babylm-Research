# budget matched design note precheck: branching experiment verification and budget-matching problem

This note verifies the earlier integration note's quantitative claims directly from `data/branching/results.json` and records why a new budget-matched full-objective comparison is necessary.

## Acquisition epochs actually used for branch starts

| seed | branch-start acq_epoch | total after +500 branch |
|---:|---:|---:|
| 42 | 300 | 800 |
| 43 | 500 | 1000 |
| 100 | 400 | 900 |

The earlier 3/3 versus 1/3 comparison is not budget-matched: branching experiment `switch_qfirst_full` receives 300–500 focused-preparation epochs plus 500 full-objective epochs, while query first binding compact summary `qfirst_full_500` receives only 500 full-objective epochs from scratch.

## branching experiment full-objective recovery at final branch epoch

| seed | total_epoch | top4 | B-swap | Q-swap | held_top4 | held_NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 800 | 0.998 | 15.100 | 14.816 | 0.480 | 6.808 |
| 43 | 1000 | 1.000 | 9.242 | 8.913 | 0.465 | 2.863 |
| 100 | 900 | 1.000 | 12.956 | 12.328 | 0.250 | 4.813 |

Threshold check (top4>0.95, B-swap>5, Q-swap>5 for all seeds): **True**.

## Context-only branch final binding check

| seed | final top4 | final B-swap | final Q-swap |
|---:|---:|---:|---:|
| 42 | 0.275 | 0.289 | 0.128 |
| 43 | 0.246 | 0.056 | -0.034 |
| 100 | 0.260 | 0.044 | 0.021 |

Context-only below top4 0.30 for all seeds: **True**.

## Blocked query-to-context evaluation at the bound checkpoint

| seed | standard top4 | blocked top4 | standard B-swap | blocked B-swap |
|---:|---:|---:|---:|---:|
| 42 | 0.990 | 0.242 | 10.185 | 0.648 |
| 43 | 0.998 | 0.256 | 7.574 | -0.141 |
| 100 | 1.000 | 0.271 | 9.361 | 0.302 |

Immediate blocked-evaluation collapse for all seeds: **True**.

## Held-entity comparison available in branching experiment

| seed | continue held_top4 | switch-full held_top4 | continue held_NLL | switch-full held_NLL |
|---:|---:|---:|---:|---:|
| 42 | 0.512 | 0.480 | 8.691 | 6.808 |
| 43 | 0.875 | 0.465 | 0.420 | 2.863 |
| 100 | 0.977 | 0.250 | 0.088 | 4.813 |

branching experiment did **not** generate held-entity B-swap or held-entity Q-swap probes; it only has a held standard probe plus train-entity B/Q swap probes. budget matched design note therefore needs to add held B-swap/Q-swap metrics in the budget-matched run.

## query first binding compact summary fresh full-objective baseline at 500 epochs (not budget matched)

| seed | top4 | B-swap | Q-swap | held_top4 |
|---:|---:|---:|---:|---:|
| 42 | 0.246 | -0.000 | -0.003 | 0.180 |
| 43 | 1.000 | 9.721 | 9.766 | 0.414 |
| 100 | 0.307 | 0.130 | 0.121 | 0.254 |

These query first binding compact summary numbers are useful historical evidence but use only 500 epochs and a different probe bank, so they cannot decide the curriculum-efficiency question.

## Consequence for budget matched design note

Run a new comparison on a common cumulative budget. For each seed, compare (i) fresh query-first full-objective training through the same total epochs, (ii) bound answer-only preparation followed by full objective, and (iii) an equally trained but unbound bag-answer preparation followed by full objective. Track acquisition/reacquisition curves and held-entity swap probes. A curriculum advantage requires earlier or more reliable strong full-objective binding under the same total experience/compute; endpoint success after extra preparation alone is insufficient.
