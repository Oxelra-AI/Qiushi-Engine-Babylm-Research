# copy and relevant update result: copy score and queried-state update split

## Why this step mattered

The earlier analysis interpretation still mixed two possible causes of the Entity crossover: exact repetition might train a general in-window copy operation, or the depth effect might be only a consequence of longer Entity contexts, recency, or benchmark text fit. copy and relevant update result directly measured copy gain and separated Entity items by updates to the queried box while seed43222 training continued asynchronously.

## Training construction fact

The MAX changed block contains 7,924 packed rows and 33,291 source/companion pairs. The first-pass stream rebuilds exactly as `source_text + rewrite_text` for VIEW and `source_text + rotated exact source-token segment` for REPEAT in 7,923/7,924 and 7,923/7,924 rows; all 33,291 pairs are source followed by their own companion. Each individual pair fits the 256-token tokenizer window in both arms; packed rows fit fully in 97.49% of VIEW rows and 98.90% of REPEAT rows.

## Direct copy-score result

Copy gain is `NLL(unrepeated control) - NLL(repeated)`, so larger means the model benefits more from an exact unmasked copy of the target/span elsewhere in the same bidirectional MLM context.

| probe | contrast | seed43022 | seed43122 | cross-seed mean |
|---|---|---:|---:|---:|
| actual source-repeat packet | RminusC | +0.588 | +0.721 | +0.654 |
| actual source-repeat packet | RminusV | +0.453 | +0.786 | +0.620 |
| actual source-repeat packet | CminusV | -0.134 | +0.065 | -0.035 |
| random token span | RminusC | -0.142 | -0.377 | -0.260 |
| random token span | RminusV | -0.102 | -0.025 | -0.063 |
| random token span | CminusV | +0.040 | +0.352 | +0.196 |

The copy score supports the copy side of the account only in the natural packet family: on actual source-repeat packets REPEAT has higher copy gain than CLEAN and VIEW in both seeds (R-C +0.588/+0.721; R-V +0.453/+0.786). On random nonsemantic token spans the REPEAT advantage is absent or negative on average, so the result is not a generic ability to exploit arbitrary repeated tokens; it is tied to the trained natural source-repeat packet structure.

## Entity queried-state update split

Parsed relevant updates to the queried box match the dataset `numops` for 6,533/6,780 official-filtered Entity items; the 247 mismatches are mainly move-contents items with extra irrelevant operations. Crucially, the `rel_updates_0` group still has mean total operation count 3.39 and mean prefix length 88.7 words, so it tests no update to the queried state in contexts that often contain irrelevant operations, not only trivial short contexts.

| relevant updates | contrast | seed43022 | seed43122 | cross-seed mean |
|---:|---|---:|---:|---:|
| 0 | RminusV | +9.69 | +9.15 | +9.42 |
| 0 | RminusC | +8.78 | +9.09 | +8.93 |
| 0 | VminusC | -0.91 | -0.06 | -0.49 |
| 1 | RminusV | -2.19 | -2.67 | -2.43 |
| 1 | RminusC | -1.39 | -3.22 | -2.31 |
| 1 | VminusC | +0.81 | -0.55 | +0.13 |
| 2 | RminusV | -4.13 | -3.19 | -3.66 |
| 2 | RminusC | -0.18 | -1.79 | -0.99 |
| 2 | VminusC | +3.95 | +1.40 | +2.67 |
| 3 | RminusV | -8.46 | -6.40 | -7.43 |
| 3 | RminusC | -5.83 | -3.09 | -4.46 |
| 3 | VminusC | +2.63 | +3.31 | +2.97 |
| 4 | RminusV | -9.14 | -8.24 | -8.69 |
| 4 | RminusC | -1.59 | -0.24 | -0.91 |
| 4 | VminusC | +7.55 | +8.00 | +7.78 |
| 5 | RminusV | -9.74 | -3.68 | -6.71 |
| 5 | RminusC | -2.06 | +2.06 | +0.00 |
| 5 | VminusC | +7.68 | +5.74 | +6.71 |

The crossover is controlled by whether the queried state is changed. REPEAT beats VIEW by about +9 points at zero relevant updates, while VIEW beats REPEAT after any relevant update and by roughly +8 to +9 points for 3-4 updates. VIEW-over-CLEAN is near zero at zero and one relevant update and grows for multiple updates, which preserves the positive residual that copying alone cannot explain.

## Reading task scope

The official Reading score is not a context retrieval task. The evaluator regresses human reading-time variables on model surprisal while controlling lexical frequency, word length, and context length; it uses `pred` and `prev_pred` as surprisal-like scalar predictors, not a discrete answer copied from the context. Therefore the copy/state trade-off should not be inferred from Reading without a separate analysis of surprisal dynamics.

## Current scientific interpretation

The strongest supported mechanism now has two parts. Exact in-window source repetition trains a natural-text copy/use-the-earlier-span computation that explains REPEAT's zero-update Entity advantage and its stale-state tendency when the copied state is superseded. Varied restatement removes that copy training and adds a separate multi-update advantage over CLEAN, suggesting better reading of context by content rather than by verbatim recurrence. The general data-efficient learning candidate is therefore a fixed-budget competition between copyable recurrence and nonidentical relational evidence, not a scalar repetition-versus-diversity value law. It still needs the third seed and, if stable, representation probes of the VIEW-over-CLEAN multi-update residual.
