# numerical repair and halfview prestate numerical repair, figure readiness, and HALF_VIEW pre-state

This Execute note repairs the numerical spine before final expression. It uses one polarity throughout: `A_T=N−T` is true-source advantage over a neutral ordinary source, `A_U=N−U` is unrelated-source advantage, and `G=U−T=A_T−A_U` is the true-vs-unrelated gain. Positive `A_T` contrasts mean the first arm uses the true source better than the comparison arm.

## Compact T/U/N identity and weighting repair

Raw row identities hold: maximum row-level identity error across compact source tables is `1.776e-15` and maximum paired-contrast identity error is `3.553e-15`. Thus the earlier apparent mismatch is not an algebra or sample-merge failure.

The difference between some anchor corrected price interpretation and ordinary heldout price probe numbers comes from weighting. Token-level summaries weight every selected masked content token (2,732 nonoverlap probe ids); pair-level summaries average the selected tokens within a pair before contrasting (1,465 pairs in seed43022). Both are computed on the same raw probe table, but answer different aggregation questions. Final central compact figures should state which weighting is used; the numerical repair and halfview prestate central compact bars use token-level masked-target weighting because that is the original probe estimand, with pair-level values saved as a robustness table.

### Central compact token-level nonoverlap summary

| family | contrast | seeds | ΔA_T=N−T | ΔA_U=N−U | ΔG=U−T |
|---|---|---:|---:|---:|---:|
| original_three_seed | RminusC | 3 | -0.8138±0.1729 | +0.0679±0.0340 | -0.8818±0.1483 |
| original_three_seed | VminusC | 3 | +0.7970±0.0758 | -0.0085±0.0536 | +0.8056±0.1090 |
| split_common_two_seed | RSminusC | 2 | -0.0624±0.1016 | +0.0481±0.0105 | -0.1105±0.1121 |
| split_common_two_seed | VSminusC | 2 | +0.0721±0.0618 | +0.0749±0.0156 | -0.0028±0.0462 |
| split_common_two_seed | RminusRS | 2 | -0.7839±0.1296 | +0.0031±0.0355 | -0.7870±0.0941 |
| split_common_two_seed | VminusVS | 2 | +0.6955±0.1410 | -0.0961±0.0537 | +0.7916±0.1947 |

The repaired polarity makes the central result visually direct: original REPEAT has negative true-source advantage relative to CLEAN on compact nonidentical targets, original VIEW has positive true-source advantage, and split residuals are much closer to zero. The local-vs-split rows show the causal attenuation directly: local REPEAT is lower than REPEAT_SPLIT on true-source advantage, while local VIEW is higher than VIEW_SPLIT.

### Token versus pair weighting examples

| seed | contrast | token n | pair n | token ΔG | pair ΔG | token ΔA_T | pair ΔA_T |
|---:|---|---:|---:|---:|---:|---:|---:|
| 43022 | RSminusC | 2732 | 1465 | -0.0313 | -0.0470 | +0.0094 | -0.0043 |
| 43022 | RminusC | 2732 | 1465 | -0.7517 | -0.7852 | -0.6828 | -0.7133 |
| 43022 | VSminusC | 2732 | 1465 | +0.0298 | +0.0040 | +0.1157 | +0.0874 |
| 43022 | VminusC | 2732 | 1465 | +0.6837 | +0.6711 | +0.7115 | +0.7012 |
| 43122 | RSminusC | 2732 | 1465 | -0.1898 | -0.1913 | -0.1343 | -0.1442 |
| 43122 | RminusC | 2732 | 1465 | -1.0433 | -1.0550 | -1.0098 | -1.0337 |
| 43122 | VSminusC | 2732 | 1465 | -0.0355 | -0.0531 | +0.0284 | +0.0005 |
| 43122 | VminusC | 2732 | 1465 | +0.8938 | +0.8891 | +0.8236 | +0.8068 |

## Wikipedia natural-domain aggregation repair

The causal lm objective generality plan natural-domain interpretation is retained, but the old `ALL` bar is now explicitly split into a class-balanced mean and a token-weighted mean. The class-balanced mean averages source-recurring and source-absent target classes equally; the token-weighted mean uses the actual frozen design ratio, 1,200 source-recurring targets and 2,400 source-absent targets.

| aggregation | target class | contrast | Δ(N−T) | Δ(U/N nuisance) | seeds |
|---|---|---|---:|---:|---:|
| pair_mean_by_target_class | overlap | RminusC | +0.1090±0.1216 | +0.0123±0.0271 | 3 |
| pair_mean_by_target_class | overlap | VminusC | +0.2376±0.0706 | +0.0069±0.0159 | 3 |
| pair_mean_by_target_class | overlap | VminusR | +0.1286±0.0533 | -0.0054±0.0355 | 3 |
| pair_mean_by_target_class | nonoverlap | RminusC | -0.2171±0.0285 | +0.0110±0.0499 | 3 |
| pair_mean_by_target_class | nonoverlap | VminusC | +0.0316±0.0398 | -0.0052±0.0255 | 3 |
| pair_mean_by_target_class | nonoverlap | VminusR | +0.2487±0.0646 | -0.0162±0.0280 | 3 |
| class_balanced_pair_mean | ALL_class_balanced | RminusC | -0.0540±0.0540 | +0.0116±0.0383 | 3 |
| class_balanced_pair_mean | ALL_class_balanced | VminusC | +0.1346±0.0539 | +0.0008±0.0168 | 3 |
| class_balanced_pair_mean | ALL_class_balanced | VminusR | +0.1887±0.0116 | -0.0108±0.0314 | 3 |
| token_weighted_target_mean | ALL_token_weighted | RminusC | -0.1084±0.0336 | +0.0114±0.0421 | 3 |
| token_weighted_target_mean | ALL_token_weighted | VminusC | +0.1003±0.0488 | -0.0012±0.0192 | 3 |
| token_weighted_target_mean | ALL_token_weighted | VminusR | +0.2087±0.0270 | -0.0126±0.0302 | 3 |

This confirms the corrected target-form result: VIEW improves true-source use for source-recurring target tokens under changed sentence form; REPEAT is reliably below CLEAN on source-absent substitutions; VIEW is near CLEAN but above REPEAT on those substitutions. The `ALL_class_balanced` and `ALL_token_weighted` values are both saved so later figures cannot accidentally present a class-balanced mean as a pooled target average.

## Hash-mixed wording repair and HALF_VIEW control

The hash-mixed seed43022 arm should not by itself be described as proving that exact-recurrence admixture interferes with changed-token/source-conditioned restatement use. It received only half the VIEW same-window rewrite pair dose. HM retaining near-full natural-copy gain while showing a smaller positive compact restatement readout is compatible with coexisting practiced relation behavior, but the VIEW shortfall can be produced either by a half-dose restatement curve or by active suppression from exact-recurrence adjacency. The observed placement makes suppression plausible because HM's unrelated-source compact term is nearly the VIEW endpoint while the true-source term is lower, but that is not a separated causal result.

Pre-stated HALF_VIEW control: keep the same deterministic hash assignment used by HM. For pairs assigned `view`, train the original source+rewrite same-window segment. For pairs assigned `repeat` in HM, keep the selected source but replace the exact-copy companion slot with length-matched CLEAN neutral text, so no exact recurrence relation is practiced in that slot. This preserves the selected sources, row lengths, 100M word budget, tokenizer/model coordinate, and training order while giving a half-dose VIEW relation without the competing exact-copy companion.

Predictions before seeing HALF_VIEW scores:
- If HALF_VIEW matches HM on compact true-source restatement terms, the HM shortfall is primarily relation-dose; the composition statement becomes coexistence of exact-copy and reduced-dose restatement behavior.
- If HALF_VIEW approaches full VIEW while HM remains lower, exact adjacency actively suppresses restatement source use when the two local relations share the diet; that would be direct evidence about composition of practiced relations under finite budget.
- In either case, the current HM result stays auxiliary until HALF_VIEW or additional seeds separate dose from competition.

## Repaired figure outputs

- `experiments/archive/relation_learning/figures/numerical_repair/fig1_compact_TUN_positive_true_source_use.png`
- `experiments/archive/relation_learning/figures/numerical_repair/fig1_compact_TUN_positive_true_source_use.pdf`
- `experiments/archive/relation_learning/figures/numerical_repair/fig2_compact_local_vs_split_true_source_advantage.png`
- `experiments/archive/relation_learning/figures/numerical_repair/fig2_compact_local_vs_split_true_source_advantage.pdf`
- `experiments/archive/relation_learning/figures/numerical_repair/fig3_wikipedia_target_class_transfer_repaired.png`
- `experiments/archive/relation_learning/figures/numerical_repair/fig3_wikipedia_target_class_transfer_repaired.pdf`

Data outputs:
- `experiments/archive/relation_learning/data/numerical_repair/compact_TUN_central_token_summary_positive_true_source_use.csv`
- `experiments/archive/relation_learning/data/numerical_repair/compact_TUN_common_contrasts_token_and_pair.csv`
- `experiments/archive/relation_learning/data/numerical_repair/compact_repair_summary.json`
- `experiments/archive/relation_learning/data/numerical_repair/compact_token_vs_pair_weighting_reconciliation.csv`
- `experiments/archive/relation_learning/data/numerical_repair/wikipedia_recompute_summary.json`
- `experiments/archive/relation_learning/data/numerical_repair/wikipedia_transfer_seed_values_recomputed.csv`
- `experiments/archive/relation_learning/data/numerical_repair/wikipedia_transfer_summary_recomputed_classbalanced_and_tokenweighted.csv`
