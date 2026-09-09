# pre result reading order and replication plan whole-pool token / WWM accounting for the 2.64x dose arms

Fixed spatial repair route status legal16k tokenizer; seq_length 256; WWM p=0.15; complete 10M pools.

| arm | rows | words | legal16k tokens | visible tokens | WWM groups | tokens/word | groups/word |
|---|---|---|---|---|---|---|---|
| max_view | 65313 | 10000000 | 14628730 | 14291088 | 9818796 | 1.46287 | 0.98188 |
| max_repeat | 65313 | 10000000 | 14569915 | 14233361 | 9819346 | 1.45699 | 0.98193 |
| max_clean | 65313 | 10000000 | 14688636 | 14338626 | 9812536 | 1.46886 | 0.98125 |
| max_breadth | 65313 | 10000000 | 14586835 | 14250629 | 9819638 | 1.45868 | 0.98196 |
| mid_view | 65041 | 10000000 | 14642995 | 14288934 | 9810159 | 1.46430 | 0.98102 |
| mid_repeat | 65041 | 10000000 | 14600997 | 14247749 | 9810535 | 1.46010 | 0.98105 |

## Relative shift versus the MAX view arm

| arm | tokens % | visible tokens % | WWM groups % | word delta |
|---|---|---|---|---|
| max_repeat | -0.4021 | -0.4039 | +0.0056 | 0 |
| max_clean | +0.4095 | +0.3326 | -0.0638 | 0 |
| max_breadth | -0.2864 | -0.2831 | +0.0086 | 0 |
| mid_view | +0.0975 | -0.0151 | -0.0880 | 0 |
| mid_repeat | -0.1896 | -0.3033 | -0.0841 | 0 |

## Interpretation

- BabyLM budget is counted in words, so exactly word-matched arms are not automatically token-matched.
- Compact rewrites tokenize less efficiently than raw sentences under the fixed legal16k tokenizer, so the view arm carries more subword targets per row than repeat/clean/breadth.
- Any view-repeat, view-clean, or view-breadth reading therefore includes a small systematic token/compute asymmetry that must be stated with the effect size.
