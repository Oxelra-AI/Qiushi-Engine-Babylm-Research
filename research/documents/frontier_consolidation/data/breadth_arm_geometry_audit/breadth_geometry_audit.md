# pre result reading order and replication plan MAX breadth arm geometry audit (view vs breadth, changed rows only)

Rows audited: 7923; seq_length 256; fixed spatial repair route status legal16k tokenizer.
Whitespace word sequence identical view vs breadth: `True`

## Changed-block totals

| quantity | view | repeat | breadth | breadth vs view |
|---|---|---|---|---|
| legal16k tokens | 1610409.0 | 1551594.0 | 1568514.0 | -2.602% |
| visible tokens (seq256) | 1608404.0 | 1550677.0 | 1567945.0 | -2.515% |
| WWM groups | 1117514.0 | 1118064.0 | 1118356.0 | +0.075% |

## Paired per-row differences (breadth minus view)

| quantity | mean | mean_abs | sd | frac_zero | p05 | p95 |
|---|---|---|---|---|---|---|
| tokens_full | -5.287769784172662 | 11.421557490849425 | 13.53505685584237 | 0.03029155622870125 | -26.0 | 17.0 |
| tokens_visible | -5.106525306070933 | 11.140855736463461 | 13.075630700584133 | 0.03319449703395179 | -25.0 | 17.0 |
| wwm_groups_visible | 0.10627287643569355 | 0.14413732172157012 | 1.3159531553442931 | 0.9796794143632462 | 0.0 | 0.0 |
| maskable_visible | -5.107913669064748 | 11.140224662375363 | 13.074533344060752 | 0.03319449703395179 | -25.0 | 17.0 |
| tokens_dropped | -0.18124447810172914 | 0.2807017543859649 | 2.491680792245725 | 0.9752618957465606 | 0.0 | 0.0 |

## Junction / punctuation (breadth minus view, per changed row)

| mark | mean delta | view mean | breadth mean |
|---|---|---|---|
| sentence_end_marks | -2.0329420673987126 | 8.577559005427236 | 6.544616938028525 |
| period | -1.969077369683201 | 8.69304556354916 | 6.72396819386596 |
| comma | -0.17102107787454246 | 7.207875804619462 | 7.03685472674492 |
| digits | -0.24574024990533888 | 5.373722074971601 | 5.1279818250662625 |
| cap_words | -1.3287895998990282 | 16.882494004796165 | 15.553704404897134 |

## Replaced companion population

- `max_compact_rewrites`: segments 33291, words 428122, tokens 654448, fertility 1.5286483759302254, TTR 0.09179859946463859, mean word chars 5.543480596652356, capitalized fraction 0.14839461648782357
- `max_selected_breadth_sentences`: segments 15123, words 428122, tokens 612553, fertility 1.430790755906027, TTR 0.09030369847847108, mean word chars 5.092758606191693, capitalized fraction 0.12380349526536828
- `max_source_sentences`: segments 33291, words 690465, tokens 955961, fertility 1.384517680114126, TTR 0.06289095030160834, mean word chars 5.091329756033977, capitalized fraction 0.10171261396305388

Unigram JS divergence (bits): rewrite vs breadth 0.1242; rewrite vs source 0.0609; breadth vs source 0.0691
Selected breadth document overlap with MAX sources: 0.9490864298911927

## Conditions to carry with any view-minus-breadth reading

- changed-block legal16k tokens differs by -2.602% (> 1.0% tolerance): carry as an explicit condition of view-minus-breadth
- changed-block visible tokens after seq256 differs by -2.515% (> 1.0% tolerance): carry as an explicit condition of view-minus-breadth
- changed rows contain -2.033 sentence-final marks per row versus view: boundary geometry differs
- breadth material shares documents with MAX sources for a large fraction of selected sentences: this is same-population sentence breadth, not new-document breadth
- replaced companion unigram distribution differs from compact rewrites (JS=0.1242 bits): allocation contrast includes a lexical-population change
