# clean control trained and representation frontier representation-allocation frontier (training-only)

Uses only the frozen legal compact-view-reinvest 10M training pool and already trained legal tokenizers. No official evaluation text, no model training, no route selection by itself.

## Integrity

- Pool rows/words: `64740` / `10000000`
- Pool SHA matched expected: `True`
- a02_step35_16k tokenizer SHA matched expected: `True`
- a01_legal16k tokenizer SHA: `4a95a2a2ead21813a409c5ad7c2c008fbf418dcf3e0dc17bd970bd9ce25ea738`
- a01_24k tokenizer SHA: `92fe399785dd4fb968b35fb686994acd07cc5598c4a53db0f81165ab46aaa7d0`
- a01_32k tokenizer SHA: `9b9c36f1048cd469bf38a2b8b3a60a251a15939832da5c07a37b1a746459e5ae`
- a01_40k tokenizer SHA matched expected: `True`
- a01_40k_minfreq25 tokenizer SHA: `311e7a20cd8b20f512ab574b8d62b574b5106408e742c32b0bce152dfe8162e0`
- a01_40k_minfreq50 tokenizer SHA: `9900f42b392fb69dd55c9c9fd7f539da09b3a33487e4e3542f9b953f48310922`

## Frontier vs legal16k baseline

| tokenizer | vocab | tok/word | visible tok/word | visible groups/word | over256 | trunc toks | raw saving | new mass frac | observed new <50 | mass<50 | source-conc low-tail mass | extra emb params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| a02_step35_16k | 16384 | 1.466452 | 1.429489 | 0.980219 | 15117 | 369626 | 4757 | 0.000324 | 0.000000 | 0.002058 | 0.001076 | 0 |
| a01_legal16k | 16384 | 1.466928 | 1.429878 | 0.980173 | 15143 | 370497 | 0 | 0 | 0 | 0.002031 | 0.001071 | 0 |
| a01_24k | 24576 | 1.428062 | 1.398853 | 0.984204 | 13039 | 292086 | 388660 | 0.025261 | 0.656066 | 0.018336 | 0.002336 | 3932160 |
| a01_32k | 32768 | 1.406627 | 1.381085 | 0.986131 | 11991 | 255416 | 603010 | 0.037544 | 0.855006 | 0.033594 | 0.003403 | 7864320 |
| a01_40k | 40000 | 1.394264 | 1.370616 | 0.987134 | 11407 | 236482 | 726632 | 0.044079 | 0.907256 | 0.041308 | 0.004042 | 11335680 |
| a01_40k_minfreq25 | 29529 | 1.413851 | 1.387134 | 0.985513 | 12335 | 267172 | 530764 | 0.033456 | 0.809154 | 0.028641 | 0.002993 | 6309600 |
| a01_40k_minfreq50 | 19609 | 1.448360 | 1.415290 | 0.982206 | 14096 | 330696 | 185676 | 0.012384 | 0.031464 | 0.003063 | 0.001662 | 1548000 |

## Source distribution of 40k raw-token savings vs legal16k

| source | raw token saving | saving/word | declared words |
|---|---:|---:|---:|
| inherited_qwen_pair_packed | 159792 | 0.096446 | 1656800 |
| simple_wiki | 143257 | 0.135271 | 1059040 |
| gutenberg | 137815 | 0.074120 | 1859360 |
| open_subtitles | 133047 | 0.072726 | 1829440 |
| childes | 64100 | 0.024894 | 2574880 |
| fineweb_source_compact_view_pair | 61253 | 0.144631 | 423511 |
| bnc_spoken | 26882 | 0.046683 | 575840 |
| switchboard | 486 | 0.023011 | 21120 |
| neutral_topup | 0 | 0.000000 | 9 |

Top 40k new-token counts vs legal16k (first 20):
- `Ġmaintenance` count=67 rows=53 top_source=fineweb_source_compact_view_pair top_frac=0.299 Hsrc=0.749
- `ĠFreda` count=67 rows=48 top_source=gutenberg top_frac=0.612 Hsrc=0.472
- `ĠPAUL` count=67 rows=25 top_source=open_subtitles top_frac=1.000 Hsrc=0.000
- `ĠCadet` count=67 rows=44 top_source=gutenberg top_frac=0.701 Hsrc=0.322
- `Ġacknowledged` count=67 rows=64 top_source=inherited_qwen_pair_packed top_frac=0.433 Hsrc=0.635
- `Ġtiming` count=67 rows=59 top_source=open_subtitles top_frac=0.567 Hsrc=0.571
- `Ġembrace` count=67 rows=64 top_source=inherited_qwen_pair_packed top_frac=0.388 Hsrc=0.592
- `Ġbanjo` count=67 rows=33 top_source=childes top_frac=0.881 Hsrc=0.232
- `Ġcalf` count=67 rows=36 top_source=childes top_frac=0.627 Hsrc=0.546
- `Ġrounded` count=67 rows=43 top_source=simple_wiki top_frac=0.478 Hsrc=0.659
- `Ġrealm` count=67 rows=62 top_source=inherited_qwen_pair_packed top_frac=0.478 Hsrc=0.496
- `ĠSeems` count=67 rows=66 top_source=open_subtitles top_frac=0.731 Hsrc=0.443
- `Ġdivorced` count=67 rows=64 top_source=open_subtitles top_frac=0.522 Hsrc=0.560
- `ĠTehran` count=67 rows=42 top_source=simple_wiki top_frac=0.657 Hsrc=0.410
- `ĠJesse` count=67 rows=37 top_source=open_subtitles top_frac=0.358 Hsrc=0.636
- `-con` count=67 rows=51 top_source=inherited_qwen_pair_packed top_frac=0.358 Hsrc=0.751
- `Ġcurly` count=67 rows=50 top_source=gutenberg top_frac=0.418 Hsrc=0.563
- `Ġchoices` count=67 rows=53 top_source=open_subtitles top_frac=0.448 Hsrc=0.691
- `Ġimpatiently` count=67 rows=64 top_source=gutenberg top_frac=0.627 Hsrc=0.415
- `Ġamidst` count=67 rows=64 top_source=inherited_qwen_pair_packed top_frac=0.687 Hsrc=0.438

## Interpretation

This asset separates compression/visibility from sparse lexical allocation. The pending legal40k official result remains a package-level result: tokenizer inventory, target-token burden, embedding/output rows, and accumulated-forward optimization all move together. This file should be joined with 40k score deltas after the collations finish and with clean-vs-reinvest vector after s50_t19 delivers.

JSON: `experiments/archive/frontier_consolidation/data/representation_allocation_frontier/representation_allocation_frontier.json`

CSV: `experiments/archive/frontier_consolidation/data/representation_allocation_frontier/frontier_summary.csv`
