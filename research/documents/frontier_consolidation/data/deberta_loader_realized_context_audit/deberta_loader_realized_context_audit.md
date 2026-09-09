# lead route assessment after source use probe exact DeBERTa-loader realized-context audit

CPU/tokenizer audit of the density cleanqwen overlay medium riskhard changed block reconstructed with lead factorial route reassessment candidate views.  It measures the actual row-level DeBERTa MLM loader geometry and a CPU proxy of fixed-seed WWM masking; no model was loaded and no training was run.

## Variant summaries
| variant | rows | words | active tokens | trunc tokens | trunc rows | source/view active | candidate groups | pair full visible | mean source→view min gap | CPU masked tokens | CPU masked view/source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 3,006 | 423,520 | 607,273 | 636 | 52 | 359,905/247,289 | 423,179 | 99.55% | 12.35 | 91,360 | 37,303/53,989 |
| compact_scrambled | 3,006 | 423,520 | 607,273 | 636 | 52 | 359,905/247,124 | 423,171 | 99.55% | 12.36 | 90,803 | 36,831/53,806 |
| sourcewide_onegap | 3,006 | 423,520 | 596,804 | 487 | 34 | 359,919/236,822 | 423,259 | 99.70% | 11.85 | 89,248 | 35,496/53,732 |
| sourcewide_onegap_scrambled | 3,006 | 423,520 | 596,804 | 487 | 34 | 359,919/236,699 | 423,253 | 99.70% | 11.85 | 88,753 | 35,180/53,499 |
| prefix_fluent | 3,006 | 423,520 | 576,463 | 157 | 19 | 359,964/216,436 | 423,424 | 99.84% | 11.03 | 86,063 | 32,255/53,767 |
| prefix_scrambled | 3,006 | 423,520 | 576,463 | 157 | 19 | 359,964/216,352 | 423,426 | 99.84% | 11.03 | 85,735 | 32,265/53,377 |

## Fixed-word-multiset contrast equality
- `compact_minus_compact_scrambled`: Δ active 0, Δ candidate groups 8, Δ view active 165, Δ trunc tokens 0, CPU-proxy Δ masked tokens 557, CPU-proxy Δ masked view/source 472/183. Nonzero-row counts for active/group/mask: 0/30/2806.
- `sourcewide_onegap_minus_sourcewide_onegap_scrambled`: Δ active 0, Δ candidate groups 6, Δ view active 123, Δ trunc tokens 0, CPU-proxy Δ masked tokens 495, CPU-proxy Δ masked view/source 316/233. Nonzero-row counts for active/group/mask: 0/18/2818.
- `prefix_fluent_minus_prefix_scrambled`: Δ active 0, Δ candidate groups -2, Δ view active 84, Δ trunc tokens 0, CPU-proxy Δ masked tokens 328, CPU-proxy Δ masked view/source -10/390. Nonzero-row counts for active/group/mask: 0/11/2757.

## Scientific reading
- The compact ordered/scrambled arm remains mechanically close under the actual row loader: legal words and pair positions are identical, active/candidate masses differ only at the token-level boundary induced by tokenization/truncation. The CPU mask proxy reveals the size of realized WWM-target drift that would remain even with identical legal words.
- Sourcewide and prefix ordered/scrambled controls are similarly clean within each word multiset, but their cross-family comparison still changes copy composition, tail mass, and active target exposure; this audit does not make them pure tail-coverage or semantic-transformation tests.
- Because this is changed-block-only and CPU-mask-proxy evidence, it supports design hygiene only. It must be combined with delivered DeBERTa grids and compact directional evidence before deciding whether any H100 training screen is worth its cost.

JSON: `experiments/archive/frontier_consolidation/data/deberta_loader_realized_context_audit/deberta_loader_realized_context_audit.json`
