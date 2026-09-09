# corrected entity gate stats — Corrected Entity discriminating-span support & SGCR gate statistics

## Entity with official `nothing` skip
- Items after skip: 6780 (was 9483 raw without skip)
- Shared tokens: 1090072
- Discriminating tokens: 87145
- disc_token_share: 0.0740
- lt20: shared_frac=0.000000, disc_frac=0.000000, ratio=None, disc_share_of_low=0.0000
- lt50: shared_frac=0.003036, disc_frac=0.015882, ratio=5.2318, disc_share_of_low=0.2949
- lt100: shared_frac=0.114400, disc_frac=0.022216, ratio=0.1942, disc_share_of_low=0.0153

## SuperGLUE support reconciliation
- Direct recount frac_lt50: 0.000000
- tokenizer support spectrum reported: 0.0816
- discriminating span support reported: 0.0738
- Total tokens: 0

## Gate statistics (K=50)
- Used types: 39320
- Mean rho: 0.4559
- Weighted mean rho: 0.935720
- Low-rho types (rho<0.5): 24854
- Low-rho mass fraction: 0.041308
- All components ge50 in low-rho: 0.9494246398969984

## Parameter budget (hidden=384, 12x384 model)
- d_comp=32: +536,960 params
- d_comp=48: +805,248 params
- d_comp=64: +1,073,536 params
- d_comp=96: +1,610,112 params
- d_comp=128: +2,146,688 params

JSON: `experiments/archive/representation_and_objectives/data/corrected_entity_gate_stats/corrected_entity_gate_stats.json`
