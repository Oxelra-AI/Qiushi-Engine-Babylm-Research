# active endpoint consumption and u256 interpretation — U256 recovered-tail quality profile

CPU-only descriptive profile of text newly visible under U256 and hidden under spatial repair route status row256. It must not be used as benchmark-conditioned data selection.

- Tail rows: 15,117; recovered tokens: 369,626; recovered full words: 197,806.
- Cleanish rows by heuristic: 5,639 (37.30%); encoding-noise rows: 91 (0.60%); all-caps/glued rows: 1,115 (7.38%).
- CHILDES-like rows: 10,430 (69.00%); subtitle-like rows: 1,155 (7.64%); list/TOC-like rows: 856 (5.66%).

## By source
- childes: 223,846 recovered tokens (60.56% of all), tail rows 10,429, cleanish 20.24%, encoding-noise 0.00%, allcaps/glued 7.74%.
- simple_wiki: 74,970 recovered tokens (20.28% of all), tail rows 3,065, cleanish 77.16%, encoding-noise 0.69%, allcaps/glued 0.42%.
- open_subtitles: 38,835 recovered tokens (10.51% of all), tail rows 1,127, cleanish 72.94%, encoding-noise 4.17%, allcaps/glued 20.94%.
- gutenberg: 29,993 recovered tokens (8.11% of all), tail rows 356, cleanish 59.83%, encoding-noise 6.46%, allcaps/glued 16.29%.
- qwen_pair_packed: 1,279 recovered tokens (0.35% of all), tail rows 82, cleanish 86.59%, encoding-noise 0.00%, allcaps/glued 0.00%.
- cleanqwen_fineweb_compact_view_reinvest: 636 recovered tokens (0.17% of all), tail rows 52, cleanish 98.08%, encoding-noise 0.00%, allcaps/glued 1.92%.
- bnc_spoken: 67 recovered tokens (0.02% of all), tail rows 6, cleanish 100.00%, encoding-noise 0.00%, allcaps/glued 0.00%.

Scientific consequence: U256 is not just a uniform token-count increase. It exposes concentrated row tails dominated by CHILDES, SimpleWiki, OpenSubtitles, and Gutenberg; a nontrivial part contains transcript, list, all-caps/glued, or encoding artifacts. Endpoint interpretation should distinguish useful long-row suffix credit from noise-driven competence rotation.

JSON: `experiments/archive/frontier_consolidation/data/u256_tail_quality/u256_tail_quality.json`
CSV: `experiments/archive/frontier_consolidation/data/u256_tail_quality/tail_quality_by_source.csv`, `experiments/archive/frontier_consolidation/data/u256_tail_quality/top_recovered_tail_noise_rows.csv`
