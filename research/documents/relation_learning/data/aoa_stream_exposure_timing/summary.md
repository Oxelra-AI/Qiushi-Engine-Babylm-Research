# earlier analysis AoA stream exposure timing

Exact lower-cased word-boundary stream counts approximate lexical exposure timing. This is a route-selection/prediction instrument for legal curriculum design, not an official AoA score and not evidence that endpoint competence changes.

## Scan

- Stream rows: 647400; consumed words from row metadata: 100000000; inferred passes over a 10M pool: 10.
- Child-AoA words in records: 406; model-AoA fitted words: 225; both: 225.

## Core correlations

- Current exposure crossing vs measured model AoA: r=-0.1479, p=0.0266, n=225.
- Current exposure crossing vs child AoA: r=-0.0096, p=0.8471, n=406.
- Measured model AoA vs child AoA (recomputed on joined words): r=-0.0349, p=0.6030, n=225.
- Whole-stream log frequency vs child AoA: r=-0.0714, p=0.1513.
- Whole-stream log frequency vs measured model AoA: r=-0.6214, p=0.0000.

## Source log-frequency relation to child AoA

| source | words | frac | r(logfreq, child AoA) | p |
|---|---:|---:|---:|---:|
| childes | 25748800 | 0.257 | -0.2047 | 0.0000 |
| neutral_cleanqwen_topup_compact_reinvest::open_subtitles | 90 | 0.000 | -0.0622 | 0.2111 |
| simple_wiki | 10590400 | 0.106 | -0.0261 | 0.5995 |
| cleanqwen_fineweb_compact_view_reinvest | 4235110 | 0.042 | -0.0071 | 0.8868 |
| qwen_pair_packed | 16568000 | 0.166 | -0.0010 | 0.9847 |
| gutenberg | 18593600 | 0.186 | 0.0058 | 0.9072 |
| open_subtitles | 18294400 | 0.183 | 0.0372 | 0.4550 |
| bnc_spoken | 5758400 | 0.058 | 0.0421 | 0.3980 |
| switchboard | 211200 | 0.002 | 0.1592 | 0.0013 |

## Candidate source-order exposure crossings

| order | r(cross, child AoA) on official fitted words | p | r(cross, model AoA) |
|---|---:|---:|---:|
| alphabetical_sources | 0.0401 | 0.5498 | -0.1373 |
| largest_sources_first | -0.0023 | 0.9722 | -0.1903 |
| source_child_aoa_enrichment_first | -0.0148 | 0.8252 | -0.2521 |
| childes_bnc_simple_first | -0.0221 | 0.7417 | -0.1287 |
| childes_spoken_first | -0.0288 | 0.6677 | -0.1178 |
| qwen_clean_first_control | -0.0405 | 0.5452 | -0.2092 |
| smallest_sources_first | NA | NA | NA |
| written_first_control | NA | NA | NA |

Full JSON: `experiments/archive/relation_learning/data/aoa_stream_exposure_timing/summary.json`
