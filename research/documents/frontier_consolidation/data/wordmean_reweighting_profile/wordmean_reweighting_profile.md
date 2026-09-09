# wordmean screen and substrate constraints word-mean reweighting profile

Training-text-only sampled actual WWM batches; no model forward, no official evaluation text, no GPU.

Sampled `79` batches, `453739` selected groups, `662183` selected tokens; mean selected tokens/group `1.4594`.

## Largest average-batch credit-share shifts

| category | wordmean share | tokenmean share | Δ share | ratio | groups | tokens | examples |
|---|---:|---:|---:|---:|---:|---:|---|
| bpe_len::1 | 69.423% | 47.595% | 21.828% | 1.459 | 314902 | 314902 | the; and; a; to; of |
| bpe_len::3 | 7.974% | 16.385% | -8.412% | 0.487 | 36190 | 108570 | *CHI:; *MOT:; *FAT:; *MAR:; *INV: |
| has_upper::True | 18.827% | 26.692% | -7.865% | 0.705 | 85483 | 176919 | *MOT:; *CHI:; I; The; *FAT: |
| has_upper::False | 81.173% | 73.308% | 7.865% | 1.107 | 368256 | 485264 | the; and; to; a; of |
| bpe_len::2 | 19.932% | 27.319% | -7.387% | 0.730 | 90517 | 181034 | I'm; don't; it's; that's; it. |
| has_any_cue::True | 25.701% | 21.394% | 4.306% | 1.201 | 116659 | 141649 | I; in; you; on; he |
| has_any_cue::False | 74.299% | 78.606% | -4.306% | 0.945 | 337080 | 520534 | the; and; to; a; of |
| cue::none | 74.299% | 78.606% | -4.306% | 0.945 | 337080 | 520534 | the; and; to; a; of |
| alpha_len::long_8_12 | 8.496% | 11.805% | -3.310% | 0.720 | 38654 | 78416 | something; February; anything; different; everything |
| source::childes | 25.146% | 28.401% | -3.254% | 0.885 | 113602 | 187529 | *MOT:; *CHI:; the; you; I |
| alpha_len::short_0_3 | 47.498% | 44.787% | 2.711% | 1.061 | 215507 | 296467 | the; and; to; a; of |
| bpe_len::4 | 1.550% | 4.247% | -2.697% | 0.365 | 7035 | 28140 | *INV1:; *SIS2:; 100; 27,; 14, |
| has_digit::True | 0.999% | 3.198% | -2.199% | 0.312 | 4544 | 21205 | *INV1:; 1; 5; 2; 3 |
| has_digit::False | 99.001% | 96.802% | 2.199% | 1.023 | 449195 | 640978 | the; and; to; a; of |
| cue::mental_social_dialogue | 9.956% | 8.010% | 1.946% | 1.243 | 45271 | 53136 | I; you; he; her; they |
| source::gutenberg | 18.592% | 16.942% | 1.650% | 1.097 | 84552 | 112416 | the; and; to; of; a |
| bpe_len::5 | 0.618% | 2.115% | -1.497% | 0.292 | 2809 | 14045 | 2017; 2022; 2021; 2016; 2020 |
| source::inherited_qwen_pair_packed | 16.947% | 15.607% | 1.339% | 1.086 | 77056 | 103496 | the; and; to; of; a |
| alpha_len::medium_4_7 | 43.635% | 42.383% | 1.252% | 1.030 | 197889 | 280500 | that; with; they; this; have |
| cue::spatial_state | 5.894% | 4.773% | 1.121% | 1.235 | 26734 | 31598 | in; on; at; from; up |
| bpe_len::6 | 0.354% | 1.457% | -1.103% | 0.243 | 1613 | 9678 | 2022,; 2019,; 2022); 2019.; 2017, |
| source::simple_wiki | 10.314% | 11.281% | -0.967% | 0.914 | 46758 | 74659 | =; the; of; and; in |
| source::bnc_spoken | 5.823% | 4.976% | 0.846% | 1.170 | 26576 | 33112 | the; to; and; I; a |
| alpha_len::verylong_13plus | 0.371% | 1.024% | -0.653% | 0.362 | 1689 | 6800 | international; International; responsibility; Massachusetts; approximately |
| cue::causal_temporal | 3.346% | 2.713% | 0.633% | 1.233 | 15155 | 17900 | if; when; would; will; could |
| bpe_len::8plus | 0.076% | 0.532% | -0.456% | 0.143 | 344 | 3511 | 200,000; 2004–05; 2003–04; CentraleSupélec; childes/CHILDES_NA/HSLLD/HV7/ET/kuret7.cha |
| source::open_subtitles | 18.557% | 18.249% | 0.308% | 1.017 | 83997 | 120531 | the; I; to; -; a |
| cue::physical_dynamics | 1.368% | 1.077% | 0.291% | 1.271 | 6178 | 7094 | put; make; take; made; give |
| cue::quantity_measure | 1.991% | 1.707% | 0.284% | 1.166 | 9015 | 11284 | one; two; more; much; three |
| bpe_len::7 | 0.073% | 0.350% | -0.277% | 0.209 | 329 | 2303 | 1970s.; 10,000; 1960s.; 1990s.; 30,000 |
| cue::material_property | 1.570% | 1.403% | 0.167% | 1.119 | 7129 | 9289 | little; good; big; long; old |
| cue::negation_modality | 2.618% | 2.493% | 0.125% | 1.050 | 11902 | 16521 | not; don't; no; if; would |
| source::fineweb_source_compact_view_pair | 4.385% | 4.314% | 0.071% | 1.016 | 20118 | 28907 | the; and; of; to; a |
| source::switchboard | 0.236% | 0.229% | 0.006% | 1.027 | 1080 | 1533 | B:; A:; I; and; the |

## Interpretation

Word-mean credit favors categories concentrated in one-piece/short/common groups and reduces credit for categories carried by multi-piece/long/name-like groups. This profile helps decide whether a score change is broad late-benefit strengthening or a redistribution away from support-thin lexical material.

Full JSON: `experiments/archive/frontier_consolidation/data/wordmean_reweighting_profile/wordmean_reweighting_profile.json`
