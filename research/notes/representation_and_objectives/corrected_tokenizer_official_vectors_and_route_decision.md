# corrected tokenizer official vectors and route decision corrected-tokenizer official vectors and route decision

## What changed

The two managed full official-compatible evaluations for the compliant 10M-trained tokenizer line completed and were collected:

- `s51_t41_tool1` / seed43022: exit code 0; controller summary `experiments/archive/representation_and_objectives/data/strictsmalltok_seed43022_posttrain_eval_controller/posttrain_eval_seed43022_summary.json`; pristine collated artifact SHA `554051aa775411557afc3651cefa6b8537b1864d255cf76a135c4ce76f385c52`.
- `s51_t42_tool1` / seed43122: exit code 0; controller summary `experiments/archive/representation_and_objectives/data/strictsmalltok_seed43122_posttrain_eval_controller/posttrain_eval_seed43122_summary.json`; pristine collated artifact SHA `39f9acc0e2d750c3d8d6044aca2bff4ca707e737575cbfb49af6f1f8e7aaa352`.

Both hardened collation summaries have empty validation error lists, current official 7,618-row EWoK, official min-context-zero AoA with 152,095 surprisal rows = 19 checkpoints × 8,005 rows, finite nine-column vectors, direct changed block overlap ancestry current official current-official GlobalPIQA files with expected SHA/counts, official SuperGLUE primary metric convention, and exact target/endpoint artifact staging.

## Official-compatible vectors under the legal tokenizer

Visible public Strict-Small leader used for comparison: `go76dof/wwm_curriculum_simplification_40k`, Overall 41.8, refreshed in `experiments/archive/representation_and_objectives/data/babylm2026_live_surface/leaderboard_parsed.json`.

### seed43022 corrected-tokenizer compact_view_reinvest

- BLiMP 66.3350104082468
- Supplement 59.28059307749794
- EWoK 49.83762097731724
- Entity 26.19560049841891
- COMPS 51.8109824000351
- SuperGLUE 69.17430600796813
- GlobalPIQA 36.10679611650485
- Reading 7.594698114753101
- AoA 0.0
- Overall 40.703956400082454
- Margin versus 41.8: -1.096043599917543

### seed43122 corrected-tokenizer compact_view_reinvest

- BLiMP 66.47890253779987
- Supplement 55.767586890202026
- EWoK 49.42614323036989
- Entity 28.58689694490618
- COMPS 52.24680579271018
- SuperGLUE 69.61148966230907
- GlobalPIQA 38.60679611650485
- Reading 8.491325047897558
- AoA 0.0
- Overall 41.023994024744404
- Margin versus 41.8: -0.7760059752555932

Two-seed mean Overall is 40.86397521241343, margin -0.9360247875865682; absolute seed spread is 0.32003762466194985.

## Comparison to inherited-tokenizer mechanism coordinate

The inherited-tokenizer seed43022 endpoint remains scientifically informative but not submission-facing because its tokenizer was learned from out-of-budget Strict 100M data. The legal tokenizer change moved the compact_view_reinvest endpoints as follows:

- seed43022 Overall: 42.0331347900748 -> 40.703956400082454, delta -1.329178389992343.
  - largest column losses: Supplement -3.9951711020236402, EWoK -3.698954617569612, SuperGLUE -1.861743520284989, Entity -1.5501405995334636.
  - GlobalPIQA increased +0.4854368932038824.
- seed43122 Overall: 41.24823958912208 -> 41.023994024744404, delta -0.22424556437767507.
  - largest column losses: Supplement -6.1847660450924025, EWoK -2.465574786913322.
  - largest gains: GlobalPIQA +3.470873786407765, Entity +2.3015579856352133, BLiMP +0.8145023966749392, COMPS +0.707611630692675.
- two-seed mean delta versus inherited-tokenizer reinvest: -0.7767119771850091 Overall, with Supplement -5.089968573558018 and EWoK -3.082264702241467 dominating the mean loss.

The corrected EWoK bridge over all 7,618 rows shows the old seed-polarized EWoK surface is largely erased rather than repaired into a stronger relation model: corrected micro accuracy is 49.264898923602 for seed43022 and 49.632449461800995 for seed43122. On the 1,766 old negative-interaction rows, corrected accuracies are 46.602491506228766 and 46.77236693091732; the inherited seed43022 relation advantage on those rows collapsed, while seed43122 no longer has the same old pattern.

## Row-level data/exposure protection completed before interpretation

The repaired corpus lineage multiplicity audit lineage script now exits 0 and wrote `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/corpus_lineage_multiplicity_audit.json` plus `research/notes/representation_and_objectives/corpus_lineage_multiplicity_audit.md`. It verifies:

- active 10M file SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`, 64,740 rows, exactly 10,000,000 words;
- active 100M stream SHA `3dd19f09deeca44d6b07340e70cd15baa4b5c7bffe0bd462ff37536e470e7691`, 647,400 rows, exactly 100,000,000 words;
- the 100M stream is exactly ten presentations of the 10M row multiset, and each 64,740-row tenth is a permutation of the 10M pool;
- 10M source-class words: 7,919,680 official BabyLM source words, 1,656,800 inherited official-source Qwen paraphrase pair words, 423,511 FineWeb source + Qwen compact rewrite pair words, and 9 neutral top-up words from an official-source row;
- compact selected pair ids are all represented in changed metadata and in the accepted rewrite pool; inherited `qwen_pair_packed` rows match the inherited packed metadata.

## Scientific consequence

The compliant single-factor tokenizer repair falsifies the practical endpoint hypothesis for the current compact_view_reinvest implementation: neither legal-tokenizer seed is close to the 41.8 leader, and the best legal seed is 41.024. The old 42.033 number depended materially on the inherited 100M-trained representation coordinate, especially for Supplement and EWoK. Compact paired views and reinvestment remain useful mechanism evidence, but this exact legal tokenizer plus 8x480 fixed-WWM recipe is not the route to submit.

The next research work should not package these models and should not extend the same failed endpoint by small patches. The strongest continuation is to rebuild the legal representation/data-learning line around the observed losses: preserve the compact-view density insight, but design a representation and curriculum that can legally recover Supplement and EWoK while keeping the legal-tokenizer gains in GlobalPIQA/Entity/COMPS. Candidate directions to consider include a legal 40k-or-hybrid tokenizer trained only on the 10M pool, sequence/token/masking curriculum closer to the public leader, relation-frame-preserving compact selection only if it can move Supplement/EWoK at learning scale, and a fresh route comparison against frontier_consolidation's independent evidence.

## Files to inspect next

- corrected two-seed comparison: `experiments/archive/representation_and_objectives/data/corrected_tokenizer_two_seed_comparison/corrected_tokenizer_two_seed_comparison.json`
- corrected subtask localization: `experiments/archive/representation_and_objectives/data/corrected_subtask_localization/corrected_subtask_localization.json`
- corrected EWoK bridge: `experiments/archive/representation_and_objectives/data/corrected_ewok_old_atlas_bridge/corrected_ewok_old_atlas_bridge.json`
- endpoint interpretation: `experiments/archive/representation_and_objectives/data/corrected_endpoint_interpretation/corrected_endpoint_interpretation.json`
- row-level lineage: `experiments/archive/representation_and_objectives/data/corpus_lineage_multiplicity_audit/corpus_lineage_multiplicity_audit.json`
