# source conditioned ordering interaction synthesis frozen-model source-use probe (chck82)

Checkpoint: `experiments/archive/frontier_consolidation/training/runs/adapter128_scale1p75_h100M100M_seed43022_official_ladder/hf_model/chck_82M`
Total target records: 0
Dry run: True
Elapsed: 13.2s

## Family summaries
| family | n_pairs | n_valid | mean I_f | se I_f | positive frac | mean NLL(ord|S) | mean NLL(scr|S) | mean NLL(ord) | mean NLL(scr) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact_vs_compact_scrambled | 20 | 60 tasks | dry run |  |  |  |  |  |  |
| prefix_fluent_vs_prefix_scrambled | 20 | 60 tasks | dry run |  |  |  |  |  |  |
| sourcewide_onegap_vs_sourcewide_onegap_scrambled | 20 | 60 tasks | dry run |  |  |  |  |  |  |

## Stratified I_f analysis

### By family
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By lex_class
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By source_match_bin
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By counterpart_visible
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By complete_bpe_copy
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By contentlike
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

### By family_x_copy_zone
| key | n_targets | n_pairs | mean I_f | se I_f | positive frac | source_eff_ord | source_eff_scr |
|---|---:|---:|---:|---:|---:|---:|---:|

## Interpretation
I_f = [NLL(V_scr|S) - NLL(V_ord|S)] - [NLL(V_scr) - NLL(V_ord)]

Positive I_f: ordered structure specifically helps source-conditioned reconstruction beyond generic fluency preference.
Near-zero I_f: ordering helps equally with and without source; generic fluency, not source-use structure.
Negative I_f: ordering helps *less* when source is present; source makes ordering irrelevant.

Records CSV: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_dryrun/source_use_probe_records.csv`
JSON: `experiments/archive/frontier_consolidation/data/frozen_source_use_probe_dryrun/frozen_source_use_probe.json`
