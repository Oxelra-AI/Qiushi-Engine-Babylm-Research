# lead factorial route reassessment target-level factor alignment audit

CPU/tokenizer-only audit of view-word targets under exact DeBERTa row truncation and WWM grouping. No model was loaded.

Target records: 1,131,956; tokenizer SHA `91b775514b9f4e2d1f37c28445ab98181007e16d14f763955168547e293ee8f9`; seq_len=256.

## Variant target mass
| variant | targets | active target tokens | WWM token mass | WWM groups | full visible | counterpart visible | whole-word copy | complete-BPE copy | seen-BPE frac | contentlike |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| compact | 161708 | 244738 | 247333 | 161424 | 99.81% | 83.47% | 83.49% | 70.21% | 81.45% | 65.82% |
| compact_scrambled | 161708 | 244739 | 247334 | 161417 | 99.81% | 83.47% | 83.49% | 70.20% | 81.45% | 65.82% |
| prefix_fluent | 161708 | 214823 | 216465 | 161631 | 99.95% | 99.79% | 99.79% | 99.74% | 100.00% | 50.79% |
| prefix_scrambled | 161708 | 214822 | 216464 | 161632 | 99.95% | 99.79% | 99.79% | 99.74% | 100.00% | 50.79% |
| sourcewide_onegap | 161708 | 234687 | 236850 | 161497 | 99.86% | 99.81% | 99.83% | 99.68% | 100.00% | 58.64% |
| sourcewide_onegap_scrambled | 161708 | 234688 | 236851 | 161492 | 99.86% | 99.81% | 99.83% | 99.68% | 100.00% | 58.64% |
| best_contiguous_span | 161708 | 224082 | 226071 | 161560 | 99.90% | 99.75% | 99.76% | 99.67% | 100.00% | 55.74% |

## Copy-zone target mass (expected WWM token opportunity)
| variant | zone | targets | active tokens | WWM mass | counterpart visible | complete-BPE copy | contentlike |
|---|---|---:|---:|---:|---:|---:|---:|
| best_contiguous_span | absent | 382 | 409 | 410 | 0.00% | 0.00% | 0.00% |
| best_contiguous_span | both_prefix_tail | 18946 | 21343 | 21378 | 100.00% | 99.92% | 19.90% |
| best_contiguous_span | prefix_only | 81492 | 112425 | 113586 | 99.99% | 99.91% | 58.08% |
| best_contiguous_span | tail_only | 60888 | 89905 | 90697 | 99.99% | 99.88% | 64.10% |
| compact | absent | 26696 | 37625 | 37752 | 0.00% | 0.00% | 67.91% |
| compact | both_prefix_tail | 12597 | 14967 | 15006 | 99.98% | 93.28% | 27.33% |
| compact | prefix_only | 74819 | 114578 | 116128 | 99.98% | 83.58% | 67.40% |
| compact | tail_only | 47596 | 77568 | 78447 | 99.97% | 82.46% | 72.36% |
| compact_scrambled | absent | 26696 | 37633 | 37760 | 0.00% | 0.00% | 67.91% |
| compact_scrambled | both_prefix_tail | 12597 | 14960 | 14999 | 99.98% | 93.25% | 27.33% |
| compact_scrambled | prefix_only | 74819 | 114501 | 116050 | 99.98% | 83.52% | 67.40% |
| compact_scrambled | tail_only | 47596 | 77645 | 78525 | 99.97% | 82.52% | 72.36% |
| prefix_fluent | absent | 332 | 348 | 350 | 0.00% | 0.00% | 0.00% |
| prefix_fluent | both_prefix_tail | 17244 | 19144 | 19166 | 100.00% | 99.94% | 17.72% |
| prefix_fluent | prefix_only | 144132 | 195331 | 196949 | 100.00% | 99.95% | 54.86% |
| prefix_scrambled | absent | 332 | 348 | 350 | 0.00% | 0.00% | 0.00% |
| prefix_scrambled | both_prefix_tail | 17244 | 19147 | 19169 | 100.00% | 99.95% | 17.72% |
| prefix_scrambled | prefix_only | 144132 | 195327 | 196945 | 100.00% | 99.94% | 54.86% |
| sourcewide_onegap | absent | 282 | 300 | 302 | 0.00% | 0.00% | 0.00% |
| sourcewide_onegap | both_prefix_tail | 19532 | 22491 | 22527 | 100.00% | 99.91% | 21.82% |
| sourcewide_onegap | prefix_only | 77310 | 111624 | 112876 | 99.99% | 99.88% | 61.56% |
| sourcewide_onegap | tail_only | 64584 | 100272 | 101145 | 99.98% | 99.82% | 66.54% |
| sourcewide_onegap_scrambled | absent | 282 | 300 | 302 | 0.00% | 0.00% | 0.00% |
| sourcewide_onegap_scrambled | both_prefix_tail | 19532 | 22492 | 22528 | 100.00% | 99.90% | 21.82% |
| sourcewide_onegap_scrambled | prefix_only | 77310 | 111540 | 112792 | 99.99% | 99.83% | 61.56% |
| sourcewide_onegap_scrambled | tail_only | 64584 | 100356 | 101229 | 99.98% | 99.88% | 66.54% |

## Design contrasts
- compact_vs_compact_scrambled_fixed_word_multiset: `{"delta_targets": 0.0, "delta_view_active_tokens": -1.0, "delta_wwm_token_mass": -1.0, "delta_counterpart_visible_frac": 0.0, "delta_complete_bpe_copy_frac": 9.275978924971895e-05}`
- sourcewide_onegap_vs_scrambled_fixed_word_multiset: `{"delta_targets": 0.0, "delta_view_active_tokens": -1.0, "delta_wwm_token_mass": -1.0, "delta_counterpart_visible_frac": 0.0, "delta_complete_bpe_copy_frac": 1.2367971899962527e-05}`
- sourcewide_onegap_vs_prefix_fluent_position_confounded: `{"delta_view_active_tokens": 19864.0, "delta_wwm_token_mass": 20385.0, "delta_complete_bpe_copy_frac": -0.0005689267073984983, "delta_counterpart_visible_frac": 0.00018551957849954892}`
- compact_vs_sourcewide_onegap_style_and_lexicon_confounded: `{"delta_view_active_tokens": 10051.0, "delta_wwm_token_mass": 10483.0, "delta_whole_word_copy_frac": -0.16334380488287525, "delta_complete_bpe_copy_frac": -0.29477205827788366}`

## Scientific reading
The table should be used to repair future controls, not to justify training. Fixed-word-multiset ordered/scrambled pairs are the cleanest current single-factor probes. Position-spread and natural-compact comparisons still require target-composition and exposure matching because lexical copy zones and active target mass can differ even when legal words and row order are fixed.

Full target records CSV: `experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/view_word_target_records.csv`
Strata summary CSV: `experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/target_strata_summary.csv`
Copy-zone summary CSV: `experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/copy_zone_summary.csv`
Samples: `experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/target_alignment_samples.jsonl`
JSON: `experiments/archive/frontier_consolidation/data/target_level_factor_alignment_audit/target_level_factor_alignment_audit.json`
