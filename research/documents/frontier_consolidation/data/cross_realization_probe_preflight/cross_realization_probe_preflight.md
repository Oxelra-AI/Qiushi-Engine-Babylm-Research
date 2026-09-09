# earlier analysis cross-realization probe preflight

Prepared events: 4500 / candidates loaded: 6949
By eval_set: {'doc_disjoint_all_accepted': 1485, 'doc_disjoint_quality': 726, 'source_disjoint_quality': 1547, 'train_fixed_probe': 742}
Skips: {'rewrite_target_at_start': 440, 'no_exact_source_occurrence_same_bpe': 982, 'target_not_content': 21, 'target_piece_count_out_of_range': 4}
Target pieces: {'n': 4500, 'min': 1, 'p05': 1, 'p25': 1, 'mean': 1.7144444444444444, 'median': 1.0, 'p75': 2, 'p95': 4, 'max': 8}
Counterfactual rel-pos |delta|: {'n': 4500, 'min': 0.0, 'p05': 0.0, 'p25': 0.0, 'mean': 0.000752208448378956, 'median': 0.0, 'p75': 0.0, 'p95': 0.0, 'max': 0.10000000000000009}
Counterfactual word-length |delta|: {'n': 4500, 'min': 0, 'p05': 0, 'p25': 0, 'mean': 0.11244444444444444, 'median': 0.0, 'p75': 0, 'p95': 0, 'max': 14}

## Arm availability
- full_compact_100M: exists=True config=True weights=True tokenizer=True path=`experiments/archive/frontier_consolidation/training/runs/cleanqwen_fineweb_compact_view_reinvest_16k_seed43022/hf_model/chck_100M`
- drop_abs_100M: exists=True config=True weights=True tokenizer=True path=`experiments/archive/representation_and_objectives/training/runs/packed_drop_abs_content_100M_fast/hf_model/chck_100M`
- drop_copied_word_100M: exists=True config=True weights=True tokenizer=True path=`experiments/archive/representation_and_objectives/training/runs/packed_drop_copied_content_wholeword_100M_fast/hf_model/chck_100M`
- repeat_100M: exists=True config=True weights=True tokenizer=True path=`experiments/archive/representation_and_objectives/training/runs/gc_compact_repeat_reinvest_16k_seed43022_r2/hf_model/chck_100M`
- adjbreak_100M: exists=True config=True weights=True tokenizer=True path=`experiments/archive/representation_and_objectives/training/runs/gc_adjbreak_reinvest_16k_seed43022_r2/hf_model/chck_100M`

Preflight pass: True

The generated JSONL stores source, compact, and counterfactual contexts plus exact target spans and BPE ids. It is a construction preflight only; no model logits were read.
