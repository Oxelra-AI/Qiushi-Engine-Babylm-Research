# mixture materialization and runner — 50/50 official + rewrite mixture materialization and runner

## Scientific purpose

pair vs shuffle 1m profile/59 showed that rewrite adjacency is a real mechanism but too weak as pure WikiAuto replacement:

- pair-adjacent minus pair-shuffled: EWoK +1.275, Entity +0.685 mean, stable sign across seeds;
- pure pair-adjacent was much worse than official WWM on BLiMP, Supplement, Reading, and absolute Entity.

The next question is whether the adjacency signal can be inserted into the official data distribution. mixture materialization and runner therefore constructed a controlled **50/50 official+rewrite mixture** experiment:

- hold the protected base fixed: BERT-WWM, baseline 16k tokenizer, fixed 256 length, AdamW, seeds 42/43;
- keep exactly the same official examples inside each seed’s adjacent and shuffled arms;
- keep exactly the same rewrite source and target multisets inside each seed’s adjacent and shuffled arms;
- differ only in whether rewrite sources are adjacent to their own target or a deranged target;
- use the repaired trainer only through `--example_jsonl`.

## Materializer

Script:

- `experiments/archive/initial_model_studies/scripts/materialize_official_rewrite_mixture.py`

Command executed:

```bash
python experiments/archive/initial_model_studies/scripts/materialize_official_rewrite_mixture.py \
  --official_words 500000 \
  --pair_words 500000 \
  --official_pool_words 10000000 \
  --words_per_official_example 160 \
  --max_seq_length 256 \
  --target_steps 98 \
  --seeds 42,43 \
  --pair_selection_seed_base 610000 \
  --pair_shuffle_seed_base 611000 \
  --slot_seed_base 612000 \
  --max_candidate_pairs 484000
```

All-seed metadata:

- `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_materialization_all_seeds.json`

Data sources:

- Official corpus: `BabyLM-community/BabyLM-2026-Strict-Small`, revision `c92ab16b4f08858304b0815706065b3354d8fc0a`
- Rewrite pairs: `GEM/wiki_auto_asset_turk`, revision `ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99`
- Baseline tokenizer: `BabyLM-community/BabyLM-2026-Baseline-GPT2-Strict`

## Materialized files and validation

### Seed 42

- adjacent JSONL: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_adjacent_seed42_official500000_pair500000_total1000000_n13587.jsonl`
- shuffled JSONL: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_shuffled_seed42_official500000_pair500000_total1000000_n13587.jsonl`
- per-seed metadata: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_materialization_seed42_official500000_pair500000_total1000000_n13587.json`
- official words: 500,000
- official examples: 3,125
- rewrite-pair words: 500,000
- rewrite-pair examples: 10,462
- total words: 1,000,000
- total examples: 13,587
- recommended training batch: 139, giving exactly 98 updates
- validation: hard validation true; aggregate opportunity identical true
- adjacent/shuffled combined kept-token delta: 0
- adjacent/shuffled WWM group delta: 0
- pair examples over 256 tokens: 0 in both arms
- total kept tokens: 1,424,551 in both arms
- total kept WWM groups: 987,521 in both arms
- total tokens lost to truncation: 23,159 in both arms, from identical official examples only

### Seed 43

- adjacent JSONL: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_adjacent_seed43_official500000_pair500000_total1000000_n13594.jsonl`
- shuffled JSONL: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_shuffled_seed43_official500000_pair500000_total1000000_n13594.jsonl`
- per-seed metadata: `experiments/archive/initial_model_studies/data/mixture_revision_61/mixture_materialization_seed43_official500000_pair500000_total1000000_n13594.json`
- official words: 500,000
- official examples: 3,125
- rewrite-pair words: 500,000
- rewrite-pair examples: 10,469
- total words: 1,000,000
- total examples: 13,594
- recommended training batch: 139, giving exactly 98 updates
- validation: hard validation true; aggregate opportunity identical true
- adjacent/shuffled combined kept-token delta: 0
- adjacent/shuffled WWM group delta: 0
- pair examples over 256 tokens: 0 in both arms
- total kept tokens: 1,425,407 in both arms
- total kept WWM groups: 987,895 in both arms
- total tokens lost to truncation: 22,567 in both arms, from identical official examples only

The materializer compiled and the metadata was sanity-checked:

```text
42 1000000 13587 {'batch_size': 139, 'actual_steps': 98, 'target_steps': 98, 'exact_target_steps': True} True True
43 1000000 13594 {'batch_size': 139, 'actual_steps': 98, 'target_steps': 98, 'exact_target_steps': True} True True
```

## Runner for next experimental step

Script:

- `experiments/archive/initial_model_studies/scripts/run_mixture_adjacent_vs_shuffled_1m.py`

This runner compiles and is ready. It will:

- validate materialization metadata before training;
- train four arms: mixture-adjacent and mixture-shuffled for seeds 42 and 43;
- use `--example_jsonl` with the exact per-seed JSONLs above;
- use `--example_jsonl_meta` with the corresponding per-seed materialization metadata;
- set `--max_word_exposure --example_pool_words --checkpoint_words 1000000`;
- use `--batch_size 139 --lr_total_steps 98` as recommended by materialization;
- keep base fixed: BERT 8x256, baseline16k, WWM, fixed 256, `max_position_embeddings=512`, AdamW LR 0.001;
- profile official `mlm`: BLiMP fast, Supplement fast, EWoK fast, Entity Tracking fast, COMPS, and Reading;
- write:
  - `experiments/archive/initial_model_studies/data/mixture_adjacent_vs_shuffled_1m_profile.json`
  - `experiments/archive/initial_model_studies/data/mixture_adjacent_vs_shuffled_1m_training_summary.json`
  - `research/notes/initial_model_studies/mixture_adjacent_vs_shuffled_1m_profile.md`
  - `research/notes/initial_model_studies/mixture_adjacent_vs_shuffled_1m_train_and_profile.log`

## Interpretation to apply after mixture adjacent vs shuffled 1m profile runs

The route strengthens only if mixture-adjacent retains the pair vs shuffle 1m profile/59 stable positive signs on EWoK and Entity over mixture-shuffled while its absolute BLiMP, Supplement, and Reading are closer to official WWM than pure pair-adjacent.

If 50/50 adjacent improves EWoK/Entity but still hurts broad columns strongly, then a lower rewrite fraction such as 25/75 may be worth materializing. If the adjacent-vs-shuffled effect disappears in the mixture, do not shrink the fraction hoping for a hidden signal; redirect toward better rewrite-data source/quality, a data distribution route, or optimizer dynamics.
