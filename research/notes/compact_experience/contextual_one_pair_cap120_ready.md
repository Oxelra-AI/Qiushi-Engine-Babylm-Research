# bidir ranking contextual training and eval plan contextual one-pair cap-120 materialization

Purpose: prevent the bidirectional pair-order test from becoming a serial bottleneck and prepare the higher-value mechanism route suggested by the current evidence: same-window original--Qwen second-view correspondence embedded in real official row context.

Non-leakage: `scripts/materialize_contextual_one_pair.py` uses only the official BabyLM training pool plus initial and clean-materialization selected Qwen rewrites and source metadata. It does not read official AoA/CDI words, child curves, AoA predictions, AoA scores, or downstream evaluation outputs.

Materialized corpus:
- Metadata: `experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_one_pair_cap120_metadata.json`
- Treatment 100M: `experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/qwen_context_onepair_cap120_100M.jsonl`
- Matched official control 100M: `experiments/archive/compact_experience/data/contextual_one_pair/cap120/training_corpora/official_context_lengthmatched_cap120_100M.jsonl`
- Pair-row meta: `experiments/archive/compact_experience/data/contextual_one_pair/cap120/contextual_pair_rows_cap120_meta.jsonl`

Key audit numbers from the metadata:
- 37,594 selected pairs; 25,486 unique original official example_ids; at most 7 selected sentences from one official row.
- One original--rewrite pair per contextual row; pair boundary preserved; no row exceeds 160 words.
- Contextual pair rows total 4,511,360 words = 45.1136% of the 10M pool.
- Qwen pair words remain exactly 1,656,800 = 16.568% of the 10M pool, as in clean clean qwen compliance and validity.
- Real official context inside contextual pair rows: 2,854,560 words, mean 75.9313 context words per pair row; only 80 extra context words were added to two rows to make the contextual block divisible by 160.
- Official filler: 5,488,640 words from 34,304 full official rows; no selected official source row had to be reused as filler after excluding selected example_ids.
- Treatment/control pool rows both 71,898, and row length sequence is exactly matched.
- SHA256 treatment 100M: `6c90cb644121f4747d5a901206b35d118cb3b6959d856a827f81191ff5903640`; matched official 100M: `ddd8291f76171359f720aecfbb1d882219a6cab810f7dc629ff81b0e0a25564a`.

Training launcher prepared:
- `experiments/archive/compact_experience/scripts/launch_contextual_one_pair_cap120_training.sh`
- Runs planned: `training/runs/qwen_context_onepair_cap120_16k_seed43022` and `training/runs/official_context_lengthmatched_cap120_16k_seed43022`, same 8x480/baseline16k/WWM/seed43022 recipe.

Scientific interpretation: this arm tests whether the validated same-window second-view signal becomes less tradeoff-heavy when the relation is embedded in enough natural official context. The target improvement profile is recovery of BLiMP, EWoK, COMPS, and Reading while retaining the clean-Qwen gains in Supplement, Entity, and SuperGLUE. It is more likely than bidirectional pair-order alone to cross the 41.8 visible leader because it changes the topology from packed relation-only rows to official-context rows without increasing generated-word dose.
