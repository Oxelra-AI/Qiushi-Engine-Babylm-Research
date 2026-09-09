# pair adjacent smoke — Rewrite-pair materialization and pair-adjacent smoke

## Why this step mattered

The current BabyLM Strict-Small route is no longer ordinary component stacking. WWM is the only stable positive base, while length scheduling, official-corpus 40k tokenization, and parameter-matched DeBERTa-v2 failed to close the SOTA gap. The current high-value mechanism question is whether **meaning-preserving rewrite adjacency** improves entity consistency and semantic learning.

A necessary control is: pair-adjacent vs pair-shuffled only isolates semantic pairing if the retained training content and supervised mask opportunity are identical. Therefore pair adjacent smoke kept pair construction outside the trainer and first materialized exact JSONL examples with no >256-token examples under the baseline tokenizer.

## Repaired/stable data path

The trainer repaired in earlier analysis remains the execution path:

- `experiments/archive/initial_model_studies/training/scripts/babylm_masked_train.py`
- Uses `--example_jsonl` to consume pre-materialized examples in exact file order.
- Does not shuffle, repack, or select a partial final example.
- Records `data_source_type=example_jsonl`, JSONL hash/rows/words, optional metadata JSON, tokenizer-coupling summary, and exact word exposure.

## Isolated materializer

Script:

- `experiments/archive/initial_model_studies/scripts/materialize_rewrite_pairs.py`

Data source:

- dataset: `GEM/wiki_auto_asset_turk`
- revision/sha: `ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99`
- file: `wiki_auto_asset_turk/train-00000-of-00001.parquet`
- known component licenses: WikiAuto CC BY-NC 3.0; ASSET CC BY-NC 4.0; TurkCorpus GPL v3.0.

Construction:

- `pair_adjacent`: `source_i + " " + target_i`
- `pair_shuffled`: `source_i + " " + target_perm[i]`, with a deranged target permutation.
- No artificial `[SEP]` marker is used in the 1M-ready materialization.
- Rows are filtered so both adjacent and shuffled examples fit within `max_seq_length=256` under the baseline 16k tokenizer.

## 10k representation and training smoke

Initial 10k materialization and smoke were used to verify the path. The 10k materialization used an earlier explicit `[SEP]` delimiter; it is retained only as a smoke artifact, not the intended final pair-comparison data.

10k meta:

- `experiments/archive/initial_model_studies/data/rewrite_pairs_revision_57/pair_materialization_target10000_actual10211_n211_sel5701_shuf5702.json`

10k pair-adjacent smoke:

- run: `experiments/archive/initial_model_studies/training/runs/babylm_pair_adjacent_wwm_smoke10k`
- summary: `experiments/archive/initial_model_studies/data/pair_adjacent_smoke_summary.json`
- note: `research/notes/initial_model_studies/pair_adjacent_smoke.md`
- words: 10,211
- examples: 211
- truncated examples: 0
- trained through repaired `--example_jsonl` path, saved root and `chck_1M`, loaded with `AutoModelForMaskedLM`, passed 180-token forward checks.
- official `mlm` smoke scores: BLiMP fast 54.24; Entity Tracking fast 18.55.
- smoke loss: 9.6661 -> 9.0082.

These smoke scores are not a scientific pair-effect result; they verify data-path compatibility.

## 1M-ready pair-vs-shuffle materialization

Command used:

```bash
python experiments/archive/initial_model_studies/scripts/materialize_rewrite_pairs.py \
  --target_words 1000000 \
  --max_seq_length 256 \
  --selection_seed 5711 \
  --shuffle_seed 5712 \
  --max_candidate_pairs 484000
```

Outputs:

- pair-adjacent JSONL: `experiments/archive/initial_model_studies/data/rewrite_pairs_revision_57/pair_adjacent_target1000000_actual999995_n21080_sel5711_shuf5712.jsonl`
- pair-shuffled JSONL: `experiments/archive/initial_model_studies/data/rewrite_pairs_revision_57/pair_shuffled_target1000000_actual999995_n21080_sel5711_shuf5712.jsonl`
- metadata/validation: `experiments/archive/initial_model_studies/data/rewrite_pairs_revision_57/pair_materialization_target1000000_actual999995_n21080_sel5711_shuf5712.json`

Validation results:

- actual words: 999,995 in each arm
- examples: 21,080 in each arm
- identical source multiset: true
- identical target multiset: true
- identical full sentence multiset: true
- identical example count: true
- identical total words: true
- no adjacent example over 256 tokens: true
- no shuffled example over 256 tokens: true
- no identity target adjacency in shuffled arm: true
- total token count delta: 0
- total WWM word-group count delta: 0
- expected WWM predicted-token delta at mask probability 0.15: 0.0
- expected selected-group delta at mask probability 0.15: 0.0

Adjacent summary:

- total tokens: 1,407,535
- total word groups: 999,995
- max tokens/example: 255
- tokens/word: 1.4075420377
- word groups/word: 1.0
- expected WWM predicted tokens at 0.15: 211,130.25
- length histogram: <=64: 11,631; 65-128: 8,510; 129-256: 939; >256: 0

Shuffled summary:

- total tokens: 1,407,535
- total word groups: 999,995
- max tokens/example: 251
- tokens/word: 1.4075420377
- word groups/word: 1.0
- expected WWM predicted tokens at 0.15: 211,130.25
- length histogram: <=64: 11,114; 65-128: 9,492; 129-256: 474; >256: 0

The two arms do not have identical per-example length histograms, because targets are reassigned across sources, but the retained sentence multiset, total tokens, total WWM groups, total words, and no-truncation guarantees are identical. This satisfies the current mechanism isolation requirement: no content is lost differently by truncation, and aggregate supervised-token opportunity is identical.

## Next exact experiment

Run the decisive 1M two-seed pair-adjacent vs pair-shuffled comparison using the repaired trainer and the 1M-ready JSONLs.

Fixed base:

- `BertForMaskedLM`
- baseline 16k tokenizer
- WWM, mask probability 0.15
- fixed `seq_length=max_seq_length=256`
- `max_position_embeddings=512`
- AdamW schedule matching previous WWM experiments
- `hidden_size=256`, `n_layer=8`, `n_head=8`, `ffn_mult=4`
- exact word exposure should be `999995`, matching the materialized JSONL total; do not request 1,000,000 or the trainer will reject partial-example selection.
- paired seed/init/RNG triples: 42/456/789 and 43/457/790.

Run IDs should distinguish four arms, e.g.:

- `babylm_pair_adjacent_wwm_seed42_999995w`
- `babylm_pair_shuffled_wwm_seed42_999995w`
- `babylm_pair_adjacent_wwm_seed43_999995w`
- `babylm_pair_shuffled_wwm_seed43_999995w`

Profile with official backend `mlm`:

- BLiMP fast
- BLiMP Supplement fast
- EWoK fast
- Entity Tracking fast
- COMPS
- Reading

Primary scientific comparison:

- pair-adjacent minus pair-shuffled by seed and mean.
- Carry the route forward only if adjacency beats shuffled on Entity and at least one of EWoK/COMPS with stable sign and without a large Supplement/Reading collapse.
