# mixture route after pair adjacency — Route after the controlled rewrite-adjacency result

## Current scientific state

The retained training/evaluation pipeline supports several controlled negative results, but none of the candidates evaluated here approaches the referenced 2026 top system.

The most important new result is pair vs shuffle 1m profile/59: semantic rewrite adjacency is real but modest.

Pair-adjacent minus pair-shuffled at 999,995 words, same sentence multiset, same token and WWM opportunity:

| column | seed42 | seed43 | mean |
|---|---:|---:|---:|
| BLiMP | +0.17 | -0.38 | -0.105 |
| Supplement | -2.00 | -0.80 | -1.40 |
| EWoK | +1.09 | +1.46 | +1.275 |
| Entity | +0.30 | +1.07 | +0.685 |
| COMPS | -0.45 | -0.20 | -0.325 |
| Reading eye | -0.28 | +0.10 | -0.09 |
| Reading SPR | -0.01 | 0.00 | -0.005 |

Interpretation: putting a complex source next to its own simplified rewrite improves EWoK and Entity under a strong same-content control. The effect is not just lower MLM loss: shuffled had lower final training loss in both seeds, while adjacent still improved EWoK/Entity.

However, pure WikiAuto-pair replacement is not a strong base. Against the official-corpus WWM base, pure pair-adjacent is weak in absolute terms:

| column | official WWM mean | pair-adjacent mean | pair-adjacent minus official |
|---|---:|---:|---:|
| BLiMP | 56.40 | 54.23 | -2.17 |
| Supplement | 50.80 | 48.40 | -2.40 |
| EWoK | 50.18 | 50.78 | +0.60 |
| Entity | 17.85 | 16.82 | -1.03 |
| COMPS | 50.32 | 49.75 | -0.57 |
| Reading eye | 9.82 | 4.46 | -5.36 |
| Reading SPR | 3.74 | 1.01 | -2.74 |

This means the useful object is not “replace the BabyLM corpus with WikiAuto pairs”. The useful object is the *adjacency mechanism* inside a better data distribution.

## Why the next experiment should be a mixture

The official corpus preserves broad linguistic and human-like behavior better than pure WikiAuto pairs, especially BLiMP, Supplement, Reading, and absolute Entity. The rewrite-pair corpus contributes a specific adjacency signal in EWoK/Entity. The next experiment should therefore test whether a mixture can combine:

- official-corpus distribution and source diversity;
- WWM base already supported by two seeds;
- rewrite-adjacency semantic signal shown by pair-adjacent vs pair-shuffled.

The next experiment is not a scale-up and not a submission candidate. It is a controlled data-composition test: does the adjacency signal transfer when rewrite pairs are only part of the 1M exposure rather than the whole corpus?

## Primary next experiment: official + rewrite mixture, adjacent vs shuffled

Use the repaired JSONL trainer path. Materialization must remain outside the trainer.

### Primary fraction

Run a 50/50 mixture first:

- 500,000 words from the official corpus, selected by the same official example-selection logic as the WWM baseline for each seed (`example_pool_words=10M`, `words_per_example=160`, seed-specific shuffle).
- about 500,000 words from `GEM/wiki_auto_asset_turk` rewrite pairs, selected as complete pair examples with an internal deranged target permutation.
- total actual words may be slightly below 1,000,000 because pair examples cannot be partially selected. The trainer must use the materialized actual total, not force exactly 1,000,000.

Why 50/50 first: the pure pair adjacency signal is only +0.685 Entity and +1.275 EWoK; a much smaller fraction may bury the signal in two-seed noise. A 50/50 transfer test is strong enough to show whether the adjacency mechanism survives in an official-data context. If 50/50 preserves the EWoK/Entity signal but harms BLiMP/Supplement/Reading too much, then a later 25/75 mixture can tune the composition. If 50/50 shows no adjacent-vs-shuffled effect, the adjacency mechanism is too weak under mixture and the route should shift to broader data distribution or optimizer dynamics.

### Arms

For each seed, train two arms:

1. `mixture_adjacent`: official examples + rewrite pair-adjacent examples.
2. `mixture_shuffled`: the exact same official examples + the exact same rewrite source and target sentence multisets, but targets deranged inside the selected rewrite subset.

The primary comparison is `mixture_adjacent - mixture_shuffled` by seed and mean.

### Fixed base

Keep the same base as pair vs shuffle 1m profile/59:

- `BertForMaskedLM`, hidden 256, 8 layers, 8 heads, FFN multiplier 4;
- baseline 16k tokenizer;
- WWM, mask probability 0.15;
- fixed `seq_length=max_seq_length=256` and `max_position_embeddings=512`;
- AdamW, LR 0.001, same seed/init/RNG triples: 42/456/789 and 43/457/790;
- no DeBERTa, no 40k tokenizer, no length schedule, no LAMB, no 10M scaling.

### Materialization requirements

Create a new isolated script, for example:

`experiments/archive/initial_model_studies/scripts/materialize_official_rewrite_mixture.py`

The script should:

1. Download or reuse the official BabyLM corpus and `GEM/wiki_auto_asset_turk` train parquet.
2. Reconstruct official examples using the same logic as the trainer: `iter_examples(files, pool_words=10M, words_per_example=160)`, assign example IDs, shuffle with the run seed, and select complete 160-word official examples up to 500,000 words. For 500k this is exactly 3,125 official examples.
3. Select rewrite pairs from GEM independently for each seed and construct an internal deranged target permutation within the selected pair subset. Do not subset the old shuffled JSONL prefix, because a prefix of the old derangement would not preserve the target multiset inside the subset.
4. Build two combined JSONLs for each seed with identical official examples and identical rewrite source/target sentence multisets. The only difference between combined arms should be pair target adjacency.
5. Interleave official and rewrite examples with a deterministic condition-independent order, e.g. a shared shuffled list of slots `official:<id>` and `pair:<id>`. The same slot order must be used for adjacent and shuffled arms.
6. Validate combined arms under the baseline tokenizer:
   - identical official example IDs/texts/word counts;
   - identical rewrite source multiset, target multiset, and full sentence multiset;
   - identical total words and example count;
   - pair examples have zero truncation at 256 in both adjacent and shuffled forms;
   - official examples may have the same truncation as the existing official baseline, but the official examples and their truncation must be identical across adjacent and shuffled arms;
   - combined total kept tokens, WWM groups, expected selected groups, and expected predicted tokens should be identical or any tiny difference must be explained before training.
7. Record dataset IDs, revisions, file hashes, licenses/provenance, official source-word mix, pair mapping, shuffle seed, mixture fraction, actual total words, token summaries, and a recommended batch size that gives 98 updates if possible.

Existing official WWM tokenization summaries show that the 160-word official chunks have about 29% examples exceeding 256 tokens and lose about 46–47k tokens at 1M. This is already part of the prior official WWM base. The mixture experiment can keep official chunks at 160 words because the official part is identical across adjacent/shuffled arms and matches the baseline distribution. The pair part must remain no-truncation as in pair adjacent smoke.

### Training and profiling

After materialization passes validation, run four arms:

- mixture_adjacent seed42
- mixture_shuffled seed42
- mixture_adjacent seed43
- mixture_shuffled seed43

Use the trainer’s `--example_jsonl` path and set `--max_word_exposure`, `--example_pool_words`, and `--checkpoint_words` to the materialized actual total words for each arm/seed. The runner should compute or read the recommended batch size so the number of optimizer updates is 98 if possible, avoiding the pair vs shuffle 1m profile update-count problem.

Profile with official backend `mlm`:

- BLiMP fast
- Supplement fast
- EWoK fast
- Entity Tracking fast
- COMPS
- Reading

## How to interpret the result

The route becomes stronger if mixture-adjacent preserves the pair vs shuffle 1m profile/59 stable signs on EWoK and Entity over mixture-shuffled, while its absolute BLiMP/Supplement/Reading are much closer to official WWM than pure pair data.

The route weakens if:

- adjacency no longer improves EWoK/Entity in the mixture;
- the mixture still collapses Reading or Supplement like pure pair replacement;
- the adjacent arm improves only EWoK/Entity but falls behind shuffled or official on too many columns to matter for Overall.

If 50/50 shows a real adjacent-vs-shuffled effect but excessive distribution damage, run a 25/75 mixture next. If 50/50 shows no adjacent-vs-shuffled effect, do not keep shrinking the fraction hoping for a miracle; redirect toward better rewrite data quality/source distribution, official+pair curriculum ordering, or optimizer dynamics.

## Why not scale now

The current result is a mechanism signal, not a SOTA candidate. Pure pair data does not beat official WWM and is far below the current top system. Scaling the pure pair route would amplify known distribution damage. The mixture experiment is needed before any 10M/100M candidate is justified.
