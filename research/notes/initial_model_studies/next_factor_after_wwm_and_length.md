# next factor after wwm and length — Next factor after cross-seed WWM and rejected length schedule

## Current controlled state

The masked route now has a real base mechanism:

- WWM versus token masking at 1M, official `mlm`, fixed 256 length, baseline 16k tokenizer, 8-layer/256-hidden BERT MLM.
- Cross-seed WWM-minus-token mean deltas:
  - BLiMP +2.15
  - EWoK +2.50
  - Entity Tracking +0.51
  - Supplement -0.40
  - COMPS -0.12
  - Reading eye -0.06
  - Reading self-paced -0.10
- Positive in both seeds: BLiMP, EWoK, Entity Tracking.

The tested short-to-long length schedule should not be carried forward:

- WWM fixed, schedule `0.0:64,0.4:128,0.8:256` versus fixed 256.
- Mean schedule-minus-fixed-WWM deltas:
  - BLiMP +0.77
  - Supplement -3.00
  - EWoK +0.36
  - Entity -0.09
  - COMPS -0.61
  - Reading eye -0.42
  - Reading self-paced +0.15

Thus the base for the next experiment is **WWM, fixed 256, baseline 16k tokenizer, current 8-layer/256-hidden BERT MLM**, not WWM plus length schedule.

## Why the next factor should be simplification-pair data, not tokenizer yet

The current top Strict-Small model card reports a 34.7M DeBERTa-v2 masked LM trained on FineWeb sentence-level simplification pairs with a 40k SentencePiece tokenizer, WWM7→Token3 masking, LAMB, and 10 epochs. Its strongest reported gains were EWoK, Entity Tracking, and GLUE-style fine-tuning. Our WWM-only base reproduces a small and stable part of this pattern on BLiMP/EWoK/Entity, but the absolute level is still far below the top system:

| quantity | our WWM base at 1M | top WWM simplification system |
|---|---:|---:|
| BLiMP | ~56 | 67.20 |
| EWoK | ~49–51 | 56.07 |
| Entity | ~17.6–18.1 | 28.45 |
| Supplement | ~50–51 | 56.01 |

The two remaining high-value factors are tokenizer granularity and paired simplification data. Paired data is the cleaner next single-factor experiment because:

1. It can be tested with the **same baseline 16k tokenizer** and the same WWM objective, keeping tokenizer granularity fixed.
2. The dataset is directly accessible and reproducible: `go76dof/Fineweb_simplification_pairs` at dataset sha `42e978d2f8313b315032d0b1c93e4cefe0a909b1`, with files:
   - `FineWeb_simplification_pairs.train`
   - `README.md`
   - `tokenizer/FineWeb_simplification_pairs_40k.model`
   - `tokenizer/FineWeb_simplification_pairs_40k.vocab`
3. The model card reports the train file as 9,999,969 whitespace-counted words, format original sentence followed by simplified rewrite, blank lines separating pairs.
4. The schedule experiment already showed that merely changing presentation length is not enough. The larger Entity/EWoK gap plausibly depends on semantic rewrite adjacency or data distribution.
5. A tokenizer experiment is valuable but less clean as the immediate next factor: reusing the top 40k tokenizer would import a tokenizer trained on the paired data condition, while training a new 40k tokenizer on official text requires extra word-budget accounting and changes vocabulary size/model embedding parameters. It should follow once the data effect is known.

## Next controlled experiment: WWM official data versus WWM simplification-pair data at 1M

### Scientific question

Does replacing the official BabyLM corpus with FineWeb simplification-pair data, while holding WWM, tokenizer, model, optimizer schedule, length, exposure, and seeds fixed, move Entity/EWoK/Supplement/COMPS toward the current SOTA profile?

### Fixed base

Keep exactly:

- backend/objective: `mlm`, WWM
- tokenizer: existing baseline 16k `PreTrainedTokenizerFast`
- model: `BertForMaskedLM`, 8 layers, hidden 256, 8 heads, ~10.7M params
- length: fixed `seq_length=max_seq_length=256`
- position capacity: `max_position_embeddings=512`
- exposure: 1,000,000 whitespace words
- pool: 10,000,000 whitespace words where available; the paired dataset has 9,999,969 words, so use that full pool and record the exact pool value
- optimizer schedule: `lr_total_steps=98`, same LR and batch size as masked 1m pos512 token vs wwm profile/44
- seeds: run seeds 42 and 43 with the same init/RNG triples used by the official-data WWM baselines
- evaluation: official `mlm` fast/local BLiMP, Supplement, EWoK, Entity Tracking, COMPS, Reading

### New factor

Only the training text condition changes:

- baseline arms already exist:
  - seed42 official data WWM: `babylm_masked_wwm_pos512_1M`
  - seed43 official data WWM: `babylm_masked_wwm_pos512_seed43_1M`
- new arms to train:
  - seed42 paired-data WWM: suggested run id `babylm_step46_masked_wwm_pairs16k_seed42_1M`
  - seed43 paired-data WWM: suggested run id `babylm_step46_masked_wwm_pairs16k_seed43_1M`

Because the corpus changes, example IDs cannot match official-data baselines. The controlled comparison should instead match random/init/training seeds and all non-data hyperparameters, while recording exact source/data manifests for the paired corpus.

### Minimal construction required

Patch or extend `babylm_masked_train.py` without disturbing the official-corpus path:

- Add a way to pass dataset repo and train file names, for example:
  - `--dataset_id go76dof/Fineweb_simplification_pairs`
  - `--dataset_revision 42e978d2f8313b315032d0b1c93e4cefe0a909b1`
  - `--train_file_names FineWeb_simplification_pairs.train`
- Keep the default official six-file behavior unchanged.
- Count whitespace words in the train file exactly and record the dataset sha, file hash, file word count, selected exposure, and actual pool words in `data_manifest.json` and `scientific_metrics.json`.
- Preserve blank-line pair adjacency by reading the train file sequentially; chunking to 160-word examples is acceptable for the first factor test because the top train format itself is original sentence followed by simplified rewrite.
- Run a 10k paired-data WWM smoke first. It must save root and `chck_1M`, load with `AutoModelForMaskedLM`, and run at least fast BLiMP and Entity under official `mlm`.
- Then train the two 1M paired-data WWM arms and profile the same fast/local columns.

### Interpretation

Carry the paired-data factor forward only if it gives a meaningful improvement over official-data WWM in at least Entity and one of EWoK/Supplement/COMPS without a large BLiMP or Reading collapse. If paired data strongly improves Entity/EWoK but hurts Supplement, the next factor should be tokenizer or masking schedule repair; if paired data is neutral, tokenizer granularity becomes the next cleaner factor.

No 10M/100M scaling or final submission work should start from this 1M data-factor experiment alone. The result must be combined with prior WWM and length-schedule evidence to decide the next one-factor step.

## Correction after access check: paired data is not immediately executable

A direct Hugging Face file download test for `go76dof/Fineweb_simplification_pairs` at sha `42e978d2f8313b315032d0b1c93e4cefe0a909b1` showed that repository metadata and file listing are visible, but the actual train file is gated:

```text
401 Client Error: Unauthorized ... Cannot access gated repo ... FineWeb_simplification_pairs.train
```

Therefore paired simplification data is **not** the next executable single-factor experiment unless authenticated access is obtained or a legal within-budget replacement corpus is constructed. The route choice must change. We should not plan a 1M paired-data run that cannot currently read the data.

## Revised next factor: tokenizer granularity on official corpus

The next immediate controlled experiment should test tokenizer granularity while keeping the cross-seed WWM base fixed and using the already available official BabyLM Strict-Small corpus.

Rationale:

1. The current top system uses a 40k tokenizer, and `Wordpiece-24-4` also suggests tokenizer/objective leverage, but paired data is gated.
2. `tokenizers` and `sentencepiece` are installed locally, so a tokenizer experiment is executable now.
3. The tokenizer can be trained on the same official 10M words, with exact word count and file hashes already available; tokenizer training must be recorded as part of the training-budget accounting.
4. This is still a single major factor if we keep WWM, data, model family, fixed length, optimizer schedule, exposure, and seeds unchanged. The model parameter count will increase because the embedding/output vocabulary changes; that is part of the tokenizer factor and should be reported explicitly.

### Revised next controlled experiment: WWM official corpus with 16k baseline tokenizer versus 40k tokenizer

Existing fixed-base arms:

- seed42 WWM 16k baseline tokenizer: `babylm_masked_wwm_pos512_1M`
- seed43 WWM 16k baseline tokenizer: `babylm_masked_wwm_pos512_seed43_1M`

New arms:

- seed42 WWM 40k tokenizer on official corpus: suggested `babylm_step46_masked_wwm_official40k_seed42_1M`
- seed43 WWM 40k tokenizer on official corpus: suggested `babylm_step46_masked_wwm_official40k_seed43_1M`

Hold fixed:

- official BabyLM Strict-Small corpus and random selected example IDs/order/source mix by seed, if possible;
- WWM objective;
- fixed `seq_length=max_seq_length=256`, `max_position_embeddings=512`;
- BERT MLM architecture depth/width/heads except tokenizer-driven vocab/embedding size;
- exposure 1M words, `lr_total_steps=98`, learning rate/batch size/seeds;
- official `mlm` fast/local profile: BLiMP, Supplement, EWoK, Entity, COMPS, Reading.

Implementation needs:

1. Add tokenizer option to `babylm_masked_train.py` without breaking the baseline tokenizer path.
2. Train or load a 40k tokenizer from official corpus text; save tokenizer artifact under `training/tokenizers/official40k/` with manifest containing tokenizer type, vocab size, source files, corpus revision, word count charged, and hashes.
3. The tokenizer should preserve `<s>`, `</s>`, `<unk>`, `<pad>`, and `<mask>`, and save as a portable `PreTrainedTokenizerFast` if possible.
4. Run a 10k WWM 40k-tokenizer smoke first: train, save, load with `AutoModelForMaskedLM`, and run official fast BLiMP and Entity under `mlm`.
5. Then run the two 1M 40k arms and compare tokenizer40k-minus-tokenizer16k against the existing WWM fixed baselines by seed.

Interpretation:

Carry 40k tokenizer forward only if it improves Entity/EWoK/Supplement or BLiMP without a large COMPS/Reading cost. If 40k mainly shifts BLiMP while hurting Entity or Supplement, tokenizer granularity alone is not the missing SOTA factor. If 40k helps strongly, combine it with WWM as the next fixed base before revisiting paired data, architecture, or optimizer.
