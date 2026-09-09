# Legal aligned-rewrite data reconstruction plan

## Route decision

Do not spend the next main execution slot on a standalone 100M LAMB run. LAMB is now numerically valid and stable, but the 10M original-data S1 result moves away from the public leader on the target columns:

- LAMB S1 10M: BLiMP 51.75, Supplement 51.63, EWoK 49.52, Entity 16.56, COMPS 50.33, GlobalPIQA 36.225, Reading 6.77.
- AdamW S1 10M: BLiMP 53.37, Supplement 52.34, Entity 17.73, COMPS 50.34, GlobalPIQA 35.21, Reading 8.35.
- LAMB final MLM loss at 10M is worse than AdamW on the same S1/data setup: 5.0807 vs 4.0523.

This does not prove a full LAMB trajectory cannot recover later, because the 10M point is still early in a 2442-step high-LR schedule. But it makes a standalone 100M original-data LAMB run lower information than a matched data-alignment experiment. Preserve LAMB as a secondary mid-trajectory or interaction test after data reconstruction exists.

## First executable data source

The strongest immediate executable source is the already verified accessible simplification-pair corpus:

- dataset: `GEM/wiki_auto_asset_turk`
- revision: `ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99`
- file: `wiki_auto_asset_turk/train-00000-of-00001.parquet`
- fields: `source`, `target`
- licenses recorded earlier: WikiAuto CC BY-NC 3.0, ASSET CC BY-NC 4.0, TurkCorpus GPL v3.0.

Reason: it is legal research material already acquired and used in these experiments, contains actual complex→simple aligned sentences, and can be used without the gated `go76dof/Fineweb_simplification_pairs` train file. Official-rule notes in `notes/official_rules_and_landscape.md` state that custom/swapped data is allowed within the ≤10M word corpus and ≤100M exposure accounting; tokenizer/augmenter/generator language exposure must also be counted.

Official-corpus-derived deterministic rewrites remain valuable as a later source-purity control, but they are not the fastest route to a meaningful first test because robust rule rewriting needs extra engineering and may produce weaker semantic pairs than the accessible human-aligned simplification corpus.

## How this differs from earlier WikiAuto tests

Earlier WikiAuto work should not be blindly repeated, but it also does not close the new experiment.

Closed or weak earlier results:

- pair adjacent smoke materialized exact 1M adjacent/shuffled JSONLs and verified identical text multiset and mask opportunity, but the intended fixed base then was small BERT 8×256 at 1M.
- Pure pair adjacency showed only mild signals: EWoK about +1.275 and Entity about +0.685 over shuffled, with weak absolute grammar/reading and lower absolute Entity than official WWM.
- 50/50 passive mixture reversed or weakened effects.
- Cross-view anchor objective failed: true-pair visible target did not reliably beat shuffled visible target.

The new experiment is different:

1. Use the current stronger S1 DeBERTa-v2 12×384/intermediate1280 backbone, not the early small-BERT screen.
2. Use 10M word exposure per arm as the first serious screen, not only 1M, because scale inversions have occurred in these experiments.
3. Use a matched-multiset design that separates correct pair alignment, seeing rewrite text at all, and same-window dual-surface exposure.
4. Keep the training objective pure WWM MLM first; do not add cross-view/InfoNCE/anchor losses until the pure data effect is measured.
5. Run full local EWoK and the standard available direct-checkpoint columns, not only fast screens.

## Required arms

Preferred first materialization uses five arms. If runtime must be reduced, source-only and rewrite-only can be collapsed only after the materialization statistics show which side dominates word count and style.

1. `aligned`: `source_i + " " + target_i`, same example/window, no artificial special token in the first version.
2. `shuffled`: `source_i + " " + target_perm[i]`, deranged permutation within buckets where possible.
3. `unpaired_mix`: same selected source and target sentences as independent examples in a deterministic mixed order, no local pair adjacency.
4. `source_only`: selected source sentences resampled/repeated deterministically to match the exact word budget.
5. `rewrite_only`: selected target sentences resampled/repeated deterministically to match the exact word budget.

The central comparisons are:

- `aligned - shuffled`: effect of correct semantic correspondence at equal text multiset.
- `aligned - unpaired_mix`: effect of local same-window pairing beyond seeing the same texts separately.
- `unpaired_mix - source_only/rewrite_only`: effect of rewrite text/style distribution independent of local pair alignment.

## Materialization invariants

Produce a manifest before training. Each record should include:

```json
{
  "pair_id": "...",
  "arm": "aligned|shuffled|unpaired_mix|source_only|rewrite_only",
  "source_dataset": "GEM/wiki_auto_asset_turk",
  "source_revision": "ac2b97468b38cb35fcebe327ac8e1cb6b55b6b99",
  "source_row_id": 0,
  "source_text_sha256": "...",
  "target_text_sha256": "...",
  "source_words": 0,
  "target_words": 0,
  "selected_words": 0,
  "cumulative_selected_words": 0,
  "token_len_baseline16k": 0,
  "wwm_word_groups_baseline16k": 0
}
```

Constraints:

- exact same total selected words for every arm;
- no use of gated FineWeb simplification-pair data;
- save dataset revision, file hash, selected row ids, and all random seeds;
- aligned and shuffled should have identical source and target text multisets;
- if using 256-token sequence length, either filter pairs so full examples fit or report source-side and target-side truncation separately for every arm;
- do not select or filter pairs by official evaluation scores.

## Fixed training base

Use the isolated leader-shape trainer, not the protected full-cycle trainer:

- script: `training/scripts/babylm_masked_train_leadershape.py`
- pass pre-materialized examples through its `--example_jsonl` path if supported; otherwise add minimal support in the isolated fork only.
- model: DeBERTa-v2, 12 layers, hidden 384, 12 heads, intermediate 1280, p2c/c2p relative attention, position buckets 256.
- tokenizer: baseline16k.
- optimizer: AdamW S1 default, not LAMB.
- mask: flat WWM, mask probability 0.15.
- sequence length: 256.
- word exposure: first serious screen 10M per arm; optional 100k smoke after materialization, but do not use smoke task scores for route decisions.

## Evaluation and interpretation

For every trained arm evaluate direct `chck_10M` on:

- BLiMP;
- BLiMP Supplement;
- Entity Tracking;
- COMPS;
- GlobalPIQA parallel/nonparallel and mean;
- Reading eye/self-paced and mean;
- full local EWoK with the existing word_tokenize pipeline.

A route-worthy signal is not absolute SOTA at 10M. It is a controlled aligned-vs-shuffled or aligned-vs-unpaired improvement on Entity and EWoK, preferably with COMPS or GlobalPIQA support, without a grammar/Reading loss large enough to erase the target benefit.

If aligned and shuffled are similar but both improve over source-only/rewrite-only, the effect is rewrite text distribution or dual-surface exposure rather than correct pair correspondence. If rewrite-only wins, the route is simplification-style data rather than pair alignment. If no arm moves Entity/EWoK, the accessible WikiAuto source is probably not enough and the next data source should be rebuilt rather than scaled blindly.

## Secondary LAMB preservation

If data materialization or training queue leaves idle capacity, a medium original-data LAMB continuation to around 30M may be useful to see whether the negative 10M target profile begins to recover after the peak-LR period. It should not displace the aligned-data experiment, and it should not be treated as a substitute for the data route unless it begins moving Entity and EWoK, not just MLM loss or GlobalPIQA nonparallel.
