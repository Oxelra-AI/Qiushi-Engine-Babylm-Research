# earlier analysis — turn same-entity hard negative into real training discriminator

## Experimental decision

The same-entity cross-clause hard-negative design is clean enough to stop construction-only iteration. The next evidence must come from real training, not another round of control design.

## 100k materialization completed

`data/same_entity_cross_clause_100k/` contains exact 100k-word arms:

- `true_pair_adjacent_100000w.jsonl`
- `hard_negative_same_entity_100000w.jsonl`
- `orig_only_100000w.jsonl`
- `shuffled_pair_adjacent_100000w.jsonl`

Metadata: `data/same_entity_cross_clause_100k/materialization_meta.json`

Key numbers: 15,000 streamed FineWeb-Edu docs; 10,947 basic-pass; 15,733 simplification items; 1,227 cross-sentence entity groups; 2,681 true pairs; 2,656 hard negatives; all four arms exact 100,000 words.

The manual examples show noise remains in deterministic simplification, but the hard negative now has the required structure: real same-document, same-entity, different-fact text. The route should be judged by training, not by more construction iteration.

## 1M materialization launched

Background task:

- label: `earlier analysis materialize same-entity pair 1M arms`
- command: `python scripts/same_entity_cross_clause_materialize.py --target_words 1000000 --max_stream_docs 160000 --out_dir data/same_entity_cross_clause_1M` with local HF cache settings.

The estimated duration based on the 100k run is roughly 1.5–2 hours; this is an estimate, not a completed-run measurement.

## Matched 1M training discriminator to launch after collection

If `data/same_entity_cross_clause_1M/materialization_meta.json` reports exact 1M words for all arms, train all four arms with the fineweb relation vs random 1m profile matched short-run configuration:

- model: DeBERTa-v2 8×480
- tokenizer: baseline16k
- objective: WWM p=0.15
- seed/init: seed 42, extra_init_seed 456, train_rng_seed 789
- batch size: 128, exact 1,000,000 word exposure
- checkpoint: `chck_1M`

Arms and intended interpretation:

1. `true_pair_adjacent`: same fact restated; primary positive condition.
2. `hard_negative_same_entity`: same entity and document, different fact; primary controlled negative.
3. `shuffled_pair_adjacent`: weak broken-topic control, retained for comparison with earlier designs.
4. `orig_only`: FineWeb source/syntax baseline.

Primary scientific contrast:

`true_pair_adjacent - hard_negative_same_entity` on Entity Tracking fast and EWoK fast.

Interpretation:

- If true_pair beats hard_negative on Entity/EWoK, aligned restatement of the same binding is a real transferable experience-organization signal.
- If true_pair ≈ hard_negative but both beat orig/shuffled, same-entity multi-fact exposure matters more than same-binding restatement.
- If no arm moves Entity/EWoK, close this deterministic pair-restatement route as a SOTA path.

## Parallel EWoK lane

fineweb relation vs random 1m profile already established a real EWoK lever: relation-explicit FineWeb improved EWoK fast by +3.27 over same-source random FineWeb at 1M, with Entity flat. This should be scaled in parallel after the pair discriminator starts, because the SOTA gap is multi-column and EWoK contributes about 0.63 Overall points.

The materializer is `scripts/fineweb_relation_matched_materialize.py`. Use it to create larger same-source random-quality and relation-explicit arms (candidate next scale: 3M or 10M, not 100M first), with token-aware packing if feasible, then train the same DeBERTa-v2 WWM configuration and evaluate fast columns. The scientific question is whether the +3.27 EWoK signal persists at larger budget without damaging Supplement/Reading/Entity too much.

Do not let the pair-control experiment prevent this EWoK lane from producing real training evidence.
