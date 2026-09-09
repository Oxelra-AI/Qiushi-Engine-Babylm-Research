# Natural re-mention probe construction report

## Purpose and status

This directory contains a CPU-built, scoring-ready natural-text instrument for testing whether the FUNCTIONAL_RELATION_STUDIES VIEW/REPEAT/CLEAN checkpoints use an earlier mention when predicting a later mention. It contains no generated rewrite and no Entity benchmark item. The probe is an instrument, not a model result: no checkpoint was scored during construction.

The frozen set has 258 records, exactly 129 in each class, with one record per source row. Every cross-class match has a stable `class_matching.pair_id`. The two class members have exactly matched intervening-word bins; 120/129 pairs additionally match re-mention length and sentence distance, seven match distance and length, and two match distance only.

## Source decision and contamination control

The proposal's preferred file, `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl`, is not held out from the current experiment. Its complete 1,118,720-word stream is the prefix of the CLEAN 10M training pool, and the earlier analysis metadata records a 133-word neutral top-up in the VIEW/REPEAT changed block. It was therefore rejected. The audit and digest are in `natural_remention_probe_summary.json`.

Candidates instead come from the natural FineWeb component of `experiments/archive/representation_and_objectives/training/data/cached_fineweb_seqsafe96_candidate/cleanqwen_seqsafe_fineweb_single_doc_10M.jsonl`. Its metadata describes quality-filtered, single-document chunks sourced from `experiments/archive/initial_model_studies/data/fineweb_relation_matched_3M/fineweb_random_quality_3000000w.jsonl`. The scanned reservoir has 21,916 chunks, 1,753,280 words, and 5,566 documents. This source may have served other research runs, so disjointness is established against the actual FUNCTIONAL_RELATION_STUDIES arm pools rather than inferred from its label.

For each candidate-bearing row, the builder normalizes case/apostrophes, extracts word tokens, and rejects the whole row if any normalized 12-word sequence occurs in any of these 10M pools:

- `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_view_dose2p64x_10M.jsonl`
- `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/compact_repeat_dose2p64x_10M.jsonl`
- `experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/cleanqwen_lengthmatched_dose2p64x_10M.jsonl`

The 30M-word scan considered 3,999 candidate-bearing rows: 3,019 were rejected for at least one arm overlap, leaving 1,384 candidate records for matching. VIEW and REPEAT each hit 3,018 rows; CLEAN hit none. This deliberately conservative rule can discard unrelated rows sharing a formulaic phrase, but makes verbatim training leakage unlikely. The full scan is cached in `training_overlap_cache.json`, keyed by source and arm SHA-256 digests.

## Extraction rules

The builder uses CPU spaCy `en_core_web_sm` annotations plus explicit rules. It also rejects common tables, multi-URL rows, dictionaries, bibliographies, multiple-choice items, outline fragments, queries/code, and word-puzzle artifacts.

`verbatim_remention` contains:

- 125 exact named-entity repeats recognized at both spans, with identical normalized text, at least three intervening words, and no more than three sentence boundaries;
- four singular indefinite-to-definite nominal repeats such as `a book` -> `the book`.

`nonidentical_remention` contains:

- 35 full-name -> acronym cases;
- 37 full PERSON-NER span -> surname/final-token cases;
- 31 person-name -> sentence-initial person-pronoun cases;
- 11 nonperson-name -> sentence-initial `it`/`its` cases;
- 15 explicit PERSON/ORG -> appositive role/description cases.

Pronoun rules require exactly one compatible recognized entity in the preceding sentence, a subject antecedent, no intervening compatible entity, no missed untyped proper-name competitor, and a sentence-initial subject/possessive pronoun. Quotation, expletive-`it`, dialogue, and several NER-boundary error patterns are excluded. Appositives require a direct dependency and role-head whitelist; organization descriptions require an article and cannot overlap another NER span.

All classes share the same intervening-word-bin counts: 0-4: 30, 5-9: 22, 10-19: 38, 20-39: 24, 40-79: 15 per class. Mean intervening distance is 17.53 words for verbatim and 16.91 for nonidentical; both medians are 12. Re-mention lengths are close but not perfectly identical because two matches relax length: verbatim 119/7/3 and nonidentical 117/7/5 in the 1/2/3+ bins. All source rows are unique; the selected set spans 239 source documents.

## Record schema and corruption

Each JSONL object stores source path/line/id/document fields; complete original text; antecedent, re-mention, word, character, sentence, and syntactic fields; local context; masking plan; acceptance reasons; contamination audit; matching pair; and a realized antecedent corruption.

The later re-mention is the MLM target. The two context conditions are:

- antecedent present: `full_text_original`;
- antecedent replaced: `corruption_plan.full_text_antecedent_replaced`.

The re-mention text is unchanged, but a replacement can change character length; `corruption_plan.remention_char_start_replaced` and `remention_char_end_replaced` are the adjusted offsets for the corrupted condition. Of 258 replacements, 246 use another selected antecedent with the same coarse type and exact whitespace-word count; 12 use a type-oriented neutral fallback. Every replacement is exactly word-count matched. Replacement can alter tokenizer subword count, casing, plausibility, gender, or local syntax; analyses should therefore include subtype and replacement-strategy sensitivity checks. Pronoun replacements preserve only coarse type, so pronoun conditioning is intrinsically weaker than acronym/surname conditioning.

## Scoring plan and pre-stated predictions

For each model/checkpoint, tokenize both conditions under that model's tokenizer, locate the later span using the original or replaced-condition offsets as appropriate, mask every model subword belonging to that span, and compute token-mean target NLL. Define antecedent gain as `NLL(replaced) - NLL(present)`, so positive values mean the earlier antecedent helps. For multi-token targets, report both token-mean gain and a sensitivity analysis that masks one word/group at a time if feasible.

Aggregate first within `class_matching.pair_id`, distance bin, subtype, sentence distance, target length, and replacement strategy. Do not interpret raw class magnitude without distance control; the 15 appositive cases have zero intervening words and should be reported separately. Use record- or pair-level bootstrap intervals and retain seed/checkpoint identity rather than treating multiple checkpoints as independent observations.

Predictions fixed before scoring:

- REPEAT should exceed CLEAN most clearly on `verbatim_remention` antecedent gain.
- VIEW should exceed CLEAN most clearly on `nonidentical_remention` antecedent gain.
- Distance-matched arm-by-class contrasts matter more than raw gain magnitude.

These are predictions, not findings. The cleanest confirmatory interaction is `(REPEAT-CLEAN)_verbatim > (REPEAT-CLEAN)_nonidentical` together with `(VIEW-CLEAN)_nonidentical > (VIEW-CLEAN)_verbatim`; subtype estimates remain diagnostic because heuristic coreference quality differs.

## Audit, limitations, and files

`natural_remention_probe_sample.md` is a stratified 100-record display with marked spans. A single-reviewer audit in `manual_sample_audit.json` judged 94 clear plausible re-mentions, six plausible but ambiguous/register-confounded, and zero likely incorrect. This is not adjudicated ground truth. Remaining confounds include named-entity errors, modifier rather than noun-phrase repetition, truncated 64-96-word chunks, titles/UI labels, list/address register, local appositives, and gender/agreement changes under corruption. Report the full set and a sensitivity subset excluding pronouns, appositives, and the six flagged audit records.

Important artifacts:

- `natural_remention_probe.jsonl`: scoring input.
- `natural_remention_probe_summary.json`: full provenance, filters, overlap audit, counts, bins, digests, and heuristic definitions.
- `natural_remention_probe_sample.md`: 100-item marked audit sample.
- `manual_sample_audit.json`: single-reviewer sample labels.
- `validation_results.json`: machine validation.
- `build_natural_remention_probe.py`: deterministic builder (`seed=9061002`).
- `validate_natural_remention_probe.py`: structural/provenance validator.
- `training_overlap_cache.json`: digest-guarded result of the expensive overlap pass.

Rebuild and validate from the repository root with:

```bash
${QIUSHI_PYTHON:-python} experiments/archive/relation_learning/analysis/natural_remention_probe/build_natural_remention_probe.py
${QIUSHI_PYTHON:-python} experiments/archive/relation_learning/analysis/natural_remention_probe/validate_natural_remention_probe.py
```

The final full cached build used one CPU process, about 160 seconds wall time, and about 1.0 GiB peak RSS. No GPU was used.
