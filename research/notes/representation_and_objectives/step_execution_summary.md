# ewok tokenizer eval interface audit execution summary — corrected-tokenizer wait, EWoK relation margins, corpus exposure

## Corrected-tokenizer retrains

The earlier analysis Strict-Small-tokenizer retrains remain the decisive work for the active BabyLM Strict-Small SOTA goal:

- seed43022, `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43022`
- seed43122, `experiments/archive/representation_and_objectives/training/runs/strictsmalltok_compact_view_reinvest_seed43122`

Per the runtime ledger at ewok tokenizer eval interface audit start both were still running; no endpoint score should be inferred yet. I did not poll them or launch another training/data route.

I reran the post-training controller dry-runs for both seeds. The only reported failures are expected because training is not finished yet: missing `chck_100M`, missing ladder checkpoints `chck_50M` through `chck_100M`, and the absent `chck_100M/tokenizer.json` SHA check. There were no remaining path/argument failures in the controller.

Post-training controller to use after completion remains:

```bash
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43022 --gpu 0
python3 -B experiments/archive/representation_and_objectives/scripts/strictsmalltok_posttrain_eval_controller.py --seed 43122 --gpu 1
```

## Official-compatible EWoK margin exporter

I wrote `experiments/archive/representation_and_objectives/scripts/official_ewok_margin_exporter.py`.

Unlike the decision framework and clean collation plan standalone PLL approximation, this script mirrors the current official MLM EWoK scoring path:

- candidate 0 is `Context1 + Target1`; candidate 1 is `Context2 + Target1`;
- completion span is `" " + Target1`, exactly as `decode_ewok` constructs it;
- tokenize the full candidate sentence with offsets;
- mask each token whose offset overlaps the completion span;
- sum the target log-probs at temperature 1.0;
- save the two candidate scores and margin before argmax.

A 10-row validation on old inherited-tokenizer four-cell endpoints exactly matched saved official predictions for all four models. Then a focused 553-row subset also matched official predictions exactly:

- clean430 match rate: 1.0
- reinv430 match rate: 1.0
- clean431 match rate: 1.0
- reinv431 match rate: 1.0

Focused margin artifacts:

- Selection script: `experiments/archive/representation_and_objectives/scripts/select_ewok_margin_targets.py`
- Selection CSV: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/ewok_margin_focus_selection.csv`
- Margin JSON: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553.json`
- Margin CSV: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553_records.csv`
- Analysis JSON: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_margin_focus553_analysis.json`
- Note: `research/notes/representation_and_objectives/official_ewok_margin_focus_analysis.md`
- Confident examples: `experiments/archive/representation_and_objectives/data/official_ewok_margin_subset/official_ewok_confident_negative_interaction_examples.csv`

## Focused EWoK relation-margin result

The 553-row subset contains all strongest negative treatment-by-seed rows in the five worst relation domains plus controls:

- material-dynamics, physical-dynamics, spatial-relations, physical-interactions, social-relations;
- controls from agent-properties, material-properties, social-interactions.

Main numbers:

- focused rows: 553
- negative treatment-by-seed rows: 318
- pattern `0110` rows (clean430 wrong, reinv430 correct, clean431 correct, reinv431 wrong): 168
- confident negative rows (`reinv430` margin > +2 and `reinv431` margin < -2): 22
- moderate negative rows (`reinv430` > +1 and `reinv431` < -1): 82
- near negative rows (both reinvest margins within ±1): 51 = 0.160 of negative rows
- pattern `0110` margin interaction mean: -6.735, median -5.638

Interpretation: the old inherited-tokenizer relation instability is not mostly infinitesimal ties or random tie-breaking. Many rows show moderate or confident opposite-signed relation preferences between the two treatment seeds. The failure mode is learned relation-preference polarization, especially in social-relations, material-dynamics, spatial-relations, physical-interactions, and physical-dynamics.

This is not an endpoint score and is enriched for unstable rows. Its main value is to define what to inspect in the corrected-tokenizer endpoints once their official EWoK predictions exist.

## Background full old four-cell EWoK margin atlas

CPU evaluation of the existing four-cell inherited-tokenizer endpoints was launched using the same official-compatible margin exporter on all 7,618 EWoK rows:

- output directory: `experiments/archive/representation_and_objectives/data/official_ewok_margin_full_old4`

Do not poll it. When delivered, read the authoritative result and use it to decide whether the focused-subset polarization generalizes across all EWoK or remains concentrated in the worst relation groups. This can become the baseline for comparing corrected-tokenizer endpoints.

## Relation lexeme corpus audit

A first regex-per-lexeme implementation timed out after syntax checking and produced no scientific result. I replaced it with a faster single-tokenization pass:

- Script: `experiments/archive/representation_and_objectives/scripts/relation_lexeme_corpus_audit_fast.py`
- JSON: `experiments/archive/representation_and_objectives/data/relation_lexeme_corpus_audit/relation_lexeme_corpus_audit_fast.json`
- Note: `research/notes/representation_and_objectives/relation_lexeme_corpus_audit.md`

The audit compared four 10M corpora:

- `clean_qwen_10M`: `experiments/archive/compact_experience/data/qwen_clean_aligned/training_corpora/qwen_aligned_10M.jsonl`
- `compact_view_reinvest_10M`: compact-view reinvest corpus used for the corrected-tokenizer retrains
- `compact_repeat_reinvest_10M`
- `lengthmatched_compact_reinvest_10M`

Selected pair-pattern row counts:

- above/below: clean 66, compact_view 67, compact_repeat 66, lengthmatched 64
- parent/child: clean 158, compact_view 180, compact_repeat 180, lengthmatched 159
- teacher/student: clean 65, compact_view 84, compact_repeat 83, lengthmatched 66
- boss/subordinate: 0 in all four corpora
- kick/drop/touch: 0 in all four corpora
- sink/float: 8 in all four corpora
- rise/fall: clean 202, compact_view 194, compact_repeat 193, lengthmatched 198
- grow/shrink: 9 in all four corpora

Selected lexeme counts:

- above: clean 1253, compact_view 1331
- below: clean 617, compact_view 692
- boss: clean 382, compact_view 359
- subordinate: clean 7, compact_view 6
- parent: clean 116, compact_view 136
- child: clean 2185, compact_view 2328
- teacher: clean 882, compact_view 914
- student: clean 303, compact_view 416
- kick/drop/touch/break slightly lower in compact_view than clean; sink equal; float lower.

Interpretation limits: this is surface lexical exposure, not relation competence. It rules out simple absence for many relation words, shows compact-view and compact-repeat are almost identical in surface counts, and shows some templatic EWoK pair combinations are absent across all corpora. Combined with the margin result, the likely issue is relation-direction/composition learning under templatic contrasts and seed-dependent representation dynamics, not only ordinary word exposure.





## Current scientific state after ewok tokenizer eval interface audit

The active route remains the corrected Strict-Small-tokenizer retrain and official-coordinate evaluation. The inherited-tokenizer 42.033 endpoint remains mechanism evidence only. The new margin evidence improves the mechanism picture: old compact-view reinvestment can strongly polarize relation preferences across seeds in EWoK relation domains. The corrected-tokenizer endpoints should be judged first by full official Overall; if they fall or differ across seed in EWoK, the ewok tokenizer eval interface audit margin tools should be applied to their official EWoK predictions to see whether the same moderate/confident polarization recurs.
