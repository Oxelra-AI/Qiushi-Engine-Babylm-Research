# roberta semantic local channel and pair audit — RoBERTa local channel semantic structure and repaired training-pair integrity

CPU/file-only analysis; no training or model evaluation was performed. The selected official-compatible RoBERTa evaluation across all 10 checkpoints remains the decisive pending downstream comparison.

## Repaired RoBERTa training-pair integrity

The earlier analysis integrity reader only accepted DeBERTa-style `event="train"` log rows, while the RoBERTa trainer writes bare training rows, so it parsed 0 rows and also compared one absolute tokenizer path to one root-relative equivalent path. Repaired `scripts/roberta_training_pair_integrity.py` to accept the actual row schema, normalize root-equivalent paths, and record expected compact/repeat corpus identity separately.

Output: `data/roberta_training_pair_integrity_repaired/roberta_training_pair_integrity.json`.

Main numbers:
- compact/repeat both have 2529 log rows, exact batch-word match, exact cumulative exposure match, exact LR match at every step, matching checkpoint names/exposures, all 10 checkpoints present, 100M counted-word exposure each, and no recipe mismatches besides expected corpus identity.
- compact endpoint training loss is 3.998428 vs repeat 4.214375, delta -0.215947. The gap is late: early_1_500 +0.006350, mid_501_1500 -0.025231, late_1501_2529 -0.256153, last_250 -0.213556.
- compact has +35,278 masked tokens and +240,000 candidate tokens across 100M, matching the roberta transfer pair scaffold ready +24,000 tokens/10M changed-block token-load difference; mean effective mask-rate delta is near zero.

This verifies the RoBERTa compact-vs-repeat pair as a matched training experiment. Training loss alone does not decide BabyLM selected competence.

## RoBERTa local channel semantic structure

Built `scripts/roberta_semantic_local_response.py`. It reads per-event local NLL losses from `data/roberta_pair_stratified_response_late_cpu_eventlosses/`, joins them to the frozen 5,470-event roberta stratified bridge and early local response event set (SHA `c172b378873553f209a6bfd9bf53a63e0bde251c1f77690f8abea0ef2d18709a`), and imports bridge dose response interpretation/234 lexical flags to ask whether the local compact-trained advantage is concentrated on relational/event-state words.

Output: `data/roberta_semantic_local_response/roberta_semantic_local_response.json`.

Late-mean `repeat_minus_compact_advantage` (positive = compact-trained model has lower local NLL) on compact `source_absent_content`:
- relational_or_event_state: +0.5667
- not_relational_or_event_state: +0.4393
- ordinary_nonrel_nonentity: +0.4405
- capitalized_or_number: +0.4272

Compact `retained_content`:
- relational_or_event_state: +0.0701
- not_relational_or_event_state: +0.3450

Novelty × relational/event interaction, `(SA_rel - SA_other) - (RC_rel - RC_other)`, is +0.4023 late-mean and positive at every late checkpoint: 60M +0.395, 70M +0.426, 80M +0.451, 90M +0.367, 100M +0.373. Source-absent relational/event bootstrap intervals are above zero at every late checkpoint.

Reading: the local compact-trained advantage transfers into the stock RoBERTa/BERT-style MLM coordinate and has the same broad semantic form as source attested fluent bridge prototype readout: strongest when source novelty and relational/event semantics coincide, not for retained relational/event content alone. This strengthens the mechanism that faithful compact views naturally introduce useful novel relational/event supervision.

Important distinction: RoBERTa local response does not reproduce name/date sign reversal; source-absent capitalized_or_number remains positive here. This local response measures model preference on fixed masked events, not the causal consequence of removing labels.

## Current status

The roberta semantic local channel and pair audit current synthesis is in `data/roberta_compact_result_synthesis_current_v2/roberta_compact_result_synthesis.json`: RoBERTa selected trajectory is not yet complete in visible files; late local response is ready and local only. No new expensive arm follows before the selected official-compatible RoBERTa scores are available.
