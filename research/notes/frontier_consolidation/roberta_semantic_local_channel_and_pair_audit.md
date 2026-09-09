# roberta semantic local channel and pair audit — RoBERTa local channel semantic structure and repaired training-pair audit

CPU/file-only analysis; no training or model evaluation was performed. The selected official-compatible RoBERTa evaluation across all 10 checkpoints remains the decisive pending downstream comparison.

## 1. Repaired RoBERTa training-pair integrity (was a reader bug in earlier analysis)

The earlier analysis integrity output was wrong because the reader only accepted DeBERTa-style
`event=="train"` log rows, while the RoBERTa trainer writes bare training rows; it therefore
parsed 0 rows and falsely flagged a tokenizer_path "mismatch" between two identical
root-relative-vs-absolute strings. Repaired `roberta_training_pair_integrity.py`:
accept bare train rows, normalize root-relative/absolute path equality, and record expected
corpus-identity differences separately. Rerun output
`data/roberta_training_pair_integrity_repaired/`:

- Both arms: 2529 log rows, exact batch-word match, exact cumulative exposure match, exact LR
  match at every step, checkpoint names/exposures match, all 10 checkpoints present, 100M
  exposure each. `mechanically_matched=true`; the only recorded differences are the expected
  compact-vs-repeat corpus identity.
- Training surface (compact − repeat): endpoint loss delta −0.215947 (compact lower). The
  difference is entirely late: window mean loss delta early_1_500 +0.006350, mid_501_1500
  −0.025231, late_1501_2529 −0.256153, last_250 −0.213556. Total masked tokens +35,278 and
  candidate tokens +240,000 (compact has +24,000 tokens/10M pass in the changed block, per
  roberta transfer pair scaffold ready), so compact sees a slightly larger denoising load; the late loss gap is not a mask
  artifact (per-step mask-rate delta ≈ 0). This confirms a mechanically matched pair with a
  real late training-loss separation, but it is not downstream evidence.

## 2. RoBERTa local channel reproduces source attested fluent bridge prototype readout semantic structure

Built `roberta_semantic_local_response.py` (CPU-only). It re-ran the roberta stratified bridge and early local response local
pair-stratified probe on the completed compact/repeat 100M arms with `--write_event_losses`
for the late band (chck_60M..chck_100M), then joined per-event local NLL to the frozen
5,470-event set (SHA `c172b378...2d18709a`, event_sha_matches_manifest=true) and stratified by
the bridge dose response interpretation/234 lexical flags (imported directly from scripts).

Late-mean `repeat_minus_compact_advantage` (positive = compact-trained model has lower local
NLL) on compact **source_absent_content** events:
- relational_or_event_state: **+0.5667**
- not_relational_or_event_state: +0.4393
- ordinary_nonrel_nonentity: +0.4405
- capitalized_or_number: +0.4272

Compact **retained_content** events: relational_or_event_state only **+0.0701**, other +0.3450.

Novelty×relational interaction (SA_rel − SA_other) − (RC_rel − RC_other) = **+0.4023**, positive
and stable across all five late checkpoints (chck_60M +0.395, 70M +0.426, 80M +0.451, 90M
+0.367, 100M +0.373). Source-absent relational/event bootstrap p05/p95 is above zero at every
late checkpoint (e.g. 100M [+0.450,+0.739]).

Reading: the compact-trained local advantage in the **stock RoBERTa** coordinate is the same
semantic object measured for DeBERTa target-selective labels — a source-absent channel
concentrated on novel relational/event content, largest exactly where source-novelty and
relational/event semantics coincide (retained relational content is near zero). This is a
real cross-architecture convergence of the *local* channel, strengthening the claim that
faithful compression naturally introduces useful novel relational/event supervision.

## 3. Honest caveats

- This is local pseudolikelihood on engineered events, not natural prevalence and not official
  competence. The roberta stratified bridge and early local response late bridge already reports source-absent selectivity margin small
  vs controls at the aggregate level; the semantic split is what makes the concentration
  visible. The decisive question is whether the **selected official** compact-minus-repeat late
  trajectory (writing `roberta_full100m_integrated/`) improves stable
  families (cheap6 no GlobalPIQA, cheap5, EWoK+Entity, Supplement). Ordered/scrambled and
  earlier analysis/077 both showed a positive local source-absent channel can dissociate from downstream
  competence, so local concentration alone does not promote.
- Our RoBERTa local channel does NOT reproduce name/date sign reversal: source-absent
  capitalized_or_number is still positive (+0.43), not negative. This is a genuine difference
  between the local RoBERTa response and DeBERTa target-selective ablation, likely because
  the local probe measures response magnitude rather than causal label ablation. Do not overstate
  the convergence as identical.

## 4. Next decision (unchanged gate)
When the selected RoBERTa evaluation completes, read `roberta_full100m_integrated/roberta_transfer_integrated.json`,
use `roberta_transfer_result_reader.py` and `selected_prediction_movement_reader.py`
(LEFT=repeat, RIGHT=compact) on late checkpoints, and rerun `roberta_compact_result_synthesizer.py`.
Promote to an independent-seed RoBERTa replication only if stable-family late transfer is positive
AND concordant with this source-absent relational/event local structure; otherwise bound the stock
RoBERTa coordinate and return to natural compact-marginal decomposition.
