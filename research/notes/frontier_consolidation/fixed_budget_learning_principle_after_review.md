# fixed budget learning principle after review fixed-budget learning principle after verifier and challenge

This note revises `fixed_budget_learning_principle_current.md` after an independent scientific read (`data/external/independent_review01_verifier1_integration.md`) and reply #362. It is a research-state note for the next phase, not final writing.

## Primary measured object

The directly measured quantity is the change in selected multi-family benchmark performance when a fixed 10M-word / 100M-exposure stream replaces one corpus slice by another:

\[
\Delta_T(A,D;B,\mathcal C,\tau,s)
=Y_T(B-D+A;\mathcal C,\tau,s)-Y_T(B;\mathcal C,\tau,s).
\]

Here \(A\) is the admitted material, \(D\) is the displaced material, \(B\) is the base stream, \(\mathcal C\) is the model/objective/tokenizer/training coordinate, \(\tau\) is the exposure horizon, and \(s\) is the basin/seed. The evidence supports studying this as an interacting substitution, not as an additive data-quality score.

A useful mechanism form is:

\[
\Delta_T \text{ follows from }
\big[\text{distinct admitted support} \times \text{terminal residual work} \times \text{coordinate connection to reusable variables}\big]
-
\big[\text{opportunity value of the displaced support under the same coordinate}\big].
\]

The central correction from fixed budget learning principle after review is that this is not a fitted predictive equation. The factors are not independently identified yet. In particular, earlier analysis and fixed budget learning principle after review show that terminal clean-prior loss reduction alone is not enough.

## What the evidence now says

### 1. DeBERTa recurrence and distinct content

The DeBERTa MAX fixed-budget decomposition remains the cleanest evidence that exact-ish recurrence loses late value while distinct admitted content can keep paying:

- late 80M–100M \(V-C_{\max}\) exEntity5: +0.3853;
- late 80M–100M \(B-C_{\max}\) exEntity5: +0.3350;
- late 80M–100M \(R-C_{\max}\) exEntity5: +0.0343.

Thus the broad late difference is not compact-view specificity: \(V-B\) was only about +0.0503 late. It is more consistent with distinct content retaining value while exact recurrence supplies little additional support near the end of the budget.

The DeBERTa own-arm loss ladder agrees qualitatively: own changed-block 60M→100M loss reduction was view 0.295710, breadth 0.222660, repeat 0.148226. But the ex-ante clean-prior terminal rates from earlier analysis were view 0.057070±0.003073, repeat 0.037811±0.004865, breadth 0.033848±0.002734, clean_displaced 0.042389±0.001187. View remains active, but breadth does not exceed repeat there. Therefore the robust statement is: recurrence exhaustion and distinct-content persistence appear in the DeBERTa downstream and own-arm training geometry; clean-prior terminal rate is one useful signal but not a complete selector.

### 2. Register substitution changes the opportunity-value term

The MAX register pair admits the identical FineWeb block and changes only which clean material is sacrificed. At seed43022:

- childspeech_removed − adultprose_removed exEntity4: −0.4925 at 80M, −0.6125 at 100M;
- childspeech_removed − adultprose_removed cheap5(no Reading): −0.5120 at 80M, −0.7140 at 100M.

This means the arm that removed CHILDES/OpenSubtitles/BNC/Switchboard performed worse than the arm that removed Gutenberg/SimpleWiki. Under this coordinate, the developmental/spoken substrate was costlier to lose than curated adult prose, despite the intuition that adult prose should be nearer to the stable evaluation surface.

The result is comparative. Both FineWeb-admission register arms were nonnegative against clean on these reported aggregates at 100M: childspeech_removed − clean exEntity4 +0.0825 / cheap5 +0.270, adultprose_removed − clean exEntity4 +0.695 / cheap5 +0.984. The register finding is not that childspeech_removed is globally harmful relative to clean; it is that removing child/speech bought much less value than removing adult prose under identical admission.

The sign also falsifies the pre-score scalar pricings: the task-aware distribution profile, the strict evaluation-text profile, and the removed-block late-loss-rate measure all expected the opposite sign. The lowercase word-unigram control pointed in the observed direction but was far too small. earlier analysis's tokenization/truncation comparison found the childspeech_removed stream has fewer subword tokens and less truncation than adultprose_removed, so that stream-level exposure difference does not explain the negative sign.

This register result still needs the currently running seed43122 pair before it becomes a durable ingredient in the principle. It is also family-heterogeneous: at 100M child-minus-adult was BLiMP −1.45, Supplement −1.56, EWoK +0.94, COMPS −0.38. The effect is an aggregate over selected families, not uniform competence gain.

### 3. Entity is an allocation surface, not state competence by itself

Entity must stay separated from the broad reading. In register seed43022, childspeech_minus_adultprose Entity is negative under the official mixture and also negative under balanced zero/nonzero weighting. That reinforces the register sign inside Entity. But the arm-vs-clean Entity gains still have the old harmful split: adultprose_minus_clean is official positive while zero-operation rows are negative and nonzero-operation rows positive; childspeech_minus_clean has the same structure more strongly. Earlier dose and second-basin measurements showed the same pattern. Therefore Entity helps locate allocation, but official Entity aggregates do not establish reusable record formation, role binding, or general state tracking.

### 4. RoBERTa makes the coordinate term unavoidable

fixed budget learning principle after review ran the no-new-training RoBERTa terminal-rate ladder on existing MAX-dose view/repeat/clean checkpoints.

RoBERTa clean-prior 80M→100M rates were:

- view_changed: 0.130971±0.001738;
- repeat_changed: 0.132226±0.000081;
- clean_displaced: 0.149805±0.001629.

This does not reproduce the DeBERTa clean-prior view>repeat pattern; the displaced clean slice is more active than the admitted view/repeat blocks under the RoBERTa clean model. RoBERTa own-arm rates still show architecture-private late fitting of the admitted view text: own changed-block 60M→100M view 0.920573 versus repeat 0.228958, and 80M→100M view 0.205920 versus repeat 0.052248. Yet RoBERTa's late selected MAX view-clean result is negative: exEntity5 −0.6873 and cheap6(no GlobalPIQA) −0.5283.

reply #362 gives the most precise reading: finite experience helps when residual work is connected to the learner's reusable coordinates. RoBERTa may fit view text late in an architecture/private/tokenization coordinate without that work becoming target-usable selected competence. Thus the coordinate term has two roles: it shapes which residual work is visible to the learner, and it shapes whether that work becomes reusable.

## What the current principle is not

The experiments rule out several simple reductions for this coordinate:

- not compact re-expression alone: breadth competes with view and \(V-B\) is small late;
- not exact recurrence: repeat has little late broad value despite being easy to fit;
- not source-attested extraction or tail coverage alone: extractive and coverage controls did not reproduce the effect;
- not monotone amount/dose: ex-Entity dose effects were nonmonotone and often within seed spread;
- not surface-profile proximity: register profile predictions were wrong in sign;
- not a single terminal-rate scalar: DeBERTa clean-prior breadth/repeat and RoBERTa clean-prior rates break the simple ordering;
- not official Entity aggregate state competence: operation strata change the sign and interpretation.

The same-coordinate mature exEntity5 resolution floor is about 0.1765, so sub-half-point effects need replicate-aware reading. The register seed43022 effect is larger than this floor, but the prior two-seed history makes the running seed43122 replicate necessary.

## Live evidence still to be incorporated

1. **Register seed43122 replication.** Training and broad-family scoring remain in progress. The file-only reader is `experiments/archive/frontier_consolidation/scripts/register_seedreplicate_readout.py`; its output under `experiments/archive/frontier_consolidation/data/register_seedreplicate_readout` shows that the replication is not yet complete.

2. **In-corpus adult-prose official result.** This is the only missing admission-identity cell: official Gutenberg/SimpleWiki replaces developmental/speech rows without FineWeb. The frozen pre-score mature clean-prior rate is 0.046387 over 442,987 words; own in-corpus admitted loss remains active after training (60M→100M 0.273919, 80M→100M 0.054853). In-corpus scoring through 100M was complete, but the full readout remained unfinished while auxiliary subdose_full scoring continued. Final interpretation requires the completed per-target result files.

3. **Item/block-conditioned conversion.** challenge suggests the next mechanistic computation: among admitted rows with high own-arm late loss change, determine which are also high under clean-prior and which selected family/item movements accompany them. This would separate residual work connected to reusable variables from residual work trapped in private token/surface coordinates.

## Current working principle

Scarce-budget language learning is governed by substitutional marginal utility: experience helps when the learner receives distinct support that still supplies reducible work near the budget horizon, when the sacrificed material has lower opportunity value for the target mixture, and when the learner's architecture/objective/tokenization coordinate connects that residual work to reusable variables. Repetition can exhaust by lowering local loss without adding enough distinct support. Distinct content can persist when it remains active near the end of training. But the same residual work can fail to improve selected competence if the coordinate fits it privately or if the displaced corpus slice carried higher task-relevant support.

This is the strongest current scientific content of the post-SOTA phase. It is not yet a general predictive formula; the next phase should use the running register replicate, the in-corpus cell, and item/block-conditioned conversion measurements to turn the mechanism from a structured account into a sharper law.
