# fixed budget learning principle after review current fixed-budget data-efficient learning principle

This is a research-facing synthesis, not final writing. It joins the durable evidence from the Strict-Small post-SOTA phase into one coordinate-specific mechanism that can be tested or revised by the still-running register seed-replication and in-corpus official scorer.

## Scientific object

Under a strict small-data/small-compute budget, a training intervention is a **substitution**, not just an addition. Its value depends on four coupled quantities:

\[
\Delta \mathrm{value}(A \leftarrow D) \approx
M(A)\,r_{\theta,\tau}(A)\,\Gamma_{\mathcal A}(A \to T)
-
M(D)\,q(D,T,\mathcal A),
\]

where:

- \(A\) is the admitted material and \(D\) is the displaced material;
- \(M(A)\) is distinct admitted mass, not merely repeated exposures or raw token count;
- \(r_{\theta,\tau}(A)\) is terminal-window active error / loss-reduction on the admitted material at the budget horizon \(\tau\), measured here by forward MLM loss reduction over 80M→100M or 60M→100M;
- \(q(D,T,\mathcal A)\) is the opportunity value of the displaced material for the target task mixture \(T\) under architecture/objective coordinate \(\mathcal A\);
- \(\Gamma_{\mathcal A}(A \to T)
\) is a conversion term: the architecture/objective/representation must turn residual prediction work on the admitted material into broad transferable competence.

The strongest lesson is negative as well as positive: no single scalar tested so far—compactness, repetition, amount, static surface proximity, terminal loss level, or clean-prior mature rate alone—predicts broad competence under the BabyLM Strict-Small constraint.

## Evidence that recurrence exhausts while distinct admitted content can persist in DeBERTa

DeBERTa MAX-dose fixed-budget experiments separate three reference arms:

- V = compact view/source packet admission;
- R = exact-ish hash-rotated repetition of the same source-side content;
- B = same-population independent whole-sentence breadth.

The late 80M–100M exEntity5 values from the matched-clean decomposition were:

- \(V-C_{\max}\): +0.3853;
- \(B-C_{\max}\): +0.3350;
- \(R-C_{\max}\): +0.0343.

Thus exact recurrence had little mature broad value, while distinct content retained value. The DeBERTa own-arm persistence ladder matched this qualitatively: from 60M→100M, own changed-block loss reduction was repeat 0.148226, breadth 0.222660, view 0.295710. Repeat rows became easy and specialized; distinct rows stayed harder and kept learning late.

The cleaner ex-ante version was weaker than the neat story. earlier analysis clean-DeBERTa 80M→100M three-replicate rates were:

- view_changed: 0.057070 ± 0.003073;
- repeat_changed: 0.037811 ± 0.004865;
- breadth_changed: 0.033848 ± 0.002734;
- clean_displaced: 0.042389 ± 0.001187.

View remains a robust terminal active-error source, but breadth and repeat overlap and do not reproduce the late downstream ordering. The mature active-error quantity is a component, not a complete law.

Evidence paths:

- `research/documents/frontier_consolidation/data/reference_decomposition_readout/reference_decomposition_summary.md`
- `experiments/archive/frontier_consolidation/data/persistence_loss_analysis/persistence_loss_results.json`
- `research/documents/frontier_consolidation/data/mature_clean_prior_rate_spread/mature_clean_prior_rate_spread.md`

## Register substitution: the opportunity value of what is displaced is not surface proximity

The MAX register pair held admitted FineWeb fixed: same 1,118,587 FineWeb admitted words, exact 10M/100M streams, fixed legal tokenizer, same DeBERTa seed43022 coordinate, and differed only in which clean material was removed.

Observed official-compatible seed43022 contrasts were:

- childspeech_removed − adultprose_removed at 80M: exEntity4 −0.4925, cheap5(no Reading) −0.5120;
- childspeech_removed − adultprose_removed at 100M: exEntity4 −0.6125, cheap5(no Reading) −0.7140.

The negative sign means that sacrificing CHILDES/OpenSubtitles/BNC/Switchboard cost more than sacrificing Gutenberg/SimpleWiki under identical FineWeb admission. This inverts the initial intuition that curated adult prose should be nearer and more valuable for the stable evaluation text.

Three substantive pre-score pricings failed in sign:

- the distribution proximity prediction task-aware distribution-profile prediction expected childspeech_removed − adultprose_removed to be positive;
- the stricter evaluation-text profile prediction also expected a positive sign;
- the register rate and spread interpretation removed-block late-loss-rate pricing expected a positive sign because adult-prose removed rows had larger clean 60M→100M loss reduction than child/speech removed rows.

The lowercase word-unigram control pointed in the observed negative direction but was much too small (around −0.068 to −0.069 exEntity5 and −0.082 to −0.085 cheap6). earlier analysis's stream tokenization comparison also found childspeech_removed has fewer subword tokens and less truncation than adultprose_removed (−111,382 pre-truncation tokens, −92,771 post-truncation tokens, −1,284 truncated rows, −18,611 truncated tokens), so effective exposure burden runs against the observed negative sign.

This makes the displaced-material term real: the opportunity value of the developmental/spoken substrate is not captured by surface resemblance to evaluation prompts or by a simple removed-block terminal-rate scalar. The sign still requires the seed43122 replication now running before it can be treated as durable rather than a one-basin allocation.

Evidence paths:

- `experiments/archive/frontier_consolidation/data/register_readout_only/decisive_contrasts.csv`
- `experiments/archive/frontier_consolidation/data/register_entity_numops_readout/entity_numops_late_summary.csv`
- `research/documents/frontier_consolidation/data/register_stream_effective_exposure_audit/effective_exposure_audit.md`

## Entity cannot carry the general principle

Entity movement repeatedly splits by operation type. In the MAX register readout:

- childspeech_minus_adultprose Entity was negative overall and negative under balanced zero/nonzero weighting;
- adultprose_minus_clean had official Entity positive, but zero-operation rows were negative while nonzero-operation rows were positive;
- childspeech_minus_clean showed the same harmful mixture structure more strongly.

Earlier DeBERTa dose and second-basin measurements likewise showed official Entity gains coming from nonzero-operation rows while zero-operation retention declined, and neutral 50/50 zero/nonzero weighting was negative. RoBERTa did not reproduce the large DeBERTa Entity allocation. Therefore Entity is a useful stress surface for allocation, but not evidence by itself for general state competence, role binding, or reusable record formation.

Evidence paths:

- `research/documents/frontier_consolidation/data/entity_balanced_and_transfer_readout_full/entity_balanced_readout_summary.md`
- `experiments/archive/frontier_consolidation/data/register_entity_numops_readout/entity_numops_late_summary.csv`

## RoBERTa terminal-rate test: architecture affects the active-error component and conversion

fixed budget learning principle after review ran the deferred CPU-only RoBERTa terminal-rate ladder on existing MAX-dose view/repeat/clean checkpoints. It performed no new training and no official benchmark evaluation. RoBERTa already had a negative broad late downstream view-clean contrast:

- RoBERTa MAX view-clean late exEntity5: −0.6873;
- RoBERTa MAX view-clean late cheap6(no GlobalPIQA): −0.5283.

The new RoBERTa clean-prior 80M→100M rates were:

- view_changed: 0.130971 ± 0.001738;
- repeat_changed: 0.132226 ± 0.000081;
- clean_displaced: 0.149805 ± 0.001629.

So RoBERTa does not preserve the DeBERTa-like ex-ante view > repeat terminal-rate ordering; generic displaced clean rows are even more active than the admitted view/repeat blocks under this sampling. Its own-arm rates still show that view text remains hard and changes late after admission (own changed 60M→100M view 0.920573 versus repeat 0.228958; own 80M→100M view 0.205920 versus repeat 0.052248), but this late fitting does not become broad selected competence in RoBERTa.

The architecture term is therefore not a small afterthought. It has at least two roles:

1. it changes the measured active-error geometry itself; and
2. it changes whether residual prediction work becomes useful transferable competence.

Evidence path:

- `research/documents/frontier_consolidation/data/roberta_terminal_rate_ladder/roberta_terminal_rate_ladder.md`

## Resolution and remaining live tests

The mature same-coordinate exEntity5 resolution floor is about 0.1765. Register seed43022's broad childspeech_minus_adultprose negative contrast is larger than this, but this program has repeatedly seen seed/basin movement change apparent signs. The seed43122 register replication is therefore still essential and is already running.

The in-corpus adult-prose arm is the missing admission-identity cell: official Gutenberg/SimpleWiki replaces developmental/speech rows without FineWeb. Its frozen pre-score mature clean-prior rate was 0.046387 over 442,987 words, with owner 60M→100M rate 0.273919 and 80M→100M rate 0.054853 after training. It predicts non-flat persistence only if the DeBERTa coordinate converts that residual error into target-relevant competence. The in-corpus chunks had been scored, but auxiliary subdose_full scoring and the final readout remained incomplete. This note therefore does not treat the partial in-corpus scores as authoritative.

## Current coordinate-specific principle

A scarce-budget learner does not simply benefit from more repetitions or from text that looks closer to the target. It benefits when the admitted experience supplies distinct, still-active prediction work at the terminal budget, when the displaced experience has lower opportunity value for the target mixture, and when the architecture/objective/representation converts that work into broad competence. Repetition can exhaust because it collapses residual error without adding enough distinct support; distinct content can persist because it continues to create reducible work near the budget horizon; but whether this work helps depends on what was sacrificed and on the model coordinate that processes it.

This principle is currently established as a coordinate-specific mechanism in the BabyLM Strict-Small legal coordinate, not as a universal law. It explicitly excludes the simpler interpretations already contradicted by evidence: compact-form advantage alone, exact recurrence, source-attested extraction, static surface-profile proximity, amount/coverage monotonicity, unbalanced Entity aggregate gains, and terminal active-error rate alone.
