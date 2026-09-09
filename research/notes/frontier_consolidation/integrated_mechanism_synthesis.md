# compact order experiment design: Integrated compact-view mechanism synthesis

## What we know after source conditioned ordering interaction synthesis and roberta stratified bridge and early local response

### Closed mechanisms

1. **Reciprocal view-topology conditioning** — CLOSED by causal fork (roberta stratified bridge and early local response/216)
   - 20M cheap7 interaction I = 0.5(FR+RF) − 0.5(FF+RR) = −0.077 (negative)
   - EWoK interaction −0.425, GlobalPIQA −0.235
   - Seeing source-then-view together provides no synergistic benefit over seeing each alone

2. **Compact-specific ordered source retrieval** — CLOSED by frozen probe (source conditioned ordering interaction synthesis)
   - Source-conditioned interaction I_f: compact 2.748, prefix 3.465, onegap 3.595
   - Compact has the SMALLEST I_f, not the largest
   - Ordered source-conditioned reconstruction is a general bidirectional MLM property
   - I_f stable 82M→100M (deltas < 0.02 nats) — does NOT track the capability decline

3. **Compact's advantage via stronger copied-token retrieval** — CLOSED by earlier analysis
   - After matching copy opportunity on copied targets, compact I_f is significantly lower than extractive families (median −0.491 nats, p>0 = 0.000)
   - Compact's lower raw I_f is composition: 21.7% source-absent fraction (vs <0.5% for extractives)
   - Source-absent I_f only 0.908 vs copied I_f > 3.0

4. **Causal GPT transfer** — CLOSED by causal transfer result synthesis
   - Compact−repeat cheap7 +0.078, cheap6 −0.145, cheap5 −0.151, EWoK+Entity −0.319
   - Positive cheap7 endpoints carried by GlobalPIQA/Reading volatility

### Surviving positive evidence

1. **Compact-view reinvestment**: +1.3464 mean7 at 80M under legal DeBERTa MLM (clean control trained and representation frontier)
   - BLiMP +1.53, Supplement +2.06, EWoK +2.03, Entity +2.33 (broad, not one-column)
   - This is the strongest validated single data-efficiency intervention

2. **source-absent target-label channel** (earlier analysis): +0.12 nats at 20M
   - Removing 7,649 source-absent content labels worsened source-absent denoising by +0.12 nats vs removing matched copied labels
   - Bootstrap CI [+0.098, +0.140], fraction above zero = 1.000
   - Persists for words never selected by WWM (+0.092 for 0-time selected)
   - Category-specific: retained_content −0.037, function_other +0.012 (within noise)
   - Not merely exact-target memorization — shapes nearby/category-level prediction

3. **Compact tail coverage** (source wide skeleton recurrence integrated): compact views cover 66.99% of source content positions vs 59.87% for prefix-repeat; tail recovery 70.26% mean, content density 65.27% vs 49.43%

4. **Budget efficiency**: compact views are shorter, freeing words for source diversity reinvestment (423,511 words reallocated in the validated intervention)

### Current mechanism model

The surviving object is **content-dense budget-efficient faithful compression with rare source-absent training targets**:

- Compact views compress source material while preserving core content, increasing content token density
- This exposes 7,649 source-absent content tokens that do not appear in the source (abstractive reformulations, content-preserving rewordings)
- The bidirectional MLM objective forces prediction of these tokens from surrounding context
- A target-removal control showed this prediction target class is causally useful: removing it costs the model more than removing equal-count copied targets
- The benefit is NOT about reciprocal source-view conditioning, NOT about compact-specific ordered retrieval, and NOT about copied-token advantage
- Budget savings from shorter views allow reinvesting in broader source diversity (more distinct examples)

### What the ordered-vs-scrambled experiment tests

The current experiment (two arms, 40M words each, both GPUs) tests:

**Does coherent word order within compact views contribute measurable downstream value at fixed compact lexical multiset?**

This separates two sub-hypotheses:
1. If ordered >> scrambled: fluent compressed syntax provides better local context for predicting source-absent content tokens → the compact view's grammatical structure matters
2. If ordered ≈ scrambled: the source-absent target signal depends on WHICH tokens appear (lexical identity), not their syntactic arrangement → content selection is the mechanism, not fluent compression

In light of finding, this maps directly:
- If scrambled degrades source-absent content denoising, ordered syntax helps the model build representations that can predict abstractive content
- If scrambled preserves source-absent content denoising, the abstractive content targets are sufficient regardless of their local context

### Connection to earlier analysis

The stronger whole-word copied-content control remains pending. Its readout will determine whether the source-absent channel survives better-matched copied controls. If it does and ordered is approximately equal to scrambled, the proposed mechanism is that faithful compact compression creates rare abstractive training targets that support generalizable representations independently of view syntax. This remains conditional, not an established conclusion.

### Evaluation plan for ordered/scrambled results

After the ordered and scrambled training results become available:

1. Check training integrity: both should produce chck_20M and chck_40M checkpoints with 20M/40M word exposure, same param counts, finite losses
2. Run selected cheap evaluation for all 4 checkpoints (ordered_20M, ordered_40M, scrambled_20M, scrambled_40M)
3. Use existing evaluator: `evaluate_compliant_endpoint.py` or `eval_custom_checkpoint.py`
4. Primary metric: cheap6 without GlobalPIQA; secondary: cheap5, relation/state, individual columns
5. Decision thresholds per experiment design note
6. Sanity check: compare ordered_40M to existing spatial repair route status reference chck_40M (cheap7 42.22) as a rough anchor (different row order, so not directly comparable, but should be in the same range)

### What comes after

Regardless of the ordered/scrambled result:
- If the source-absent target channel survives stronger control AND ordered ≈ scrambled: the transferable principle is "faithful compression naturally creates abstractive compact-side target events" → test whether the natural compact-data distribution transfers to another objective or architecture without explicitly prioritizing those targets.
- If the source-absent channel survives AND ordered >> scrambled: the principle includes "fluent compressed context helps learn from abstractive targets" → test natural compact data in a matched model/objective coordinate before changing loss allocation.
- If the source-absent channel does NOT survive control: the mechanism is more about budget efficiency and content density exposure, not label-specific learning → a different kind of transferable principle.

The stronger whole-word copied-content control has completed, and the local source-absent compact-side channel survives. This does not justify source-absent target weighting: a direct strict-innovation intervention improved its intended target loss while damaging mature broad BabyLM competence. If the pending ordered/scrambled comparison supports the channel, follow-up tests must preserve the distinction between naturally arising faithful compression under ordinary WWM and hand-prioritized targets. Candidate comparisons are transfer to another bidirectional MLM coordinate or matched faithful-compression pools with different naturally arising source-absent density, not explicit source-absent mask/loss selection.

## Protected assets (unchanged)

- chck_82M: public submitted 41.94, SHA `93ceb76...`
- chck_84M: projected Overall(AoA0) 42.0189, HF `040284de...`
- coherent86 α=0.75: projected Overall(AoA0) 42.1210, carrier SHA `40181994...`

## Files

- This note: `notes/integrated_mechanism_synthesis.md`
- Experiment design: `notes/compact_order_experiment_design.md`
- Launcher: `scripts/compact_order_mechanism_experiment.py`
- causal fork: `experiments/archive/representation_and_objectives/data/compact_directional_cheap7_eval_r2`
- target-selective: `experiments/archive/representation_and_objectives/data/target_selective_readout`
- mechanism synthesis: `research/notes/representation_and_objectives/mechanism_route_transition_after_causal_closure.md`
