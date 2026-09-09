# whole system route comparison draft — whole-learning-system route comparison after FW boundary

## Starting evidence

The active goal remains a legal BabyLM 2026 Strict-Small Overall SOTA. The best complete legal endpoint is spatial repair route status compact-view reinvest with a compliant same-pool 16k tokenizer: Overall 41.257770896404615. The live visible target is 41.8, so the remaining gap is +0.5422 Overall. If only the seven cheap columns moved and SuperGLUE/AoA remained unchanged, the 100M cheap7 requirement is about +0.6972.

fw absolute progress decision discipline established a real boundary for the compact-view data family. In the repaired expanded FineWeb comparison, aligned compact recurrence beat independent whole-sentence breadth inside the FW companion-budget family at 100M (+0.5443 cheap7). But the best endpoint was only +0.1757 cheap7 above the existing legal trajectory and moved only 3/7 cheap columns positively: BLiMP +1.00, Entity +0.96, GlobalPIQA +2.57, but Supplement -2.31, EWoK -0.14, COMPS -0.17, Reading -0.683. Therefore simply increasing compact-view dose inside the current legal tokenizer + DeBERTa-v2 8x480 + fixed WWM + AdamW system mainly redistributes competence and is not a SOTA-scale path.

The compact semantic second-view substrate is still protected: within the legal spatial repair route status tokenizer coordinate, reinvest minus clean is +1.2921 mean7 at 70M and +1.3464 mean7 at 80M. The next mechanism should preserve this substrate but need not be pair-local and should not be derived from remaining weak official columns.

## Important prior evidence that constrains route choice

- Global word-mean MLM and minfreq50 support-floor tokenization both shifted strength toward GlobalPIQA-nonparallel while weakening Supplement/EWoK. More global lexical-credit or vocabulary reshuffling is not the right next move.
- Strict source-absent innovation masking made the intended local targets easier but damaged broad cheap7 by -1.0529 at 80M. Local target learnability is not enough.
- Positive source/rewrite identity consistency is saturated; naive source-view agreement is not a useful training signal.
- EWoK crossed interaction is an EWoK readout but not a reliable broad competence proxy.
- Tail optimizer rephase produced only a small original-stream +0.0845 cheap7 and is closed as a SOTA route.
- Legal40k/minfreq/depth alone are closed by results; plain added depth does not solve the gap.
- Same-stack token-shift MNTP auxiliary in COMPACT_EXPERIENCE was fully evaluated and negative: Overall 39.7256 vs clean-Qwen 41.3443, with EWoK/Reading gains but broad losses and AoA -13.4. This closes that exact MNTP auxiliary, not all dense or discriminative objectives.
- High-LR LAMB x data screen in COMPACT_EXPERIENCE did not show a broad persistent interaction: interaction equal7 was negative at 3M and 5M and only mildly positive at 10M, partly GlobalPIQA-driven with Supplement negative. A simple optimizer-copy route is not enough.
- Adaptive MLM / hard difficulty masking has external BabyLM evidence, but local 100M official-corpus screens were redistributive: AMLM-hard vs fixed WWM at 100M was BLiMP +2.73 and GlobalPIQA +4.84, but Supplement -5.2, EWoK -2.81, Entity -0.66, Reading -0.59, weighted fast proxy -0.124. It should not be promoted as ordinary token difficulty masking.
- GPT-BERT evidence: the paper reports that a low causal/MNTP component (around 1:15) can improve bidirectional BabyLM scores, and GPT-BERT adds attention gating, layer weighting, batch and mask scheduling. But previous local MNTP auxiliary and leader-bundle imitations show that copying one piece naively is unsafe.
- DeBERTaV3 / ELECTRA evidence: RTD is explicitly framed as more sample-efficient than MLM and supplies binary supervision over every token; GDES was introduced to avoid generator/discriminator embedding-gradient interference. But pure RTD discriminator is not directly official likelihood-compatible under the current BabyLM evaluator, which supports mlm/causal/mntp backends. Any RTD route must retain or distill into an MLM/MNTP likelihood head for official-compatible scoring.

## Candidate whole-system hypotheses

### H1. Official-compatible dense discriminative pretraining: small legal generator + DeBERTa MLM head with RTD auxiliary

Scientific idea: under a 10M-word, 10-epoch word budget, plain WWM gives sparse supervised signals and can fit token identity without making local implausibility detection necessary. An ELECTRA-like auxiliary uses in-budget generated corruptions to provide all-token contrast: every context position asks whether the observed token is locally coherent. If attached to the existing DeBERTa-v2 MLM model with a retained MLM head, it could regularize representations toward broad contextual compatibility while preserving official MLM pseudo-likelihood scoring. The compact-view substrate would provide aligned second views; RTD would make the model judge local contextual replacements across both original and compact contexts.

Why this is genuinely different: it changes the learning signal from sparse reconstruction to dense plausibility discrimination. It is not pair-local and not an official-column target. It attacks broad reusable competence through local distributional compatibility and negative evidence.

Official-compatibility requirement: final submitted model must still expose an MLM or MNTP likelihood head accepted by the official pipeline. Therefore the first version should train standard WWM-MLM plus an RTD auxiliary head on the same encoder, not submit a pure discriminator. Corruptions must be generated without external uncounted text or hidden teacher logits. Lowest-risk corruptions for an initial test are in-batch same-tokenizer substitutions drawn from model-free unigram/frequency/POS bins or from a tiny generator trained only on the legal 10M pool. A learned generator increases engineering burden; model-free replacements are less ELECTRA-like but sufficient to test whether dense plausibility gradients complement WWM.

Lowest-cost reliable test before 100M:
1. CPU/GPU-smoke a collator that preserves baseline MLM tensors, adds replacement labels for all non-special tokens, records replacement rate, same-word-group preservation, and exact word exposure unchanged.
2. Frozen-batch gradient test on spatial repair route status legal chck_80M or init: MLM gradient vs RTD auxiliary gradient cosine/norm by layer, and whether RTD gradients are not dominated by embeddings/head. This is cheaper than training and can rule out severe conflict or trivial shortcutting.
3. If gradient and shortcut checks pass, run a 10M or 20M screen on the legal compact-view reinvest corpus, not 100M. Compare to an existing same-horizon spatial repair route status checkpoint if the schedule is matched or run a paired baseline if not. Read cheap columns plus an internal corruption generalization probe on held-out legal corpus replacements. Continue only if gains are broad, not just GlobalPIQA, and Supplement/Reading are not damaged.

Main risks: model-free replacements may be too easy and create surface anomaly detection; learned generator doubles computation and may conflict with official model interface; previous MNTP auxiliary warns that dense auxiliary losses can damage acquisition dynamics even when locally coherent.

### H2. Compact-view-compatible knowledge consolidation by weight averaging / EMA over the existing legal trajectory

Scientific idea: the compact-view substrate produces late-emerging broad gains, but the legal endpoint may be a narrow point in weight space with some competence redistributed by late updates. Competence might be more stable in a temporal ensemble over the mature trajectory. Unlike tail rephase, this is a post-training consolidation operation using existing checkpoints; unlike earlier simple 90/100 averaging, it should be evaluated as a systematic mature-window soup/EMA over spatial repair route status legal checkpoints with official-compatible MLM head and no new pretraining.

Why this is genuinely different: it changes knowledge consolidation rather than data, tokenizer, or objective. It is cheap and can test whether mature representations are fragile across nearby checkpoints.

Prior constraint: existing ladder checkpoint route state old-tokenizer averaging did not beat raw90, and late stopping/averaging was closed for EWoK repair. But BabySteps reports tail averaging improved SuperGLUE by about 1.6, and our legal endpoint gap includes SuperGLUE/Supplement/EWoK/Reading. This route is only worth a cheap posthoc check, not a main H100 training line.

Lowest-cost reliable test:
1. Build 3-4 legal spatial repair route status mature-window soups from existing checkpoints: 80-100 every 5M/10M, 90-100, EMA-like exponentially weighted 70-100, and perhaps best-loss-free uniform 70-100. Use only existing checkpoint weights; save full HF models with the same tokenizer.
2. Evaluate cheap7 first for all soups and, if any exceeds spatial repair route status legal100 by >= +0.35 cheap7 without losing Supplement/Reading, run SuperGLUE on that soup. No new training required. Stop if cheap7 movement is small or column-redistributive.

Main risks: prior old-tokenizer averaging was small; averaging can blur specialized MLM calibration; cannot by itself create the +0.542 Overall gap unless SuperGLUE also rises.

### H3. Minimal representation/inductive-bias change: gated residual/GEGLU attention-FFN block compatible with DeBERTa-v2

Scientific idea: the current 8x480 DeBERTa-v2 has enough parameters but may not have the right small-data computation. GPT-BERT reports attention gating and layer-output weighting as architecture-level modifications, with attention gating relatively cheap and not dependent on data pairs. A gate on attention/FFN outputs could allow the model to allocate limited updates between lexical reconstruction and relation/event features, possibly reducing destructive interference under compact-view reinvestment.

Why different: changes inductive bias/capacity use rather than data or objective; not derived from any official weak column. It could complement compact views by making multi-view relation signals usable rather than just more numerous.

Prior constraint: 12x384 depth alone closed; leader shape/curriculum/40k/LAMB bundle imitations failed; adding parameters without a mechanism is not justified. Therefore this route must be a minimal single architectural intervention, not another shape/depth run.

Lowest-cost reliable test:
1. Inspect Transformers DeBERTaV2 modules and implement a tiny fork adding a learnable scalar/vector gate to attention output and/or FFN output initialized near identity, without changing tokenizer/data/exposure and with parameter count recorded.
2. CPU/GPU smoke for save/load as AutoModelForMaskedLM and loss match near baseline at initialization when gate is identity.
3. 10M or 20M legal compact-view reinvest screen with matched schedule and cheap columns. Continue only if broad columns improve over same-horizon baseline without the familiar GPIQA-only or BLiMP-only redistribution.

Main risks: architecture screens are weak predictors at short horizons; changes can silently break official loading; GPT-BERT gating evidence is from a different stack and coupled to objective/schedule.

### H4. Tokenization-robust character/subword side channel constrained to preserve baseline 16k interface

Scientific idea: the legal-tokenizer loss is not explained by simple fragmentation, but the remaining gap appeared when moving from inherited broad tokenizer to legal in-budget tokenizer. Instead of changing vocabulary size, add a character/subtoken composition side channel (e.g., token n-hot substring projection, CharCNN-like residual) to recover productive morphology and rare-form sharing while keeping the exact legal 16k tokenizer and MLM head. This targets representation robustness rather than vocabulary expansion.

Prior evidence: AMLM paper's n-hot raised adjective nominalization but hurt BLiMP and some tasks; local global word-mean/minfreq results show lexical-interface changes can damage syntax/EWoK. Therefore this is scientifically plausible but high-risk, and should not be next unless an implementation can be tiny and an internal probe shows it helps held-out morphology without harming baseline logits.

Lowest-cost reliable test:
1. Build static substring features from the legal tokenizer vocabulary only; no external data. Add a zero-initialized residual projection to token embeddings so baseline path is preserved at init.
2. Run held-out morphology/segmentation probes and a 5M-10M screen before any 100M. Stop if BLiMP/Supplement/Reading move down or if gains are confined to morphology-like internal probes.

Main risks: known n-hot broad damage; parameter overhead; gains may not affect official Overall.

## Current route judgment before independent_review

H1 is the strongest candidate for a new whole-learning-system route because it changes the sample-efficiency of the learning signal over all tokens and has direct DeBERTaV3/ELECTRA prior support, while preserving the compact-view substrate and official MLM interface if implemented as an auxiliary. It needs careful cheap checks for shortcutting and gradient conflict before any H100 training.

H2 is the cheapest worthwhile immediate check because it uses existing legal checkpoints and could capture mature knowledge consolidation with almost no training cost. It is unlikely to be enough alone but can be run while H1 is constructed, if tools/time permit in a later Execute step.

H3 is a serious fallback if H1 fails mechanical checks or early training, because compact views may require a more interference-resistant computation. It should be implemented only as a single identity-initialized gating change.

H4 is lower priority because prior n-hot/lexical-credit evidence warns of broad damage, but it remains a representation path distinct from vocabulary-size changes.

No new 100M training is authorized from this note. The next expensive work, if any, should be a small discriminating H1 or H3 screen after CPU/smoke evidence, while H2 can be evaluated from existing checkpoints without training.
