# earlier analysis uniform-reference experiment and route reassessment

## Why this step exists

The current BabyLM Strict-Small goal is still to exceed the visible 41.8 Overall leader under the current official constraints, with a real, legal, reproducible sample-efficient learning principle. The strongest complete clean result remains clean-Qwen seed43022 `chck_100M` at Overall `41.34429066479573`, below the visible leader.

mask endpoint full eval summary-48 created a serious but incomplete continuation story:

- clean-tail/uniform restart from clean-Qwen `chck_80M` is byte-identical and improved no-AoA equal7 at `chck_90M` to about `43.434`, a real within-seed late optimization/rephasing signal;
- inverse-priority masking improved no-AoA further, with endpoint-matched true-100M inverse minus uniform `+0.4936` equal7 and `+0.4875` equal6 without GlobalPIQA;
- evidence-visible masking improved no-AoA in a different profile, but mostly through GlobalPIQA at the true endpoint;
- full nine-column evaluation then showed both non-uniform allocations fail official-style Overall because AoA becomes strongly negative:
  - inverse_priority `chck_95M`: Overall `39.961218980736284`, AoA raw `-0.1719406337063024`;
  - inverse_priority `chck_100M`: Overall `39.714154598526164`, AoA raw `-0.18344781331440352`;
  - evidence_visible `chck_90M`: Overall `40.034892698105594`, AoA raw `-0.15456019530556697`;
  - evidence_visible `chck_100M`: Overall `40.08313318762824`, AoA raw `-0.14666542859729043`.

The missing separator is therefore the byte-identical uniform restart full evaluation. It is not a routine reference: it determines whether the negative AoA is a common 80M restart/rephasing/tail-continuation effect or mainly produced by target redistribution. Full evaluation had started for `chck_90M` and true `chck_100M`; completed measurements were pending in this record.

## What the uniform result can establish

Use only aggregate official-style AoA values (`aoa_raw_correlation`, `aoa_leaderboard_score`, `aoa_status`) and complete Overall arithmetic. Do not inspect or use AoA/CDI words, per-word curves, or checkpoint acquisition curves for new training design.

### Case A: uniform AoA is near clean-Qwen's AoA behavior

If uniform restart keeps AoA near `0.0` while inverse/evidence-visible are strongly negative, then the common restart is not the main source of the acquisition penalty. The no-AoA gains remain scientifically meaningful but target redistribution must be made more conservative or temporally localized. The best next construction should not repeat the failed full-tail high-dose target allocation. A better design would keep the clean restart/uniform acquisition behavior and insert a legally defined, training-statistics-only redistribution in a smaller late window or with a milder mixture:

- limited duration: e.g. apply non-uniform target allocation for only part of the 80M->100M tail, then return to uniform before final checkpoints, without using AoA to select the switch;
- smaller amplitude: interpolate target scores toward uniform rather than hard inverse/evidence allocation;
- cleaner mechanism: decouple selection RNG from corruption RNG and separate low-priority identity from target length/count using `scripts/decoupled_mask_control_trainer.py` and its length-matched random control;
- fixed endpoint policy: true `chck_100M` remains the submission-relevant target, and any frozen earlier endpoint is only a scientific measurement.

This branch preserves the continuation principle but reconstructs it around acquisition-safe consolidation, not maximizing no-AoA proxy columns.

### Case B: uniform restart also has strongly negative AoA

If uniform restart also becomes strongly negative, then the negative AoA is probably a common consequence of the late 80M restart/rephasing/tail family. In that case, do not spend seed43122 or decoupled-control H100 time on the current continuation route as a SOTA path. The route's useful contribution is a mechanistic clue: late optimization can raise broad frozen no-AoA columns while damaging acquisition-fit. The active goal then needs a different load-bearing route rather than another local tail variant.

The next route should be rebuilt at a more fundamental layer: objective/architecture/corpus construction from scratch, while retaining only verified lessons:

- same-window generated second views are legal and useful beyond exact duplication/separated coexistence, but clean-Qwen saturates at 41.344 and needs a stronger backbone or objective;
- simple global developmental order, cap-length geometry, high-dose contextual pairs, natural shared-anchor clusters, pair-agreement training, and simple SWA should remain closed unless a new mechanism changes the premise;
- lower training loss is not a reliable selector; complete official-compatible measurement decides candidates.

## Source-grounded route families if Case B holds

The Knowledge read in earlier analysis suggests three serious route families, not final choices:

1. **Leader-compatible DeBERTa MLM refinement.** `Mask and You Shall Receive` (`\cite{edman2025maskc}`) used DeBERTa-v2, 40k BPE, LAMB, sequence length 64->256, decaying mask ratio 40%->15%, and adaptive MLM/n-hot variants. Its public table reports a 41.9 final score for n-hot hard, but with uneven task tradeoffs: n-hot greatly helps adjective nominalization/AoA-like morphology while hurting BLiMP and other zero-shot columns; hard decay improves SuperGLUE/Entity. The earlier experiments found the 40k/LAMB/stagewise recipe weaker than the 16k DeBERTa route and found simple hard/evidence target allocation unsafe at 100M, so a direct copy is not enough. The scientific opportunity is to combine the clean same-window second-view data with a backbone/objective that better preserves morphology/acquisition and avoids the full-tail AoA penalty.

2. **GPT-BERT/LTG-BERT objective-architecture route.** Official findings (`\cite{charpentier2025findingsa}`) describe GPT-BERT/LTG-BERT as using DeBERTa-style disentangled attention, both pre- and post-layer normalization, span masking, GEGLU, and masked next-token prediction mixed with autoregressive training. The 2025 Strict-Small baseline used about 31M parameters, 12 layers/6 heads, token batch 16384, LR `7e-3`, mask ratio decaying from 0.3 to 0.15, and sequence schedule 128->256->512. This is a genuinely different objective/architecture layer from the current MLM-only 8x480 model. The high-value question is whether clean same-window generated second views plus a GPT-BERT-style mixed objective can improve sample efficiency and acquisition behavior without the DeBERTa-MLM tail penalty.

3. **Recursive/shared-block causal architecture route.** RecGPT-10M's model card reports a 34.17M parameter recursive causal LM for 2026 Strict-Small with a shared Transformer block repeated 16 times, hidden size 768, embedding size 192, FFN 12288, 32,768-token BPE, Muon for recursive block and AdamW for embedding-related parameters, sequence length 256, and no KV-cache generation. It reports strong BLiMP/COMPS/GlobalPIQA but weak Entity and lower SuperGLUE. This is not directly a current frontier for our Overall without AoA/Reading information, but it is a different inductive-bias family: parameter-efficient depth through repeated computation. It may be valuable if subsequent comparisons move beyond DeBERTa MLM.

## Interpretation after uniform-reference evaluation

When the uniform full eval returns:

1. Run or read `scripts/interpret_all_mask_endpoint_full_eval.py` output to compare inverse, evidence-visible, and uniform using aggregate scores only.
2. If uniform preserves AoA while non-uniform damages it, the proposed follow-up is a milder/temporal redistribution experiment with decoupled RNG and length-matched controls, ideally after a cheap no-AoA or small-tail pilot but with the true 100M endpoint as the meaningful target.
3. If uniform also has strongly negative AoA, route back to serious Explore/Construct around a different objective or architecture. The two most scientifically promising next directions are GPT-BERT/LTG-BERT mixed objective with clean-Qwen same-window corpus, and a leader-compatible morphology/acquisition-preserving DeBERTa variant; RecGPT-like recurrence is an architecture alternative if implementation bandwidth permits.
4. In no case should seed43122 inverse replication be launched as a SOTA replication from the failed inverse/evidence full scores.
