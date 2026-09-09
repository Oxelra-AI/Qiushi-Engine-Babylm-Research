# mixed objective measurement and mechanism plan source-grounded objective-route state

## Why the full mixed-objective measurement matters now

The current active experiment (`mixed_causal50_qwen_seed43022` and `mixed_causal15_qwen_seed43022`) is not just another no-AoA column optimization. It is the first test in this series of changing the learning signal throughout the whole 100M exposure on the established clean-Qwen same-window corpus, after tail restart/mask allocation showed that late continuation can raise selected frozen columns while destroying aggregate acquisition behavior.

The mixed objective measurement and mechanism plan measurement plan therefore evaluates both true-100M arms on the complete nine-column official-style score. The seven-column trajectory is descriptive; it must not decide whether SuperGLUE or AoA are worth measuring. AoA remains only a terminal aggregate readout from the frozen checkpoint ladder.

## What the GPT-BERT source contributes

The paper `GPT or BERT: why not both?` says the key idea is to combine masked and causal language modeling in one stack by aligning outputs as next-token predictions: traditional MLM predicts a masked token at its original position, whereas masked next-token prediction masks position `k+1` and predicts it from position `k` (lines 47-60 of the Knowledge object). The paper argues that hybrid pretraining can combine strengths of CLM and MLM without extra parameters, and it explicitly reports that adding as little as 6.25% MNTP can improve bidirectional performance in its setting (lines 179-188), with final ratio chosen as 1:15 causal-to-masked (line 192).

This supports testing a small causal fraction and explains why the `causal15` arm is scientifically important even if `causal50` is too strong a perturbation. However, the paper's system is not a clean mechanism isolate: it also uses LTG-BERT-derived architecture, attention gating, layer weighting, batch-size scheduling, mask scheduling, and a corpus mixture (lines 64-101, 138-146, 214-225). Its ablation table indicates that mask scheduling, batch scheduling, attention gating, and layer weighting have separable effects, so importing the whole recipe would make our causal inference opaque.

## What BabyLM findings contribute

The first BabyLM findings paper identifies LTG-BERT/ELC-BERT-style architecture as a major successful factor in early Strict/Strict-Small submissions: additional layer normalization, GEGLU, DeBERTa-like disentangled attention, scaled initialization, and ELC-BERT layer mixing (lines 324-326). It also reports that curriculum learning attempts were common but usually did not improve broadly (lines 347-349), while architecture modifications, objectives, preprocessing, and hyperparameter searches were more consistently useful (lines 347-391). The 2025 findings paper similarly says training objectives and architecture modifications remained effective and compute alone was not strongly predictive in Strict-Small (lines 304-308).

This supports moving away from closed global-ordering/tail-continuation routes and toward objective/architecture mechanisms. It does not justify bundling all architecture changes before the objective mechanism has full-regime evidence.

## What causal attention verification actually tests

causal attention verification tests an architecture-matched **causal next-token batch interleaving** package:

- same DeBERTa-v2 8×480 architecture and 16k tokenizer as clean-Qwen;
- same clean-Qwen corpus, ordering, initialization, optimizer, LR schedule, batch size, sequence length, and word exposure;
- MLM batches: normal bidirectional WWM reconstruction at 15%;
- causal batches: uncorrupted input, lower-triangular attention, hidden state at `t` predicts token `t+1`;
- deterministic low-discrepancy objective allocation at causal fractions 0.50 and 0.15.

Implementation checks from causal attention verification showed zero future attention probability and exact bidirectional forward equivalence, so any result will not be due to a future-token leak or forward-path bug.

## What causal attention verification does not yet separate

A score advantage or damage belongs initially to the full package. It does **not** yet separate:

- dense prediction target count from objective complementarity;
- uncorrupted causal inputs from masked MNTP-style inputs;
- fewer MLM updates from positive causal updates;
- changed masking RNG trajectory from objective identity;
- pure causal visibility from prefix/bidirectional-masked next-token designs.

The first separator, if the full package has useful signal, should be a dense-MLM target-density control calibrated from causal attention verification training metrics, not from evaluation scores. Script prepared: `scripts/calibrate_dense_mlm_mask_probability.py`.

## Immediate next actions after causal attention verification training result

1. Verify both arms completed exactly 100M word exposure, saved the AoA-required checkpoint ladder, and realized the intended causal fractions.
2. Launch `scripts/launch_mixed_objective_full_eval_100M.sh` as the decisive complete measurement for both true-100M arms.
3. Use `scripts/score_mixed_objective_full_eval.py` to compare against clean-Qwen Overall `41.34429066479573`, clean-Qwen equal7 `43.112857142857145`, and visible leader `41.8`.
4. Only after the full mixed-package result is known, decide whether to run the dense-MLM separator, revise objective form toward MNTP/prefix-LM, or return to broader route construction.
