# identity address stress replication — Route reassessment after all-suffix CPC diagnostics

## Current evidence that changes the route

The current all-suffix CPC-MLM objective is not scale-ready.

Evidence from cpc profiles and prefix probe:

- Same-source CPC at chck_1M moved some fast columns relative to matched WWM: BLiMP +0.54, EWoK +2.27, Entity +1.10, but Supplement −1.60, COMPS −0.30, Reading −0.26.
- Forced-cross CPC moved BLiMP +0.75 and EWoK +1.54 but Entity −0.70 and COMPS −0.41.
- Both CPC variants ended with the margin still fully active (`margin_active_frac_last = 1.0`) and true-minus-negative log-prob near zero. The objective was not actually satisfied.
- The prefix probe does not support the desired same-source prefix mechanism. Same-source CPC has strongly negative CPC-minus-WWM same-vs-cross specificity: all cases mean −0.658, high-WWM-sensitivity quartile mean −0.839. Forced-cross CPC specificity is essentially zero.

The representative rows show why the same-source CPC task movement is unsafe to scale: for many BNC cases the CPC_same checkpoint is nearly invariant to full/deleted/shuffled/same-source prefixes but changes by order 0.5–2.0 log-prob when the prefix is cross-source. That is a source/register/topic contrast, not a true earlier-state to later-token correspondence. This is the source-confounding failure mode the control was intended to detect.

Important nuance: this does **not** prove BabyLM lacks state-continuation dependencies. The all-suffix CPC loss averages over thousands of suffix masked targets that are locally predictable or insensitive to prefix. The current probe itself found WWM prefix sensitivity to be extremely small for most constructed token targets (mean WWM sensitivity score around 0.0004; high quartile around 0.0013), so broad all-suffix margins mostly train on irrelevant targets.

## Route decision

Do not run seed43, 3M, 10M, or 100M for current all-suffix CPC. Do not repair it by only changing margin, lambda, or source-mode, because the primary problem is target specificity rather than source pairing alone.

A high-precision CPC repair remains scientifically possible but should not be the immediate main route unless we can mine many targets whose gold logit is demonstrably prefix-dependent before training. Previous attempts to mine clean official-text state-counterfactual cases were sparse and polluted (related experiments), so this repair may become another extractor bottleneck.

The stronger next experimental route is therefore the transfer gain per word gate reserve: **dual-interface dense continuation / GPT-BERT-style objective on the protected DeBERTa backbone**.

## Why dual-interface continuation is now the best next route

1. It tests a different causal hypothesis from CPC.
   - CPC failed because hard-negative margins over broad suffix targets were not aligned with prefix-dependent tokens.
   - A continuation objective gives dense gradient over suffix tokens conditioned on prefix, without requiring a mined negative prefix to be semantically incompatible.

2. It is supported by BabyLM field evidence.
   - The Third BabyLM findings state that effective methods were often training-objective or architecture changes.
   - GPT-BERT/LTG-BERT baselines combine masked and autoregressive losses; the report describes GPT-BERT as using both masked and autoregressive language modeling, including a masked next-token prediction interface, and notes GPT-BERT/LTG-BERT remain strong baselines.
   - The report also highlights training cadence and sequence-length choices as high-impact; these can be tested without inventing fragile semantic miners.

3. It keeps the most valuable protected assets.
   - DeBERTa-v2 8×480 WWM is the strongest verified internal backbone.
   - WWM, DeBERTa disentangled attention, and direct-checkpoint evaluation remain protected.
   - The new active factor is a continuation interface, not another data filter or local anomaly classifier.

4. It creates useful evidence even if it fails.
   - If dense continuation improves BLiMP/Supplement but not Entity/EWoK, the bottleneck is not simply sparse MLM supervision.
   - If it improves Entity/EWoK without true-prefix logit specificity, the task gains are likely broad distributional or endpoint effects.
   - If it improves true-prefix specificity and task profile, it becomes a real route for larger validation.

## Proposed next experiment: WWM + prefix continuation objective

### Model and data

- Reuse `babylm_masked_train.py` infrastructure, tokenizer, official corpus selection, DeBERTa-v2 8×480 n_head=8 config, checkpoint saving, and direct-checkpoint evaluation.
- Same seed/control geometry as prior data event binding panel: official 1M, batch 128, lr_total_steps 49, seeds 42/456/789.
- Matched WWM baseline exists: `training/runs/wwm_debertav2_8x480_official_1M_b128_seed42_matched/`.

### Objective

Train with ordinary WWM on the full sequence plus an auxiliary continuation loss over suffix tokens.

A simple first implementation:

- Split each example into prefix and suffix by token position.
- Use an attention mask for the continuation forward such that suffix token position `t` can see prefix tokens and earlier suffix tokens, but not later suffix tokens.
- Use the same MLM decoder/head to predict suffix token `x_t` from the hidden state at a shifted or masked-next-token position.
- Total loss: `L = L_WWM + lambda_cont * L_cont`, with lambda chosen small enough that WWM loss is not dominated.

If causal masking inside DeBERTa is cumbersome, a first proxy can use masked-next-token prediction on suffix positions: replace a suffix token with `[MASK]`, allow prefix + previous suffix context, and predict the next/current token under a controlled mask. The key is dense continuation-conditioned gradient, not a separate classifier.

### Required controls

- Matched WWM baseline (already exists for seed42 1M; rerun only if direct compatibility differs).
- WWM + continuation, seed42 1M.
- A cheap corrupted-prefix continuation control if implementation cost is low: same continuation loss with prefix deleted or block-shuffled for the continuation forward, not for the WWM forward. This distinguishes useful prefix-conditioned continuation from pure local suffix modeling.

### Required probes

Use the same style as identity edge state transport gate, but with improved target selection:

- Fixed later token targets.
- Variants: true prefix, deleted prefix, block-shuffled prefix, same-source-near prefix, cross-source-near prefix.
- Apply variants to WWM and continuation model.
- Report candidate-minus-WWM deltas and same-source specificity.
- Stratify by target class if possible: BNC vs CHILDES, content vs function-ish tokens, high vs low WWM prefix sensitivity.

The route survives only if:

- candidate-minus-WWM true/deleted or true/same-near specificity is positive in MLM logits;
- any fast EWoK/Entity gain is not bought by Supplement/COMPS/Reading damage;
- effects are not only cross-source/register sensitivity.

### 1M decision after run

- If 1M continuation shows broad damage or no prefix specificity: stop this first continuation variant; do not run seed43/3M.
- If it shows small task gains but no specificity: treat as endpoint/regularization effect, not mechanism; consider only if two-seed profile is unusually strong and no core columns are harmed.
- If it shows true-prefix specificity and no core damage: run seed43 and a modest 3M trajectory before any larger budget.

## What not to do next

- Do not keep changing CPC margin/lambda/source mode without mined prefix-sensitive targets.
- Do not build another generic contrastive or RTD head.
- Do not repeat relation-explicit data filtering, paired-restatement adjacency, relation-word mask priority, or downstream content-mask priority.
- Do not launch 100M from any one-seed 1M bump.

## Reserve route if continuation fails

If the continuation route fails cleanly, return to Explore with two alternatives:

1. High-precision CPC/state target mining: build a small, manually/automatically inspected inventory of suffix tokens where true prefix already changes WWM logits over deleted/same-near controls, then train only on those targets.
2. Architecture/cadence pivot: test a GPT-BERT/LTG-BERT-like mixed-objective and sequence-length schedule package or smaller-batch optimization-noise package, because BabyLM findings repeatedly show architecture/objective/cadence effects can rival data changes.
