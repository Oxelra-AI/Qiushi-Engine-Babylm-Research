# Mechanism-level diagnosis and proposed relational training

Historical status: This note records an early causal hypothesis and a proposed research design. The strong causal interpretations below were hypotheses at the time, not validated mechanisms or general impossibility results. The proposed RDGA interventions and conditional follow-ups are not reported experimental outcomes.

## Evaluation setting and research question

### 1. Interpretation of the scoring rules

**The interpretation used here is:**
- Use 19 checkpoints (the specified subset from chck_1M to chck_100M) to identify a **single training endpoint with the best Overall score**.
- Compute 8 capability scores (BLiMP, Supplement, EWoK, Entity, COMPS, SuperGLUE, GlobalPIQA, Reading) using **that same final checkpoint**.
- AoA separately uses the full learning trajectory (multiple checkpoints).
- The equally weighted mean of the 9 scores = Overall.

**Implications for evaluation:**
- Do not cherry-pick a different best checkpoint for each metric.
- Identify one checkpoint that performs as well as possible across all 8 metrics together.
- The existing protected reference evaluated only chck_100M; the trajectory data show that WWM seed42's 7-column mean at 80M (42.43) is actually higher than at 100M (42.41).
- If SuperGLUE at 80M is also competitive, 80M may be a better endpoint. This was an immediate, untested opportunity.

### 2. Why SOTA had not been reached — a historical mechanism hypothesis

**Proposed root cause, inferred at the time from the negative results:**

The hypothesis was that the loss minimum of bidirectional MLM (WWM) **does not require** broad relational representations. Late in training, the model was hypothesized to discover and converge to a narrow local-statistical solution, causing systematic decay of the Entity/EWoK/GlobalPIQA capabilities formed mid-training.

The evidence and its historical interpretation were:

| Experiment | Finding | Historical mechanistic interpretation (hypothesis) |
|---|---|---|
| AMLM 40M peak (two seeds) | Broad mid-training capability improvement +0.993 | The model **has the capacity** to form broad representations |
| AMLM 100M decay | Systematic decay over 80-100M to mean -0.089 | Late MLM gradients no longer require broad representations → representational narrowing |
| WESS (related experiments) | Persistent state can be learned with labeled addresses → transfer without labels fails | External modules cannot transfer through the MLM interface |
| R1/SCMLM-R (related experiments) | Frozen readout both-correct = 0; unfrozen memory does not generalize | The DeBERTa encoder does not form sequential-state variables under MLM |
| RecGPT on official (earlier analysis) | Overall 34.27 | The causal/recursive interface alone does not change the representation; the data distribution carries the effect |
| Structured-WWM | EWoK/BLiMP ↑ but Entity unchanged, Supplement ↓ | Semantic diversity helps world knowledge but not entity persistence |
| Custom corpus | EWoK +1.15, losses on several other metrics | Data reorganization under plain WWM does not solve the central problem |
| Leader decomposition | Individual components (shape/curriculum/LAMB/40k) ineffective in isolation | The leader's advantage emerges from multi-component interactions or data construction |

**Combined historical mechanism hypothesis:**

1. **Representational narrowing as the central decay mechanism**: As training converges, local statistics increasingly suffice to minimize MLM loss → gradients no longer maintain broad representational dimensions → Entity/EWoK/GlobalPIQA capabilities deteriorate.
2. **Local predictability as the driver of narrowing**: Most masked tokens were hypothesized to be predictable from the neighboring ±5 tokens → learning local patterns suffices → maintaining long-range entity/relation representations is unnecessary.
3. **The then-current leader might make local statistics insufficient through data construction**: The combination of simplified text + curriculum + 40k tokenizer might require broad representations throughout training to minimize loss.

### 3. Proposed insight from the negative results

**Central historical hypothesis:** The issue is not "insufficient model capacity" but "an objective that does not require the capability at convergence."

The AMLM 40M peak was interpreted as demonstrating sufficient DeBERTa-v2 capacity to form Entity/EWoK/GlobalPIQA representations. The proposed problem was not inability to learn, but **forgetting after learning**, because subsequent gradients point toward a simpler solution. This causal explanation remained a hypothesis.

The early interpretation went further, claiming that any approach based on "injecting external knowledge" or "changing the data distribution" was bound to fail, for the following reasons. These were historical generalizations from bounded experiments, not validated impossibility claims:
- Injecting external knowledge through the MLM interface → WESS was interpreted as showing infeasibility.
- Changing the data under the same MLM objective → custom-corpus/structured-WWM was interpreted as showing insufficiency.
- Changing the masking schedule → AMLM was interpreted as showing a transient effect.

**Proposed requirement: Make the loss minimum itself require broad representations.**

### 4. Exact arithmetic of the SOTA gap

Protected reference (chck_100M): Overall 40.527
Public leader: Overall 41.801
Gap: 1.274 (total gap across 9 metrics: 11.47 points)

Main deficits: Entity -5.83, GlobalPIQA -4.03, EWoK -3.88, SuperGLUE -1.77, COMPS -1.38
Main surpluses: Supplement +3.84, Reading +2.20

**How much could perfect preservation of the AMLM 40M advantage contribute?**
Two-seed mean 40M delta: Entity +1.6, EWoK +0.24, GlobalPIQA +1.21, BLiMP +1.31, Supplement +2.01, COMPS +0.27, Reading +0.32
Total: +6.96 / 9 = +0.77 to Overall → reaches ~41.30, still 0.50 below the leader

**Conclusion: Preserving the AMLM mid-training advantage alone is insufficient for SOTA. An additional ~4.5 points of improvement is needed.**

---

## Proposed mechanism: Relational Dependency Gradient Amplification (RDGA)

### Scientific rationale

The hypothesis is that identifying masked tokens whose correct prediction **requires long-range context** ("relationally dependent tokens") and amplifying their gradient contributions would make MLM gradients **continually** require broad relational representations.

Proposed distinctions from the closed routes, as hypothesized at the time:

| Closed route | Historical explanation of failure | Hypothesized RDGA distinction |
|---|---|---|
| AMLM | Amplifies "globally difficult" tokens → unrelated to relational dependency → effect decays as the model improves | Amplifies "locally unpredictable but globally predictable" tokens → hypothesized to persist without decay |
| WESS | External labeled-state module → cannot transfer to unlabeled text | No external module; only gradient weights change |
| R1/SCMLM-R | Synthetic data + ranking objective → DeBERTa does not learn | No synthetic data; relational dependency is measured directly on official text |
| Structured-WWM | Changes the data distribution → helps EWoK but not Entity | Keeps the data unchanged; changes each token's gradient-contribution weight |

### Operational definition of relational dependency

For each masked position i:
- L_full(i) = cross-entropy loss with the full context
- L_local(i) = cross-entropy loss with only the ±k local window (implemented through an attention mask)
- Relational dependency RD(i) = L_local(i) - L_full(i)

The proposed interpretation of high RD was: the token is locally unpredictable but globally predictable → its prediction **requires long-range relational information**. The identification of context dependence with relational information was part of the hypothesis.

Entity mentions, state verbs, causal outcomes, and anaphoric pronouns were hypothesized to have naturally high RD.

### Training loss

L = Σ_i w(i) × CE(predicted_i, target_i)

where w(i) = softmax(RD(i) / τ) over all masked positions in the batch

τ = temperature, controlling amplification strength
- τ → ∞: uniform weights = standard WWM
- τ → 0: only highest-RD token gets gradient

### Implementation options

**Version A: Online dual pass (exact, but 2× compute)**
1. For each batch, run a full-context forward pass → obtain L_full per token.
2. Run a local-context forward pass (attention mask ±k) → obtain L_local per token.
3. Compute RD = L_local - L_full and use RD to weight the full-context loss.
4. Backpropagate the weighted loss.

**Version B: Periodic offline measurement + static weight map (practical version)**
1. Every 10M words, run a full-vs-local comparison on the official corpus.
2. Build a token-position-level importance map.
3. Look up weights from the map during training.
4. Update every 10M.

**Version C: Linguistic-prior approximation (zero additional compute)**
- Predefine high-RD categories: pronouns, demonstratives, state verbs (is/was/has/had), causal connectives (so/because/therefore), temporal markers (then/before/after/now), entity re-mentions (definite NPs: the X).
- Multiply mask probability for these categories by 2.5; leave other categories unchanged.
- This was proposed as equivalent to amplifying gradients at these relational positions.

### First-round test design

**Scale: 50M words (expected to be sufficient to observe the decay pattern)**

5 arms, all on protected DeBERTa-v2 8×480, baseline16k, official data, batch 256, seq 256:

| Arm | Description | Purpose |
|---|---|---|
| 1 | Flat WWM 0→50M | Control baseline |
| 2 | AMLM 0→50M | Replicates known decay |
| 3 | WWM + RDGA-C (linguistic prior) 0→50M | Tests whether relational masking bias helps |
| 4 | AMLM 0→20M → RDGA-B (periodic) 20→50M | Tests anchored RDGA after AMLM builds broad repr |
| 5 | RDGA-A (online dual-pass) 0→50M | Tests exact RDGA from scratch |

Evaluation checkpoints: 10M, 20M, 30M, 40M, 50M (5 points per arm = 25 evaluations)
Evaluation columns: BLiMP, Supplement, Entity, EWoK, COMPS, GlobalPIQA, Reading (7-col)

### Success criteria

1. **Persistence**: By 50M, the Entity/EWoK/GlobalPIQA advantages of Arm 3/4/5 do not decay (versus the decay in Arm 2).
2. **Compatibility**: Supplement and Reading losses ≤ 0.5 relative to Arm 1.
3. **Separation**: RDGA arms show stronger relational representations than Arm 1/2 on held-out probes.
4. **Scale**: The best-checkpoint 7-col mean of any RDGA arm > that of Arm 1's best checkpoint.

### If the first round succeeds

- Scale to 100M + full 9-column official evaluation.
- Combine with tokenizer exploration (40k on the same protected geometry) to test whether the gains stack.
- If Overall > 41.80 → the proposed SOTA submission criterion is met.

### If the first round fails

Analyze the RD distribution. If high-RD tokens are genuinely amplified but Entity still does not improve, the historical proposed interpretation was:
→ The problem lies in DeBERTa encoder capacity rather than gradient allocation (possibly requiring a deeper model or a different attention pattern).
→ Pivot to dual-interface or intermediate-state supervision.

This was a proposed diagnostic interpretation, not a validated exclusion of other causes.

---

## Parallel analysis: best-checkpoint search

Under the scoring interpretation above, the historical proposal described a **zero-cost** check of whether existing trajectories contain a better Overall endpoint than chck_100M. It reuses trained checkpoints; the SuperGLUE evaluation below is still required.

Required: Evaluate SuperGLUE for WWM seed42's chck_80M (7-column scores already exist) to determine whether the 9-column Overall exceeds 40.527. If SuperGLUE ≥ 68.02 at 80M, then 80M may be a better endpoint.

Similarly, AMLM seed42's chck_80M has a 7-col mean of 42.45 (higher than the protected 100M value of 42.41). If its SuperGLUE performance is competitive, this may already represent a small but real Overall improvement.

---

## Evidence files

Evidence:
- `data/paired_wwm_amlm_trajectory_summary.json`
- `notes/paired_wwm_amlm_trajectory_summary.md`
- `data/existing_runs_exposure_trajectory.json`
- `plans/route_reconstruction_after_amlm_100m.md`
- All closed route evidence (related experiments)
