# lead frozen anchor fastpath route Lead synthesis — end of selector program, frozen-anchor fast-path replay route

## Decisions grounded in evidence

### 1. The selector program is closed (three independent failures)
- earlier analysis corpus target-token NLL: selects 100M on scale1.75/U256/spatial repair route status; never 82M.
- earlier analysis structural contrast-margin v1: composite Pearson vs cheap7 -0.29, peaks at 77M.
- crossed sign probe design and smoke crossed-sign interaction: composite Pearson vs cheap7 -0.44, peaks at 77M; entity_state cs=0.0
  and temporal cs~0.04 are flat across the whole late window, spatial_put at ceiling.
- partial deberta grid and endpoint branch micro-world v3 independently: template/tokenizer-sensitive, wrong sign vs full EWoK,
  D-state failure is real but too narrow, and coupled sparse20 is not rewarded.
- **Conclusion:** No label-free corpus quantity found so far tracks the 82M official peak. The synthetic
  flat state/temporal failures are NOT valid training targets (they do not move across the late window,
  so they cannot explain why 82M wins). Stop building selectors and stop training on those probes.

### 2. The scientific object stands: continued legal-corpus learning REALLOCATES competence
- 82M -> 100M: MLM NLL keeps improving while EWoK relations, high-op Entity, GlobalPIQA, SuperGLUE decline
  (chck82 reproducibility and peak characterization/145). This is competence-allocation-over-time, not global forgetting and not scalar convergence.

### 3. Endpoint state
- Protected public rank-1: reproduced scale1.75 chck_82M, Overall 41.942481167385985, submitted,
  challenge-counted, persistent rank 1 (41.94). This remains the winner.
- Truthful shuffled86 carrier materialized from EXISTING outputs only:
  `experiments/archive/frontier_consolidation/data/truthful_shuffled86_carrier/all_full_preds_truthful_shuffled86_mlm.json`
  SHA256 `8c9885efa84727c4997c4d6d492b8cc8ae78ec74775ac94698ea059bce636764`, validator PASS,
  candidate-native Overall(AoA0) 41.98991359885548, +0.047432 vs chck_82M.
  - AoA scalar 0 only, `fast_eval_results` OMITTED (no borrowed 82M history). Truthful.
  - Model SHA `7a090773...083bd6c`, deterministic-replay-reproducible.
  - This is a BASELINE / null, NOT a mechanism: its +0.047 is SuperGLUE seed variance + a non-significant
    GlobalPIQA flip surplus (14v10, p=0.541). Aligned lost to shuffled and had worse source-free NLL.
    Do not promote it as science and do not submit it as a "better result" on this evidence.

## Next route: frozen-82M slow anchor + reversible fast path on BROAD natural legal replay

The proposed design treats chck_82M as an immutable slow function and permits late learning only through a sparse reversible fast path trained on coherent natural legal replay, not auxiliary rewrites, synthetic probes or official labels. The scientific claim under test is:

> Under limited data/capacity, late learning can be made more sample-efficient by preserving an earlier
> broad-competence basin as an immutable function and admitting subsequent information through sparse or
> validated residual computation, so that acquisition does not overwrite the fragile relation/state basin.

### Existing infrastructure that already fits (no new trainer needed for the core arms)
- `frozen82_private_tail_trainer.py` freezes all non-private tensors, attaches zero-output private
  residual adapters, trains private-only, and supports `--mode {aligned,shuffled,neutral_only}`.
- `neutral_only` mode = pure main-stream MLM replay through the frozen anchor with private-only gradients +
  deterministic neutrality KL. This is exactly the "coherent broad replay" arm with NO auxiliary rewrites.
- Exact boundary: initial 82,012,495 words, skip_rows 530,944, cap remaining 17,987,505.
- Verified: private-OFF recovers protected chck_82M logits exactly (earlier analysis), trusted class
  `FrozenSlowPrivateDebertaV2ForMaskedLM`, native non-trust load is the WRONG function.

### Minimum decisive experiment (verifier-designed, matched controls)
Fix one frozen 82M backbone, same ~1M-param private architecture, optimizer, masking budget, schedule,
trusted-load path. Match total charged exposure to the shuffled86 baseline (86,005,413 words) so results
are directly comparable, and checkpoint at ~84M/85M/86M to separate transient turnover from accumulation.

- Arm 0 Anchor: frozen chck_82M (reference, already scored).
- Arm 1 Shuffled86: existing materialized model (empirical null for private-tail redistribution).
- Arm 2 Coherent broad replay: `neutral_only` mode, main MLM on the coherent legal suffix, private-only.
- Arm 3 Content-matched structure-destroyed replay: same token/mask/step budget with within-row token
  order destroyed (isolates coherent-context contribution beyond frequency/private adaptation).
- Arm 4 Coherent replay + explicit anchor retention: Arm 2 + frozen-teacher KL on a SEPARATE legal replay
  stream, retention coefficient fixed from legal held-out divergence (NOT official scores).

Key contrasts: coherent-structure effect = A2 - A3; retention effect = A4 - A2; shuffled86 = observed noise.

### Predeclared measurement (fixed BEFORE scoring)
- Paired official item transitions vs 82M per task, especially declining families: EWoK physical/material/
  interaction/spatial, Entity high-op (regular_5, move_contents_4/3, ambiref_3), GlobalPIQA, SuperGLUE;
  BLiMP/COMPS gains reported separately so they cannot mask relational losses.
- >=3 SuperGLUE finetune seeds per final arm; paired-item bootstrap for zero-shot; Jaccard of gained/lost
  item sets across two fast-path seeds (reproducible item identity, not just repeated aggregate).
- Independent legal held-out: teacher-student KL/JS, top-choice agreement, coherent-vs-shuffled context
  discrimination fixed identically across arms; MLM NLL only secondary.
- Fast-path mechanism: residual-logit norm vs slow logits, fraction of slow-top-choice flips, frequency
  concentration of residual means, low-margin vs high-confidence behavior; private-OFF exact equality.

### Continue / stop / modify
- CONTINUE only if enabled model retains nearly all anchor-correct items in the DECLINING EWoK/Entity
  families while adding correct decisions elsewhere (a reproducible Pareto pattern), coherent replay beats
  the structure-destroyed control, retention reduces lost-correct without killing acquisition, and Overall
  exceeds shuffled86 beyond seed uncertainty.
- STOP as private-tail redistribution if coherent and destroyed replay give similar family turnover, or if
  gains hover ~+0.05 Overall with unstable item identity.
- A GENERAL principle additionally needs the same family-level pattern on >=1 independent trajectory; we do
  not yet have late-window official ground truth there, so that is a later, separate expense.

## Immediate low-cost first action (before the full 5-arm panel)
Run Arm 2 (`neutral_only` coherent replay, 4M charged, matched to shuffled86) and Arm 3 (structure-destroyed
matched control) as the first pair, since Arm 2 needs no new trainer and Arm 1/0 are already measured. This
is the cheapest test that can already reject the route: if A2 ~ A3 ~ shuffled86 on cheap7 and family
transitions, the frozen-anchor coherent-replay idea is just private-tail redistribution and we stop before
building Arm 4 retention machinery.
