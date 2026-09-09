# lead strategic synthesis Lead Investigator strategic synthesis

## Where the research actually stands (facts, not hope)

### Secured benchmark milestone
- Protected endpoint: reproduced scale1.75 `chck_82M`, **Overall 41.942481167385985**, submitted and
  **live-persistent at displayed rank 1** on the BabyLM 2026 Strict-Small leaderboard (earlier analysis/144).
  Legal, bit-reproducible from corpus, fast-inclusive carrier validated.
- This clears the prior public displayed top (41.80) by +0.14. **The benchmark objective at the ~41.94
  level is achieved and defended.** But the goal is a *generalizable data-efficient learning principle*
  plus a real, robust SOTA, not a single noisy checkpoint pick.

### Fragile +0.04 hypothesis (86M shuffled private tail)
- Repeat SuperGLUE delivered: 69.8192 (orig 69.7662) -> projected Overall **41.98991**,
  delta **+0.0474** vs chck_82M. Lead did not erase; it slightly grew on re-eval.
- BUT: the entire lead is SuperGLUE variance (~+/-0.05 Overall between finetune seeds) + a GlobalPIQA flip
  surplus (14 vs 10 items, p=0.541). This is **noise-level**, not a mechanism. It also lacks a transparent
  official full+fast/AoA carrier for 86M. Keep as fallback numeric hypothesis; do NOT chase it further.

### The real scientific object: 82M->100M vector loss
- The winning scale1.75 surface PEAKS at 82M then declines to 100M (cheap7 43.96 -> 43.54, Overall 41.94 -> 41.57).
- Item-flip localization (chck82 reproducibility and peak characterization): the 82M->100M loss is a **weighted capability reallocation**, not global forgetting:
  - EWoK -0.975 (physical-dynamics, material-properties, social-interactions, physical-interactions, spatial-relations)
  - Entity -0.849 (high-operation state tracking: regular_5_ops, move_contents_4_ops, ambiref_3_ops, move_contents_3_ops)
  - GlobalPIQA -1.471, SuperGLUE -0.432
  - Offset by raw BLiMP/COMPS gains -> net cheap7 falls.
- **Crucial:** legal-corpus MLM NLL keeps IMPROVING to 100M on all trajectories while official competence peaks earlier.
  So scalar/stratified MLM likelihood is blind to the phenomenon (earlier analysis negative result, closed).

## The two candidate routes, judged as Lead

### Route A: build a benchmark-independent CONTRASTIVE-MARGIN signal
- Instead of P(target token | context), measure margin between a coherent corpus example and a *controlled
  minimal perturbation* (entity swap, role/argument swap, polarity flip, relation reversal, temporal reorder).
- This is exactly the structure the official EWoK/Entity/COMPS/GlobalPIQA tasks use (minimal-pair scoring),
  so a corpus-built minimal-pair margin can plausibly track relation/state preservation where token-NLL cannot.
- Fix transformations + summary, then test unchanged on scale1.75 late window (has official cheap7 ground truth)
  AND an independent trajectory. If it tracks the capability peak, it becomes:
    (a) a legal early-stopping / checkpoint-selection principle (immediate SOTA robustness value), AND
    (b) a measurement that can guide a consolidation objective that PREVENTS the 82M->100M loss.
- Value: high. This is the difference between "we happened to pick 82M" and "we understand and can secure the peak."

### Route B: directly attack the 82M->100M loss with a legal training intervention
- E.g. protect the fragile families via a consolidation mechanism during the last ~18M legal words.
- This is not justified before a label-free signal that tracks the movement exists,
  because we will again optimize a compelling local quantity that does not transfer. Every prior training
  intervention (Muon, LAMB, RTD, U256, private tails, dual-view) reproduced this failure mode.

## Scientific Decision
- Do NOT launch another private consolidation intervention yet.
- The measurement is the bottleneck. A signal that tracks the capability movement is the prerequisite for
  BOTH a robust selection principle and a principled intervention. Build Route A first.
- Design constraint: the contrastive margin must be computed ONLY from corpus structure (entity lists, role
  positions, negation/polarity markers, relation verbs, temporal connectives derivable from the legal 10M pool),
  NEVER from official benchmark items/labels. It must be minimal-pair (coherent vs perturbed), scored by
  the model's own pseudo-log-likelihood margin, aggregated by a FIXED summary decided before evaluation.
- Success test (fixed in advance): does mean/median margin correlate with official cheap7 across the 8 late
  scale1.75 checkpoints (must pick or rank-align 82M), then transfer UNCHANGED to a second trajectory?
- If it works: it is a publishable generalizable principle AND directly usable for endpoint security/improvement.
- If it fails: we have ruled out corpus-minimal-pair margin as a retention proxy and must reconsider whether
  the 82M peak is even predictable from any label-free corpus signal (which would itself be an important result
  bounding what "data-efficient consolidation" can be measured without benchmark leakage).

## Independent Ground Truth
Scale1.75 has late-window official cheap7 results, whereas U256 and the baseline have only 100M and partial results. A transfer test requires late-window cheap7 for at least one independent trajectory. Reuse complete results if available; otherwise a proposed minimal comparison covers chck_80/82/100M on U256 or the baseline. This 3-checkpoint evaluation is justified only after the corpus-margin signal is built and rank-aligns on scale1.75.
