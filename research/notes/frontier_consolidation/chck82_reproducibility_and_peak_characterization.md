# chck82 reproducibility and peak characterization — chck_82M reproducibility closure and 82M→100M peak characterization

Two reproducibility questions organize this analysis: (1) whether the score-bearing endpoint reproduces end-to-end from corpus, and (2) whether the 82M endpoint is a stable plateau or a narrow peak, and whether a legal benchmark-independent signal predicts it. Both are examined using existing evidence, with no new training or inference.

## 1. Reproducibility of chck_82M — CLOSED (bit-identical + independent hardened eval)

The independent from-corpus fixed-selection reproduction (`adapter128_scale1p75_from_corpus_fixed82M_seed43022`)
trained the full scale1.75 trajectory from random init on the legal corpus/tokenizer and produced
**bit-identical checkpoints at every ladder point compared**:

| ckpt | original SHA256 | reproduction SHA256 | hash_equal |
|---|---|---|---|
| chck_20M | b37e243b…7b031 | b37e243b…7b031 | true |
| chck_50M | 995c1678…2034 | 995c1678…2034 | true |
| chck_80M | c37f6665…edda | c37f6665…edda | true |
| chck_82M | 93ceb76a…591b3 | 93ceb76a…591b3 | true |
| chck_100M | 73494758…6f52 | 73494758…6f52 | true |

Source: `experiments/archive/representation_and_objectives/data/from_corpus_reproduction_compare/from_corpus_reproduction_compare.json` (status PASS).

Independent hardened re-evaluation (`matched graph transfer probe result`) reproduces the score:
- graph packet v0 construction Overall 41.942254213869 vs matched graph transfer probe result hardened Overall **41.942481167386**, delta **+0.000227**.
- All nine columns match within ±0.006; hardened == collated on every column
  (`chck82_submission_artifact_verification.json`, all `equal: true`).
- Collated prediction counts complete for all official tasks: BLiMP 59875, Supplement 5218,
  EWoK 7618, Entity 6780, COMPS 91028, Reading 1726, GLUE 29577, GlobalPIQA 103+100,
  AoA ladder 19 steps × 8005 rows.

**Conclusion:** the 41.9425 endpoint is now legal-exposure, loadable, repeated-score, and
reproducible from corpus with identical weights. The reproducibility gate is satisfied.
Remaining pre-submission item: the collated prediction file lacks the official `fast_eval_results`
block required for a *full* leaderboard submission (`generalization experiment contract` open_work); this is a packaging
detail, not a score question.

## 2. 82M is a narrow peak, not a stable plateau

Legal per-checkpoint cheap7 (7 cheap columns, fully evaluated per checkpoint) across the mature window
(`scale1p75_checkpoint_sweep`):

| ckpt | cheap7 |
|---|---:|
| 77M | 43.282 |
| 78M | 43.702 |
| 79M | 43.579 |
| 80M | 43.812 |
| 81M | 43.649 |
| **82M** | **43.960** |
| 83M | 43.808 |

- Window mean 43.685, population std 0.201, full range 0.678 across single-M steps.
- 82M stands **+0.231 cheap7 above its immediate-neighbor mean** (81M 43.649, 83M 43.808).
- Trajectory oscillates by ~0.7 cheap7 within 6M words; there is no monotone plateau.

Full-scored (own SuperGLUE + AoA) endpoints in the window:
- **chck_80M Overall 41.77163** (SuperGLUE 69.260, AoA 0.0), cheap7 43.812.
- **chck_82M Overall 41.94248** (SuperGLUE 69.766, AoA 0.0), cheap7 43.959.

So the window contains one clearly-above-neighbor peak (82M, 41.942) and one strong secondary
point (80M, 41.772). Only 82M clears the 41.8 frontier. The peak is real and legally selected
(the 1M-spaced ladder is an intrinsic part of the ≤100M run and was reproduced bit-identically),
but it is a single-checkpoint peak in a noisy trajectory, not a robust plateau.

## 3. No legal benchmark-independent signal currently predicts the peak

- Per-batch MLM training loss in the window is 2.36–2.57 and does NOT track cheap7: the 82M-region
  batch loss (~2.49) is higher than the 79M/80M-region batch loss (~2.36). Consistent with the
  cross-experiment finding that lower MLM loss does not predict official competence.
- The run's `dynamics_traces.jsonl` is present but its per-checkpoint freq-band loss/accuracy and
  prediction-entropy fields are empty (0.0), so no precomputed internal signal exists to correlate.
- Therefore, at present the peak is only identifiable through the (legal) cheap7 ladder itself, not
  through a cheaper benchmark-independent proxy.

## 4. Research implications

- The private-pathway correspondence family (broad dual-view, sparse separated, frozen-82M tail)
  is exhausted as a source-free transfer object: 4M aligned tail true-free NLL is +0.018 *worse*
  than shuffled, and both tails only redistribute competence (shuffled cheap7 +0.053, aligned
  −0.047 vs 82M; both lose Supplement ~3pts). Do not build another variant of this family.
- The score-bearing SOTA candidate (41.9425) is secure and reproducible. The open scientific
  questions that could *raise* it are: (a) whether a legal internal early-stopping/consolidation
  signal (e.g. cheap probe of prediction-entropy or representation stability computed at each
  ladder checkpoint) can identify or slightly exceed the peak; (b) whether a different
  benchmark-independent selection would recover a higher-cheap7 checkpoint that also holds SuperGLUE.
  Any such probe must be cheap, legal, and benchmark-independent, and must not become a new
  expensive training family without a route-changing forecast.

Sources:
- `experiments/archive/frontier_consolidation/data/chck82_independent_verification/chck82_independent_verification.json`
- `experiments/archive/representation_and_objectives/data/from_corpus_reproduction_compare/from_corpus_reproduction_compare.json`
- `research/documents/representation_and_objectives/data/chck82_eval_comparison/chck82_eval_comparison.md`
- `experiments/archive/representation_and_objectives/data/scale1p75_checkpoint_sweep/summary/scale1p75_checkpoint_sweep_summary.json`
- `experiments/archive/representation_and_objectives/data/chck82_submission_artifact_verification/chck82_submission_artifact_verification.json`
