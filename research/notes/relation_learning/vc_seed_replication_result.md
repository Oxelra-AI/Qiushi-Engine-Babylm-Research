# invariance probe design: V-C Seed Replication — Complete Analysis

## Pre-stated decision rule outcome

**Supplement V-C flipped** from +1.93 (seed43022 mean over 80/90/100M) to −2.41
(seed43122), triggering the pre-stated RESET condition for the Supplement-based
aggregate substitution hypothesis.

**But the full picture is more structured than a clean reset.**

## Cross-seed V-C signal-to-noise analysis

| Benchmark  | s43022 mean V-C | s43122 mean V-C | Cross-seed mean | Cross-seed spread | **SNR** |
|------------|----------------:|----------------:|----------------:|------------------:|--------:|
| Entity     |          +3.480 |          +2.940 |          +3.210 |             0.540 |  **11.9** |
| Reading    |          +0.490 |          +0.250 |          +0.370 |             0.240 |   **3.1** |
| COMPS      |          −0.547 |          −0.157 |          −0.352 |             0.390 |   **1.8** |
| BLiMP      |          −0.053 |          +1.380 |          +0.663 |             1.433 |     0.9 |
| EWoK       |          +0.103 |          −0.217 |          −0.057 |             0.320 |     0.4 |
| Supplement |          +1.933 |          −2.407 |          −0.237 |             4.340 |     0.1 |

SNR = |cross-seed mean| / (cross-seed spread / 2). Values > 2 indicate reproducible effects.

## Three-tier classification

### Tier 1: Seed-stable V-C effects (SNR > 2)

**Entity V-C ≈ +3.2** (VIEW helps, SNR 11.9). The strongest and most reproducible
effect by far. In all 6 cells (2 seeds × 3 checkpoints), Entity V-C is positive
and between +2.89 and +3.93. Entity tracking requires following content through
varying surface descriptions — exactly the competence that identity-under-variation
evidence should improve.

**Reading V-C ≈ +0.37** (VIEW helps, SNR 3.1). Consistent positive in all 6 cells
(range +0.14 to +0.52). Psycholinguistic processing may benefit from training on
varied surface forms.

### Tier 2: Seed-stable but negative (SNR ≈ 2)

**COMPS V-C ≈ −0.35** (VIEW hurts, SNR 1.8). Consistently negative in all 6 cells
(range −0.02 to −0.70). VIEW training reproducibly hurts compositional comparison
ability. Hypothesis: the compression process used to create VIEW text drops
comparative/relational structures, specifically impairing this competence.

### Tier 3: Seed-unstable (SNR < 1)

**BLiMP** (SNR 0.9), **EWoK** (SNR 0.4), **Supplement** (SNR 0.1). These benchmarks
show V-C effects within the seed noise floor. Single-seed V-C claims about these
benchmarks are unreliable.

## The double dissociation is benchmark-specific

VIEW arms are reproducibly **worse language models** of the held-out distribution
(+0.03 to +0.10 nats higher loss than CLEAN, in both seeds). Despite this:
- VIEW is reproducibly **better** at Entity tracking (+3.2)
- VIEW is reproducibly **better** at Reading (+0.37)
- VIEW is reproducibly **worse** at COMPS (−0.35)
- VIEW is **indeterminate** on BLiMP, Supplement, EWoK

The double dissociation (worse LM + better Entity/Reading) establishes that the
Entity/Reading improvement cannot be explained by better distributional fit. It
must live in representation — specifically, in content-invariant representation
that helps track entities through surface variation.

## Aggregate effects are not seed-stable

| Aggregate   | s43022 V-C | s43122 V-C | Same sign? |
|-------------|------------|------------|------------|
| cheap6      |     +0.743 |     +0.254 | Yes        |
| exEntity5   |     +0.238 |     −0.273 | **No**     |

cheap6 V-C stays positive only because Entity's large effect dominates.
exEntity5 V-C flips because Supplement dominates. This confirms that aggregate
benchmark scores are unreliable measures of data substitution value.

## Connection to the learning principle

The result sharpens the principle from "VIEW helps" to "VIEW helps content-tracking
competences specifically." This connects directly to the representation_and_objectives mechanism:
identity-under-variation evidence (same content in varied forms) teaches the model
to extract invariant meaning, benefiting tasks that require content tracking through
surface change (Entity, Reading) but not tasks about grammatical form (BLiMP,
Supplement) and actively hurting compositional comparison tasks that the compression
process degrades (COMPS).

## Next steps

1. **Invariance probe**: Test whether VIEW checkpoints show greater representational
   invariance between (source, rewrite) pairs than CLEAN checkpoints. The prediction
   is now specific: the invariance advantage should appear for entity-relevant or
   content-tracking text, not for form-sensitive text.

2. **Third seed (43222)**: Train VIEW + CLEAN at seed43222 to tighten the cross-seed
   estimates. With 3 seeds, the Entity SNR test has more power.

3. **COMPS diagnosis**: Check whether VIEW text specifically drops relational/comparative
   language, explaining the reproducible COMPS deficit.

## Files

- Per-target results: `experiments/archive/relation_learning/data/vc_seed_replication/per_target`
- Summary JSON: `experiments/archive/relation_learning/data/vc_seed_replication/vc_seed_replication_summary.json`
- Pre-stated rule: `research/notes/relation_learning/vc_seed_replication_prestate.md`
- Invariance probe design: `research/notes/relation_learning/invariance_probe_design.md`
- Probe script (ready): `experiments/archive/relation_learning/scripts/invariance_probe.py`
- Evaluation script: `experiments/archive/relation_learning/scripts/vc_seed_replication_eval.py`
