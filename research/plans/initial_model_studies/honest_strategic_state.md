# Evidence limits and candidate mechanisms

## The Overall gap is small; the column structure is the real story

Protected 8×480 baseline16k WWM 100M (our complete internal coordinate): Overall 40.527.
Public leader `go76dof/wwm_curriculum_simplification_40k`: Overall 41.80. Gap ~1.17.

Per-column (ours vs leader):

| Column | Ours | Leader | Delta (leader-ours) |
|---|---:|---:|---:|
| BLiMP | 66.76 | 67.20 | +0.44 |
| Supplement | 59.88 | 56.04 | **-3.84 (we win)** |
| EWoK | 52.19 | 56.07 | +3.88 |
| Entity | 22.62 | 28.45 | +5.83 |
| COMPS | 52.19 | 53.57 | +1.38 |
| GlobalPIQA | 35.64 | 39.67 | +4.03 |
| SuperGLUE | 68.02 | 69.79 | +1.77 |
| Reading | 7.62 | 5.42 | **-2.20 (we win)** |
| AoA | -0.174 | 0.0 | small |

Leader wins the relational/knowledge/understanding cluster (EWoK, Entity, GlobalPIQA, SuperGLUE, COMPS). We win Supplement and Reading. BLiMP near tie.

## What has been definitively tested and closed

Model-side (leader-shape isolation, related experiments, all at real exposure):
- **S1 (12×384 leader shape, our tokenizer/data, AdamW, flat WWM), 100M**: BLiMP 66.84, Supplement 60.31, EWoK 52.02, **Entity 20.24**, COMPS 52.26, GlobalPIQA 37.605, Reading 7.25. Leader shape RAISES GlobalPIQA (+~2) but does NOT fix Entity (worse than our 8×480's 22.62) or EWoK.
- **S2 (S1 + length/mask curriculum), 100M**: harms the target cluster (Entity 18.47, BLiMP 64.24).
- **S3 (S1 + legal 40k tokenizer), 10M**: shifts PIQA, damages Reading badly, no Entity/EWoK fix.
- **LAMB**: worse pretraining loss than AdamW at 10M, does not move the cluster.
- **official40k on 8×480**: Reading collapse 7.62→0.89, no Entity fix.

Objective/interface-side (all closed): AMLM, RTD+MLM, CPC-MLM, strict prefix-continuation, factorized cadence (length/mask). Each moves ~1 fast column and fails matched specificity or damages EWoK.

Representation/architecture-side (all closed): WESS (gold works, unlabeled fails), CLBH (copy not binding, clbh binding repair result), morphology/surface adapters, vocab-size variants.

Data-side (closed): custom corpus mix, relation-explicit FineWeb (EWoK +3.6 at 1M, collapses by 3M), paired-restatement adjacency (true_pair worse than controls under matched no-truncation), RecGPT full recipe on official data (Overall 394→39.4).

## The honest pattern

Every route has been a **single mechanism moving a single column ~1 point** on the fixed 8×480 backbone, then failing a matched control or damaging a protected column. The leader achieves its cluster through a **coherent full-system recipe** (deeper shape + 40k + LAMB + curriculum + FineWeb simplification pairs) whose data component is access-restricted and whose legal reconstruction (paired-restatement) did not reproduce the gain.

## Strategic options now

**Option A — Consolidate our own strengths into a net-Overall gain.**
We win Supplement (+3.84) and Reading (+2.20) = +6.04 summed on two columns. The leader wins five columns by +16.9 summed. But Overall is a 9-column equal-weight mean. Instead of chasing the leader's best columns, ask: is there a configuration that KEEPS our Supplement/Reading advantage while recovering part of EWoK/GlobalPIQA? S1 (12×384) already gives GlobalPIQA +1.97 and keeps Supplement +0.43 and Reading ~-0.37 vs our 8×480. S1's Overall was never computed as a full 9-column coordinate. **This is a real gap in our evidence: we do not have S1's Overall.**

**Option B — Multi-seed endpoint + ensemble/robustness of the existing best.**
Single-seed fast-EWoK is noisy. The true internal best coordinate may be higher than 40.53 with proper seed selection and endpoint choice, but cadence all arms interpretation showed chck_80M is not better than chck_100M on 6 columns. Limited headroom.

**Option C — A genuinely new mechanism targeting relation representation** (motivated by the CLBH binding-repair result). But the last 8 mechanisms failed, so the bar for a NEW mechanism must be: it changes relation representation before output, passes the binding-switch test, and beats a param-matched control at 1M with two seeds.

## The most information-rich under-tested fact

S1 (leader-shape 12×384) at 100M raised GlobalPIQA to 37.605 and kept Supplement 60.31 (better than leader's 56.04) and Reading 7.25 (better than leader's 5.42). If S1's full 9-column Overall is competitive, then **combining our winning columns (Supplement/Reading) with leader-shape's GlobalPIQA gain** may already beat 40.53 — without any new mechanism. This has never been computed. It is the cheapest high-value next measurement.

## Decision

Before building another mechanism, compute the full 9-column Overall for the existing S1 (12×384) 100M checkpoint if it still exists, and compare to the protected 40.53. This tests whether we already have an un-scored configuration near or above our current best. In parallel, define the bar for any new relation-representation mechanism using the clbh binding repair result binding-switch test.
