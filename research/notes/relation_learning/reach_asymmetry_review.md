# Frequency-concentration synthesis: reach asymmetry and curve correction

Created: 2026-09-06T13:05Z

This note reviews causal gpt relation boundary and the new frequency concentration probe checks against the active goal: derive a general data-efficient learning principle from the BabyLM SOTA history. The scientific center is no longer merely that relation type matters. The stronger current statement is a **reach asymmetry**: local exact recurrence installs a liability that travels across registers and objectives when later targets require changed form, while corresponding restatement installs a positive competence whose reach is more tied to the practiced register/format and to whether the target token itself recurs in the source.

## Files produced or inspected in this step

New frequency concentration probe evidence:

- `experiments/archive/relation_learning/scripts/replicate_paired_context_aln_off_seed43122.py`
- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_probe`
- `research/notes/relation_learning/paired_context_aln_off_seed43122_probe.md`
- `experiments/archive/relation_learning/scripts/paired_context_aln_off_seed43122_ordinary_heldout.py`
- `experiments/archive/relation_learning/data/paired_context_aln_off_seed43122_ordinary_heldout`
- `experiments/archive/relation_learning/scripts/score_half_view_curve.py`
- `experiments/archive/relation_learning/data/half_view_curve_probe`
- `research/notes/relation_learning/half_view_curve_probe.md`
- `experiments/archive/relation_learning/scripts/half_view_curve_ordinary_heldout.py`
- `experiments/archive/relation_learning/data/half_view_curve_ordinary_heldout`
- `experiments/archive/relation_learning/scripts/frequency_concentration_probe.py`
- `experiments/archive/relation_learning/data/frequency_concentration_probe`
- `research/notes/relation_learning/frequency_concentration_probe.md`

Load-bearing causal gpt relation boundary evidence inspected:

- `research/notes/relation_learning/paired_context_relation_design_probe.md`
- `experiments/archive/relation_learning/data/paired_context_relation_design_probe`
- `research/notes/relation_learning/causal_gpt_relation_boundary.md`
- `experiments/archive/relation_learning/data/causal_gpt_relation_boundary`
- `research/notes/relation_learning/paired_context_ordinary_heldout_loss.md`
- `experiments/archive/relation_learning/data/paired_context_ordinary_heldout_loss`
- `research/notes/relation_learning/model_tiedness_probe.md`

## 1. COMPACT_EXPERIENCE ALN--OFF crossing replicated at seed43122

The reach-asymmetry finding survives a second seed using forward passes only. The available second seed has only `qwen_clean_aligned` and `official_lengthmatched`, so SHUF/DUP/SEP remain seed43022-only, but ALN--OFF itself can now be treated as a two-seed result.

| comparison | seed43022 | seed43122 | reading |
|---|---:|---:|---|
| COMPACT_EXPERIENCE ALN--OFF, Wikipedia overlap Δ(N−T) | +3.5975 ± 0.1236 | +4.1998 ± 0.1379 | large near-register source-recurring source use |
| COMPACT_EXPERIENCE ALN--OFF, Wikipedia nonoverlap Δ(N−T) | −0.5934 ± 0.0502 | −0.8806 ± 0.0528 | source-absent substitutions are not helped and are lower than OFF |
| COMPACT_EXPERIENCE ALN--OFF, compact nonoverlap ΔA_T | +0.0592 | −0.1931 | compact source-absent readout does not receive the same competence |
| COMPACT_EXPERIENCE ALN--OFF, ordinary held-out Δloss | −0.0125 ± 0.0022 | −0.0128 ± 0.0023 | not explained by broad language-fit damage |

This is a double dissociation with the designed compact VIEW intervention. Designed VIEW--CLEAN is large on the matching FineWeb-compact nonoverlap readout (seed43022 ΔA_T +0.7104 here; three-seed mean +0.7970 ± 0.0758) but small on Wikipedia overlap (three-seed +0.2376 ± 0.0706). Inherited Qwen aligned restatement is the mirror: huge on the near-register Wikipedia source-recurring readout, near zero or negative on compact nonoverlap. The compact probe is sensitive in the same COMPACT_EXPERIENCE run because DUP--OFF is −1.9263 and ALN--DUP is +1.9855 on compact nonoverlap. Thus ALN's compact near-null is a real reach boundary, not an insensitive instrument.

The correct generalization is not "restatement always transfers." It is: **correspondence-based source use is installed with the register, format, and target relation practiced during finite-budget learning.**

## 2. Exact recurrence has broader negative reach on changed-form/source-absent targets

The cost side is less confined than the competence side:

- Designed DeBERTa REPEAT--CLEAN compact nonoverlap: three-seed ΔA_T = −0.8138 ± 0.1729.
- COMPACT_EXPERIENCE DUP--OFF compact nonoverlap: ΔA_T = −1.9263 at seed43022.
- COMPACT_EXPERIENCE DUP--OFF Wikipedia nonoverlap: Δ(N−T) = −1.1840 ± 0.0659 at seed43022.
- Designed causal GPT R--C compact nonoverlap: ΔA_T = −1.1532, with R--RS ΔA_T = −1.2599 under a left-context next-token readout.

Exact recurrence is still beneficial when target form recurs or state is unchanged: COMPACT_EXPERIENCE DUP--OFF natural-copy gain is +1.7497, Wikipedia overlap is +3.3422, and Entity rel_eq0 is +15.51 points. The important scientific point is the reach asymmetry: the exact-recurrence liability for changed-form prediction travels from compact MLM to Wikipedia substitutions and to causal next-token learning, whereas the positive restatement competence remains more format/register-shaped.

## 3. SHUF supports practiced non-correspondence only in the matching register

The seed43022 `qwen_shuffled_control` keeps local rewrite-register adjacency and the original/rewrite multisets while breaking source--own-rewrite correspondence. Its preregistered non-correspondence pattern is strong in the near-register Wikipedia source-recurring readout:

- Wikipedia overlap SHUF--OFF Δ(N−T) = −0.9942 ± 0.0682.
- Wikipedia overlap SHUF--OFF Δ(U-related-to-N) = +0.1302 ± 0.0355.
- Ordinary held-out SHUF--OFF Δloss = +0.0130 ± 0.0020.

The same pattern is not present in compact nonoverlap: SHUF--OFF ΔA_T = −0.0845, ΔA_U = −0.1144, ΔG = +0.0298. Therefore practiced wrong-restatement/non-correspondence is also reach-limited; it discounts source-recurring near-register evidence, but should not be generalized to all compact changed-form contexts.

## 4. SEP should not carry causal weight; no immediate Qwen split retraining is needed

The existing `qwen_separated_pair` arm is informative as lower-locality coexistence but remains unsuitable as the decisive SOTA-ingredient locality comparison:

- It has known row-count/packing differences relative to aligned/official arms.
- It is broadly worse on ordinary held-out MLM: SEP--OFF Δloss = +0.1384 ± 0.0027, far larger than SHUF--OFF +0.0130 and ALN--OFF −0.0125.
- Its source-use readouts are much lower than ALN, but those differences mix relation locality with broad degraded fit.

The current evidence does not warrant spending new H100 time on exact row-length-preserving Qwen split training. SHUF carries the correspondence test with matched packing and near-identical broad fit, while the designed compact split controls already establish within-window locality at two DeBERTa seeds. A new Qwen split would be useful only if a later paper version needs a separate SOTA-block-locality panel beyond the correspondence result.

## 5. HALF_VIEW corrects the hash-mix interpretation

The HALF_VIEW training task completed and was scored. It overturns the earlier expectation that half-dose VIEW might already match full VIEW on the compact source-use readout.

| role/contrast | compact nonoverlap A_T or ΔA_T | compact nonoverlap G or ΔG | Wikipedia overlap Δ(N−T) | Wikipedia nonoverlap Δ(N−T) | natural-copy Δgain |
|---|---:|---:|---:|---:|---:|
| C absolute | +1.5267 | +1.2213 | reference | reference | +3.8990 |
| HV absolute | +1.5444 | +1.3896 | -- | -- | +3.8776 |
| V absolute | +2.2371 | +1.9428 | -- | -- | +4.3037 |
| HM absolute | +1.8723 | +1.4871 | -- | -- | +4.4247 |
| HV--C | +0.0177 | +0.1683 | +0.0324 ± 0.0484 | −0.0407 ± 0.0339 | −0.0215 |
| V--C | +0.7104 | +0.7214 | +0.2978 ± 0.0576 | +0.0485 ± 0.0356 | +0.4047 |
| HM--C | +0.3456 | +0.2658 | +0.3243 ± 0.0536 | −0.2136 ± 0.0372 | +0.5257 |
| HM--HV | +0.3279 | +0.0975 | +0.2918 ± 0.0467 | −0.1728 ± 0.0332 | +0.5471 |

This demands a repair of the frequency concentration probe half-view note's first reading: HM is **above**, not below, HV on compact nonoverlap A_T. HALF_VIEW is near C on true-source advantage, while full VIEW is large. That pattern is more compatible with a threshold/superlinear dose response or with the particular half assignment missing a necessary density than with a concave saturation curve. Hash-mix is not evidence that exact recurrence suppressed the half-dose restatement signal relative to HV; instead, exact recurrence added copy behavior and moved source-recurring/compact A_T upward while introducing the expected source-absent Wikipedia cost. HM remains below full VIEW on compact nonoverlap A_T, so it still shows that the off-axis exact/rewrite mixture does not reproduce full restatement competence.

The ordinary held-out broad-fit companion helps keep the interpretation narrow: HV--C Δloss = +0.0706, less than V--C +0.0994, HM--C +0.0928, and R--C +0.1074. These broad-fit shifts are not enough to explain the T/U/N source-use pattern by themselves.

## 6. Tied-softmax and frequency concentration

causal gpt relation boundary established that representative DeBERTa, RoBERTa, and GPT coordinates all have tied input embeddings and output readout weights. A tied lexical-row path is possible, so this step tested a simple row-frequency explanation for the COMPACT_EXPERIENCE ALN--OFF Wikipedia effect by counting target-token frequencies in the 10M OFF and ALN pools.

The result does not support simple frequency concentration:

| target class | two-seed mean ALN--OFF Δ(N−T) | mean corr with log OFF frequency | mean corr with ALN--OFF log-frequency change |
|---|---:|---:|---:|
| Wikipedia overlap | +3.8986 | −0.0127 | +0.0116 |
| Wikipedia nonoverlap | −0.7370 | −0.0752 | +0.0043 |

The large source-recurring effect remains large across OFF-frequency quartiles, and correlations with marginal token-frequency change are near zero. This does not identify the internal computation, but it weakens the alternative that the ALN source-recurring result is merely low-frequency lexical-row exposure under tied embeddings.

## 7. Current paper-level scientific statement

The strongest present principle is:

**Finite-budget pretraining converts local sequence relations into source-conditioned routines with asymmetric reach. Exact recurrence trains reliable exact/unchanged retrieval but installs a changed-form liability that travels across register and even across MLM-to-causal objectives. Corresponding restatement trains useful source use when the deployment target matches the practiced relation--especially source-recurring content in the practiced or nearby register--but this competence does not automatically transfer to source-absent substitutions or to a different compact register. Wrong local restatement can install a matching-register non-correspondence routine rather than acting as inert adjacency.**

This is a stronger data-efficient learning principle than scalar data value: the same finite data item is not valuable in isolation; its value is shaped by the relation it practices inside the learner's context, the target form later required, and the objective/learner coordinate.

## 8. Immediate next scientific work

1. Patch the argument map and any research-facing figure/table notes so they use the reach-asymmetry statement and the corrected HALF_VIEW interpretation. In particular, remove the old axis that placed inherited Qwen as a point on the compact designed-register dose curve; for compact nonoverlap the designed-dose curve is C → HV → V, with HM off-axis.
2. Treat ALN--OFF as two-seed, but label SHUF, DUP, and SEP as seed43022-only.
3. Do not launch Qwen split training now; use SHUF for correspondence and designed splits for locality.
4. Send functional_learning the tiedness and frequency-concentration results, emphasizing that tied rows are possible but simple lexical-frequency concentration is not supported.
5. Keep RoBERTa N-anchor scoring and deeper contextual-source-recognition probes as useful boundaries, not prerequisites for expressing the central reach-asymmetry result.
