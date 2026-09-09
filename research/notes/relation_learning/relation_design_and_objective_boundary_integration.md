# causal gpt relation boundary integration: SOTA-ingredient relation design and causal-objective boundary

This note integrates the new causal gpt relation boundary execution results. It is research-facing: it records what the computations establish about the active goal of deriving a general data-efficient learning principle from the BabyLM SOTA work.

## What was run and checked

- Completed causal-GPT training. The three 8×480 GPT2LMHeadModel arms C/R/RS trained successfully for 100M legal charged words at seed43022. Final training losses were C 3.5366, R 3.5829, RS 3.6955; manifests are under `experiments/archive/relation_learning/training/runs/causal_gpt_{c,r,rs}_dose2p64x_seed43022/training_manifest.json`.
- HALF_VIEW was still training at this stage, so no HALF_VIEW scientific result is inferred here.
- Added the pre-number branch that `SHUF` might practice non-correspondence/discounting rather than act as inert adjacency to `notes/paired_context_relation_design_prestate.md` before the successful scoring run.
- Repaired `scripts/score_paired_context_relation_design.py`: removed the duplicate `global CKPTS`, added a mixed-schema scorer for compact/copy records and Wikipedia records, and fixed inherited short-arm copy aggregation. The successful run scored 487,500 rows across five inherited Qwen arms × three checkpoints and wrote `notes/paired_context_relation_design_probe.md` plus CSVs under `data/paired_context_relation_design_probe/`.
- Added ordinary held-out loss for the same five inherited Qwen arms with `scripts/paired_context_ordinary_heldout_loss.py`, writing `notes/paired_context_ordinary_heldout_loss.md` and `data/paired_context_ordinary_heldout_loss/`.
- Wrote and ran `scripts/score_causal_gpt_relation_boundary.py`, scoring 292,284 causal-GPT rows with a left-context next-token readout. Outputs are in `data/causal_gpt_relation_boundary/` and `notes/causal_gpt_relation_boundary.md`.

## COMPACT_EXPERIENCE relation design: own restatement, wrong restatement, exact duplication, separated coexistence, and no selected Qwen-pair practice

The five-arm COMPACT_EXPERIENCE design now turns the inherited SOTA ingredient into direct mechanism evidence rather than background analogy.

### Own aligned restatement is strongly source-recurring and correspondence-dependent

`ALN` (local original + own Qwen rewrite) greatly increases source-recurring extraction in the near-register Wikipedia/Simple-English probe:

- Wikipedia overlap `ALNminusOFF` Δ(T vs N) = +3.5975 ± 0.1236.
- Wikipedia overlap `ALNminusSHUF` = +4.5917 ± 0.1339.
- Wikipedia overlap `ALNminusSEP` = +6.1434 ± 0.1486.

The same direction appears in compact overlap targets:

- compact overlap `ALNminusOFF` ΔA_T = +1.6489.
- compact overlap `ALNminusSHUF` ΔA_T = +2.3514.
- compact overlap `ALNminusSEP` ΔA_T = +4.1179.

This is not just broader model quality: ordinary held-out `ALNminusOFF` is slightly better, Δloss = -0.0125 ± 0.0022, and `ALNminusSHUF` is only -0.0255 ± 0.0024. The correspondence between source and rewrite is therefore doing real work beyond same-window rewrite-register adjacency.

### Source-absent substitutions remain a boundary of restatement practice

The same aligned restatement arm does not solve source-absent substitution. In Wikipedia nonoverlap targets, `ALNminusOFF` Δ(T vs N) = -0.5934 ± 0.0502, while `ALNminusDUP` = +0.5906 ± 0.0687. In compact nonoverlap, `ALNminusOFF` ΔA_T is only +0.0592, but `ALNminusDUP` is +1.9855. Thus own restatement mainly installs form-robust use of content that recurs under changed sentence form; it does not generally invent or prefer target tokens absent from the source.

### Wrong local restatement is not inert adjacency

The `SHUF` arm keeps original and rewrite multisets and local rewrite-register adjacency but breaks own-source correspondence. It is below OFF for source-recurring use while ordinary held-out degradation is tiny:

- Wikipedia overlap `SHUFminusOFF` Δ(T vs N) = -0.9942 ± 0.0682, with Δ(U vs N) = +0.1302 ± 0.0355.
- compact overlap `SHUFminusOFF` ΔA_T = -0.7024 and ΔA_U = -0.0709.
- ordinary held-out `SHUFminusOFF` Δloss = +0.0130 ± 0.0020.

This supports the pre-stated non-correspondence branch for source-recurring targets: wrong restatement can practice a discounting relation between adjacent source-like spans rather than acting as a harmless adjacency control. The effect is target-form specific, because Wikipedia nonoverlap `SHUFminusOFF` is +0.1765 ± 0.0298 and compact nonoverlap is only -0.0845 on A_T. The general principle should therefore include practiced mismatch as a third relation, but only with the target-class boundary preserved.

### Exact duplication reappears as exact/unchanged benefit plus changed-form cost

`DUP` is a local selected-original + selected-original exact-recurrence analogue. It reproduces the qualitative recurrence dissociation on the SOTA ingredient axis:

- held-out natural-copy `DUPminusOFF` Δgain = +1.7497.
- Entity `rel_eq0` `DUPminusOFF` = +15.51 accuracy points.
- compact nonoverlap `DUPminusOFF` ΔA_T = -1.9263.
- Wikipedia nonoverlap `DUPminusOFF` Δ(T vs N) = -1.1840 ± 0.0659.
- Entity `rel_ge3` `DUPminusOFF` = -2.26 accuracy points.

For source-recurring Wikipedia overlap, `DUPminusOFF` is +3.3422 ± 0.1395, close to aligned restatement's +3.5975 ± 0.1236, showing that exact recurrence is not simply bad: it is powerful when the target form recurs and costly when the later target must differ.

### Separated coexistence is informative but not a clean row-matched locality analogue

`SEP` is much lower than `ALN` on source-use readouts but also worse on ordinary held-out MLM:

- Wikipedia overlap `SEPminusOFF` = -2.5459 ± 0.0915.
- compact overlap `SEPminusOFF` ΔA_T = -2.4690.
- ordinary held-out `SEPminusOFF` Δloss = +0.1384 ± 0.0027.

This says that separated source/rewrite coexistence did not install the same adjacent-source use as local aligned restatement in the available COMPACT_EXPERIENCE arm. Because `SEP` has a known row-count/packing mismatch and broad loss, it should remain a lower-locality evidence point, not the exact analogue of the matched DeBERTa `VIEW_SPLIT` design. If the final argument needs a clean SOTA-ingredient locality statement, an exact row-length-preserving Qwen split remains scientifically justified.

## Causal GPT objective boundary

The causal-GPT readout uses only left context, so absolute levels cannot be compared to MLM T/U/N. The signs are nevertheless important.

Exact recurrence still produces a changed-form source-conditioned cost and a local-vs-split contrast under the causal objective:

- compact nonoverlap `RminusC` ΔA_T = -1.1532, ΔA_U = +0.0806, ΔG = -1.2338.
- compact nonoverlap `RSminusC` ΔA_T = +0.1068, ΔA_U = +0.1784, ΔG = -0.0716.
- compact nonoverlap `RminusRS` ΔA_T = -1.2599, ΔA_U = -0.0978, ΔG = -1.1621.

The exact-copy routine also survives the causal objective:

- held-out natural-copy `RminusC` Δgain = +1.7855.
- `RSminusC` = +0.2517.
- `RminusRS` = +1.5338.

Natural restatement remains target-form dependent under the causal readout: Wikipedia nonoverlap `RminusC` is -0.5355 ± 0.0351 and `RminusRS` is -0.3561 ± 0.0329, while Wikipedia overlap `RminusC` is +0.0416 ± 0.0500 and `RminusRS` is +0.2705 ± 0.0484. Thus the causal objective supports the recurrence-cost locality mechanism for changed-form/substitution targets and exact-copy gain, but not a uniform recurrence disadvantage over all natural target forms.

## Current scientific update

The strongest current principle is no longer a two-way contrast between exact recurrence and own restatement. Under finite data and fixed training budget, local sequence relations install relation-specific source-conditioned routines:

1. exact recurrence promotes exact or unchanged retrieval but can actively mis-handle changed-form targets;
2. own restatement promotes source use when source content recurs under changed sentence form, especially near the practiced register;
3. wrong restatement can practice non-correspondence/discounting for source-recurring targets, separating correspondence from mere adjacency.

The mechanism remains bounded by learner coordinate, objective/deployment format, register, target form, and row packing. The new causal-GPT result makes the exact-recurrence part less MLM-specific, while the COMPACT_EXPERIENCE five-arm result makes the SOTA ingredient itself part of the evidence rather than a narrative bridge.

Important open work after this step:

- collect HALF_VIEW when delivered and place it on the restatement-dose curve; do not infer its result from runtime status;
- determine whether the inherited Qwen `SEP` residual requires exact row-length-preserving Qwen split training;
- score RoBERTa with the neutral `N` anchor;
- probe whether source-recurring effects concentrate in low-frequency lexical rows or reflect contextual source recognition independent of token-row exposure, and record tied embedding/readout structure across DeBERTa/RoBERTa/GPT coordinates for the tied-softmax synthetic bridge.
