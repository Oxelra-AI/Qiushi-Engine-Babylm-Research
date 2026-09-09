# earlier analysis — Route reset after the 4M paired BSM result

## Why the route must change

The 4M paired BSM trace is strong negative mechanism evidence for the current synthetic-binding route.

Evidence:

- Training completed for all four arms: `data/4m_training_status.json`.
- Evaluation is saved in `data/4m_trace_eval.json` and `notes/4m_trace_eval.md`.
- Coherent, swapped, and token-matched official reference all stayed at essentially zero continuous switch margin and 0.000 pair-level both-correct across available checkpoints, including exact 4M.
- Targeted losses fell, so the model learned something about the rows, but not cross-row entity→value consistency.
- At 4M, coherent−swapped gives EWoK +3.09 but Entity −0.04 and GlobalPIQA mean −1.975. Coherent−token-matched-reference gives EWoK +3.55 but Entity −0.15 and GlobalPIQA mean −4.93.

This means the current BSM recipe does not create the intended reusable binding mechanism and does not move the BabyLM Strict-Small SOTA target in the needed joint direction. Longer exposure of the same recipe is not justified by the evidence.

## Overall arithmetic correction

The leaderboard Overall is not a simple mean of the nine visible columns. From the public row:

\[
NLP=\frac{\text{BLiMP}+\text{Supplement}+\text{EWoK}+\text{Entity}+\text{COMPS}+\text{GlobalPIQA}+\text{SuperGLUE}}{7},
\]

\[
HumanLike=\frac{\text{Reading}+\text{AoA}}{2},
\]

\[
Overall=\frac{3\,NLP+HumanLike}{4}.
\]

Therefore each NLP-column point is worth \(3/28\approx0.1071\) Overall, and each Reading/AoA point is worth \(1/8=0.125\) Overall.

The previous shorthand that the gap is concentrated only in Entity/EWoK/GlobalPIQA was incomplete because it ignored our positive margins on Supplement and Reading.

Current best internal coordinate (`data/current_best_internal_coordinate.json`): Overall 40.7028. Parsed public leader: about 41.80. Gap about 1.10.

Approximate leader−internal column differences and Overall contributions:

| column | leader−internal | Overall contribution |
|---|---:|---:|
| BLiMP | +0.66 | +0.071 |
| Supplement | -4.99 | -0.535 |
| EWoK | +5.63 | +0.603 |
| Entity | +6.25 | +0.670 |
| COMPS | +0.57 | +0.061 |
| GlobalPIQA | +2.08 | +0.222 |
| SuperGLUE | +1.535 | +0.164 |
| Reading | -1.88 | -0.235 |

Entity is the largest single visible deficit, but closing it entirely would still not close the full Overall gap. It is also the least tractable route in our evidence so far.

## What the leaderboard rows imply

The top rows show that high Entity is not necessary for 41+ Overall:

- Rank 2 RecGPT-10M has Entity 16.59, below our internal best, but reaches Overall 41.53 through very high BLiMP 73.11, Supplement 61.73, and COMPS 55.43.
- Rank 3 Wordpiece-24-4 has Entity 21.82, close to ours, but gets BLiMP 70.20 and Supplement 66.52.
- The top row name `wwm_curriculum_simplification_40k` points to a data/curriculum/tokenizer recipe, not a dedicated entity-state architecture.

So the next route should optimize weighted multi-column competence, especially BLiMP/Supplement/COMPS/EWoK while protecting GlobalPIQA and Reading, rather than treating Entity-binding as the central mechanism by default.

## Important boundary: do not simply reopen official40k as a standalone factor

A strict 1M official-corpus 40k-tokenizer WWM test already exists:

- Evidence: `data/wwm_official40k_1m_profile.json` and `notes/wwm_official40k_1m_profile.md`.
- Mean official40k−16k deltas at 1M were harmful: BLiMP −0.475, Supplement −3.20, Entity −0.06, COMPS −0.405, Reading eye −8.315, Reading self-paced −3.56, with only EWoK +1.775.
- This rejects official40k alone as a direct replacement under that early BERT/WWM setting.

Later DeBERTa official40k smoke and launch notes also indicate that 40k is an intervention package: segmentation plus larger embeddings and different target density, not a pure tokenizer effect. The next route must not be “try 40k again because the leader has 40k.”

The useful lesson is different: strong public rows use representation/data/curriculum/tokenization *together* to raise core competence columns. The route should test such combinations with strict controls, not single-factor 40k repetition.

## Current best next scientific direction

### Core-competence data/representation route

The next line should ask:

**Can we construct a legal ≤10M-word experience structure that improves BLiMP, Supplement, COMPS, and EWoK together while preserving GlobalPIQA and Reading?**

This is broader than entity-binding. It should inherit what worked:

- WWM remains the stable objective base.
- DeBERTa-v2 8×480 remains the protected backbone.
- The internal model already has strong Supplement and Reading relative to the top row; those must be protected.
- Single-seed fast EWoK is noisy; any route using EWoK must also monitor GlobalPIQA, Supplement, and Reading.

It should avoid what failed:

- Synthetic binding density without explicit relation-credit assignment.
- Pair adjacency/restatement as a standalone lever.
- Static relation-density selection.
- Official40k as a standalone tokenizer replacement.
- Simple RTD/CPC/continuation objectives that did not create true prefix specificity.

### Candidate mechanisms worth constructing next

1. **Weighted core-competence corpus selection from existing legal text pools.**
   Instead of selecting relation-heavy text, select text that jointly improves grammar/semantic plausibility and preserves reading. A practical screen can compare official text, leader-like simplified/high-clarity text, and mixed curricula by the weighted Overall proxy:
   \[
   \Delta Overall_{proxy}=\frac{3}{28}\Delta\sum_{7\,NLP}+\frac{1}{8}\Delta Reading.
   \]
   This must evaluate BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading together, not only EWoK/Entity.

2. **Curriculum/simplification as an experience-quality mechanism, not adjacent restatement.**
   Prior paired-restatement experiments tested adjacency of fact restatements and closed. They did not fully test a leader-like replacement curriculum in which the whole 10M experience distribution is simpler, clearer, and higher signal-to-noise. A new construction should use a same-word-budget replacement design with random-source controls and preserve the exact weighted score accounting.

3. **Tokenizer only as an interaction term.**
   Because official40k alone hurt early scores, any future 40k test should be embedded in a 2×2 with data/curriculum and possibly a non-embedding-capacity control. It is not the immediate single-factor test.

4. **Credit-assignment objectives only if they directly optimize relation consistency.**
   If the route returns to binding, it must not be another density recipe. It would need a direct coherent-vs-swapped contrastive or dual-query loss and must first show nonzero switch margin on held-out surfaces. This is lower priority than the core-competence route because current evidence shows poor tractability and bad GlobalPIQA tradeoff.

## Immediate next work

The proposed core-competence redirection requires comparison with the accumulated evidence and competing mechanisms before another training run.

Files to read:

- `notes/4m_trace_eval.md`
- `data/4m_trace_eval.json`
- `data/current_best_internal_coordinate.json`
- `notes/strict_small_leaderboard_anchor.md`
- `data/public_leader_available_coordinate.json`
- `data/wwm_official40k_1m_profile.json`
- 
- this file: `plans/route_reset_after_bsm_4m.md`

Conditional on that comparison, the proposed construction is a compact experiment design for a **weighted core-competence corpus/curriculum screen**. It should specify legal data sources, exact word accounting, arms, controls, target columns, weighted proxy formula, and the smallest 1M/4M screen that can detect a real multi-column signal without repeating closed entity-binding routes.
