# earlier analysis — route reconstruction after the 1M XSpan result

## Current position relative to the goal

Highest goal remains BabyLM 2026 Strict-Small Overall SOTA. The current protected model is baseline16k DeBERTa-v2 8×480 WWM at complete local 9-column Overall 40.5269, below the parsed Strict-Small top row `wwm_curriculum_simplification_40k` Overall 41.80. Main deficits remain Entity, EWoK, GlobalPIQA, with smaller deficits on COMPS/SuperGLUE/BLiMP and advantages on Supplement/Reading.

The compact-v4 XSpan discovery remains scientifically useful but the current training route is not worth scaling:

- Protected 100M WWM model showed real s1-specific likelihood on compact XSpan spans: true−wrong +0.717, broad across target types.
- The 1M primary-objective trainer learned the target spans but did not learn correct-s1 binding: true-s1 improves true−wrong only +0.00287 logprob/token over the wrong-s1 control.
- Available task subset at 1M: true-s1 XSpan gives Entity +0.19 vs WWM, exactly matched by wrong-s1; GlobalPIQA −1.95, Supplement −4.34, Reading −0.32. This is target-span familiarity plus guard-column damage, not transferable relation learning.

Therefore the next work must not be more XSpan span filtering, lower-rho hoping, or 10M/100M scaling of the same objective.

## What the literature/source rereading changes

### GPT-BERT and AntLM are now the strongest inherited route family

`GPT or BERT: why not both?` reports that a shared LTG-BERT-style model trained with a hybrid masked/causal objective outperforms pure masked or pure causal systems on BabyLM 2024. Key details worth inheriting rather than copying blindly:

- The winning small model uses a low causal-to-masked ratio; the paper chooses 1:15 for the final model after ablation.
- Adding a small causal component can improve bidirectional performance, while keeping bidirectional evaluation strongest.
- GPT-BERT also uses attention gating, layer weighting, batch-size scheduling, and mask scheduling; the paper notes layer weighting is costly/less clearly useful, attention gating and mask scheduling look better supported.
- Corpus mixture contributed to EWoK in 2024, but 2026 Strict-Small must keep exact exposure accounting and legality.

`AntLM` independently supports the same mechanism family: alternating CLM and MLM phases improves BabyLlama and LTG-BERT baselines, with LTG-BERT 10M results showing strong EWoK and BLiMP improvements in the reported 2024-style suite. Its best pattern places CLM at the beginning and end around a longer MLM block, suggesting CLM may act as fast sequence-structure initialization and late consolidation rather than as an always-on replacement.

### RecGPT says pure causal recurrence is not sufficient for the missing cluster

RecGPT-10M is strong on BLiMP (73.11), COMPS (55.43), GlobalPIQA (40.68), and competitive Overall (41.53), but Entity is only 16.59 and EWoK 52.62. This matters: causal/recurrent modeling can help sequence plausibility and GlobalPIQA-like columns, but it does not by itself solve Entity. The next route should therefore be hybrid, not pure causal replacement.

### ACLM/2025 hybrid results highlight two later levers, not the immediate route

The 2025 ACLM-over-GPT-BERT paper suggests small tokenizers (4k/6k), 1:1 hybrid ratios, and smaller batches can raise Entity in some settings, but the results are unstable and use a different year/task surface. Dynamic curriculum also requires careful exposure accounting and self-selection mechanics. This is useful later, but too many factors for the immediate next experiment.

## Route comparison after XSpan

| route | scientific value | main risk | near-term use |
|---|---|---|---|
| Current XSpan span-mask objective | data object revealed real s1-specific corpus structure | training learns target spans without correct-s1 binding; harms Supplement/GlobalPIQA | stop as-is; retain data only for probes |
| Ordinary WWM on relational windows | tests whether relation-dense evidence distribution alone helps | previous data/order routes often failed; small relational pool may overfit | side control if cheap, not main route |
| Hybrid causal/masked objective on protected DeBERTa family | inherited from GPT-BERT/AntLM; changes information-flow and supervision on every token while preserving WWM scoring | engineering causal attention in DeBERTa; causal stream may help BLiMP/GPIQA but not Entity | best immediate next route |
| Constrained sentence memory bottleneck | strongest structural way to force cross-sentence information through one channel | more engineering, optimization risk, official eval may not activate memory | reserve for after hybrid result or if hybrid fails cleanly |
| Pure recursive/causal model | RecGPT shows high BLiMP/GPIQA/COMPS and near-leader Overall | poor Entity; does not inherit our Supplement/Reading strengths | not primary unless combined with WWM |
| Tokenizer/curriculum route | 2025 ACLM and tokenizer papers suggest strong Entity/WUG sensitivity | multi-factor and unstable; 40k already hurt Reading in our 2026 run | revisit after hybrid baseline, especially 4k/6k not 40k |

## Chosen next direction: low-ratio causal/masked hybrid on the protected DeBERTa-v2 WWM backbone

### Mechanistic hypothesis

Our WWM DeBERTa learns syntax and reading well but underuses sequential event/state information. XSpan showed that the corpus contains s1-specific semantic dependencies, but narrow span masks from random initialization become target familiarity. A small causal/next-token stream uses every token as supervised signal and forces hidden states to maintain left context throughout normal text, not only selected spans. The WWM branch keeps the bidirectional pseudo-likelihood interface that protects BLiMP, Supplement, Reading, and official `mlm` evaluation.

The expected useful profile is not “causal replaces WWM.” It is: low-ratio causal training improves GlobalPIQA/EWoK/sequence plausibility and possibly Entity through better state/history representations, while WWM preserves the grammar and reading advantages.

### Minimal executable design

Build a trainer fork from the trusted full-cycle WWM trainer with an additional primary causal-next-token stream through the same vocab head:

1. WWM branch: unchanged ordinary WWM, same tokenizer, same DeBERTa-v2 8×480 geometry, same official corpus.
2. Causal branch: use the same token sequence with a causal attention mask; label position t with token t+1, no `[MASK]` token. This must prevent the model from seeing the predicted token at its own position.
3. Combined loss: `L=(1-λ)L_WWM + λL_CLM`, starting with low λ inspired by GPT-BERT (about 1/16 ≈ 0.0625). A second higher-signal 1M arm at λ≈0.5 may be useful only after the low-ratio plumbing is clean.
4. Standard HF checkpoint remains `AutoModelForMaskedLM` compatible for official `mlm` evaluation.
5. Optional but not first: mask-rate schedule 30%→15% and attention gating. Do not stack these until the basic hybrid effect is measured.

### 1M evidence to collect

At 1M, compare at least:

- WWM-only matched baseline (already available or rerun if trainer code changes example order).
- WWM+CLM low-ratio λ≈0.0625.
- If cheap after smoke: WWM+CLM λ≈0.5 as a high-signal stress arm.

Measure:

- ordinary WWM loss and masked target density;
- CLM loss and token supervision count;
- heldout adjacent-sentence next-token NLL under true-s1, wrong-s1, and no-s1 contexts, with early s2 tokens and content spans separated from local later tokens;
- available BabyLM subset: BLiMP, Supplement, Entity, COMPS, GlobalPIQA parallel/nonparallel, Reading.

Proceed to 10M only if the low-ratio hybrid improves at least one target gap column (GlobalPIQA or EWoK proxy plus preferably Entity/COMPS) while keeping Supplement and Reading close to WWM, and if the true/wrong/no-s1 next-token probe shows more history sensitivity than WWM-only or pure span familiarity. If hybrid improves only BLiMP while leaving Entity/GlobalPIQA weak, then RecGPT’s pattern is being reproduced and a memory/data route is still needed.

## What not to do next

- Do not build v5 XSpan filters.
- Do not run current XSpan rho0.15 at 10M/20M.
- Do not jump to full RecGPT reproduction as the main route: it is close to SOTA but misses Entity, one of our largest gaps.
- Do not introduce ACLM, tokenizer changes, attention gating, layer weighting, and mask scheduling simultaneously. They may matter, but the next experiment must first decide whether hybrid information flow helps our protected DeBERTa route.

## Next experiment

The proposed construction and smoke test use a hybrid WWM+CLM trainer with careful causal masking and no leakage, then run a small matched 1M screen. The causal attention implementation is the central engineering risk; verify with a tiny synthetic test that changing a future token cannot change earlier hidden/logit predictions under the causal branch.
