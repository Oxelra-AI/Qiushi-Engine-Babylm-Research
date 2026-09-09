# route pivot review after three controlled failures — Route pivot after memory, order, and surface routes failed

## Why the previous main line should stop

Three controlled mechanism families have now failed as direct scale-up candidates:

1. **Prefix-average memory** did not produce stable positive deltas under paired shared-core initialization, fixed schedule, identical example order, two pools, and two seeds.
2. **Pure single-pass order** over exactly the same selected examples/source mix moved scores, but source staging traded Entity/Supplement away for BLiMP/EWoK and readability interleaving gave only a small BLiMP signal.
3. **Learned shared char/n-gram surface composition** plus a parameter-scale lookup adapter produced no broad gain at 1M. The char route averaged BLiMP +0.21 but Supplement -2.40, EWoK -1.23, Entity -0.14, COMPS -0.15, Reading-eye -0.39. The lookup control was also negative on Supplement/EWoK/Entity/COMPS/Reading-eye. Char-minus-lookup was mixed and seed-unstable.

These results do not say architecture, ordering, or representation can never help. They say the current causal dense backbone plus small input or memory side path is not attacking the large 2026 Strict-Small gap. Continuing to patch this family would likely spend compute on low-value local variants.

## What the current 2026 leaderboard implies

The live Strict-Small snapshot puts the frontier well outside our current causal profile:

| rank | model | Overall | NLP | Human-like | BLiMP | Supp | EWoK | Entity | COMPS | GLUE | GlobalPIQA | Reading | AoA |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `wwm_curriculum_simplification_40k` | 41.80 | 52.97 | 2.71 | 67.20 | 56.01 | 56.07 | 28.45 | 53.57 | 69.79 | 39.67 | 5.42 | 0.00 |
| 2 | `RecGPT-10M` | 41.53 | 52.40 | 3.46 | 73.11 | 61.73 | 52.62 | 16.59 | 55.43 | 66.64 | 40.68 | 6.92 | 0.00 |
| 3 | `Wordpiece-24-4` | 41.31 | 52.85 | 0.92 | 70.20 | 66.52 | 52.10 | 21.82 | 54.04 | 69.18 | 36.08 | 1.84 | 0.00 |

The top row is not a causal GPT-2 variant. Its model card states:

- DeBERTa-v2 style encoder, `DebertaV2ForMaskedLM`.
- 34.68M parameters, hidden 384, intermediate 1280, 12 layers, 12 heads.
- 40k SentencePiece BPE tokenizer trained for the data condition.
- 9,999,969 whitespace-counted words of FineWeb sentence-level simplification pairs: original sentence followed by simplified rewrite.
- Masked LM objective.
- LAMB optimizer, max LR 0.007, cosine schedule.
- 10 epochs.
- Sequence length curriculum 64 -> 256.
- Masking curriculum: epochs 1-7 whole-word masking, epochs 8-10 token-level masking.

The official evaluator supports `mlm` and `mntp` as well as `causal` for zero-shot, AoA, and collation. Therefore the model comparison need not be restricted by the causal-only training path.

## Scientific interpretation

The current SOTA evidence suggests the winning lever is not a small side mechanism on top of causal LM. It is the combination of:

1. **More supervised prediction density** from masked or MNTP-style objectives.
2. **Bidirectional contextual scoring** for minimal-pair and fine-tuning tasks.
3. **Whole-word masking early in training**, which forces all subword pieces of a selected word to be predicted from context and may improve entity/world-knowledge abstraction.
4. **A large 40k tokenizer**, reducing token fragmentation and token-sequence length relative to 16k BPE.
5. **Paired rewrite data**, which presents semantically aligned variants and may explain high EWoK/Entity/GLUE.
6. **Length and masking schedules**, which increase training difficulty across epochs rather than using a single pass.
7. **Strong small-model optimization** through LAMB or Muon-like updates.

The leaderboard also separates strengths:

- RecGPT shows recursive causal structure can produce very high BLiMP/Supplement/COMPS, but its Entity is weak. It is not the best next route for the unresolved Entity/overall problem unless combined with a separate data/objective solution.
- WWM simplification is the only visible top route that simultaneously lifts Entity to 28.45 and keeps competitive NLP columns.
- Wordpiece-24-4 supports tokenizer/objective leverage but its technical artifact was not directly available in this step.

## What to inherit, what to improve

### Inherit now

- Official-compatible `mlm`/`mntp` evaluation target.
- Masked/encoder or GPT-BERT-style training path.
- Whole-word masking implementation.
- Sequence-length schedule 64 -> 256.
- 40k tokenizer option, at least as a comparison arm.
- 10-epoch budget accounting and `chck_1M`... checkpoint naming.
- Fast/local profile first: BLiMP, Supplement, EWoK, Entity, COMPS, Reading.

### Do not merely copy

A direct reproduction of the current top model is useful as an anchor only if time allows, but it is not a breakthrough by itself. The research route should use the top system to identify which factor actually matters, then alter one high-value factor in a way that can beat the current Overall. The strongest potential advance is to combine the WWM simplification route’s Entity/EWoK/GLUE strength with a better way to preserve RecGPT/Wordpiece-like BLiMP/Supplement strength.

## Next executable route

### Route A — masked/encoder pipeline and WWM schedule anchor

Build an official-compatible masked LM trainer before designing new architecture. Start with a compact DeBERTa-v2 or BERT-like masked encoder small enough to train quickly, because the official top row itself is DeBERTa-v2 MLM.

First 1M experiment should not try to reproduce the full 10-epoch top model. It should answer whether the new pipeline produces the expected task-family movement relative to causal dense.

Arms for a controlled 1M route-sense experiment:

1. **MLM-token-random**: official corpus or available paired-rewrite corpus, token-level random masking, 40k or baseline tokenizer, length 128/256.
2. **MLM-WWM**: same examples, same tokenizer, same model, same exposure, whole-word masking.
3. **MLM-WWM + short-to-long**: same total word exposure with sequence length 64 early, then 256.

If paired simplification data is accessible and compliant, add a paired-data arm only after the basic MLM/WWM path works:

4. **MLM-WWM-pairs**: FineWeb simplification pairs or a legally constructed paired corpus, with exact 10M whitespace-word accounting.

Useful signal at 1M:

- WWM should raise Entity/EWoK or Supplement relative to token masking without hurting BLiMP severely.
- If all MLM arms immediately outperform causal dense on BLiMP/Supplement but not Entity, paired data is probably the missing WWM-top-row factor.
- If WWM improves Entity but hurts Supplement, the next innovation should target a masking schedule rather than architecture.
- If 40k tokenizer is essential, compare 16k vs 40k only after the basic MLM pipeline has a working smoke and one small profile.

### Route B — GPT-BERT/MNTP + Muon/Mixed objective

If DeBERTa-v2 construction is too slow or unstable, a GPT-BERT/MNTP path is the next best execution route because the official evaluator supports `mntp`, and BabySteps/RecGPT evidence suggests optimizer and hybrid objective can greatly improve NLP columns.

Do not begin by adding another memory or surface adapter. Build the official-compatible backend first.

### Route C — RecGPT-style recursive causal

RecGPT is valuable for BLiMP/Supplement/COMPS but weak on Entity. It should not be the next main route unless the masked/WWM path fails to run. It may later supply architecture ideas to strengthen BLiMP/Supplement after an Entity-strong masked/data route exists.

## Immediate construction work

The proposed implementation is the minimal masked pipeline, not a full final system:

1. Inspect official `babylm-eval` `mlm` and `mntp` scoring calls enough to know required HF classes and tokenizer fields.
2. Create a small masked-LM trainer under `training/scripts/`, separate from the verified causal trainer to avoid breaking it.
3. Support: tokenizer loading/training path, exact whitespace-word exposure accounting, masking mode `token` vs `wwm`, sequence length schedule, root/`chck_1M` save, `AutoModelForMaskedLM` loading, and evaluator backend `mlm` or `mntp`.
4. Run 10k smokes for token MLM and WWM, checking save/reload and official fast BLiMP/Supplement/EWoK/Entity execution.
5. Only after smokes pass, launch the first controlled 1M MLM token-vs-WWM comparison.

Do not scale to 10M/100M, run full official evaluation, or write final deliverables until the masked route produces real controlled signal in the fast/local suite.
