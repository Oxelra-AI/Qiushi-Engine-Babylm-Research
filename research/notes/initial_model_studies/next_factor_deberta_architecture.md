# next factor deberta architecture — Next single-factor route after WWM, length schedule, and official40k tokenizer

## Current evidence state

The supported masked-route base is now:

- official BabyLM Strict-Small corpus;
- baseline 16k tokenizer;
- `BertForMaskedLM` 8 layers, hidden 256, 8 heads, intermediate 1024;
- WWM, fixed `seq_length=max_seq_length=256`, `max_position_embeddings=512`;
- exact 1M whitespace-word exposure, paired seeds 42/43;
- official `mlm` fast/local profiling.

Controlled evidence:

1. **WWM is protected as a base component.** Across seeds, WWM-minus-token mean deltas are BLiMP +2.15, EWoK +2.50, Entity +0.51, Supplement -0.40, COMPS -0.12, Reading eye -0.06, Reading self-paced -0.10. It is positive in both seeds for BLiMP, EWoK, and Entity, but does not by itself approach the current top system.

2. **The tested short-to-long length schedule is rejected.** With WWM fixed, `0.0:64,0.4:128,0.8:256` versus fixed 256 has mean deltas BLiMP +0.77, Supplement -3.00, EWoK +0.37, Entity -0.09, COMPS -0.61, Reading eye -0.42, self-paced +0.16. The Supplement/COMPS losses and lack of stable Entity gain mean this schedule should not be stacked.

3. **Official-corpus 40k ByteLevel-BPE tokenizer is rejected as the next base.** With WWM fixed and raw examples/order/source mix matched to the 16k WWM baselines, official40k-minus16k mean deltas are BLiMP -0.475, Supplement -3.20, EWoK +1.775, Entity -0.06, COMPS -0.405, Reading eye -8.315, Reading self-paced -3.56. The coupled representation package increased embedding capacity and reduced tokens/truncation, but did not improve Entity and severely damaged Reading.

4. **Simplification-pair data remains scientifically important but not executable now.** The top model card reports 9,999,969 words of FineWeb simplification pairs and strong EWoK/Entity/GLUE gains, but direct download of the train file currently returns gated access. This blocks a faithful data-factor experiment unless access is obtained or a defensible replacement with a clean control is built.

The current top `wwm_curriculum_simplification_40k` system uses DeBERTa-v2 style relative/disentangled attention, simplification-pair data, 40k tokenizer, LAMB, WWM→token masking curriculum, length curriculum, 10 epochs, and 34.7M parameters. Our controls show that WWM alone captures only a small part of the pattern; the tested length schedule and official-corpus 40k tokenizer do not explain the gap.

## Route decision

The next executable single-factor experiment should test **DeBERTa-v2-style architecture at matched tokenizer/data/objective/exposure**:

\[
\text{BERT-WWM-16k-official-fixed256}
\rightarrow
\text{DeBERTaV2-WWM-16k-official-fixed256}.
\]

Reasons:

- It is executable now with Hugging Face `DebertaV2ForMaskedLM` and official `mlm` evaluator support.
- It isolates a major remaining factor from the top system without using the gated data, rejected length schedule, or harmful official40k tokenizer.
- Mechanistically, disentangled relative attention (`relative_attention=true`, `pos_att_type=["p2c","c2p"]`) may improve small-data learning of syntactic relations, entity state tracking, and masked-word inference from context/position relations.
- It avoids the high-confound optimizer path: LAMB at 1M/98 steps would immediately entangle optimizer, peak LR, trust ratio, batch construction, and warmup.
- It avoids constructing a weak replacement for paired data. If the original paired dataset becomes accessible, or if a controlled pair-vs-shuffle replacement is built, data should be revisited.

This route does **not** imply that architecture is proven to explain the SOTA gap. It is the best current single-factor test because the previous executable factors were negative and the strongest data factor is gated.

## Parameter probe and primary configuration

Baseline BERT-WWM has 10,727,168 parameters, with 4,194,304 input-embedding parameters (`vocab=16384`, `hidden=256`). A parameter probe is saved at:

`experiments/archive/initial_model_studies/data/deberta_param_probe.json`

The primary DeBERTa-v2 configuration should be:

| field | value |
|---|---:|
| model | `DebertaV2ForMaskedLM` |
| vocab | 16384 |
| hidden_size | 240 |
| num_hidden_layers | 8 |
| num_attention_heads | 6 |
| head_dim | 40 |
| intermediate_size | 960 |
| relative_attention | true |
| pos_att_type | `p2c,c2p` |
| position_buckets | 256 |
| max_relative_positions | 256 |
| max_position_embeddings | 512 |
| total params | 10,733,104 |
| delta from BERT | +5,936 (+0.055%) |
| input embedding params | 3,932,160 |

Why this configuration:

- It preserves the **8-layer depth** and **4× FFN ratio** of the current BERT baseline.
- Total parameters are almost exactly matched to the BERT-WWM baseline.
- It introduces the DeBERTa-v2 relative/disentangled attention mechanism without a hidden capacity jump.
- It does change hidden size/head count/embedding size (240/6 heads versus 256/8 heads), which must be reported. If results are ambiguous, a later diagnostic can test a same-hidden DeBERTa config such as hidden 256/layers 8/heads 8/intermediate 768 (+1.24% params) to separate hidden-shape from relative-attention effects. The immediate next experiment should use the primary matched-total configuration only.

## Controlled experiment design

### Existing baselines

- seed42 BERT-WWM-16k-fixed256: `babylm_masked_wwm_pos512_1M`
- seed43 BERT-WWM-16k-fixed256: `babylm_masked_wwm_pos512_seed43_1M`

### New arms

- seed42 DeBERTaV2-WWM-16k-fixed256: suggested run id `babylm_step52_masked_debertav2_wwm_seed42_1M`
- seed43 DeBERTaV2-WWM-16k-fixed256: suggested run id `babylm_step52_masked_debertav2_wwm_seed43_1M`

### Hold fixed

- official BabyLM Strict-Small corpus and exact raw selected examples/order/source mix by seed;
- baseline 16k tokenizer;
- WWM objective, mask probability 0.15, same masking RNG;
- fixed `seq_length=max_seq_length=256`, no length schedule;
- `max_position_embeddings=512`; `max_relative_positions=256`, `position_buckets=256` for DeBERTa;
- exact 1,000,000 whitespace-word exposure from the same 10M pool;
- batch size 64, `lr_total_steps=98`, AdamW, LR 0.001, warmup fraction 0.05, weight decay 0.01;
- seed/init/RNG triples: seed42 uses 42/456/789; seed43 uses 43/457/790;
- official `mlm` profiles: BLiMP fast, Supplement fast, EWoK fast, Entity Tracking fast, COMPS, Reading.

### Implementation requirements

Patch `babylm_masked_train.py` without breaking BERT:

- add `--model_type {bert,deberta_v2}` default `bert`;
- add DeBERTa args if needed: `--deberta_relative_attention`, `--deberta_pos_att_type`, `--position_buckets`, `--max_relative_positions`;
- import `DebertaV2Config`, `DebertaV2ForMaskedLM`;
- in `build_model`, choose BERT or DeBERTa based on `args.model_type`;
- record model type, config fields, total parameters, input embedding params, and if convenient non-embedding params;
- keep tokenizer-coupling and exact example manifests unchanged.

Run a 10k DeBERTa WWM smoke first:

- small or primary config is acceptable for smoke, but it must save root and `chck_1M`;
- load with `AutoModelForMaskedLM`;
- pass a 180-token forward test;
- complete official fast BLiMP and Entity with backend `mlm`.

Then train and profile both 1M DeBERTa arms. The runner should assert exact raw example IDs/order/source match to the corresponding BERT-WWM baseline.

## Carry-forward signals

Carry DeBERTa-v2 forward only if the two-seed 1M result shows a task-family profile consistent with better relation/position modeling:

- Entity Tracking improves in both seeds, ideally mean gain clearly above the WWM-only +0.51 effect;
- BLiMP improves in both seeds or at least does not fall meaningfully;
- Supplement does not show the large losses seen for length schedule and official40k;
- EWoK or COMPS improves in at least one meaningful, repeatable direction;
- Reading does not collapse like official40k;
- training loss/MLM dynamics are not so much worse that the result mainly reflects optimizer incompatibility.

If DeBERTa improves Entity/EWoK/BLiMP without severe losses, WWM+baseline16k+DeBERTa becomes the next base for either optimizer or legal data experiments. If DeBERTa is neutral or harmful under the fixed AdamW recipe, the next route should not immediately scale; it should evaluate whether optimizer dynamics (LAMB) or data semantics is the missing factor.

## Remaining high-value alternatives

- **Simplification/rewrite data:** still the most semantically plausible explanation for the top system's Entity/EWoK/GLUE profile, but currently blocked by gated access. If access becomes available, test data while holding tokenizer/model/objective fixed. If building a replacement, use a design like pair-adjacent versus pair-shuffled on the same sentence set, not a broad corpus swap.
- **LAMB/optimizer:** executable but high-risk as a coupled training-recipe factor. Test after architecture unless DeBERTa construction is impossible or unstable; define trust ratio, LR, warmup, and batch rules before running.
- **40k tokenizer and length schedule:** do not carry forward from the controlled results just obtained.
