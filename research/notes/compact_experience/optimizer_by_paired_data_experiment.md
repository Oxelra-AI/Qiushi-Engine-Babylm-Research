# optimizer by paired data experiment — optimizer × paired-data construction

## Scientific question

The clean-Qwen same-window corpus is the strongest validated positive mechanism in these experiments, but its complete Overall (41.3443) remains below the visible 41.8 leader. The immediate question is not whether LAMB generically lowers MLM loss. It is whether layer-wise adaptive update scaling changes how the fixed 8×480/16k DeBERTa-v2 extracts transferable structure from aligned same-window rewrites, relative to official-only text.

This requires separating:

1. **optimizer identity at matched step scale:** AdamW 1e-3 versus LAMB 1e-3, with identical beta/epsilon/schedule;
2. **optimizer identity at matched high step scale:** AdamW 7e-3 versus LAMB 7e-3 if AdamW remains numerically stable;
3. **coupled leader-like recipe:** LAMB 7e-3 with beta2=0.95 is distinct from the matched-beta comparison and must not be attributed solely to layer adaptation;
4. **generic optimizer improvement versus data interaction:** measure Qwen minus official under each optimizer, not only Qwen absolute score.

## Fixed experimental coordinate

- corpora: `data/qwen_clean_aligned/training_corpora/{official_only,qwen_aligned}_100M.jsonl`
- tokenizer: baseline 16k from the existing tokenizer interface result and route checkpoint
- model: DeBERTa-v2 8 layers × 480 hidden, 8 heads, FFN 1920
- fixed sequence length 256
- pure WWM-MLM, mask probability 0.15, BERT 80/10/10 replacement
- batch 128
- initialization seed 43022, training RNG seed 43023
- cosine schedule, 5% warmup, weight decay 0.01
- no AoA/CDI information in construction or route selection

## Pilot

`scripts/optimizer_study_trainer.py` adds AdamW/LAMB choice and records gradient norm, masked-token count, pair-row/non-pair-row loss, and LAMB trust ratios grouped by layer. `scripts/launch_optimizer_pilot.sh` launches an eight-arm 1M-word stability study:

- AdamW 1e-3 × official/Qwen
- LAMB 1e-3 × official/Qwen
- AdamW 7e-3 × official/Qwen
- LAMB 7e-3 × official/Qwen

The 1e-3 pair isolates optimizer identity. The 7e-3 pair isolates identity only if high-LR AdamW is stable; otherwise LAMB 7e-3 remains a coupled optimizer-plus-step-scale recipe. If AdamW 7e-3 diverges, a 5e-3 AdamW smoke already ran stably for 39 short-sequence CPU steps, but a proper seq256 1M bracket is still required before using it as the highest stable AdamW comparator.

## Interpretation across learning time

A 1M run is only a stability and update-dynamics pilot. The clean-Qwen data effect changes substantially from 20M to 100M, so an optimizer route cannot be selected from a single early endpoint. The promoted compact study must preserve checkpoints and evaluate a trajectory (at least 5M, 10M, and 20M) for both corpora under the selected optimizer settings. At each endpoint compute:

\[
\Delta_{data}^{(o,t)} = S(Qwen,o,t)-S(Official,o,t),
\]

and the interaction

\[
I_t = \Delta_{data}^{(LAMB,t)}-\Delta_{data}^{(AdamW,t)}.
\]

Inspect the column vector, not just its mean: BLiMP, Supplement, EWoK, Entity, COMPS, GlobalPIQA, and Reading. Promotion requires a persistent or strengthening broad interaction, not a transient single-task swing analogous to tokenizer interface result and route's GlobalPIQA artifact. Loss acceleration without broad downstream interaction is not sufficient.

## Scale-up logic

After pilot stability:

1. choose matched 1e-3 identity pair and, only if stable, matched 7e-3 identity pair;
2. run official/Qwen arms to 20M with 1M checkpoints;
3. no-AoA evaluate 5M, 10M, and 20M endpoints, reusing tokenizer interface result and route AdamW 1e-3 references where configuration hashes match;
4. compute optimizer main effects, data effects, and optimizer × data interactions at every endpoint;
5. move one setting to 100M only if the interaction is broad and develops coherently with exposure;
6. complete all nine official columns, including terminal aggregate AoA, for any 100M candidate.

The trusted complete result remains clean-Qwen seed43022 `chck_100M`, Overall 41.34429066479573. No SOTA has been established.

## 1M pilot result

All eight arms completed 51 updates with finite losses. The corpus prefix ended at the last complete row, 999,866 words; this pilot is therefore a dynamics study, not a legal 1M checkpoint or evaluation endpoint. No checkpoint was saved because the complete-row exposure did not cross 1,000,000 words.

Final-batch losses (single noisy observations) were:

| optimizer/LR | official | Qwen |
|---|---:|---:|
| AdamW 1e-3 | 6.7127 | 6.8841 |
| LAMB 1e-3 | 8.6825 | 8.6909 |
| AdamW 7e-3 | 6.7904 | 6.9465 |
| LAMB 7e-3 | 6.6890 | 6.8654 |

At matched 1e-3, LAMB learned much more slowly because its trust ratios began near 0.61 for transformer layers and 0.41 for embeddings, so its effective updates were smaller than AdamW's. At matched 7e-3, both AdamW and LAMB were stable on both corpora and reached similar losses; LAMB was marginally lower on the final sampled batch, but that is not downstream evidence. Thus the asymmetric-LR confound is resolved: a matched 7e-3 AdamW comparator exists and is stable over this pilot.

The Qwen prefix contained 1,236 pair rows and 167,386 pair words. Pair-row MLM loss remained higher than non-pair loss under both high-LR optimizers near the end (AdamW 7e-3: 7.4093 vs 6.9408; LAMB 7e-3: 7.3410 vs 6.8609), with a slightly smaller gap under LAMB in that sampled batch. This is suggestive only; the decisive evidence is the downstream optimizer × data interaction trajectory.

A matched 10M high-LR four-arm run had started; completed measurements were pending in this record. It will generate 1M..10M checkpoints for AdamW/LAMB × official/Qwen at 7e-3, beta2=0.98, eps=1e-8. The next step should evaluate multiple checkpoints without AoA and compute the column-wise interaction trajectory. The 1e-3 LAMB coordinate is not promoted because its reduced effective step scale makes a 20M comparison likely uninformative unless later needed to complete optimizer-identity interpretation.
