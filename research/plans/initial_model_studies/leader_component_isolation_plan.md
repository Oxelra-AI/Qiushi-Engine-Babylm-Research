# earlier analysis — leader-package staging result and executable route decision

## What is now established

Model-side artifacts for the visible Strict-Small leader `go76dof/wwm_curriculum_simplification_40k` are staged locally and audited in `data/leader_package_revision_124/model_side/model_side_stage_audit.json`.

Exact leader architecture (config.json):
- DeBERTa-v2, hidden 384, 12 layers, 12 heads, intermediate 1280
- vocab 40000, relative attention true, pos_att_type [p2c, c2p]
- position_buckets 256, max_relative_positions -1, max_position_embeddings 1024
- norm_rel_ebd layer_norm, layer_norm_eps 1e-7, gelu
- 34,677,952 parameters (from card)

Tokenizer: 40000-piece SentencePiece, loads cleanly.

Training recipe (card): LAMB, max LR 0.007, cosine, 10 epochs, sequence length curriculum 64→256, mask curriculum WWM epochs 1–7 then token masking epochs 8–10.

## Hard blocker

The exact training data `go76dof/Fineweb_simplification_pairs` (`FineWeb_simplification_pairs.train`, 9,999,969 words) is **gated (403)** in this environment. The dataset-side 40k tokenizer file is also gated. Only dataset metadata/README is accessible.

Consequence: we cannot faithfully reproduce the leader's exact data condition, and we must not claim leader-package reproduction. But the leader's gains are a coupled package of (data) + (tokenizer) + (architecture) + (optimizer) + (curriculum). Several components are legal and recoverable, and can be tested on the official Strict-Small corpus we already have verified.

## Scientific decomposition of the leader's advantage over our protected model

Our protected baseline16k DeBERTa-v2 **8×480** WWM (34.5M) vs leader **12×384** DeBERTa-v2 (34.7M), both ~34–35M:

| column | ours | leader | gap |
|---|---:|---:|---:|
| BLiMP | 66.76 | 67.20 | −0.44 |
| Supplement | 59.88 | 56.01 | **+3.87** |
| EWoK | 52.19 | 56.07 | −3.88 |
| Entity | 22.62 | 28.45 | −5.83 |
| COMPS | 52.19 | 53.57 | −1.38 |
| SuperGLUE | 68.02 | 69.79 | −1.77 |
| GlobalPIQA | 35.64 | 39.67 | −4.03 |
| Reading | 7.62 | 5.42 | **+2.20** |

The leader wins the relational/knowledge cluster (Entity, EWoK, GlobalPIQA) and SuperGLUE; we win Supplement and Reading. This is exactly the cluster all our mechanism routes failed to move. The leader achieves it with a *different architecture shape (deeper/narrower), a much larger tokenizer (40k), and simplification-pair data + curriculum* — not an exotic objective.

## Confound with our prior official40k result

Earlier we found official40k on the official corpus hurt Reading badly (7.62→0.89) and did not fix Entity. But that was: our 8×480 backbone + official corpus + our tokenizer training + our schedule. The leader differs in **architecture depth (12×384), data (FineWeb simplification pairs), optimizer (LAMB), and curricula (length + WWM→token)** simultaneously. So our official40k result does NOT falsify the leader package. The open question is which legal component(s) carry the relational-cluster gain.

## Chosen next experiment: legal component-isolation on official corpus

Since the exact data is gated, isolate the **architecture + curriculum** levers, which are fully legal on our verified official 10M corpus, before deciding whether data access is essential.

Design a matched set at a screening budget (start 10M exposure, i.e. 1 epoch-equivalent screen, then extend), all on the official Strict-Small corpus with exact word accounting, standard HF DeBERTa-v2 MLM checkpoints:

- **Arm P (protected reference)**: 8×480, baseline16k tokenizer, flat WWM, our schedule. Already characterized.
- **Arm S1 (leader shape)**: 12×384 DeBERTa-v2, baseline16k tokenizer, flat WWM, matched exposure. Tests architecture depth/width alone.
- **Arm S2 (leader shape + curricula)**: 12×384, baseline16k tokenizer, length curriculum 64→256, mask curriculum WWM→token. Tests architecture + curriculum with our legal tokenizer/data.
- **Arm S3 (leader shape + curricula + 40k)**: 12×384, our own 40k SentencePiece trained on the official corpus (legal, counts toward budget as before), length + mask curricula. Tests whether large-vocab helps under the leader shape rather than our 8×480.

If a legal arm (S1/S2/S3) moves Entity + EWoK/GlobalPIQA toward the leader while acceptably trading Supplement/Reading, the relational-cluster gain is reproducible without the gated data, and we can push it to 100M and multi-seed. If none of them approach the leader's relational cluster, then the gated FineWeb simplification data is likely the load-bearing factor, and the next move is authorized-access pursuit or a legal simplification-pair reconstruction under strict word accounting.

## Why this over the hybrid route right now

- The hybrid (GPT-BERT/MNTP) route is competitive but the *visible leader is not hybrid*; hybrids in the snapshot (instanton 40.78, BabySteps ~) do not clearly beat the leader's relational cluster.
- The leader shows the relational-cluster gain comes from a masked DeBERTa with different shape/tokenizer/data/curriculum — closer to our protected model, so component isolation is cheaper and more directly informative than building/validating a new leak-tested causal branch.
- The hybrid pilot remains a valid parallel/next route if component isolation shows architecture/curriculum alone cannot close the relational gap.

## Engineering notes for the executor

- Our trusted `babylm_masked_train_fullcycle.py` already supports `model_type deberta_v2`, `--hidden_size/--n_layer/--n_head`, `--seq_len_schedule`, and mask modes. A 12×384 arm is a config change, not new code.
- Mask curriculum WWM→token needs a small scheduled switch of `mask_mode`; if not already supported, add a minimal epoch/word-threshold switch rather than a new trainer.
- Training our own 40k SentencePiece on the official corpus is legal and was done before; reuse that path. Do NOT use the gated leader tokenizer/data.
- Keep exact word-exposure accounting and standard HF checkpoints for official evaluation.

## Do not

- Do not claim leader reproduction; the data is gated.
- Do not train on `go76dof/Fineweb_simplification_pairs` files (gated/unauthorized).
- Do not scale XSpan.
- Do not stack all leader components in one arm without the S1/S2/S3 isolation, or the causal attribution is lost.
