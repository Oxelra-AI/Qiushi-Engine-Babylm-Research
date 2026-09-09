# s1 10m available coordinate — S1 12×384 leader-shape 10M result and required comparison

## Run validity

S1 run: `training/runs/babylm_leadershape_s1_10M_aligned_micro128/`

The run completed successfully and is valid as a legal architecture-shape isolation screen:

- isolated fork: `training/scripts/babylm_masked_train_leadershape.py`
- protected full-cycle trainer remains restored and untouched;
- alignment evidence: `data/leadershape_micro_split_alignment.json` confirms exact example order, tokenization, masking counts, exposure, optimizer steps, and LR trajectory under the micro-split fork;
- training: official 10M corpus, baseline16k tokenizer, flat WWM, effective batch 256, micro batch 128, `lr_total_steps=2442`;
- model: DeBERTa-v2 12 layers, hidden 384, 12 heads, intermediate 1280, p2c/c2p relative attention;
- parameters: 29,329,792 total, 6,291,456 embedding, 23,038,336 non-embedding;
- checkpoint: direct path `hf_model/chck_10M` exists and was used for evaluation;
- training loss: 9.7892 → 4.0523 over 245 optimizer steps / 10M words.

This is not leader reproduction: exact `go76dof/Fineweb_simplification_pairs` data remains gated, and S1 uses baseline16k rather than the leader's 40k SentencePiece data condition.

## Direct-checkpoint available coordinate

Evaluation script: `scripts/eval_s1_10m_available_coordinate.py`

Evidence JSON: `data/s1_10m_available_coordinate.json`

Direct-checkpoint path was used as `model_path_or_name`, avoiding the known local-root `revision_name` bug.

| column/task | S1 10M score |
|---|---:|
| BLiMP | 53.37 |
| BLiMP Supplement | 52.34 |
| Entity Tracking | 17.73 |
| COMPS | 50.34 |
| GlobalPIQA parallel | 19.42 |
| GlobalPIQA nonparallel | 51.00 |
| GlobalPIQA mean | 35.21 |
| Reading eye-tracking | 11.69 |
| Reading self-paced | 5.01 |
| Reading mean | 8.35 |

Missing in this quick coordinate: full EWoK, SuperGLUE, AoA.

## Scientific reading without over-interpreting

The result is not a positive early signal for the target relational/knowledge cluster: Entity is 17.73, COMPS 50.34, and GlobalPIQA mean 35.21. Reading is strong at 10M (mean 8.35), but the central gap to the visible leader is Entity/EWoK/GlobalPIQA, not only Reading. BLiMP 53.37 is also far below the protected model's 100M endpoint, as expected for an early checkpoint.

However, the architecture route should not be eliminated by smoke/1M evidence; this is the first valid 10M architecture evidence. The decisive next comparison is the matched protected 8×480 WWM `chck_10M`, evaluated by the same direct-checkpoint script. That comparison is required before deciding whether S1 is a slower-learning architecture that might catch up with exposure, or whether the leader's gains are mainly curriculum/tokenizer/data rather than shape alone.

## Required next action

Run the same direct-checkpoint available-coordinate evaluation on:

`training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/hf_model/chck_10M`

using the same tasks and parser as `eval_s1_10m_available_coordinate.py`. Then compare:

- S1 12×384 10M vs protected 8×480 10M on BLiMP, Supplement, Entity, COMPS, GlobalPIQA mean, Reading;
- both 10M profiles against protected 8×480 100M endpoint to see learning-curve direction;
- S1 parameter allocation: 23.04M non-embedding and 6.29M embedding versus protected 8×480 34.47M total, because S1 has lower total capacity under baseline16k.

Only after that comparison should the next route be chosen:

- If S1 is competitive or shows a distinctive target-cluster direction at 10M, add leader-style curriculum S2 (length curriculum and WWM→token mask switch) before judging the shape route.
- If S1 is clearly behind the protected 8×480 10M on the target cluster and learning curve, shape alone is unlikely to explain the leader; then prioritize curriculum/tokenizer/data or the separate GPT-BERT/MNTP hybrid branch.
