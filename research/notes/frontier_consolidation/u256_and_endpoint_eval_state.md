# u256 and endpoint eval state — U256 screen, endpoint evaluation hardening, and active tasks

## Scale1.75 100M endpoint evaluation path

The scale1.75 adapter128 endpoint remains the lead fully legal endpoint trajectory because its 80M cheap7 was 43.8121, above the approximate cheap7 needed for a flat-SuperGLUE/AoA Overall 41.8 endpoint. The 100M training task from earlier analysis is still the decisive measurement path.

u256 and endpoint eval state found two problems in the active tasks and eval pipeline full-evaluation draft before the endpoint arrived:

1. `reference_100M_REF["cheap7"]` was missing and would crash summary writing.
2. The script ran cheap columns and SuperGLUE/AoA concurrently into the same per-target JSON and output tree, creating a race where one process could overwrite or lose the other process's task records.

A hardened evaluator was written:

- `scripts/full_eval_scale1p75_100M_hardened.py`
- Output root for its launched task: `data/scale1p75_100M_full_eval_hardened/`

The hardened evaluator waits for the 100M endpoint and AoA ladder, checks first loss 9.837543487548828, evaluates each zero-shot/Reading column in a separate output root, runs SuperGLUE in its own output root, runs AoA in its own output root, stages all predictions into the exact earlier analysis pristine-collator path layout, then runs fail-closed pristine collation. It avoids per-target JSON races and computes exact deltas against spatial repair route status 100M Overall 41.257770896404615. After evaluation started, the file was additionally patched to write the pristine score vector back into the staged candidate payload for later item-flip analysis; if the running evaluation had already loaded the pre-patch script, this backfill may need to be done manually from the summary after completion, but the core scoring/collation remains race-free.

Self-test on the completed spatial repair route status endpoint verified endpoint readiness/loss/ladder logic and correctly refused to stage absent split parts. AST syntax check passed after the final patch.

## U256 20M evidence

U256 faithful experience utilization changes the training example object while keeping the same legal 10M/100M compact-view reinvest corpus, legal spatial repair route status tokenizer, DeBERTa-v2 8x480 architecture, AdamW/WWM recipe, stream order, and seeds. The mechanism is faithful visibility of every charged word's tokenizer tokens rather than row-level fixed-256 truncation.

Training evidence:

- Run: `training/runs/eu_U256_legal16k_seed43022_20M/`
- First loss: 9.826857208144903 (close to spatial repair route status 9.837543, difference -0.1086%, expected from changed token presentation)
- Last loss: 3.718591442220565
- Steps/words: 506 steps, exactly 20,000,000 charged words
- Active tokens per 10M: 14,664,519 versus spatial repair route status row256 about 14,295,000 (+2.59%)
- Checkpoints: 20 checkpoint directories chck_1M..chck_20M

Evaluation evidence:

- Summary: `data/eu_U256_20M_summary/eu_U256_20M_eval_summary.json`
- Payload: `data/eu_U256_20M_eval/per_target/eu_U256_20M_seed43022.json`
- U256 cheap7: 40.581428571428575
- spatial repair route status 20M cheap7 reference: 39.66357142857142
- Delta: +0.9178571428571445
- Column deltas: BLiMP +0.90, Supplement +2.44, EWoK -0.48, Entity -0.28, COMPS +0.53, GlobalPIQA +4.455, Reading -1.14.

CPU item-flip analysis:

- `data/u256_20m_item_flips/u256_vs_step35_20M.{json,md}`
- The discrete-column reconstruction exactly matched the payload delta.
- BLiMP net +586 item flips, with positives in principle_A_domain_1, npi_present_1, wh_island, adjunct_island, but losses in existential_there_quantifiers_2, irregular plural agreement, tough_vs_raising_2, and left_branch_island_simple_question.
- Supplement net +130, mainly subject_aux_inversion and small QA gains.
- EWoK net -50: material-properties and spatial-relations improve, but material-dynamics (-83/770), social-interactions, social-properties, physical-relations, and quantitative-properties weaken.
- Entity is near-neutral at the item level (+11 net) but official aggregation is -0.28; high-operation regular_5_ops is negative.
- COMPS net +479 broad but small relative to its row count.
- GlobalPIQA net +9 items over only 203 items, so the +4.455 point gain is fragile.

Scientific interpretation: the U256 20M result is strong enough to justify a full maturation test because the predeclared continuation threshold was +0.3 cheap7 and the observed gain was +0.9179. It is not endpoint evidence. Like Muon and residual adapters, it may reverse or mature into a tradeoff. The specific risk to watch is whether the EWoK material-dynamics/quantitative/social losses and Reading loss persist while early BLiMP/Supplement/GlobalPIQA gains fade.

## Full U256 trajectory launched

Because U256 crossed the predeclared threshold and tests a distinct model/data-structure mechanism, u256 and endpoint eval state launched the full 100M U256 trajectory on GPU1:

- Run: `training/runs/eu_U256_legal16k_seed43022_100M/`
- Command family: `python -B scripts/stream_order_eu_trainer.py --arm U256 --tokenizer_path data/compliant_tokenizer --tokenizer_label legal16k --output_dir ... --hidden_size 480 --n_layer 8 --n_head 8 --ffn_mult 4 --learning_rate 0.001 --weight_decay 0.01 --warmup_fraction 0.06 --mask_prob 0.15 --seed 43 --extra_init_seed 43022 --train_rng_seed 43023 --checkpoint_words 1000000 --log_every 50`
- Intended endpoint: 100M charged words, 2529/2530-ish optimizer steps depending exact step partition, full 1M checkpoint ladder for AoA.

The expensive-work decision: no cheaper measurement can decide mature viability because earlier routes repeatedly showed early gains that reversed by 80M/100M. The 100M U256 run will decide whether faithful visibility is a true legal endpoint route or another early-surface redistribution.

## Pending Training and Evaluation

- Scale1.75 100M training: in progress.
- Hardened full official-compatible scale1.75 100M evaluation: conditional on completed training and endpoint verification.
- U256 100M training: in progress.

Do not launch another full evaluation that can race these jobs. After the scale1.75 evaluation completes, inspect `data/scale1p75_100M_full_eval_hardened/summary/scale1p75_100M_full_eval_hardened_summary.{json,md}` and the pristine collation summary. After U256 training completes, verify the U256 endpoint and decide when to evaluate it, preferably after scale1.75 evaluation has freed resources.
