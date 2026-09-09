# fixedinit and phase2 dynamics — fixed-init data effect and corrected Phase 2 dynamics

## Fixed-initialization 16k replication

Both inherited-recipe arms completed training:

- `training/runs/official_fixedinit16k_seed43022/`: 100,000,000 word exposure, 2,442 steps, loss 9.81849 → 2.57820, `extra_init_seed=43022`, `train_rng_seed=43023`.
- `training/runs/mix25_fixedinit16k_seed43022/`: 100,000,000 word exposure, 2,442 steps, loss 9.80520 → 2.57731, `extra_init_seed=43022`, `train_rng_seed=43023`.

Evaluation script: `scripts/eval_fixedinit_replication.py`.
Summary JSON: `data/fixedinit_eval/fixedinit_eval_summary.json`.
Per-target JSON: `data/fixedinit_eval/per_target/official_fixedinit.json`, `data/fixedinit_eval/per_target/mix25_fixedinit.json`.

| target | BLiMP | Supp | EWoK | Entity fast | Entity full | COMPS | GPIQA | Reading | equal7 fast | equal7 fullEnt | wproxy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| official_fixedinit | 67.94 | 60.00 | 50.64 | 21.16 | 22.32 | 51.96 | 36.62 | 7.31 | 42.233 | 42.399 | 48.053 |
| mix25_fixedinit | 67.56 | 63.20 | 50.91 | 20.81 | 22.16 | 51.47 | 38.59 | 8.01 | 42.936 | 43.129 | 48.757 |

`mix25_fixedinit - official_fixedinit`: equal7 fast +0.703, equal7 fullEntity +0.730, weighted fast proxy +0.703. The robust data effect is not an Entity lift under shared init; it is broad improvement through Supplement (+3.20), GlobalPIQA mean (+1.97), Reading (+0.70), and EWoK (+0.27), with BLiMP (-0.38), COMPS (-0.49), and Entity full (-0.16) slightly lower.

## Phase-2 control correction

The original phase2 sota plan Phase 2 launcher was not run. The sequence curriculum in `scripts/phase2_sota_trainer.py` truncates 160-word examples to 64/128 tokens while still counting all 160 words, and keeps batch size fixed at 256. A 2M pilot of that trainer confirmed the problem and also showed an OOM when it reached seq256 at batch256. In the pilot, seq64 saw only about 0.4 visible token positions per counted word, while seq256/batch256 exceeded H100 memory.

A shared tokenizer was rebuilt from only the 7.5M official subset contained in both official-only and mix25 pools, avoiding aligned text in the official arm's tokenizer:

- Extracted text: `data/shared_tokenizer/official_subset_7p5M.txt` (46,875 rows, 7,500,000 words).
- Trainer: `scripts/train_shared_tokenizer.py`.
- Tokenizer: `data/shared_tokenizer/hf_tokenizer_40k_shared/`.
- Metadata: `data/shared_tokenizer/shared_tokenizer_meta.json`.

A memory/dynamics probe using the real 12×384/40k model found safe constant-position batches:

- seq64 batch512: ok, peak 58.9 GB, 32,768 visible padded positions/update.
- seq128 batch256: ok, peak 57.0 GB, 32,768 positions/update.
- seq256 batch128: ok, peak 59.8 GB, 29,274 actual visible tokens/update in the sample; batch160 was too close to memory limit and batch256 OOM.

Initial 40w/80w chunks still lost several percent of tokens in early stages, so the final low-truncation materialization uses 32w/64w/160w chunks:

- Materializer: `scripts/materialize_phase2_stagewise_v2.py`.
- Summary: `data/phase2_stagewise_v2/stagewise_v2_summary.json`.
- Stage design: 0–30M words: 32-word chunks at seq64/batch512; 30–60M words: 64-word chunks at seq128/batch256; 60–100M words: 160-word chunks at seq256/batch128.
- Official files: `official_stage1_seq64_30M_v2.jsonl`, `official_stage2_seq128_30M_v2.jsonl`, `official_stage3_seq256_40M_v2.jsonl`.
- Mix25 files: `mix25_stage1_seq64_30M_v2.jsonl`, `mix25_stage2_seq128_30M_v2.jsonl`, `mix25_stage3_seq256_40M_v2.jsonl`.

Stagewise trainer: `scripts/phase2_stagewise_trainer.py`. It initializes one model, trains stages 1→2→3 with dynamic batches, uses one cosine schedule over 5,984 total updates, switches WWM→token at 70M counted words, saves 10M checkpoints, and records visible-token/visible-word-group exposure.

Smoke run: `training/runs/stagewise_smoke_official/` completed 60 updates over all three stages. Stage metrics showed nearly all counted words visible:

- stage1 seq64/batch512: visible tokens/word 1.4481, visible word groups/word 0.9977.
- stage2 seq128/batch256: visible tokens/word 1.4418, visible word groups/word 0.9995.
- stage3 seq256/batch128: visible tokens/word 1.4103, visible word groups/word 0.9805.

This repairs the scientific dynamics relative to the old Phase 2 trainer.

## Running long Phase 2 task

Launcher: `scripts/launch_phase2_stagewise.sh`.

The first attempt failed before training because its working directory and file paths were inconsistent; it produced no training evidence.

After correcting the working-directory/path mismatch, both arms started. Their completed measurements were pending in this record:

- `training/runs/phase2s_official_40k_12x384_seed43200/` on GPU0.
- `training/runs/phase2s_mix25_40k_12x384_seed43200/` on GPU1.

Required checks after training: verify both runs reach 100,000,000 words, 5,984 updates, 12×384 architecture, 40k shared tokenizer, and valid `hf_model/chck_100M`; inspect stage metrics for visible exposure; then evaluate phase2 official vs phase2 mix25 with `scripts/eval_phase2.py` or a repaired target mapping using the mixture eval repaired/fixedinit and phase2 dynamics evaluator pattern.
