# leadershape micro split alignment — S1 aligned 10M launch record

## Why this run is now justified

The protected `babylm_masked_train_fullcycle.py` has been restored and is no longer modified. The S1 route uses isolated fork `training/scripts/babylm_masked_train_leadershape.py`.

Fork alignment evidence:

- `data/leadershape_micro_split_alignment.json`
- `notes/leadershape_micro_split_alignment.md`

The fork preserves the protected trainer's load-bearing semantics at short-run level:

- same example order and tokenization summary;
- same HF config;
- same word exposure and source words;
- same masked-token totals/per-step masked tokens;
- same sequence length, optimizer steps, and LR trajectory.

Tensor differences are expected from dropout RNG grouping under micro-split forward/backward, but data, masks, loss reduction, optimizer-step schedule, and checkpoint accounting are aligned.

S1 memory/config smoke evidence:

- `training/runs/babylm_s1_eb256_micro128_smoke40k/`

This smoke verified:

- DeBERTa-v2 12 layers, hidden 384, 12 heads, intermediate 1280;
- baseline16k tokenizer, flat WWM;
- effective batch 256, micro_batch_size 128;
- lr_total_steps 2442, one optimizer step at 40k words;
- parameter_count 29,329,792 (lower than leader because baseline16k vocab is used);
- standard HF checkpoint saved and config loadable;
- no OOM.

## 10M run command

Run ID: `babylm_leadershape_s1_10M_aligned_micro128`

Scientific interpretation: legal S1 architecture-shape isolation on the official corpus, not leader reproduction. It tests 12×384/intermediate1280 deeper-narrower allocation with baseline16k embeddings and flat WWM. The exact FineWeb simplification-pair leader data remains gated and is not used.

The 10M result should be judged against existing protected WWM learning curves/coordinates and should not be treated as a final SOTA candidate unless evaluation supports it. The 1M/smoke results must not be used to rule out the 12×384 route; this 10M aligned checkpoint is the first meaningful architecture evidence.
