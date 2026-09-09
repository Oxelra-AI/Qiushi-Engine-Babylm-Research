# highlr 10M training completion — high-LR optimizer × data 10M training completion

The four matched LR=0.007, beta2=0.98 runs are present and passed `scripts/verify_highlr_10M_training.py`:

| arm | corpus | optimizer | words | steps | checkpoints | final sampled MLM loss |
|---|---|---|---:|---:|---:|---:|
| `adamw_lr0.007_official_10M_seed43022` | official | AdamW | 10,000,000 | 503 | chck_1M..10M | 6.785714 |
| `lamb_lr0.007_official_10M_seed43022` | official | LAMB | 10,000,000 | 503 | chck_1M..10M | 6.786146 |
| `adamw_lr0.007_qwen_10M_seed43022` | qwen-aligned | AdamW | 10,000,000 | 503 | chck_1M..10M | 6.808361 |
| `lamb_lr0.007_qwen_10M_seed43022` | qwen-aligned | LAMB | 10,000,000 | 503 | chck_1M..10M | 6.808129 |

All four arms share:

- DeBERTa-v2 8×480, FFN 1920, 8 heads, 34,467,424 params;
- vocab 16,384 from the baseline16k tokenizer path;
- batch128, fixed seq256, WWM mask probability 0.15;
- init seed 43022, train RNG seed 43023;
- warmup 25 steps under a 503-step 10M cosine horizon;
- identical checkpoint cumulative-word values: chck_3M=3,002,527; chck_5M=5,012,079; chck_10M=10,000,000.

Corpus hashes match within corpus and differ across corpora:

- official: `312629a8f64498007cdc235a97528a7394dea32b08fc55828f94ee9724d4867c`
- Qwen-aligned: `728192f8e8c5855aaa52a6b6940ff3a4fd6aa8f87c98aca4ff018dbc04cb0345`

The Qwen arms contain 12,236 pair rows and 1,656,800 pair words; official arms contain none.

## Dynamics interpretation before downstream evaluation

The final sampled MLM losses are almost identical between AdamW and LAMB within each corpus at 10M, so any downstream difference should not be interpreted as a simple gross loss/undertraining effect. Conversely, identical final loss does not imply identical representation or transfer.

The stdout trace shows LAMB trust ratios grow far beyond the initial ~0.61 layer scale: after a few million words, many transformer-layer trust ratios are around 40, embeddings are often in the 20s, while the head remains around ~0.82–0.87. This confirms that the tested object is a concrete LAMB optimizer recipe with substantial layer-wise update rescaling, not just a small perturbation of AdamW. It also reinforces the interpretation caveat that a positive interaction would not isolate "trust ratio alone" from the implementation's coupled weight-decay/adaptation behavior.

No-AoA trajectory evaluation was pending in this record. The downstream result, not training loss, will determine whether a fresh 20M four-arm run is scientifically justified.
