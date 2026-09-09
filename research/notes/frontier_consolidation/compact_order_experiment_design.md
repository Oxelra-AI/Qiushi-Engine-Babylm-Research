# compact order experiment design: Compact ordered-vs-scrambled mechanism experiment

## Scientific question

Does coherent word order within compact-view text contribute measurable downstream value at fixed compact lexical multiset, under the stock DeBERTa-v2 MLM coordinate?

## Design

Two arms, trained from scratch, differing ONLY in whether compact-pair view text retains its natural word order or is word-shuffled:

| Arm | Pool | Row count | Words | Description |
|-----|------|-----------|-------|-------------|
| compact_ordered | compact_ordered_40M.jsonl | 258,960 | 40,000,000 | 10M pool × 4 passes, original compact-view text order |
| compact_scrambled | compact_scrambled_40M.jsonl | 258,960 | 40,000,000 | 10M pool × 4 passes, compact-view text word-scrambled |

Both pools:
- Identical non-pair rows (same text, same order)
- Identical pair source text
- Same word multiset within every compact-pair view row
- Same example_id, source labels, word counts per row
- Pool SHAs recorded at build time in lead route assessment after source use probe scaffold

## Pool provenance (from lead route assessment after source use probe)

- compact_ordered_10M SHA = original 10M pool SHA `215944978157394dbecf2039f2c1e1806bfcbb9701421920f78605eb58975a23`
- compact_ordered_40M SHA `48313173a4cd93e768f490c8a5eaaa850fc828fabc9cfb21ebad73651bef85dc` (258,960 rows, 40M words)
- compact_scrambled_10M SHA `07101d1391fbc1090f1318ead2c780a8a0071f98e2da06b6be30e9a19ca14b1d`
- compact_scrambled_40M SHA `7b8dcd13655ab3208938ff1cdc633bb1a695cd2d1765700256b052c96fda26e3` (258,960 rows, 40M words)

## Why both arms are trained fresh

The spatial repair route status reference used `cleanqwen_fineweb_compact_view_reinvest_100M.jsonl` (SHA `3dd19f09...`), which has a DIFFERENT row order than the 10M pool × 10. Empirically, 647,387 of 647,400 lines differ from 10M-cycled positions. Same unique example_id sets per cycle but shuffled ordering. Therefore the spatial repair route status chck_40M is NOT comparable to compact_ordered_40M — a fresh ordered arm is required.

## Training configuration (both arms identical except pool)

Exact match to compliant tokenizer retrain status legal reference recipe:
- Trainer: `experiments/archive/compact_experience/scripts/masking_curriculum_trainer.py`
- Tokenizer: `experiments/archive/frontier_consolidation/data/compliant_tokenizer` (vocab 16384, SHA `91b775...`)
- Architecture: DeBERTa-v2 8×480, n_head=8, ffn_mult=4 (stock, no adapter)
- Seeds: --seed 43, --extra_init_seed 43022, --train_rng_seed 43023
- Optimizer: AdamW lr=0.001, wd=0.01, betas=(0.9, 0.98), warmup=0.06
- Masking: wwm_fixed, p=0.15
- Batch: 256, seq_length=256, max_seq_length=256
- Exposure: **--max_word_exposure 40000000** (40M words = 4 epochs)
- Checkpoint: **--checkpoint_words 20000000** (saves at 20M and 40M)
- LR schedule: **--lr_total_steps 2529** (100M horizon, NOT 40M)
  - This makes the LR at 40M identical to spatial repair route status's 40M-checkpoint LR
  - Prevents confounding order effect with schedule compression
- GPU: ordered on GPU0, scrambled on GPU1 (parallel)

## Interpretation framework

### What each outcome establishes

| Outcome | Meaning | Does NOT prove |
|---------|---------|----------------|
| ordered >> scrambled | Coherent compact-view order has downstream value | Not that fluent syntax is the dominant mechanism; tail coverage, budget efficiency, semantic transformation remain coupled |
| ordered ≈ scrambled | Order within compact views does not matter | Not that compact views are irrelevant (lexical recurrence, content coverage, budget reallocation untested) |
| scrambled >> ordered | Unexpected — debug | — |

### Decision metrics (not GlobalPIQA/Reading carried)

Primary: cheap6 = (BLiMP + Supplement + EWoK + Entity + COMPS + Reading) / 6 [without GlobalPIQA]
Secondary: cheap5 = (BLiMP + Supplement + EWoK + Entity + COMPS) / 5 [without GlobalPIQA/Reading]
Relation/state: (EWoK + Entity) / 2
Column-level: Supplement, Entity, COMPS individually

GlobalPIQA and Reading movements noted but do not drive the decision.

### Thresholds and extension rules

| |Δcheap6| at 40M | Decision |
|---|---|
| ≥ 0.3 | Decisive. Record direction and close the comparison. |
| 0.1 – 0.3 | Suggestive. Extend both arms to 80M (8 passes) for confirmation. |
| < 0.1 | No order effect at this budget. Check 20M: if order matters early but washes out, extend to 80M; otherwise close and remove ordered structure from the compact-view mechanism. |

### Extension to 80M (if needed)

Same pools extended to 80M words (8 passes of 10M). Same settings except `--max_word_exposure 80000000`. Score at 40M/60M/80M with checkpoint_words=20000000. Interpret using same cheap6/cheap5/relation-state criteria at each point.

## Evaluation plan

After training completes:
1. Run selected cheap evaluation on chck_20M and chck_40M for both arms
2. Check integrity: verify exact word exposure, model param counts, loss curves
3. Compare cheap6, cheap5, relation/state, column-level for ordered-minus-scrambled
4. Note spatial repair route status reference chck_40M cheap7=42.22 as sanity anchor (different row order, not directly comparable)
5. If decisive, record finding and route next mechanism work
6. If ambiguous, extend per rules above

## What this experiment cannot do

- Cannot separate tail coverage from content density from semantic transformation (all three are properties of the compact views, preserved identically in ordered vs scrambled)
- Cannot establish whether compact-view gains transfer to other architectures (GPT2 causal negative already in evidence)
- Cannot determine whether the effect is mature or early-only at 40M (the validated reinvestment benefit emerged late; if ordering matters early but washes out, it may not be the load-bearing mechanism)

## Files

- Pools: `data/compact_order_factorial_pool_scaffold/`
- Launcher: `scripts/compact_order_mechanism_experiment.py`
- This note: `notes/compact_order_experiment_design.md`
