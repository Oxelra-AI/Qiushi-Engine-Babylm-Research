# debertav2 official40k smoke — official40k DeBERTa launch decision

## Evidence collected this step

1. Four-point update-geometry comparison persisted:
   - `data/update_geometry_coordinate_comparison.json`
   - `notes/update_geometry_coordinate_comparison.md`

   Matched BERT b256 did not reproduce DeBERTa gains. At matched batch/update geometry, DeBERTa-v2 b256 minus BERT b256 is:

   - BLiMP +12.51
   - Supplement +8.97
   - Entity +6.03
   - COMPS +2.08
   - GlobalPIQA parallel +3.88
   - GlobalPIQA mean −0.56 because BERT b256 nonparallel is higher

   This confirms the DeBERTa-v2 package/backbone, not doubled updates alone, is the protected route.

2. official40k tokenizer smoke:
   - script: `scripts/run_debertav2_official40k_smoke.py`
   - smoke run: `training/runs/babylm_smoke_debertav2_8x480_official40k_20k/`
   - evidence JSON: `data/debertav2_official40k_smoke.json`

   The smoke training/save/load succeeded. Only the post-run note writer failed first because it looked for a non-existent key `kept_tokens_per_word`; the data file exists and records the correct key `kept_tokens_per_whitespace_word`.

   Key smoke facts:

   - tokenizer: `training/tokenizers/official40k`, 40,000 vocab, portable fast tokenizer
   - model: DeBERTa-v2 8×480, 8 heads, p2c/c2p relative attention
   - parameters: 45,826,720 total; 19,200,000 embedding; 26,626,720 non-embedding
   - 20k-word smoke: 32 optimizer steps, loss 10.6119 → 6.7070
   - loadability: root and `chck_1M` load via `PreTrainedTokenizerFast` + `DebertaV2ForMaskedLM`
   - tokenization: 1.19765 kept tokens per whitespace word in smoke, no truncation at 20k smoke scale

3. official40k parameter geometry probe:
   - script: `scripts/probe_deberta40k_param_configs.py`

   Results show two possible experiment styles:

   - **Same successful transformer backbone:** hidden 480 with official40k preserves non-embedding capacity (26,626,720 vs 26,603,104 baseline16k) but increases total params to 45,826,720 through embeddings.
   - **Total-param matched:** hidden 400 with official40k gives total 34,579,600, close to baseline DeBERTa 34,467,424, but reduces non-embedding transformer capacity to 18,579,600 (about −8.0M relative to the successful backbone).

## Decision

Launch the full run with **same 8×480 DeBERTa-v2 backbone + official40k tokenizer**, not hidden-400.

Reasoning:

- The experiment targets official BabyLM Strict-Small Overall. The recorded constraints are word budget/exposure/epochs/tokenizer accounting, not a model parameter cap.
- The breakthrough came from the DeBERTa-v2 package at ~26.6M non-embedding parameters. Reducing hidden size to 400 would mix tokenizer with a large transformer-capacity cut, making any failure uninterpretable and likely weakening the near-leading BLiMP/Supplement/Entity gains.
- Keeping hidden 480 makes the scientific intervention clearer: same DeBERTa transformer capacity, same data/objective/update geometry, tokenizer/vocabulary/embedding allocation changed. The total-parameter increase is real and must be reported as part of the intervention.
- The 40k analysis showed reduced content/entity-token fragmentation and lower tokens/word; this is the best current single variable likely to help Entity, GlobalPIQA, and possibly EWoK without copying simplification/curriculum.

Interpretation boundary: any gain from this run is a **tokenizer + embedding-capacity + segmentation** effect, not pure tokenization. If it improves substantially, a later hidden-400 or embedding-factorized control can separate total-param fairness from performance route. If it fails or harms BLiMP/Supplement, do not continue blindly with 40k; return to entity/relation objective or data mechanism.

## Full run launched

Run id planned/launched:

`babylm_fullcycle_debertav2_8x480_official40k_wwm_seed42_100M_b256`

Matched to DeBERTa b256 baseline except tokenizer:

- official 10M words, 100M exposure, 10 passes
- WWM 0.15, fixed length 256
- DeBERTa-v2 8×480, 8 heads, FFN 1920, p2c/c2p
- batch 256, `lr_total_steps=2442`, lr 0.001
- seeds 42/456/789
- tokenizer path `training/tokenizers/official40k`, label `official40k`
