# 34m coordinate comparison — DeBERTa-v2 8×480 batch-256 rescue validation

Run: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/`

This is the rescue run launched after the matched batch-512 DeBERTa-v2 arm failed by CUDA OOM in disentangled relative attention. It is a full-scale system run but not a clean A/B match to BERT 8×512 because it uses `batch_size=256` and `lr_total_steps=2442`, whereas the BERT capacity control used `batch_size=512` and `lr_total_steps=1221`.

Validated from `scientific_metrics.json`, `example_order_manifest.json`, `training_log.jsonl`, `tokenization_coupling_summary.json`, and AutoModel/AutoTokenizer load checks:

- model: `DebertaV2ForMaskedLM`
- architecture: 8 layers, hidden 480, 8 heads, FFN 1920, relative attention true, `p2c,c2p`, 256 buckets / relative positions
- parameters: 34,467,424 total; 7,864,320 embedding; 26,603,104 non-embedding
- tokenizer/data/objective: baseline16k, official 10M corpus, full-cycle 10 passes, WWM 0.15, fixed length 256
- exposure: 100,000,000 selected/exposed words
- optimizer geometry: 2,442 optimizer steps, batch size 256, `lr_total_steps=2442`
- loss: 9.7792 → 2.6975
- checkpoints: 100 checkpoint directories `chck_1M` … `chck_100M`
- loadability: root, `chck_1M`, `chck_10M`, `chck_50M`, and `chck_100M` all load with `PreTrainedTokenizerFast` + `DebertaV2ForMaskedLM` and 34,467,424 params
- tokenization/truncation: same as WWM100M/BERT34 arms: 625,000 examples, 148,966,880 untruncated tokens, 144,444,290 kept tokens, 29.2048% truncated at length 256, 21,672,238 selected masked tokens

Active evaluation started in 34m coordinate comparison:

- runner: `scripts/run_available_coordinate_for_model.py`
- output JSON target: `data/debertav2_b256_available_coordinate.json`
- note target: `notes/debertav2_b256_available_coordinate.md`
- log target: `notes/debertav2_b256_available_coordinate.log`
- raw eval output: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M_b256/eval_results_available/`

At the last 34m coordinate comparison status check the evaluator was running the first BLiMP task. Once it completes, run `scripts/compare_34m_coordinates.py` to join the 10.7M BERT, 34M BERT, and DeBERTa-v2 batch-256 coordinates.
