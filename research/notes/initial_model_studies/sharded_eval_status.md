# sharded eval status — Sharded evaluation throughput repair (SCORE FREEZE)

## Status: evaluation infrastructure repaired; 9/9 Overall in progress

### AoA — COMPLETE (throughput bottleneck solved)
- Old official loop: ~15-16 min/checkpoint, one forward per context, no partial output → repeatedly killed at code 143.
- New batched scorer `score_freeze_sharded_eval.py aoa_shard`: preserves official causal phrase-mask
  surprisal (context.strip()+' '+word, offset-based target span, shift [:-1]/[1:], sum logprob over phrase_mask[1:]),
  but batches contexts on GPU and writes one durable JSON per checkpoint.
- Result: all 19 checkpoints done at ~18s each. Each part has 6,560 entries (328 words × 20 contexts).
- Parts: `training/runs/recgpt_official_aoa_sharded/parts/surprisal_chck_{1..9,10..100}M.json`

### SuperGLUE — RUNNING across both H100s (durable per-task)
- Root cause of prior boolq failure: official `classifier_model.py` used `AutoModel.from_pretrained`, but RecGPT
  registers only `AutoModelForCausalLM`. Patched (eval-only) to fall back to AutoModelForCausalLM, request
  hidden_states, and classify from final hidden state. Verified: finite logits shape [1,2] on GPU.
- boolq (GPU0, task T586...): confirmed training, 5890 steps @ ~1.7 it/s, patched classifier works.
- GPU0 chain T26AF...: multirc, qqp, mnli
- GPU1 chain TC14...: rte, wsc, mrpc
- Each task writes `training/runs/recgpt_official_superglue_sharded/task_json/<task>.json`.

### Aggregation
- `score_freeze_sharded_eval.py aggregate` combines: base 7/9
  (`data/recgpt_official_final_causal_scores.json`), AoA (via official AoAEvaluator over the
  19 parts), and SuperGLUE (mean of 7 task accuracies) → `data/recgpt_official_full_overall.json`.
- Compare to protected DeBERTa Overall 40.5269 and public leader 41.8011.

### Known caveat carried forward
- Reading (-4.89 vs leaderboard after space-fix) and GlobalPIQA (-1.97) local-vs-public evaluator offsets
  remain documented for the RecGPT reference and must stay explicit in Overall interpretation.

### Frozen candidate 7/9 (already weak)
BLiMP 55.13, Supplement 44.23, Entity 16.38, COMPS 50.70, GlobalPIQA mean 32.725, EWoK 50.49, Reading mean 0.215.
NLP-7 mean 35.70. Far below protected DeBERTa on every measured column.
