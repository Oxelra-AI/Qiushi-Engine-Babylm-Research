# 34m training validation and deberta repair — 34M training validation and DeBERTa OOM repair plan

## BERT 8×512 full run validated

Run: `training/runs/babylm_fullcycle_bert8x512_wwm_seed42_100M/`

Verified from files and AutoModel loading:

- model: `BertForMaskedLM`, 34,151,936 parameters, 25,763,328 non-embedding parameters
- exposure: 100,000,000 words, official-corpus fullcycle, 10 × 10M-word passes
- updates: 1,221 optimizer steps with `batch_size=512`, `lr_total_steps=1221`
- loss: 9.7866 → 6.7510
- checkpoints: 100 directories `chck_1M` … `chck_100M`
- loadability: root, `chck_1M`, `chck_10M`, `chck_50M`, `chck_100M` all load with `PreTrainedTokenizerFast` + `BertForMaskedLM`
- tokenization/truncation matches the protected WWM100M baseline: 29.2048% examples truncated at length 256, 21,671,247 masked tokens

This is a valid capacity arm for comparison against the 10.7M BERT-WWM baseline.

## DeBERTa-v2 8×480 failed by OOM at batch 512

Run: `training/runs/babylm_fullcycle_debertav2_8x480_wwm_seed42_100M/`

Failure is specifically CUDA OOM during the first forward pass in DeBERTa-v2 disentangled relative attention:

- location: `modeling_deberta_v2.py`, `disentangled_attention_bias`, `torch.gather`
- attempted allocation: 1024 MiB
- GPU 0 memory in use: 78.75 GiB / 79.18 GiB
- no `training_log.jsonl` entries; data manifest, raw dataset, tokenization summary were produced

This is not negative scientific evidence about DeBERTa; it is a batch-memory failure. The 20k smoke already showed the architecture can train/save/load.

## Repair choice

The current trusted full-cycle trainer has no gradient accumulation support. A clean matched comparison would require a trainer clone with microbatch 128 or 256 and gradient accumulation to preserve effective batch 512 and 1,221 optimizer updates.

For immediate evidence, 34m training validation and deberta repair launches a recorded rescue DeBERTa-v2 run at `batch_size=256`, `lr_total_steps=2442` (same 100M exposure, but twice the optimizer updates). This is a useful full-scale DeBERTa system coordinate, but it is not a perfectly matched A/B against BERT 8×512. If it improves key columns, later work should build the accumulation trainer or a matched BERT batch-256 control to locate update-geometry effects.
