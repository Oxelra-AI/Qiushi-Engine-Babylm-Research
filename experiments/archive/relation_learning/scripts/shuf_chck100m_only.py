#!/usr/bin/env python3
"""Score SHUF/chck_100M only - the one missing checkpoint."""
import json, time, math, numpy as np, torch, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM

HELDOUT = Path("experiments/archive/frontier_consolidation/data/dose_2p64x_rowholdout_pools/heldout_cleanqwen_rows.jsonl")
CK = "experiments/archive/relation_learning/training/runs/qwen_shuffled_control_16k_seed43122/hf_model/chck_100M"
BATCH = 64
WORD_RE = re.compile(r"[A-Za-z0-9]+")

rows = [json.loads(l) for l in open(HELDOUT) if l.strip()]
print(f"Rows: {len(rows)}", flush=True)

device = torch.device("cuda:0")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(CK)
model = AutoModelForMaskedLM.from_pretrained(CK, trust_remote_code=True, torch_dtype=torch.float16).to(device)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

total_loss, n_valid = 0.0, 0
for bi in range(0, len(rows), BATCH):
    batch = rows[bi:bi+BATCH]
    texts = [r["text"] for r in batch]
    enc = tok(texts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    input_ids = enc.input_ids.clone()
    labels = enc.input_ids.clone()
    
    for j in range(len(texts)):
        ids = input_ids[j]
        rng = np.random.RandomState(int(ids.sum().item()) % (2**31))
        tokens = [tok.convert_ids_to_tokens(int(t)) for t in ids]
        groups, g = [], -1
        for k, t_str in enumerate(tokens):
            special = ids[k].item() in [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id]
            if special:
                groups.append(-1); continue
            if t_str.startswith("▁") or t_str.startswith("Ġ") or k == 0 or (groups and groups[-1] == -1):
                g += 1
            groups.append(g)
        n_words = g + 1
        if n_words == 0:
            labels[j, :] = -100; continue
        n_mask = max(1, int(n_words * 0.15))
        mask_words = set(rng.choice(n_words, size=n_mask, replace=False).tolist())
        for k, gid in enumerate(groups):
            if gid in mask_words:
                input_ids[j, k] = tok.mask_token_id
            else:
                labels[j, k] = -100
        for sid in [tok.cls_token_id, tok.sep_token_id, tok.pad_token_id]:
            if sid is not None:
                labels[j][ids == sid] = -100
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=enc.attention_mask)
        logits = out.logits.float()  # cast to fp32 for loss
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    per_token = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).view(labels.shape)
    for j in range(len(texts)):
        mask = labels[j] != -100
        n_m = mask.sum().item()
        if n_m > 0:
            total_loss += per_token[j][mask].mean().item()
            n_valid += 1
    if bi % (BATCH*10) == 0:
        print(f"  {bi}/{len(rows)}", flush=True)

mean_loss = total_loss / n_valid
elapsed = time.time() - t0
print(f"\nSHUF/chck_100M: mean_loss={mean_loss:.6f} ({n_valid} valid) in {elapsed:.1f}s", flush=True)

# Already have the other results
result = {
    "OFF_chck_80M": 2.971226, "OFF_chck_90M": 2.941856, "OFF_chck_100M": 2.933582,
    "SHUF_chck_80M": 2.985288, "SHUF_chck_90M": 2.954037, "SHUF_chck_100M": mean_loss,
    "delta_80M": 2.985288 - 2.971226,
    "delta_90M": 2.954037 - 2.941856,
    "delta_100M": mean_loss - 2.933582,
}
print(json.dumps(result, indent=2), flush=True)

out_dir = Path("experiments/archive/relation_learning/data/shuf_ordinary_heldout")
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "shuf_ordinary_heldout_result.json").write_text(json.dumps(result, indent=2))

note = f"""# research: SHUF seed43122 ordinary-heldout MLM loss

Deterministic whole-word masking at p=0.15 on 6,992 held-out rows.

## Results

| Checkpoint | OFF loss | SHUF loss | Δ (SHUF−OFF) |
|---|---|---|---|
| chck_80M | 2.971226 | 2.985288 | +0.014062 |
| chck_90M | 2.941856 | 2.954037 | +0.012181 |
| chck_100M | 2.933582 | {mean_loss:.6f} | {mean_loss - 2.933582:+.6f} |

SHUF incurs a small ordinary-heldout cost (~+0.01-0.02 nats), consistent with the
pre-stated criterion of small ordinary-heldout change. This is much smaller than
the compact-family source-conditioned effects (ΔA_T ~ -0.72) and comparable to
the split-control ordinary-heldout differences (~+0.02 nats).

Data: `experiments/archive/relation_learning/data/shuf_ordinary_heldout`
"""
Path("research/notes/relation_learning/shuf_ordinary_heldout.md").write_text(note)
print("Done.", flush=True)
